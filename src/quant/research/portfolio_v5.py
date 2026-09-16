"""Leakage-safe labelled portfolio evaluation and V5 experiment matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .leakage import assert_label_after_signal


@dataclass(frozen=True)
class V5PortfolioResult:
    metrics: dict[str, float]
    results: pd.DataFrame
    holdings: pd.DataFrame


class V5PortfolioEvaluator:
    """Evaluate precomputed signal scores on forward labels with explicit costs and constraints."""

    def __init__(self, base_transaction_cost: float = 0.0018) -> None:
        if base_transaction_cost < 0:
            raise ValueError("base_transaction_cost must be non-negative")
        self.base_transaction_cost = base_transaction_cost

    def evaluate(
        self,
        panel: pd.DataFrame,
        *,
        score_column: str = "ensemble_score",
        horizon: int = 20,
        top_n: int = 20,
        rebalance_frequency: int = 20,
        weighting: str = "equal_weight",
        industry_limit: float = 0.20,
        max_stock_weight: float = 0.10,
        risk_overlay: str = "v4_baseline",
        transaction_cost_multiplier: float = 1.0,
    ) -> V5PortfolioResult:
        """Evaluate fixed score outputs; labels are used only after their maturity date."""
        if top_n < 1 or rebalance_frequency < 1:
            raise ValueError("top_n and rebalance_frequency must be positive")
        if weighting not in {"equal_weight", "score_proportional", "volatility_adjusted"}:
            raise ValueError("unsupported portfolio weighting")
        if risk_overlay not in {"off", "v4_baseline", "regime_only"}:
            raise ValueError("unsupported risk overlay")
        if not 0 < industry_limit <= 1 or not 0 < max_stock_weight <= 1 or transaction_cost_multiplier < 0:
            raise ValueError("invalid portfolio constraint or transaction-cost multiplier")
        label, end_date = f"future_return_{horizon}d", f"label_end_date_{horizon}d"
        benchmark_label = f"benchmark_future_return_{horizon}d"
        required = {"date", "code", "industry", score_column, label, benchmark_label, end_date, "market_regime", "volatility_regime"}
        missing = required.difference(panel.columns)
        if missing:
            raise ValueError(f"portfolio panel missing: {', '.join(sorted(missing))}")
        # The matrix must not duplicate the complete feature panel for every
        # scenario.  Portfolio construction only needs these audit columns.
        optional = {"amount_20", "realized_vol_20"}
        keep = list(required | (optional & set(panel.columns)))
        frame = panel.loc[:, keep].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame[end_date] = pd.to_datetime(frame[end_date], errors="coerce")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        valid = frame[end_date].notna()
        assert_label_after_signal(frame.loc[valid, "date"], frame.loc[valid, end_date])
        date_groups = {date: values for date, values in frame.loc[valid].groupby("date", sort=True)}
        return self._evaluate_groups(
            date_groups, score_column=score_column, horizon=horizon, top_n=top_n,
            rebalance_frequency=rebalance_frequency, weighting=weighting, industry_limit=industry_limit,
            max_stock_weight=max_stock_weight, risk_overlay=risk_overlay,
            transaction_cost_multiplier=transaction_cost_multiplier,
        )

    def _evaluate_groups(
        self,
        date_groups: dict[pd.Timestamp, pd.DataFrame],
        *,
        score_column: str,
        horizon: int,
        top_n: int,
        rebalance_frequency: int,
        weighting: str,
        industry_limit: float,
        max_stock_weight: float,
        risk_overlay: str,
        transaction_cost_multiplier: float,
    ) -> V5PortfolioResult:
        label, end_date = f"future_return_{horizon}d", f"label_end_date_{horizon}d"
        benchmark_label = f"benchmark_future_return_{horizon}d"
        dates = sorted(date_groups)[::rebalance_frequency]
        previous: dict[str, float] = {}
        equity = 1.0
        records: list[dict[str, object]] = []
        holdings: list[dict[str, object]] = []
        for date in dates:
            current = date_groups[date]
            selected = self._select(current, score_column, top_n, industry_limit)
            weights = self._weights(selected, score_column, weighting, max_stock_weight)
            exposure = _risk_exposure(selected, risk_overlay)
            weights *= exposure
            target = dict(zip(selected["code"], weights, strict=False))
            turnover = _turnover(previous, target)
            portfolio_return = float((weights * pd.to_numeric(selected[label], errors="coerce")).sum())
            net_return = portfolio_return - turnover * self.base_transaction_cost * transaction_cost_multiplier
            benchmark_return = float(pd.to_numeric(selected[benchmark_label], errors="coerce").dropna().iloc[0]) if selected[benchmark_label].notna().any() else np.nan
            equity *= 1 + net_return
            records.append(
                {
                    "date": date,
                    "label_end_date": selected[end_date].max(),
                    "portfolio_return": net_return,
                    "gross_return": portfolio_return,
                    "benchmark_return": benchmark_return,
                    "equity": equity,
                    "turnover": turnover,
                    "exposure": exposure,
                    "market_regime": str(selected["market_regime"].iloc[0]),
                    "volatility_regime": str(selected["volatility_regime"].iloc[0]),
                }
            )
            for row, weight in zip(selected.itertuples(index=False), weights, strict=False):
                holdings.append({"date": date, "code": row.code, "industry": row.industry, "weight": float(weight), "score": float(getattr(row, score_column)), "amount_20": float(getattr(row, "amount_20", np.nan))})
            previous = target
        results = pd.DataFrame(records)
        return V5PortfolioResult(_metrics(results, rebalance_frequency), results, pd.DataFrame(holdings))

    @staticmethod
    def _select(frame: pd.DataFrame, score_column: str, top_n: int, industry_limit: float) -> pd.DataFrame:
        current = frame.copy()
        current[score_column] = pd.to_numeric(current[score_column], errors="coerce")
        current = current.dropna(subset=[score_column]).sort_values([score_column, "code"], ascending=[False, True])
        per_industry = max(1, int(np.floor(top_n * industry_limit)))
        selected: list[pd.Series] = []
        counts: dict[str, int] = {}
        for _, row in current.iterrows():
            industry = str(row["industry"])
            if counts.get(industry, 0) >= per_industry:
                continue
            selected.append(row)
            counts[industry] = counts.get(industry, 0) + 1
            if len(selected) == top_n:
                break
        return pd.DataFrame(selected) if selected else current.head(0)

    @staticmethod
    def _weights(selected: pd.DataFrame, score_column: str, weighting: str, cap: float) -> np.ndarray:
        if selected.empty:
            return np.array([], dtype=float)
        if weighting == "equal_weight":
            raw = np.ones(len(selected), dtype=float)
        elif weighting == "score_proportional":
            values = pd.to_numeric(selected[score_column], errors="coerce").to_numpy(dtype=float)
            raw = values - np.nanmin(values) + 1e-8
        else:
            volatility = pd.to_numeric(selected.get("realized_vol_20", pd.Series(np.nan, index=selected.index)), errors="coerce").to_numpy(dtype=float)
            raw = 1 / np.where(np.isfinite(volatility) & (volatility > 0), volatility, np.nan)
            raw = np.where(np.isfinite(raw), raw, 0.0)
        weights = raw / raw.sum() if raw.sum() > 0 else np.full(len(selected), 1 / len(selected))
        return _capped_weights(weights, cap)


def run_experiment_matrix(
    evaluator: V5PortfolioEvaluator,
    panel: pd.DataFrame,
    *,
    score_column: str = "ensemble_score",
    horizon: int = 20,
) -> pd.DataFrame:
    """Store all requested portfolio configurations, including weak candidates."""
    label_end = f"label_end_date_{horizon}d"
    label = f"future_return_{horizon}d"
    benchmark_label = f"benchmark_future_return_{horizon}d"
    required = {
        "date", "code", "industry", score_column, label, benchmark_label, label_end,
        "market_regime", "volatility_regime",
    }
    optional = {"amount_20", "realized_vol_20"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"portfolio matrix panel missing: {', '.join(sorted(missing))}")
    prepared = panel.loc[:, list(required | (optional & set(panel.columns)))].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise").dt.normalize()
    prepared[label_end] = pd.to_datetime(prepared[label_end], errors="coerce")
    date_groups = {
        date: values for date, values in prepared.loc[prepared[label_end].notna()].groupby("date", sort=True)
    }
    records: list[dict[str, object]] = []
    for top_n in (10, 20, 30, 50):
        for rebalance in (5, 10, 20, 40):
            for weighting in ("equal_weight", "score_proportional", "volatility_adjusted"):
                for industry_limit in (0.10, 0.15, 0.20, 0.25):
                    for max_weight in (0.05, 0.075, 0.10):
                        for overlay in ("off", "v4_baseline", "regime_only"):
                            result = evaluator._evaluate_groups(
                                date_groups, score_column=score_column, horizon=horizon, top_n=top_n,
                                rebalance_frequency=rebalance, weighting=weighting, industry_limit=industry_limit,
                                max_stock_weight=max_weight, risk_overlay=overlay,
                            )
                            records.append({
                                "top_n": top_n, "rebalance_frequency": rebalance, "weighting": weighting,
                                "industry_limit": industry_limit, "max_stock_weight": max_weight,
                                "risk_overlay": overlay, "stability": float((result.results["portfolio_return"] > 0).mean()),
                                **result.metrics,
                            })
    matrix = pd.DataFrame(records)
    if matrix.empty:
        return matrix
    components = (
        matrix["sharpe"].rank(pct=True).fillna(0)
        + matrix["alpha"].rank(pct=True).fillna(0)
        + (-matrix["max_drawdown"]).rank(pct=True).fillna(0)
        + (-matrix["turnover"]).rank(pct=True).fillna(0)
        + matrix["stability"].rank(pct=True).fillna(0)
    )
    matrix["research_rank"] = components.rank(ascending=False, method="dense").astype(int)
    return matrix.sort_values("research_rank").reset_index(drop=True)


def transaction_cost_stress(
    evaluator: V5PortfolioEvaluator,
    panel: pd.DataFrame,
    *,
    score_column: str = "ensemble_score",
    horizon: int = 20,
) -> pd.DataFrame:
    """Measure all requested cost multipliers without dropping unfavorable scenarios."""
    rows: list[dict[str, object]] = []
    for multiplier in (0.5, 1.0, 1.5, 2.0, 3.0):
        result = evaluator.evaluate(
            panel, score_column=score_column, horizon=horizon, transaction_cost_multiplier=multiplier
        )
        rows.append({"transaction_cost_multiplier": multiplier, **result.metrics})
    return pd.DataFrame(rows)


def _capped_weights(weights: np.ndarray, cap: float) -> np.ndarray:
    output = np.minimum(weights, cap)
    for _ in range(len(output)):
        remaining = 1 - output.sum()
        eligible = output < cap - 1e-12
        if remaining <= 1e-12 or not eligible.any():
            break
        allocation = weights[eligible]
        allocation = allocation / allocation.sum() if allocation.sum() > 0 else np.full(eligible.sum(), 1 / eligible.sum())
        output[eligible] = np.minimum(cap, output[eligible] + remaining * allocation)
    return output


def _risk_exposure(selected: pd.DataFrame, overlay: str) -> float:
    if overlay == "off" or selected.empty:
        return 1.0
    regime = str(selected["market_regime"].iloc[0])
    exposure = {"bull": 1.0, "sideways": 0.6, "bear": 0.3}.get(regime, 0.6)
    if overlay == "v4_baseline" and str(selected["volatility_regime"].iloc[0]) == "high":
        exposure *= 0.5
    return exposure


def _turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    return 0.5 * sum(abs(previous.get(code, 0.0) - current.get(code, 0.0)) for code in set(previous).union(current))


def _metrics(results: pd.DataFrame, period_sessions: int) -> dict[str, float]:
    if results.empty:
        return {name: np.nan for name in ("annual_return", "sharpe", "max_drawdown", "alpha", "beta", "turnover", "information_ratio")}
    returns = results["portfolio_return"].dropna()
    annualization = 252 / period_sessions
    annual_return = float((1 + returns).prod() ** (annualization / len(returns)) - 1) if len(returns) else np.nan
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(annualization)) if len(returns) > 1 and returns.std(ddof=0) > 0 else np.nan
    drawdown = float((results["equity"] / results["equity"].cummax() - 1).min())
    aligned = results.dropna(subset=["portfolio_return", "benchmark_return"])
    variance = aligned["benchmark_return"].var(ddof=0)
    beta = float(aligned["portfolio_return"].cov(aligned["benchmark_return"], ddof=0) / variance) if len(aligned) > 1 and variance > 0 else np.nan
    alpha = float((aligned["portfolio_return"] - beta * aligned["benchmark_return"]).mean() * annualization) if pd.notna(beta) else np.nan
    excess = aligned["portfolio_return"] - aligned["benchmark_return"]
    information_ratio = float(excess.mean() / excess.std(ddof=0) * np.sqrt(annualization)) if len(excess) > 1 and excess.std(ddof=0) > 0 else np.nan
    return {"annual_return": annual_return, "sharpe": sharpe, "max_drawdown": drawdown, "alpha": alpha, "beta": beta, "turnover": float(results["turnover"].sum()), "information_ratio": information_ratio}
