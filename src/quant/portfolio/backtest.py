"""Risk-aware portfolio-construction backtest for v1.0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..backtest.diagnostics import drawdown_periods, worst_holding_periods, yearly_attribution
from ..backtest.metrics import calculate_metrics, drawdown_series, monthly_returns
from ..backtest.portfolio import Portfolio, TradingCosts
from ..data.universe import Universe
from ..research import FactorEvaluator, FactorResearchDataBuilder
from ..risk import RiskController, RiskSettings
from .optimizer import EqualWeightOptimizer
from .scoring import CompositeScorer


@dataclass
class PortfolioBacktestResult:
    """Outputs from the v1.0 construction and risk-control workflow."""

    portfolio: Portfolio
    equity_curve: pd.DataFrame
    benchmark_curve: pd.DataFrame
    metrics: dict[str, float]
    scores: pd.DataFrame
    holdings_history: pd.DataFrame
    sector_exposure: pd.DataFrame
    factor_contribution: pd.DataFrame
    yearly_attribution: pd.DataFrame
    drawdown_series: pd.DataFrame
    drawdown_periods: pd.DataFrame
    worst_holding_periods: pd.DataFrame
    monthly_returns: pd.DataFrame


class PortfolioBacktestEngine:
    """Compose factor processing, equal weights, and risk controls into a backtest."""

    def __init__(
        self,
        features_dir: str | Path = "data/features",
        raw_dir: str | Path | None = None,
        factor_config_path: str | Path = "configs/factor_weights.yaml",
        initial_cash: float = 100_000.0,
        rebalance_interval: int = 20,
        top_n: int = 20,
        benchmark_code: str = "000300",
        benchmark_path: str | Path | None = None,
        risk_settings: RiskSettings | None = None,
        universe: Universe | None = None,
        commission: float = 0.0003,
        stamp_tax: float = 0.0005,
        slippage: float = 0.001,
    ) -> None:
        if rebalance_interval < 1:
            raise ValueError("rebalance_interval must be at least 1")
        self.features_dir = Path(features_dir)
        self.raw_dir = Path(raw_dir) if raw_dir is not None else self.features_dir.parent / "raw"
        self.initial_cash = initial_cash
        self.rebalance_interval = rebalance_interval
        self.benchmark_code = str(benchmark_code).zfill(6)
        self.benchmark_path = Path(benchmark_path) if benchmark_path is not None else None
        self.scorer = CompositeScorer.from_yaml(factor_config_path)
        self.optimizer = EqualWeightOptimizer(top_n=top_n)
        self.risk = RiskController(risk_settings)
        self.universe = universe or Universe()
        self.costs = TradingCosts(commission=commission, stamp_tax=stamp_tax, slippage=slippage)

    def run(self) -> PortfolioBacktestResult:
        """Run v1.0, trading each signal at the following session's open."""
        panel = self._load_feature_panel()
        scores = self._score_panel(panel)
        dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        targets = self._rebalance_targets(scores, dates)
        benchmark_prices = self._load_benchmark_prices(dates)
        portfolio = Portfolio(initial_cash=self.initial_cash, costs=self.costs)
        holdings_records: list[dict[str, object]] = []
        prior_weights: dict[str, float] = {}
        turnover = 0.0

        for current_date in dates:
            day = panel.loc[panel["date"] == current_date].set_index("code")
            if current_date in targets:
                signal_date, selected = targets[current_date]
                raw_weights = selected.set_index("code")["weight"].to_dict()
                adjusted_weights, exposure = self.risk.target_weights(
                    raw_weights,
                    benchmark_prices,
                    portfolio.equity_frame(),
                    as_of=signal_date,
                )
                turnover += _target_turnover(prior_weights, adjusted_weights)
                prior_weights = adjusted_weights
                portfolio.rebalance_weights(adjusted_weights, day["open"].to_dict())
                self._record_holdings(
                    holdings_records, current_date, selected, adjusted_weights, exposure
                )
            portfolio.record(current_date, day["close"].to_dict())

        equity_curve = portfolio.equity_frame()
        benchmark_curve = self._benchmark_curve(benchmark_prices, dates)
        metrics = calculate_metrics(equity_curve, benchmark_curve)
        metrics["volatility"] = metrics["annual_volatility"]
        metrics["turnover"] = float(turnover)
        holdings_history = pd.DataFrame.from_records(
            holdings_records,
            columns=("date", "code", "weight", "composite_score", "sector", "exposure"),
        )
        sector_exposure = self._sector_exposure(holdings_history)

        research_panel = FactorResearchDataBuilder(
            features_dir=self.features_dir, raw_dir=self.raw_dir
        ).build()
        factor_contribution = FactorEvaluator(
            factors=self.scorer.processor.factor_names
        ).contribution_analysis(research_panel)
        return PortfolioBacktestResult(
            portfolio=portfolio,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            metrics=metrics,
            scores=scores,
            holdings_history=holdings_history,
            sector_exposure=sector_exposure,
            factor_contribution=factor_contribution,
            yearly_attribution=yearly_attribution(equity_curve, benchmark_curve),
            drawdown_series=drawdown_series(equity_curve),
            drawdown_periods=drawdown_periods(equity_curve),
            worst_holding_periods=worst_holding_periods(equity_curve),
            monthly_returns=monthly_returns(equity_curve, benchmark_curve),
        )

    def _load_feature_panel(self) -> pd.DataFrame:
        paths = sorted(
            path
            for path in self.features_dir.glob("*.parquet")
            if path.stem.isdigit() and len(path.stem) == 6
        )
        if not paths:
            raise RuntimeError(f"no per-stock feature files found in {self.features_dir}")
        frames: list[pd.DataFrame] = []
        for path in paths:
            frame = pd.read_parquet(path)
            required = {"date", "open", "close"}
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"{path.name} missing feature columns: {', '.join(missing)}")
            frame = self.scorer.processor.align_factor_columns(frame)
            columns = ["date", "open", "close", *self.scorer.processor.factor_names]
            frame = frame.loc[:, columns].copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            for column in columns[1:]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["code"] = path.stem
            frames.append(frame)

        panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
        common_dates = panel.groupby("date")["code"].nunique()
        common_dates = common_dates[common_dates == len(paths)].index
        if len(common_dates) < 2:
            raise RuntimeError("feature files do not share enough trading dates for backtesting")
        return panel.loc[panel["date"].isin(common_dates)].reset_index(drop=True)

    def _score_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Build v1.0 composite scores; subclasses may supply other alpha scores."""
        return self.scorer.score_and_store(panel, self.features_dir / "composite_score.parquet")

    def _rebalance_targets(
        self, scores: pd.DataFrame, dates: pd.DatetimeIndex
    ) -> dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]]:
        targets: dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]] = {}
        for index in range(0, len(dates) - 1, self.rebalance_interval):
            signal_date = dates[index]
            selected = self.optimizer.construct(scores.loc[scores["date"] == signal_date])
            if not selected.empty:
                targets[dates[index + 1]] = (signal_date, selected)
        return targets

    def _load_benchmark_prices(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        path = self._benchmark_file()
        frame = pd.read_parquet(path)
        if not {"date", "close"}.issubset(frame.columns):
            raise ValueError(f"{path.name} benchmark requires date and close columns")
        benchmark = frame.loc[:, ["date", "close"]].copy()
        benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise")
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
        benchmark = benchmark.dropna().drop_duplicates("date", keep="last").set_index("date")
        benchmark = benchmark.reindex(dates)
        benchmark.index.name = "date"
        if benchmark["close"].isna().any():
            raise ValueError(f"{self.benchmark_code} benchmark is missing strategy trading dates")
        return benchmark.reset_index()

    def _benchmark_curve(
        self, benchmark_prices: pd.DataFrame, dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        normalized = self.initial_cash * benchmark_prices["close"] / benchmark_prices["close"].iloc[0]
        return pd.DataFrame({"date": dates, "benchmark": normalized.to_numpy()})

    def _benchmark_file(self) -> Path:
        if self.benchmark_path is not None:
            return self.benchmark_path
        direct = self.raw_dir / f"{self.benchmark_code}.parquet"
        if direct.exists():
            return direct
        legacy = self.raw_dir / "index" / "benchmark.parquet"
        if legacy.exists():
            return legacy
        raise FileNotFoundError(
            f"benchmark {self.benchmark_code} not found; expected {direct} or {legacy}"
        )

    def _record_holdings(
        self,
        records: list[dict[str, object]],
        date: pd.Timestamp,
        selected: pd.DataFrame,
        adjusted_weights: dict[str, float],
        exposure: float,
    ) -> None:
        sectors = self._sectors()
        for row in selected.itertuples(index=False):
            records.append(
                {
                    "date": date,
                    "code": row.code,
                    "weight": adjusted_weights.get(row.code, 0.0),
                    "composite_score": row.composite_score,
                    "sector": sectors.get(row.code, "Unknown"),
                    "exposure": exposure,
                }
            )

    def _sectors(self) -> dict[str, str]:
        return {
            str(stock["code"]).zfill(6): str(stock["sector"] or "Unknown")
            for stock in self.universe.stocks()
        }

    @staticmethod
    def _sector_exposure(holdings_history: pd.DataFrame) -> pd.DataFrame:
        if holdings_history.empty:
            return pd.DataFrame(columns=["date", "sector", "weight"])
        return (
            holdings_history.groupby(["date", "sector"], as_index=False)["weight"]
            .sum()
            .sort_values(["date", "sector"])
            .reset_index(drop=True)
        )


def _target_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    codes = set(previous).union(current)
    return 0.5 * sum(abs(current.get(code, 0.0) - previous.get(code, 0.0)) for code in codes)
