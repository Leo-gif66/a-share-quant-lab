"""Chronological portfolio walk-forward simulation for institutional validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.metrics import calculate_metrics
from ..backtest.portfolio import Portfolio, TradingCosts
from ..memory import TradeDecisionSnapshot, TradeMemoryStore
from ..portfolio.allocator import PortfolioAllocator
from ..regime import MarketRegimeDetector
from ..research.decision_explanation import DecisionExplanationEngine


RESULT_COLUMNS = (
    "date",
    "equity",
    "benchmark",
    "portfolio_return",
    "benchmark_return",
    "turnover",
    "rebalance",
    "period",
    "market_regime",
    "training_end_date",
    "training_observations",
)


@dataclass(frozen=True)
class WalkForwardSettings:
    """Windows are trading sessions; every window is strictly point-in-time."""

    train_window: int = 40
    validation_window: int = 20
    rebalance_frequency: int = 10
    retrain_frequency: int = 1
    top_n: int = 20
    initial_cash: float = 1_000_000.0
    commission: float = 0.0003
    stamp_tax: float = 0.0005
    slippage: float = 0.001

    def __post_init__(self) -> None:
        if min(self.train_window, self.validation_window, self.rebalance_frequency, self.retrain_frequency, self.top_n) < 1:
            raise ValueError("walk-forward windows, frequencies, and top_n must be positive")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")


@dataclass(frozen=True)
class WalkForwardSimulationResult:
    results: pd.DataFrame
    holdings: pd.DataFrame
    metrics: dict[str, float]
    benchmark_comparison: pd.DataFrame
    memory_path: Path
    output_path: Path


class WalkForwardSimulator:
    """Fit score-return calibrations only on outcomes matured before a signal.

    The simulator pre-computes historical labels to make the experiment fast,
    but every fit applies the invariant ``exit_date < signal_date``.  That
    permits efficient replay without granting a historical decision access to
    its own (or any later) realized return.
    """

    def __init__(self, settings: WalkForwardSettings | None = None) -> None:
        self.settings = settings or WalkForwardSettings()
        self.regime_detector = MarketRegimeDetector()
        self.explanations = DecisionExplanationEngine()

    def run(
        self,
        scores: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark: pd.DataFrame,
        industries: Mapping[str, str] | None = None,
        memory: TradeMemoryStore | None = None,
        output_path: str | Path = "data/validation/walk_forward_results.parquet",
        benchmarks: Mapping[str, pd.DataFrame | None] | None = None,
    ) -> WalkForwardSimulationResult:
        score_frame = self._scores(scores)
        price_frame = self._prices(prices)
        benchmark_frame = self._benchmark(benchmark)
        dates = self._dates(score_frame, price_frame, benchmark_frame)
        if len(dates) <= self.settings.train_window + 1:
            raise ValueError("not enough common sessions for the requested train window")
        labels = self._labels(score_frame, price_frame, dates)
        price_index = price_frame.set_index(["date", "code"]).sort_index()
        portfolio = Portfolio(
            initial_cash=self.settings.initial_cash,
            costs=TradingCosts(self.settings.commission, self.settings.stamp_tax, self.settings.slippage),
        )
        plans = self._plans(score_frame, benchmark_frame, labels, dates, price_index, industries or {})
        holdings: list[dict[str, object]] = []
        decisions: list[TradeDecisionSnapshot] = []
        observations: list[dict[str, object]] = []
        previous_weights: dict[str, float] = {}
        last_close: dict[str, float] = {}
        active_state: dict[str, object] = {
            "turnover": 0.0,
            "rebalance": False,
            "period": -1,
            "market_regime": "sideways",
            "training_end_date": pd.NaT,
            "training_observations": 0,
        }
        first_entry = min(plans) if plans else None
        for current_date in dates:
            current_date = pd.Timestamp(current_date)
            if first_entry is None or current_date < first_entry:
                continue
            day = self._day_prices(price_index, current_date)
            active_state["rebalance"] = False
            active_state["turnover"] = 0.0
            if current_date in plans:
                plan = plans[current_date]
                open_prices = day["open"].to_dict()
                # Selling a suspended holding at an invented price would make
                # results less reliable; defer this rebalance instead.
                if all(code in open_prices and open_prices[code] > 0 for code in portfolio.positions):
                    target = plan["weights"]
                    turnover = _turnover(previous_weights, target)
                    portfolio.rebalance_weights(target, open_prices)
                    previous_weights = dict(target)
                    active_state.update(
                        {
                            "turnover": turnover,
                            "rebalance": True,
                            "period": plan["period"],
                            "market_regime": plan["regime"],
                            "training_end_date": plan["training_end_date"],
                            "training_observations": plan["training_observations"],
                        }
                    )
                    holdings.extend(plan["holdings"])
                    decisions.extend(plan["decisions"])
            close_prices = day["close"].to_dict()
            last_close.update({code: value for code, value in close_prices.items() if value > 0})
            marked = dict(close_prices)
            marked.update({code: last_close[code] for code in portfolio.positions if code not in marked and code in last_close})
            if not all(code in marked and marked[code] > 0 for code in portfolio.positions):
                continue
            equity = portfolio.record(current_date, marked)
            benchmark_value = float(benchmark_frame.loc[benchmark_frame["date"] == current_date, "close"].iloc[0])
            observations.append({"date": current_date, "equity": equity, "benchmark_close": benchmark_value, **active_state})
        results = self._results(observations)
        if len(results) < 2:
            raise RuntimeError("walk-forward simulation produced fewer than two marked sessions")
        equity_curve = results.loc[:, ["date", "equity"]]
        benchmark_curve = results.loc[:, ["date", "benchmark"]]
        metrics = calculate_metrics(equity_curve, benchmark_curve)
        metrics["turnover"] = float(results["turnover"].sum())
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(target, index=False)
        store = memory or TradeMemoryStore("data/validation/walk_forward_trades.parquet")
        if decisions:
            store.upsert(decisions)
            store.realize(as_of=dates[-1])
        comparison_inputs = {"CSI300": benchmark}
        comparison_inputs.update(benchmarks or {})
        from .benchmarks import benchmark_comparison
        comparison = benchmark_comparison(equity_curve, comparison_inputs)
        return WalkForwardSimulationResult(
            results=results,
            holdings=pd.DataFrame(holdings),
            metrics=metrics,
            benchmark_comparison=comparison,
            memory_path=store.path,
            output_path=target,
        )

    def _plans(
        self,
        scores: pd.DataFrame,
        benchmark: pd.DataFrame,
        labels: pd.DataFrame,
        dates: pd.DatetimeIndex,
        prices: pd.DataFrame,
        industries: Mapping[str, str],
    ) -> dict[pd.Timestamp, dict[str, object]]:
        plans: dict[pd.Timestamp, dict[str, object]] = {}
        coefficients: np.ndarray | None = None
        for event, index in enumerate(range(self.settings.train_window, len(dates) - 1, self.settings.rebalance_frequency)):
            signal_date, entry_date = dates[index], dates[index + 1]
            lower_bound = dates[max(0, index - self.settings.train_window)]
            matured = labels.loc[(labels["exit_date"] < signal_date) & (labels["signal_date"] >= lower_bound)].copy()
            retrained = event % self.settings.retrain_frequency == 0
            if retrained:
                coefficients = self._fit(matured)
            # No warm-up fallback is used. A portfolio begins only where a
            # real, historically matured training sample supports a model.
            if coefficients is None:
                continue
            current = scores.loc[scores["date"] == signal_date, ["date", "code", "composite_score"]].copy()
            if current.empty:
                continue
            current["model_prediction"] = self._predict(current["composite_score"], coefficients)
            snapshot = self.regime_detector.detect(benchmark, as_of=signal_date)
            current["symbol"] = current["code"]
            current["industry"] = current["code"].map(industries).fillna("Unknown")
            current["score"] = current["composite_score"]
            current["risk"] = 1.0 - snapshot.exposure
            current["regime"] = snapshot.state
            allocation = PortfolioAllocator().allocate(current, as_of=signal_date)
            if allocation.holdings.empty:
                continue
            exit_index = index + self.settings.rebalance_frequency + 1
            exit_date = dates[exit_index] if exit_index < len(dates) else pd.NaT
            holdings: list[dict[str, object]] = []
            decisions: list[TradeDecisionSnapshot] = []
            for row in allocation.holdings.itertuples(index=False):
                score = float(current.loc[current["code"] == row.symbol, "composite_score"].iloc[0])
                holdings.append(
                    {
                        "date": entry_date,
                        "code": row.symbol,
                        "industry": row.industry,
                        "weight": float(row.weight),
                        "model_prediction": float(row.model_score) if pd.notna(row.model_score) else np.nan,
                        "composite_score": score,
                        "period": (index - self.settings.train_window) // self.settings.validation_window,
                    }
                )
                decisions.append(
                    TradeDecisionSnapshot(
                        date=entry_date,
                        symbol=row.symbol,
                        strategy="walk_forward_v4",
                        model_score=float(row.model_score),
                        factor_scores={"composite_score": score},
                        industry=str(row.industry),
                        market_regime=snapshot.state,
                        entry_price=self._open_at(prices, entry_date, row.symbol),
                        exit_price=self._open_at(prices, exit_date, row.symbol) if pd.notna(exit_date) else None,
                        exit_date=exit_date if pd.notna(exit_date) else None,
                        weight=float(row.weight),
                        market_volatility=float(snapshot.volatility),
                        decision_reason=self.explanations.explain(
                            {"composite_score": score}, snapshot.state, snapshot.exposure, float(row.weight)
                        ),
                    )
                )
            plans[entry_date] = {
                "weights": allocation.holdings.set_index("symbol")["weight"].to_dict(),
                "holdings": holdings,
                "decisions": decisions,
                "period": (index - self.settings.train_window) // self.settings.validation_window,
                "regime": snapshot.state,
                "training_end_date": matured["exit_date"].max() if not matured.empty else pd.NaT,
                "training_observations": int(len(matured)),
            }
        return plans

    def _labels(self, scores: pd.DataFrame, prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
        index = prices.set_index(["date", "code"])["open"]
        frames: list[pd.DataFrame] = []
        for position in range(0, len(dates) - self.settings.rebalance_frequency - 1, self.settings.rebalance_frequency):
            signal, entry, exit_date = dates[position], dates[position + 1], dates[position + self.settings.rebalance_frequency + 1]
            day = scores.loc[scores["date"] == signal, ["code", "composite_score"]].copy()
            entry_prices = index.xs(entry, level="date")
            exit_prices = index.xs(exit_date, level="date")
            day["entry_price"] = day["code"].map(entry_prices)
            day["exit_price"] = day["code"].map(exit_prices)
            day["future_return"] = day["exit_price"] / day["entry_price"] - 1.0
            day["signal_date"] = signal
            day["exit_date"] = exit_date
            frames.append(day.loc[:, ["signal_date", "exit_date", "code", "composite_score", "future_return"]])
        if not frames:
            return pd.DataFrame(columns=["signal_date", "exit_date", "code", "composite_score", "future_return"])
        return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna().sort_values("signal_date")

    def _results(self, observations: list[dict[str, object]]) -> pd.DataFrame:
        result = pd.DataFrame(observations)
        result = result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        result["benchmark"] = result["equity"].iloc[0] * result["benchmark_close"] / result["benchmark_close"].iloc[0]
        result["portfolio_return"] = result["equity"].pct_change().fillna(0.0)
        result["benchmark_return"] = result["benchmark"].pct_change().fillna(0.0)
        return result.loc[:, RESULT_COLUMNS]

    @staticmethod
    def _fit(labels: pd.DataFrame) -> np.ndarray | None:
        if len(labels) < 30 or labels["composite_score"].nunique() < 2:
            return None
        return np.linalg.lstsq(
            np.column_stack([np.ones(len(labels)), labels["composite_score"].to_numpy(dtype=float)]),
            labels["future_return"].to_numpy(dtype=float),
            rcond=None,
        )[0]

    @staticmethod
    def _predict(scores: pd.Series, coefficients: np.ndarray) -> np.ndarray:
        return coefficients[0] + coefficients[1] * scores.to_numpy(dtype=float)

    @staticmethod
    def _scores(scores: pd.DataFrame) -> pd.DataFrame:
        code = "code" if "code" in scores else "symbol" if "symbol" in scores else None
        value = "composite_score" if "composite_score" in scores else "score" if "score" in scores else None
        if code is None or value is None or "date" not in scores:
            raise ValueError("scores require date, code/symbol, and composite_score/score")
        frame = scores.loc[:, ["date", code, value]].copy()
        frame.columns = ["date", "code", "composite_score"]
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["composite_score"] = pd.to_numeric(frame["composite_score"], errors="coerce")
        return frame.dropna().drop_duplicates(["date", "code"], keep="last").sort_values(["date", "code"])

    @staticmethod
    def _prices(prices: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "open", "close"}
        code = "code" if "code" in prices else "symbol" if "symbol" in prices else None
        if code is None or not required.issubset(prices.columns):
            raise ValueError("prices require date, code/symbol, open, and close")
        frame = prices.loc[:, ["date", code, "open", "close"]].copy()
        frame.columns = ["date", "code", "open", "close"]
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        for column in ("open", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna().loc[lambda values: (values["open"] > 0) & (values["close"] > 0)].drop_duplicates(["date", "code"], keep="last").sort_values(["date", "code"])

    @staticmethod
    def _benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
        if not {"date", "close"}.issubset(benchmark.columns):
            raise ValueError("benchmark requires date and close")
        frame = benchmark.loc[:, ["date", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        return frame.dropna().loc[lambda values: values["close"] > 0].drop_duplicates("date", keep="last").sort_values("date")

    @staticmethod
    def _dates(scores: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DatetimeIndex:
        counts = prices.groupby("date")["code"].nunique()
        score_counts = scores.groupby("date")["code"].nunique()
        eligible = score_counts.index.intersection(counts.index).intersection(pd.DatetimeIndex(benchmark["date"]))
        eligible = [date for date in eligible if score_counts.loc[date] >= 5 and counts.loc[date] >= 5]
        return pd.DatetimeIndex(sorted(eligible))

    @staticmethod
    def _day_prices(prices: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        return prices.xs(date, level="date")

    @staticmethod
    def _open_at(prices: pd.DataFrame, date: pd.Timestamp, code: str) -> float | None:
        try:
            value = prices.loc[(pd.Timestamp(date), str(code).zfill(6)), "open"]
        except KeyError:
            return None
        return float(value) if pd.notna(value) and value > 0 else None


def _turnover(previous: Mapping[str, float], current: Mapping[str, float]) -> float:
    codes = set(previous).union(current)
    return 0.5 * sum(abs(float(current.get(code, 0.0)) - float(previous.get(code, 0.0))) for code in codes)
