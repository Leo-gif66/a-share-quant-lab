"""Leakage-safe, time-ordered live simulation with periodic score calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..memory import TradeDecisionSnapshot, TradeMemoryStore
from .decision_explanation import DecisionExplanationEngine


SIMULATION_COLUMNS = (
    "date",
    "entry_date",
    "exit_date",
    "symbol",
    "model_score",
    "composite_score",
    "entry_price",
    "exit_price",
    "future_return",
    "training_end_date",
    "training_observations",
    "retrained",
    "weight",
    "market_regime",
)


@dataclass(frozen=True)
class LiveSimulationResult:
    snapshots: pd.DataFrame
    trades: pd.DataFrame
    output_path: Path


class LiveSimulationEngine:
    """Simulate only decisions available at each rebalance timestamp.

    The calibrator is intentionally a simple linear score-to-return model.
    Its aim is diagnosis: every training observation has an exit date strictly
    before the decision timestamp, and each retraining boundary is persisted.
    """

    def __init__(
        self,
        rebalance_interval: int = 20,
        retrain_interval: int = 1,
        top_n: int = 20,
        strategy: str = "live_simulation_v35",
    ) -> None:
        if min(rebalance_interval, retrain_interval, top_n) < 1:
            raise ValueError("simulation intervals and top_n must be positive")
        self.rebalance_interval = rebalance_interval
        self.retrain_interval = retrain_interval
        self.top_n = top_n
        self.strategy = strategy
        self.explanations = DecisionExplanationEngine()

    def run(
        self,
        scores: pd.DataFrame,
        prices: pd.DataFrame,
        memory: TradeMemoryStore | None = None,
        industries: Mapping[str, str] | None = None,
        regime_provider: Callable[[pd.Timestamp], tuple[str, float, float]] | None = None,
        output_path: str | Path = "data/memory/live_simulation.parquet",
    ) -> LiveSimulationResult:
        score_frame = self._scores(scores)
        price_frame = self._prices(prices)
        dates = pd.DatetimeIndex(sorted(score_frame["date"].unique()))
        price_index = price_frame.set_index(["date", "code"])["open"]
        labels = self._labels(score_frame, dates, price_index)
        snapshots: list[dict[str, object]] = []
        decisions: list[TradeDecisionSnapshot] = []
        coefficients: np.ndarray | None = None
        train_end = pd.NaT
        for event_number, index in enumerate(range(0, len(dates) - 1, self.rebalance_interval)):
            signal_date, entry_date = dates[index], dates[index + 1]
            exit_index = index + self.rebalance_interval + 1
            exit_date = dates[exit_index] if exit_index < len(dates) else pd.NaT
            matured = labels.loc[labels["exit_date"] < signal_date]
            retrained = event_number % self.retrain_interval == 0
            if retrained:
                coefficients = self._fit(matured)
                train_end = matured["exit_date"].max() if not matured.empty else pd.NaT
            day = score_frame.loc[score_frame["date"] == signal_date].copy()
            day["model_score"] = self._predict(day["composite_score"], coefficients)
            selected = day.nlargest(self.top_n, "model_score", keep="all").head(self.top_n).copy()
            if selected.empty:
                continue
            selected["weight"] = 1.0 / len(selected)
            regime, regime_exposure, volatility = (
                regime_provider(signal_date) if regime_provider is not None else ("sideways", 0.70, np.nan)
            )
            for row in selected.itertuples(index=False):
                entry_price = _price_at(price_index, entry_date, row.code)
                exit_price = _price_at(price_index, exit_date, row.code) if pd.notna(exit_date) else np.nan
                future_return = exit_price / entry_price - 1.0 if entry_price > 0 and exit_price > 0 else np.nan
                snapshots.append(
                    {
                        "date": signal_date,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "symbol": row.code,
                        "model_score": row.model_score,
                        "composite_score": row.composite_score,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "future_return": future_return,
                        "training_end_date": train_end,
                        "training_observations": len(matured),
                        "retrained": retrained,
                        "weight": row.weight,
                        "market_regime": regime,
                    }
                )
                factor_scores = {"composite_score": float(row.composite_score)}
                decisions.append(
                    TradeDecisionSnapshot(
                        date=entry_date,
                        symbol=row.code,
                        strategy=self.strategy,
                        model_score=float(row.model_score),
                        factor_scores=factor_scores,
                        industry=(industries or {}).get(str(row.code).zfill(6), "Unknown"),
                        market_regime=regime,
                        entry_price=entry_price if entry_price > 0 else None,
                        exit_price=exit_price if exit_price > 0 else None,
                        exit_date=exit_date if pd.notna(exit_date) else None,
                        weight=float(row.weight),
                        market_volatility=float(volatility) if pd.notna(volatility) else None,
                        decision_reason=self.explanations.explain(
                            factor_scores, regime, float(regime_exposure), float(row.weight)
                        ),
                    )
                )
        frame = pd.DataFrame(snapshots, columns=SIMULATION_COLUMNS)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        store = memory or TradeMemoryStore()
        if decisions:
            store.upsert(decisions)
        return LiveSimulationResult(frame, store.realize(), target)

    @staticmethod
    def _scores(scores: pd.DataFrame) -> pd.DataFrame:
        if not {"date", "code", "composite_score"}.issubset(scores.columns):
            raise ValueError("scores require date, code, and composite_score")
        frame = scores.loc[:, ["date", "code", "composite_score"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["composite_score"] = pd.to_numeric(frame["composite_score"], errors="coerce")
        return frame.dropna().drop_duplicates(["date", "code"], keep="last").sort_values(["date", "code"])

    @staticmethod
    def _prices(prices: pd.DataFrame) -> pd.DataFrame:
        if not {"date", "code", "open"}.issubset(prices.columns):
            raise ValueError("prices require date, code, and open")
        frame = prices.loc[:, ["date", "code", "open"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
        return frame.dropna().drop_duplicates(["date", "code"], keep="last").sort_values(["date", "code"])

    def _labels(
        self, scores: pd.DataFrame, dates: pd.DatetimeIndex, prices: pd.Series
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for index in range(0, len(dates) - 1, self.rebalance_interval):
            exit_index = index + self.rebalance_interval + 1
            if exit_index >= len(dates):
                continue
            signal_date, entry_date, exit_date = dates[index], dates[index + 1], dates[exit_index]
            frame = scores.loc[scores["date"] == signal_date, ["code", "composite_score"]].copy()
            entry = prices.xs(entry_date, level="date")
            exit = prices.xs(exit_date, level="date")
            frame["entry_price"] = frame["code"].map(entry)
            frame["exit_price"] = frame["code"].map(exit)
            frame["future_return"] = frame["exit_price"] / frame["entry_price"] - 1.0
            frame["signal_date"] = signal_date
            frame["exit_date"] = exit_date
            frames.append(frame.loc[:, ["signal_date", "exit_date", "code", "composite_score", "future_return"]])
        if not frames:
            return pd.DataFrame(columns=["signal_date", "exit_date", "code", "composite_score", "future_return"])
        return pd.concat(frames, ignore_index=True).dropna().sort_values("signal_date")

    @staticmethod
    def _fit(labels: pd.DataFrame) -> np.ndarray | None:
        if len(labels) < 30 or labels["composite_score"].nunique() < 2:
            return None
        x = labels["composite_score"].to_numpy(dtype=float)
        y = labels["future_return"].to_numpy(dtype=float)
        return np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]

    @staticmethod
    def _predict(scores: pd.Series, coefficients: np.ndarray | None) -> np.ndarray:
        values = scores.to_numpy(dtype=float)
        if coefficients is None:
            return np.zeros(len(values), dtype=float)
        return coefficients[0] + coefficients[1] * values


def _price_at(prices: pd.Series, date: pd.Timestamp, code: str) -> float:
    value = prices.get((pd.Timestamp(date), str(code).zfill(6)), np.nan)
    return float(value) if pd.notna(value) and value > 0 else np.nan
