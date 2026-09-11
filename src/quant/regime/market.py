"""Point-in-time market-regime classification for intelligent portfolio overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeSettings:
    trend_window: int = 200
    breadth_window: int = 20
    volatility_window: int = 20
    high_volatility_threshold: float = 0.30
    bull_breadth_threshold: float = 0.55
    bear_breadth_threshold: float = 0.45
    bull_exposure: float = 1.0
    sideways_exposure: float = 0.70
    bear_exposure: float = 0.40
    high_volatility_exposure: float = 0.50

    def __post_init__(self) -> None:
        if min(self.trend_window, self.breadth_window, self.volatility_window) < 2:
            raise ValueError("regime windows must be at least two sessions")
        if not 0 < self.high_volatility_threshold:
            raise ValueError("high_volatility_threshold must be positive")
        if not 0 <= self.bear_breadth_threshold <= self.bull_breadth_threshold <= 1:
            raise ValueError("breadth thresholds must be ordered in [0, 1]")


@dataclass(frozen=True)
class RegimeSnapshot:
    date: pd.Timestamp
    state: str
    exposure: float
    close: float
    trend: float
    breadth: float
    volatility: float


class MarketRegimeDetector:
    """Detect bull, bear, sideways, and high-volatility states without leakage."""

    def __init__(self, settings: RegimeSettings | None = None) -> None:
        self.settings = settings or RegimeSettings()

    def detect(
        self,
        benchmark: pd.DataFrame,
        breadth: float | pd.Series | pd.DataFrame | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> RegimeSnapshot:
        prices = self._history(benchmark, as_of)
        date = pd.Timestamp(prices["date"].iloc[-1])
        close = float(prices["close"].iloc[-1])
        ma = prices["close"].rolling(self.settings.trend_window).mean().iloc[-1]
        trend = float(close / ma - 1.0) if pd.notna(ma) and ma > 0 else 0.0
        returns = prices["close"].pct_change().dropna().tail(self.settings.volatility_window)
        volatility = float(returns.std(ddof=0) * np.sqrt(252)) if len(returns) >= 2 else 0.0
        breadth_value = self._breadth_at(breadth, date)
        if volatility >= self.settings.high_volatility_threshold:
            state, exposure = "high_volatility", self.settings.high_volatility_exposure
        elif trend > 0 and breadth_value >= self.settings.bull_breadth_threshold:
            state, exposure = "bull", self.settings.bull_exposure
        elif trend < 0 and breadth_value <= self.settings.bear_breadth_threshold:
            state, exposure = "bear", self.settings.bear_exposure
        else:
            state, exposure = "sideways", self.settings.sideways_exposure
        return RegimeSnapshot(
            date=date,
            state=state,
            exposure=float(exposure),
            close=close,
            trend=trend,
            breadth=float(breadth_value),
            volatility=volatility,
        )

    def market_breadth(
        self, raw_dir: str | Path, as_of: pd.Timestamp | None = None, benchmark_code: str = "000300"
    ) -> float:
        """Return the fraction of locally stored stocks above their short MA."""
        directory = Path(raw_dir)
        states: list[bool] = []
        cutoff = pd.Timestamp(as_of) if as_of is not None else None
        for path in directory.glob("*.parquet"):
            if not path.stem.isdigit() or path.stem == str(benchmark_code).zfill(6):
                continue
            frame = pd.read_parquet(path, columns=["date", "close"])
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame = frame.dropna().sort_values("date")
            if cutoff is not None:
                frame = frame.loc[frame["date"] <= cutoff]
            if len(frame) < self.settings.breadth_window:
                continue
            close = float(frame["close"].iloc[-1])
            average = float(frame["close"].tail(self.settings.breadth_window).mean())
            states.append(close > average)
        return float(np.mean(states)) if states else 0.5

    @staticmethod
    def _history(benchmark: pd.DataFrame, as_of: pd.Timestamp | None) -> pd.DataFrame:
        if not {"date", "close"}.issubset(benchmark.columns):
            raise ValueError("benchmark requires date and close columns")
        frame = benchmark.loc[:, ["date", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna().drop_duplicates("date", keep="last").sort_values("date")
        if as_of is not None:
            frame = frame.loc[frame["date"] <= pd.Timestamp(as_of)]
        if frame.empty:
            raise ValueError("benchmark has no data on or before as_of")
        return frame.reset_index(drop=True)

    @staticmethod
    def _breadth_at(breadth: float | pd.Series | pd.DataFrame | None, date: pd.Timestamp) -> float:
        if breadth is None:
            return 0.5
        if isinstance(breadth, (int, float, np.floating)):
            return float(np.clip(breadth, 0.0, 1.0))
        if isinstance(breadth, pd.DataFrame):
            if not {"date", "breadth"}.issubset(breadth.columns):
                raise ValueError("breadth frame requires date and breadth")
            values = breadth.set_index(pd.to_datetime(breadth["date"], errors="raise"))["breadth"]
        else:
            values = breadth.copy()
            values.index = pd.to_datetime(values.index, errors="raise")
        values = pd.to_numeric(values, errors="coerce").dropna().sort_index()
        eligible = values.loc[values.index <= date]
        return float(np.clip(eligible.iloc[-1], 0.0, 1.0)) if not eligible.empty else 0.5


class RegimeAwareRiskOverlay:
    """Multiply the existing institutional risk target by a transparent regime cap."""

    def __init__(
        self,
        base_risk: object,
        detector: MarketRegimeDetector,
        breadth_provider: Callable[[pd.Timestamp], float],
        benchmark_history: pd.DataFrame | None = None,
    ) -> None:
        self.base_risk = base_risk
        self.detector = detector
        self.breadth_provider = breadth_provider
        self.benchmark_history = benchmark_history
        self.history: list[RegimeSnapshot] = []

    def target_weights(
        self,
        weights: dict[str, float],
        benchmark: pd.DataFrame,
        equity_curve: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> tuple[dict[str, float], float]:
        base_weights, base_exposure = self.base_risk.target_weights(
            weights, benchmark, equity_curve, as_of
        )
        source = self.benchmark_history if self.benchmark_history is not None else benchmark
        snapshot = self.detector.detect(source, self.breadth_provider(pd.Timestamp(as_of)), as_of)
        self.history.append(snapshot)
        adjusted = {code: float(weight) * snapshot.exposure for code, weight in base_weights.items()}
        return adjusted, float(base_exposure) * snapshot.exposure
