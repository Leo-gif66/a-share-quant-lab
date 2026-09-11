"""Benchmark-trend, volatility-targeting, and position-cap risk controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskSettings:
    benchmark_ma_window: int = 200
    normal_exposure: float = 1.0
    bear_exposure: float = 0.5
    volatility_target: float = 0.20
    volatility_lookback: int = 60
    max_position: float = 0.10

    def __post_init__(self) -> None:
        if self.benchmark_ma_window < 1 or self.volatility_lookback < 2:
            raise ValueError("risk windows must be positive")
        if not 0 <= self.bear_exposure <= self.normal_exposure <= 1:
            raise ValueError("exposures must satisfy 0 <= bear <= normal <= 1")
        if self.volatility_target <= 0 or not 0 < self.max_position <= 1:
            raise ValueError("volatility target and maximum position must be positive")


class RiskController:
    """Calculate a cash-aware target weight vector under v1.0 risk limits."""

    def __init__(self, settings: RiskSettings | None = None) -> None:
        self.settings = settings or RiskSettings()

    def trend_exposure(self, benchmark: pd.DataFrame, as_of: pd.Timestamp) -> float:
        """Return normal exposure unless CSI 300 closes below its MA200."""
        if not {"date", "close"}.issubset(benchmark.columns):
            raise ValueError("benchmark requires date and close columns")
        frame = benchmark.loc[:, ["date", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna().sort_values("date")
        frame = frame.loc[frame["date"] <= pd.Timestamp(as_of)]
        if len(frame) < self.settings.benchmark_ma_window:
            return self.settings.normal_exposure
        moving_average = frame["close"].rolling(self.settings.benchmark_ma_window).mean().iloc[-1]
        return (
            self.settings.bear_exposure
            if frame["close"].iloc[-1] < moving_average
            else self.settings.normal_exposure
        )

    def realized_volatility(self, equity_curve: pd.DataFrame) -> float:
        """Return trailing annualized daily volatility, or zero before warm-up."""
        if "equity" not in equity_curve:
            raise ValueError("equity curve requires an equity column")
        equity = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
        returns = equity.pct_change().dropna().tail(self.settings.volatility_lookback)
        if len(returns) < 2:
            return 0.0
        return float(returns.std(ddof=0) * np.sqrt(252))

    def volatility_exposure(self, equity_curve: pd.DataFrame) -> float:
        """Scale exposure down when realized portfolio volatility exceeds target."""
        volatility = self.realized_volatility(equity_curve)
        if volatility <= self.settings.volatility_target or volatility == 0:
            return 1.0
        return float(self.settings.volatility_target / volatility)

    def target_weights(
        self,
        weights: Mapping[str, float],
        benchmark: pd.DataFrame,
        equity_curve: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> tuple[dict[str, float], float]:
        """Cap positions and apply combined benchmark-trend and volatility exposure."""
        exposure = self.trend_exposure(benchmark, as_of) * self.volatility_exposure(equity_curve)
        adjusted = {
            str(code): min(max(float(weight), 0.0), self.settings.max_position) * exposure
            for code, weight in weights.items()
        }
        return adjusted, float(exposure)
