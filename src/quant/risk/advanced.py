"""Regime-aware, volatility-targeted risk controls for the institutional mode."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdvancedRiskSettings:
    """Conservative defaults for the v1.3 risk overlay."""

    bull_exposure: float = 1.0
    neutral_exposure: float = 0.70
    bear_exposure: float = 0.40
    volatility_target: float = 0.15
    volatility_lookback: int = 60
    drawdown_warning: float = 0.10
    drawdown_critical: float = 0.20
    warning_scale: float = 0.70
    critical_scale: float = 0.50
    max_position: float = 0.10

    def __post_init__(self) -> None:
        if not 0 <= self.bear_exposure <= self.neutral_exposure <= self.bull_exposure <= 1:
            raise ValueError("exposures must satisfy bear <= neutral <= bull and be in [0, 1]")
        if self.volatility_target <= 0 or self.volatility_lookback < 2:
            raise ValueError("volatility target must be positive and lookback must be at least two")
        if not 0 < self.drawdown_warning < self.drawdown_critical < 1:
            raise ValueError("drawdown thresholds must satisfy 0 < warning < critical < 1")
        if not 0 < self.critical_scale <= self.warning_scale <= 1:
            raise ValueError("drawdown scales must satisfy 0 < critical <= warning <= 1")
        if not 0 < self.max_position <= 1:
            raise ValueError("max_position must be in (0, 1]")


@dataclass(frozen=True)
class MarketRegime:
    """A point-in-time CSI 300 regime decision and its observable features."""

    as_of: pd.Timestamp
    state: str
    exposure: float
    close: float
    ma20: float
    ma60: float
    ma120: float
    ma200: float


@dataclass(frozen=True)
class RiskDecision:
    """Breakdown of the exposure multiplier used for a rebalance."""

    regime: MarketRegime
    realized_volatility: float
    volatility_scale: float
    drawdown: float
    drawdown_scale: float
    exposure: float


class MarketRegimeModel:
    """Classify CSI 300 into Bull, Neutral, or Bear without look-ahead."""

    WINDOWS = (20, 60, 120, 200)

    def __init__(self, settings: AdvancedRiskSettings | None = None) -> None:
        self.settings = settings or AdvancedRiskSettings()

    def evaluate(self, benchmark: pd.DataFrame, as_of: pd.Timestamp) -> MarketRegime:
        prices = _benchmark_history(benchmark, as_of)
        moving_averages = {
            window: prices["close"].rolling(window).mean().iloc[-1]
            if len(prices) >= window
            else np.nan
            for window in self.WINDOWS
        }
        close = float(prices["close"].iloc[-1])
        complete = all(pd.notna(value) for value in moving_averages.values())
        if not complete:
            state, exposure = "Neutral", self.settings.neutral_exposure
        elif (
            close >= moving_averages[60]
            and close >= moving_averages[120]
            and close >= moving_averages[200]
            and moving_averages[20] >= moving_averages[60]
        ):
            state, exposure = "Bull", self.settings.bull_exposure
        elif (
            close <= moving_averages[60]
            and close <= moving_averages[120]
            and close <= moving_averages[200]
            and moving_averages[20] <= moving_averages[60]
        ):
            state, exposure = "Bear", self.settings.bear_exposure
        else:
            state, exposure = "Neutral", self.settings.neutral_exposure
        return MarketRegime(
            as_of=pd.Timestamp(as_of),
            state=state,
            exposure=float(exposure),
            close=close,
            ma20=float(moving_averages[20]),
            ma60=float(moving_averages[60]),
            ma120=float(moving_averages[120]),
            ma200=float(moving_averages[200]),
        )


class AdvancedRiskController:
    """Apply regime, realized-volatility, and drawdown controls to weights.

    It intentionally exposes the same ``target_weights`` method as the v1.0
    controller, so it can be substituted into a portfolio engine without
    changing order accounting.
    """

    def __init__(self, settings: AdvancedRiskSettings | None = None) -> None:
        self.settings = settings or AdvancedRiskSettings()
        self.regime_model = MarketRegimeModel(self.settings)

    def decision(
        self, benchmark: pd.DataFrame, equity_curve: pd.DataFrame, as_of: pd.Timestamp
    ) -> RiskDecision:
        regime = self.regime_model.evaluate(benchmark, as_of)
        volatility = self.realized_volatility(equity_curve)
        volatility_scale = (
            min(1.0, self.settings.volatility_target / volatility) if volatility > 0 else 1.0
        )
        drawdown = self.current_drawdown(equity_curve)
        if drawdown <= -self.settings.drawdown_critical:
            drawdown_scale = self.settings.critical_scale
        elif drawdown <= -self.settings.drawdown_warning:
            drawdown_scale = self.settings.warning_scale
        else:
            drawdown_scale = 1.0
        exposure = regime.exposure * volatility_scale * drawdown_scale
        return RiskDecision(
            regime=regime,
            realized_volatility=volatility,
            volatility_scale=float(volatility_scale),
            drawdown=drawdown,
            drawdown_scale=float(drawdown_scale),
            exposure=float(exposure),
        )

    def target_weights(
        self,
        weights: Mapping[str, float],
        benchmark: pd.DataFrame,
        equity_curve: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> tuple[dict[str, float], float]:
        decision = self.decision(benchmark, equity_curve, as_of)
        adjusted = {
            str(code): min(max(float(weight), 0.0), self.settings.max_position)
            * decision.exposure
            for code, weight in weights.items()
        }
        return adjusted, decision.exposure

    def realized_volatility(self, equity_curve: pd.DataFrame) -> float:
        if equity_curve.empty:
            return 0.0
        if "equity" not in equity_curve:
            raise ValueError("equity curve requires an equity column")
        equity = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
        returns = equity.pct_change().dropna().tail(self.settings.volatility_lookback)
        if len(returns) < 2:
            return 0.0
        return float(returns.std(ddof=0) * np.sqrt(252))

    @staticmethod
    def current_drawdown(equity_curve: pd.DataFrame) -> float:
        if equity_curve.empty:
            return 0.0
        if "equity" not in equity_curve:
            raise ValueError("equity curve requires an equity column")
        equity = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
        if equity.empty:
            return 0.0
        return float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1.0)


def _benchmark_history(benchmark: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if not {"date", "close"}.issubset(benchmark.columns):
        raise ValueError("benchmark requires date and close columns")
    frame = benchmark.loc[:, ["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna().drop_duplicates("date", keep="last").sort_values("date")
    frame = frame.loc[frame["date"] <= pd.Timestamp(as_of)]
    if frame.empty:
        raise ValueError("benchmark has no prices on or before as_of")
    return frame.reset_index(drop=True)
