import numpy as np
import pandas as pd

from quant.regime import MarketRegimeDetector, RegimeAwareRiskOverlay, RegimeSettings


def test_regime_detector_uses_only_available_trend_breadth_and_volatility():
    dates = pd.bdate_range("2023-01-02", periods=220)
    bull = pd.DataFrame({"date": dates, "close": 100 * 1.001 ** np.arange(len(dates))})
    detector = MarketRegimeDetector(RegimeSettings(high_volatility_threshold=0.5))

    snapshot = detector.detect(bull, breadth=0.8, as_of=dates[-1])

    assert snapshot.state == "bull"
    assert snapshot.breadth == 0.8
    assert snapshot.date == dates[-1]


def test_regime_detector_prioritizes_high_volatility():
    dates = pd.bdate_range("2023-01-02", periods=220)
    changes = np.where(np.arange(len(dates)) % 2 == 0, 1.08, 0.92)
    prices = 100 * np.cumprod(changes)

    snapshot = MarketRegimeDetector().detect(pd.DataFrame({"date": dates, "close": prices}), breadth=0.9)

    assert snapshot.state == "high_volatility"
    assert snapshot.exposure < 1


def test_regime_overlay_uses_full_csi300_history_not_only_accounting_calendar():
    class _Risk:
        def target_weights(self, weights, benchmark, equity_curve, as_of):
            return weights, 1.0

    dates = pd.bdate_range("2023-01-02", periods=220)
    full_history = pd.DataFrame({"date": dates, "close": 100 * 1.001 ** np.arange(len(dates))})
    short_calendar = full_history.tail(10)
    overlay = RegimeAwareRiskOverlay(
        _Risk(), MarketRegimeDetector(RegimeSettings(high_volatility_threshold=0.5)), lambda _date: 0.8,
        benchmark_history=full_history,
    )

    _weights, exposure = overlay.target_weights({"000001": 0.1}, short_calendar, pd.DataFrame(), dates[-1])

    assert overlay.history[-1].state == "bull"
    assert exposure == 1.0
