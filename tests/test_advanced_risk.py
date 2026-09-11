import pandas as pd
import pytest

from quant.risk import AdvancedRiskController, AdvancedRiskSettings, MarketRegimeModel


def test_market_regime_uses_only_prices_available_at_rebalance_date():
    dates = pd.bdate_range("2023-01-02", periods=220)
    benchmark = pd.DataFrame({"date": dates, "close": [100 + index for index in range(220)]})
    model = MarketRegimeModel()

    regime = model.evaluate(benchmark, dates[-1])

    assert regime.state == "Bull"
    assert regime.exposure == 1.0
    assert regime.ma20 > regime.ma60 > regime.ma120 > regime.ma200


def test_advanced_risk_scales_for_volatility_and_drawdown():
    dates = pd.bdate_range("2023-01-02", periods=220)
    benchmark = pd.DataFrame({"date": dates, "close": [400 - index for index in range(220)]})
    equity = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=6),
            "equity": [100.0, 130.0, 80.0, 110.0, 70.0, 75.0],
        }
    )
    controller = AdvancedRiskController(
        AdvancedRiskSettings(volatility_lookback=5, volatility_target=0.10)
    )

    decision = controller.decision(benchmark, equity, dates[-1])
    weights, exposure = controller.target_weights({"000001": 0.5}, benchmark, equity, dates[-1])

    assert decision.regime.state == "Bear"
    assert decision.drawdown <= -0.20
    assert decision.drawdown_scale == 0.50
    assert decision.volatility_scale < 1
    assert exposure == pytest.approx(decision.exposure)
    assert weights["000001"] <= 0.10 * exposure
