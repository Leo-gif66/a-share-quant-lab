import pandas as pd

from quant.risk import RiskController, RiskSettings


def test_risk_controls_reduce_exposure_and_cap_single_positions():
    dates = pd.bdate_range("2023-01-02", periods=200)
    benchmark = pd.DataFrame({"date": dates, "close": [100.0] * 199 + [90.0]})
    equity_curve = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=5),
            "equity": [100.0, 120.0, 80.0, 125.0, 75.0],
        }
    )
    controller = RiskController(
        RiskSettings(volatility_target=0.10, volatility_lookback=4, max_position=0.10)
    )

    weights, exposure = controller.target_weights(
        {"000001": 0.70, "000002": 0.30}, benchmark, equity_curve, dates[-1]
    )

    assert controller.trend_exposure(benchmark, dates[-1]) == 0.5
    assert controller.volatility_exposure(equity_curve) < 1.0
    assert exposure < 0.5
    assert max(weights.values()) <= 0.10
