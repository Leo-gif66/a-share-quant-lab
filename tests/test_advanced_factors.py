import pandas as pd
import pytest

from quant.factors import AdvancedFactorEngine, merge_quality_factors


def _prices(periods: int = 130) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=periods)
    close = pd.Series([10 + index * 0.1 for index in range(periods)])
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "volume": 1_000 + pd.Series(range(periods)),
            "amount": 10_000 + pd.Series(range(periods)) * 10,
            "turnover": 1.0,
        }
    )


def test_advanced_price_factor_engine_calculates_requested_factor_families():
    prices = _prices()
    benchmark = pd.DataFrame({"date": prices["date"], "close": prices["close"] * 2})

    factors = AdvancedFactorEngine().calculate(prices, benchmark)

    expected = {
        "rsi_14",
        "macd",
        "bollinger_position",
        "atr_14",
        "volatility_breakout_20",
        "momentum_120",
        "turnover",
        "amount",
        "volume_change",
        "beta_60",
        "volatility_20",
        "drawdown_60",
    }
    assert expected.issubset(factors.columns)
    assert factors.loc[len(factors) - 1, "rsi_14"] == pytest.approx(100.0)
    assert pd.notna(factors.loc[len(factors) - 1, "beta_60"])


def test_quality_merge_uses_availability_date_not_future_report_data():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "code": ["000001"] * 3,
            "momentum_20": [1.0, 2.0, 3.0],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "code": ["000001"],
            "available_date": ["2024-01-03"],
            "roe": [0.12],
            "profit_growth": [0.20],
            "revenue_growth": [0.10],
        }
    )

    merged = merge_quality_factors(prices, fundamentals)

    assert pd.isna(merged.loc[0, "roe"])
    assert merged.loc[1, "roe"] == pytest.approx(0.12)
    with pytest.raises(ValueError, match="available_date"):
        merge_quality_factors(prices, fundamentals.drop(columns="available_date"))
