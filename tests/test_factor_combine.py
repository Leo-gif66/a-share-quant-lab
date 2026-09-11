import pandas as pd
import pytest

from quant.factors import FactorCombiner


def _panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            "date": [date for date in dates for _ in range(2)],
            "code": ["000001", "000002"] * 3,
            "momentum": [1, 2, 3, 4, 5, 6],
            "volatility": [2, 1, 4, 3, 6, 5],
        }
    )


def test_factor_combiner_supports_equal_ic_icir_and_optimized_weights_without_lookahead():
    panel = _panel()
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"]),
            "factor_name": ["momentum", "volatility"] * 3,
            "IC": [0.2, -0.1, 0.3, -0.2, 99.0, -99.0],
        }
    )
    combiner = FactorCombiner(["momentum", "volatility"], directions={"volatility": -1})

    equal = combiner.combine(panel, "equal")
    ic = combiner.combine(panel, "ic", history)
    icir = combiner.combine(panel, "icir", history)
    optimized = combiner.combine(panel, "rolling_optimized", history)

    assert list(equal.scores.columns) == ["date", "code", "factor_score"]
    assert len(ic.scores) == len(panel)
    assert len(icir.weights) == len(panel["date"].unique()) * 2
    assert optimized.weights["weight"].notna().all()
    target_date = pd.Timestamp("2024-01-04")
    # The extreme IC published on the target date must not influence that date.
    target_weights = ic.weights.loc[ic.weights["date"] == target_date, "weight"]
    assert target_weights.abs().sum() == pytest.approx(1.0)
