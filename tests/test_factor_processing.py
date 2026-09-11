import pandas as pd
import pytest

from quant.portfolio.factor_processing import FactorProcessor, FactorSpec, standardize, winsorize


def test_winsorization_and_standardization_are_cross_sectional_by_date():
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 21,
            "code": [f"{index:06d}" for index in range(21)],
            "momentum_5": [0.0] * 20 + [100.0],
        }
    )

    clipped = winsorize(panel, ["momentum_5"])
    standardized = standardize(clipped, ["momentum_5"])

    assert clipped["momentum_5"].max() < 100.0
    assert standardized["momentum_5_z"].mean() == pytest.approx(0.0)
    assert standardized["momentum_5_z"].std(ddof=0) == pytest.approx(1.0)


def test_processor_resolves_feature_aliases_and_creates_z_scores():
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 2,
            "code": ["000001", "000002"],
            "mom_5": [1.0, 3.0],
            "volatility_20": [3.0, 1.0],
        }
    )
    processor = FactorProcessor(
        {
            "momentum_5": FactorSpec(weight=0.5, direction=1),
            "volatility": FactorSpec(weight=0.5, direction=-1),
        }
    )

    result = processor.process(panel)

    assert {"momentum_5", "volatility", "momentum_5_z", "volatility_z"}.issubset(result)
    assert result.loc[1, "momentum_5_z"] == pytest.approx(1.0)
    assert result.loc[1, "volatility_z"] == pytest.approx(-1.0)
