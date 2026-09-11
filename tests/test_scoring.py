import pandas as pd
import pytest

from quant.portfolio.factor_processing import FactorProcessor, FactorSpec
from quant.portfolio.scoring import CompositeScorer


def test_composite_score_applies_weights_and_directions(tmp_path):
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 2,
            "code": ["000001", "000002"],
            "momentum_5": [1.0, 3.0],
            "volatility": [3.0, 1.0],
        }
    )
    scorer = CompositeScorer(
        FactorProcessor(
            {
                "momentum_5": FactorSpec(weight=0.5, direction=1),
                "volatility": FactorSpec(weight=0.5, direction=-1),
            }
        )
    )

    result = scorer.score_and_store(panel, tmp_path / "composite_score.parquet")

    assert list(result.columns) == ["date", "code", "composite_score"]
    assert result.loc[result["code"] == "000002", "composite_score"].iloc[0] == pytest.approx(1.0)
    assert (tmp_path / "composite_score.parquet").exists()
