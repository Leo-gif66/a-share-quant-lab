import pandas as pd
import pytest

from quant.portfolio.optimizer import EqualWeightOptimizer


def test_equal_weight_optimizer_selects_top_n_per_date():
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 3,
            "code": ["000003", "000001", "000002"],
            "composite_score": [0.1, 0.9, 0.5],
        }
    )

    result = EqualWeightOptimizer(top_n=2).construct(scores)

    assert result["code"].tolist() == ["000001", "000002"]
    assert result["weight"].tolist() == pytest.approx([0.5, 0.5])
    assert result["weight"].sum() == pytest.approx(1.0)
