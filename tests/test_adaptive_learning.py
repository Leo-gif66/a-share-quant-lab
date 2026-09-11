import pandas as pd
import pytest

from quant.portfolio.factor_processing import FactorSpec
from quant.research import AdaptiveFactorWeightEngine


def test_combined_adaptive_weights_use_ic_icir_and_realized_contribution(tmp_path):
    engine = AdaptiveFactorWeightEngine(
        {"momentum_5": FactorSpec(0.5, 1), "volatility": FactorSpec(0.5, -1)}
    )
    statistics = pd.DataFrame(
        {"factor": ["momentum_5", "volatility_20"], "IC": [0.01, 0.02], "ICIR": [0.5, 1.0]}
    )
    contribution = pd.DataFrame(
        {"factor_name": ["momentum_5", "volatility"], "contribution": [0.03, 0.01]}
    )

    result = engine.calculate("combined_adaptive", statistics, realized_contribution=contribution)
    path = engine.save(result, tmp_path / "learned_factor_weights.yaml")

    assert result.method == "combined"
    assert result.weights["weight"].tolist() == pytest.approx([11 / 24, 13 / 24])
    assert path.exists()
    assert "method: combined" in path.read_text(encoding="utf-8")
