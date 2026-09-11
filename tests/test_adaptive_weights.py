import pandas as pd
import pytest

from quant.portfolio.factor_processing import FactorSpec
from quant.research import AdaptiveFactorWeightEngine


def test_adaptive_factor_weights_support_all_requested_methods_and_save_yaml(tmp_path):
    engine = AdaptiveFactorWeightEngine(
        {"momentum_5": FactorSpec(0.5, 1), "volatility": FactorSpec(0.5, -1)}
    )
    statistics = pd.DataFrame(
        {"factor": ["momentum_5", "volatility_20"], "IC": [0.02, -0.04], "ICIR": [0.5, -1.0], "average_return": [0.01, -0.03]}
    )

    equal = engine.calculate("equal", statistics)
    ic = engine.calculate("ic", statistics)
    icir = engine.calculate("icir", statistics)
    performance = engine.calculate("rolling_performance", rolling_performance=statistics)
    path = engine.save(icir, tmp_path / "adaptive_factor_weights.yaml")

    assert equal.weights["weight"].tolist() == pytest.approx([0.5, 0.5])
    assert ic.weights["weight"].tolist() == pytest.approx([1 / 3, 2 / 3])
    assert icir.weights["weight"].tolist() == pytest.approx([1 / 3, 2 / 3])
    assert performance.weights["weight"].tolist() == pytest.approx([0.25, 0.75])
    assert "direction: -1" in path.read_text(encoding="utf-8")
