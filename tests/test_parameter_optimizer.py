import pytest

from quant.research import ParameterOptimizer, factor_weight_candidates


def test_parameter_optimizer_enumerates_grid_ranks_objective_and_persists_results(tmp_path):
    optimizer = ParameterOptimizer({"top_n": [10, 20], "rebalance": [10, 20]})

    result = optimizer.run(
        lambda params: {"sharpe": float(params["top_n"]) / params["rebalance"], "return": 0.1},
        output_dir=tmp_path,
    )

    assert len(result) == 4
    assert result.iloc[0]["sharpe"] >= result.iloc[-1]["sharpe"]
    assert (tmp_path / "parameter_search.csv").exists()
    weights = factor_weight_candidates(["a", "b"], values=[0.0, 1.0])
    assert all(sum(candidate.values()) == pytest.approx(1.0) for candidate in weights)
