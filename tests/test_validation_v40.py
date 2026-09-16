import numpy as np
import pandas as pd
import yaml

from quant.evolution import StrategyVersionStore
from quant.research import PerformanceAttributionEngine
from quant.validation import (
    RobustnessConfig,
    RobustnessTester,
    WalkForwardSettings,
    WalkForwardSimulator,
    benchmark_comparison,
)


def _market_data(periods: int = 70):
    dates = pd.bdate_range("2024-01-02", periods=periods)
    codes = ["000001", "000002", "000003", "000004", "000005"]
    scores = pd.concat(
        [
            pd.DataFrame({"date": dates, "code": code, "composite_score": rank + np.arange(periods) * 0.002})
            for rank, code in enumerate(codes, start=1)
        ],
        ignore_index=True,
    )
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "code": code,
                    "open": 10 + rank + np.arange(periods) * (0.01 * rank),
                    "close": 10 + rank + np.arange(periods) * (0.011 * rank),
                }
            )
            for rank, code in enumerate(codes, start=1)
        ],
        ignore_index=True,
    )
    benchmark = pd.DataFrame({"date": dates, "close": 4_000 + np.arange(periods) * 1.5})
    industries = {code: "Bank" if index < 3 else "Technology" for index, code in enumerate(codes)}
    return scores, prices, benchmark, industries


def test_walk_forward_is_chronological_and_only_trains_on_matured_labels(tmp_path):
    scores, prices, benchmark, industries = _market_data()
    result = WalkForwardSimulator(
        WalkForwardSettings(train_window=40, validation_window=10, rebalance_frequency=5, top_n=5)
    ).run(
        scores,
        prices,
        benchmark,
        industries=industries,
        output_path=tmp_path / "walk_forward_results.parquet",
        memory=None,
    )

    assert result.output_path.exists()
    assert not result.results.empty
    rebalances = result.results.loc[result.results["rebalance"]]
    assert (pd.to_datetime(rebalances["training_end_date"]) < pd.to_datetime(rebalances["date"])).all()
    assert result.metrics["turnover"] >= 0
    assert result.memory_path.exists()


def test_strategy_versions_are_immutable_and_written_only_for_weight_changes(tmp_path):
    store = StrategyVersionStore(tmp_path / "versions")
    first = pd.DataFrame({"factor": ["momentum", "volatility"], "weight": [0.5, 0.5], "direction": [1, -1]})
    second = pd.DataFrame({"factor": ["momentum", "volatility"], "weight": [0.6, 0.4], "direction": [1, -1]})

    version_one = store.record_if_changed(first, "baseline")
    duplicate = store.record_if_changed(first, "unchanged")
    version_two = store.record_if_changed(second, "validated adjustment", {"sharpe": 1.0}, {"sharpe": 1.1})

    assert version_one.path.name == "v001.yaml"
    assert not duplicate.changed
    assert version_two.path.name == "v002.yaml"
    payload = yaml.safe_load(version_two.path.read_text(encoding="utf-8"))
    assert {"timestamp", "factors", "reason", "performance_before", "performance_after"}.issubset(payload)
    assert len(store.history()) == 2


def test_performance_attribution_reconciles_components_to_simple_portfolio_return():
    dates = pd.bdate_range("2024-01-02", periods=5)
    equity = pd.DataFrame({"date": dates, "equity": [100, 101, 103, 102, 104]})
    benchmark = pd.DataFrame({"date": dates, "benchmark": [100, 100.5, 101, 100.8, 101.5]})
    holdings = pd.DataFrame(
        {"date": [dates[0], dates[0]], "code": ["1", "2"], "industry": ["Bank", "Technology"], "weight": [0.5, 0.5]}
    )
    prices = pd.concat(
        [
            pd.DataFrame({"date": dates, "code": code, "close": values})
            for code, values in [("1", [10, 10.2, 10.4, 10.1, 10.5]), ("2", [20, 20.0, 20.4, 20.3, 20.6])]
        ],
        ignore_index=True,
    )
    result = PerformanceAttributionEngine().decompose(equity, benchmark, holdings, prices)

    values = result.summary.set_index("component")["contribution"]
    components = values.loc[
        ["market_beta_contribution", "industry_allocation_contribution", "stock_selection_contribution", "factor_timing_contribution"]
    ].sum()
    assert np.isclose(components, values["portfolio_simple_return"])


def test_robustness_executes_real_runner_for_each_requested_scenario():
    calls: list[dict[str, float | int]] = []

    def runner(parameters):
        calls.append(dict(parameters))
        return {"annual_return": 0.1, "sharpe": 1.0, "max_drawdown": -0.1, "turnover": 2.0}

    config = RobustnessConfig(
        rebalance_periods=(5,),
        transaction_cost_multipliers=(1.0,),
        factor_weight_perturbations=(0.05,),
        universe_fractions=(0.8,),
    )
    results = RobustnessTester(config).run(runner)

    assert len(calls) == 4
    assert set(results["status"]) == {"available"}


def test_benchmark_comparison_marks_missing_indexes_unavailable_without_proxying():
    dates = pd.bdate_range("2024-01-02", periods=4)
    equity = pd.DataFrame({"date": dates, "equity": [100, 101, 102, 103]})
    csi300 = pd.DataFrame({"date": dates, "close": [4_000, 4_020, 4_030, 4_040]})
    result = benchmark_comparison(equity, {"CSI300": csi300, "CSI500": None, "CSI1000": None})

    assert result.loc[result["benchmark"] == "CSI300", "status"].iloc[0] == "available"
    assert set(result.loc[result["benchmark"] != "CSI300", "status"]) == {"unavailable"}
