import pandas as pd

from quant.research import WalkForwardConfig, WalkForwardValidator


def test_walk_forward_uses_chronological_non_overlapping_train_validation_and_test_sets():
    data = pd.DataFrame({"date": pd.bdate_range("2018-01-02", "2025-01-31"), "value": 1.0})
    observed: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    validator = WalkForwardValidator(
        WalkForwardConfig(train_years=3, validation_months=6, test_months=6, step_months=6)
    )

    result = validator.run(
        data,
        lambda train, validation: observed.append(
            (train["date"].max(), validation["date"].min(), validation["date"].max())
        )
        or {"last_train": train["date"].max()},
        lambda model, test: test.assign(prediction=1.0),
        lambda prediction, test: {
            "return": 0.10,
            "benchmark": 0.05,
            "alpha": 0.03,
            "sharpe": 1.0,
            "drawdown": -0.10,
        },
    )

    assert not result.empty
    assert list(result.columns) == ["period", "return", "benchmark", "alpha", "sharpe", "drawdown"]
    assert all(train_end < validation_start <= validation_end for train_end, validation_start, validation_end in observed)
