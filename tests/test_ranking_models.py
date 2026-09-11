import pandas as pd

from quant.models import (
    RankingModel,
    add_future_excess_return,
    prediction_ic_metrics,
    time_ordered_split,
)


def _ranking_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=80)
    return pd.DataFrame(
        {
            "date": [date for date in dates for _ in range(2)],
            "code": ["000001", "000002"] * len(dates),
            "close": [10 + day * (0.03 if code == "000001" else 0.01) for day in range(len(dates)) for code in range(2)],
            "factor_a": [float(day) for day in range(len(dates)) for _ in range(2)],
            "factor_b": [float(code) for _ in range(len(dates)) for code in (1, -1)],
        }
    )


def test_ranking_model_uses_time_split_and_exposes_feature_importance(tmp_path):
    panel = _ranking_panel()
    labelled = panel.assign(future_excess_return_20d=panel["factor_b"] * 0.01)
    split = time_ordered_split(labelled, purge_dates=2)
    model = RankingModel("random_forest", {"n_estimators": 10})
    model.fit(split.train, ["factor_a", "factor_b"], "future_excess_return_20d")

    prediction = model.predict(split.test)
    importance = model.feature_importance()
    metrics = prediction_ic_metrics(split.test, prediction, "future_excess_return_20d", min_cross_section=2)
    path = model.save(tmp_path / "models" / "ranker.joblib")

    assert split.train["date"].max() < split.validation["date"].min() < split.test["date"].min()
    assert len(prediction) == len(split.test)
    assert set(importance["feature"]) == {"factor_a", "factor_b"}
    assert {"prediction_IC", "prediction_Rank_IC"}.issubset(metrics)
    assert path.exists()


def test_future_excess_return_label_subtracts_same_horizon_benchmark_return():
    panel = _ranking_panel().loc[lambda frame: frame["date"] < pd.Timestamp("2024-01-12")]
    dates = sorted(panel["date"].unique())
    benchmark = pd.DataFrame({"date": dates, "close": [100 + index for index in range(len(dates))]})

    labelled = add_future_excess_return(panel, benchmark, horizon=2)

    row = labelled.loc[(labelled["code"] == "000001") & (labelled["date"] == dates[0])].iloc[0]
    assert row["future_excess_return_20d"] == row["future_return_20d"] - row["benchmark_future_return_20d"]
