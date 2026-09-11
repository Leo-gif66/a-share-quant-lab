import numpy as np
import pandas as pd
import pytest

from quant.models import RankingModel, encode_ranking_labels, validate_ranking_labels
from quant.research import FactorEvaluator, ResearchPreprocessor


def _ml_panel() -> pd.DataFrame:
    rows = []
    for date in pd.bdate_range("2024-01-02", periods=12):
        for code, value in enumerate((0.01, 0.02, 0.03), start=1):
            rows.append(
                {
                    "date": date,
                    "code": f"{code:06d}",
                    "factor_a": value * 10,
                    "factor_b": value * 20,
                    "future_excess_return_20d": value,
                }
            )
    return pd.DataFrame(rows)


def test_preprocessor_replaces_infinities_imputes_missing_and_records_coverage():
    panel = _ml_panel()
    panel.loc[0, "factor_a"] = np.inf
    panel.loc[1, "factor_a"] = np.nan

    result = ResearchPreprocessor().process(panel, ["factor_a", "factor_b"])

    assert np.isfinite(result.data[["factor_a", "factor_b"]].to_numpy(dtype=float)).all()
    coverage = result.coverage.set_index("factor")
    assert coverage.at["factor_a", "nan_ratio"] > 0
    assert coverage.at["factor_a", "imputed_samples"] == 2


def test_ml_training_never_receives_infinite_values():
    panel = _ml_panel()
    panel.loc[0, "factor_a"] = np.inf
    panel.loc[1, "factor_b"] = -np.inf
    model = RankingModel("linear")

    model.fit(panel, ["factor_a", "factor_b"], "future_excess_return_20d")
    prediction = model.predict(panel)

    assert prediction.notna().all()
    assert np.isfinite(prediction.to_numpy()).all()


def test_lightgbm_relevance_labels_are_continuous_per_time_window():
    panel = _ml_panel()
    panel.loc[panel.index[:3], "future_excess_return_20d"] = [268.0, 120.0, 31.0]

    encoded = encode_ranking_labels(panel, "future_excess_return_20d")

    assert encoded.number_of_classes == 3
    assert encoded.labels.min() >= 0
    assert encoded.labels.max() < encoded.number_of_classes
    validate_ranking_labels(encoded.labels, encoded.groups, encoded.number_of_classes)
    for _, group in pd.DataFrame({"date": panel["date"], "label": encoded.labels}).groupby("date"):
        assert sorted(group["label"].tolist()) == [0, 1, 2]


def test_lightgbm_fit_accepts_validated_window_labels():
    model = RankingModel(
        "lightgbm", {"n_estimators": 5, "min_child_samples": 1, "verbosity": -1}
    )

    model.fit(_ml_panel(), ["factor_a", "factor_b"], "future_excess_return_20d")

    assert model.ranking_label_info == {
        "minimum_label": 0,
        "maximum_label": 2,
        "number_of_classes": 3,
    }


def test_factor_evaluation_survives_missing_data_and_hides_insufficient_ic():
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 3,
            "momentum_5": [1.0, np.inf, np.nan],
            "future_return_20d": [0.01, 0.02, 0.03],
        }
    )

    result = FactorEvaluator(factors=("momentum_5",), min_ic_samples=2).analyze(panel)

    row = result.iloc[0]
    assert row["status"] == "insufficient_data"
    assert pd.isna(row["IC"])
    assert row["nan_ratio"] > 0
    assert row["IC_sample_count"] == 0


def test_factor_ic_remains_stable_after_clean_validation():
    rows = []
    for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
        for factor, target in zip((1.0, 2.0, 3.0), (0.01, 0.02, 0.03), strict=True):
            rows.append({"date": date, "momentum_5": factor, "future_return_20d": target})

    result = FactorEvaluator(factors=("momentum_5",), min_ic_samples=2).evaluate(
        pd.DataFrame(rows)
    )

    assert result.loc[0, "status"] == "ok"
    assert result.loc[0, "IC"] == pytest.approx(1.0)
    assert result.loc[0, "IC_sample_count"] == 2
