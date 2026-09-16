from __future__ import annotations

import json

import pandas as pd
import pytest

from quant.data.fundamental import FUNDAMENTAL_COLUMNS, FundamentalStore
from quant.research import ExperimentRegistry
from quant.research.leakage import (
    assert_feature_available,
    assert_fundamental_available,
    assert_label_after_signal,
    assert_training_before_test,
)
from quant.validation import ValidationCoverageAuditor
from quant.validation.walk_forward import WalkForwardSettings


def test_experiment_registry_writes_required_metadata_without_overwrite(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    target = registry.record(
        "v5-safety",
        data_coverage={"dates": 10},
        universe={"stocks": 2},
        factor_set=["momentum_20"],
        model="linear",
        parameters={"seed": 42},
        result_metrics={"sharpe": -0.1},
        notes="negative result retained",
        git_commit="test-commit",
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload) == {
        "experiment_id", "timestamp", "git_commit", "data_coverage", "universe", "factor_set",
        "model", "parameters", "result_metrics", "notes",
    }
    with pytest.raises(FileExistsError):
        registry.record(
            "v5-safety", data_coverage={}, universe={}, factor_set=[], model="none", parameters={},
            result_metrics={}, notes="duplicate", git_commit="test-commit",
        )


def test_leakage_assertions_reject_invalid_temporal_order():
    dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert_feature_available(dates, dates)
    assert_fundamental_available(dates, dates)
    assert_label_after_signal(dates, dates + pd.Timedelta(days=1))
    assert_training_before_test(dates, pd.Series(pd.to_datetime(["2024-01-04"])))
    with pytest.raises(AssertionError):
        assert_label_after_signal(dates, dates)
    with pytest.raises(AssertionError):
        assert_training_before_test(dates, dates)


def test_fundamental_store_only_attaches_announced_disclosures(tmp_path):
    records = pd.DataFrame(
        [{column: 1.0 for column in FUNDAMENTAL_COLUMNS if column not in {"code", "report_period", "announcement_date", "effective_date"}}]
    )
    records["code"] = "1"
    records["report_period"] = "2023-12-31"
    records["announcement_date"] = "2024-03-20"
    records["effective_date"] = "2024-03-21"
    store = FundamentalStore(tmp_path / "fundamentals.parquet")
    joined = store.asof_join(
        pd.DataFrame({"date": pd.to_datetime(["2024-03-20", "2024-03-21"]), "code": ["1", "1"]}),
        records,
    )

    assert pd.isna(joined.loc[0, "roe"])
    assert joined.loc[1, "roe"] == pytest.approx(1.0)
    assert joined.loc[1, "fundamental_announcement_date"] == pd.Timestamp("2024-03-20")


def test_coverage_audit_identifies_short_score_history(tmp_path):
    raw_dir, feature_dir = tmp_path / "raw", tmp_path / "features"
    raw_dir.mkdir()
    feature_dir.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=8)
    for code in ("000001", "000002"):
        pd.DataFrame({"date": dates, "close": range(10, 18)}).to_parquet(raw_dir / f"{code}.parquet")
        pd.DataFrame({"date": dates, "close": range(10, 18)}).to_parquet(feature_dir / f"{code}.parquet")
    scores = pd.DataFrame(
        [{"date": date, "code": code, "composite_score": number} for date in dates[-3:] for number, code in enumerate(("000001", "000002"))]
    )
    score_path = feature_dir / "composite_score.parquet"
    scores.to_parquet(score_path)
    benchmark = raw_dir / "000300.parquet"
    pd.DataFrame({"date": dates, "close": range(100, 108)}).to_parquet(benchmark)

    result = ValidationCoverageAuditor(
        raw_dir=raw_dir,
        features_dir=feature_dir,
        scores_path=score_path,
        benchmark_path=benchmark,
        settings=WalkForwardSettings(train_window=2, validation_window=1, rebalance_frequency=1, top_n=2),
    ).audit()

    assert result.eligible_signal_dates == 3
    assert result.rebalances == 0
    assert "composite-score history" in result.bottleneck
