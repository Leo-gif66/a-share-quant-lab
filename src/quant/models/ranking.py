"""Time-ordered machine-learning ranking models for factor research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from ..research.preprocessing import ResearchPreprocessor


ModelName = Literal["lightgbm", "random_forest", "linear", "xgboost"]


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class RankingLabelEncoding:
    """Per-date integer relevance labels accepted by LightGBM LambdaRank."""

    labels: pd.Series
    groups: list[int]
    number_of_classes: int


def encode_ranking_labels(
    data: pd.DataFrame, label: str, date_column: str = "date"
) -> RankingLabelEncoding:
    """Encode each training-date cross-section as ``0 .. group_size - 1``.

    Ranking relevance is calculated independently for each date.  That makes
    a target's encoding depend only on its contemporaneous cross-section, not
    observations from later windows, and avoids LightGBM's invalid sparse or
    oversized label-mapping values.
    """
    if date_column not in data or label not in data:
        raise ValueError("ranking labels require date and target columns")
    frame = data.loc[:, [date_column, label]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    frame[label] = pd.to_numeric(frame[label], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if frame[label].isna().any():
        raise ValueError("ranking labels require finite target values")

    labels = pd.Series(index=frame.index, dtype="int64")
    groups: list[int] = []
    for _, group in frame.groupby(date_column, sort=True):
        relevance = group[label].rank(method="first", ascending=True).astype("int64") - 1
        labels.loc[group.index] = relevance
        groups.append(len(group))
    number_of_classes = max(groups, default=0)
    validate_ranking_labels(labels, groups, number_of_classes)
    return RankingLabelEncoding(labels.astype("int32"), groups, number_of_classes)


def validate_ranking_labels(
    labels: pd.Series | np.ndarray, groups: Sequence[int], number_of_classes: int
) -> None:
    """Assert the LambdaRank relevance contract before a model is fitted."""
    values = np.asarray(labels, dtype=np.int64)
    if number_of_classes < 1 or len(values) != sum(groups):
        raise ValueError("ranking labels and groups are inconsistent")
    assert values.min() >= 0
    assert values.max() < number_of_classes
    offset = 0
    for size in groups:
        group = np.sort(values[offset : offset + size])
        assert np.array_equal(group, np.arange(size))
        offset += size


def time_ordered_split(
    data: pd.DataFrame,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    purge_dates: int = 20,
) -> TimeSplit:
    """Split whole dates in chronological order with a post-train purge gap."""
    if "date" not in data:
        raise ValueError("ranking data requires date")
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("train and validation ratios must be positive and sum to less than one")
    if purge_dates < 0:
        raise ValueError("purge_dates must be non-negative")
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    train_count = int(len(dates) * train_ratio)
    validation_count = int(len(dates) * validation_ratio)
    validation_start = train_count + purge_dates
    test_start = validation_start + validation_count + purge_dates
    if train_count < 1 or validation_count < 1 or test_start >= len(dates):
        raise ValueError("not enough distinct dates for chronological train/validation/test split")
    return TimeSplit(
        train=frame.loc[frame["date"].isin(dates[:train_count])].copy(),
        validation=frame.loc[frame["date"].isin(dates[validation_start : validation_start + validation_count])].copy(),
        test=frame.loc[frame["date"].isin(dates[test_start:])].copy(),
    )


def add_future_excess_return(
    panel: pd.DataFrame, benchmark: pd.DataFrame, horizon: int = 20
) -> pd.DataFrame:
    """Create a future 20-day excess-return label from stock and index closes."""
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    if not {"date", "code", "close"}.issubset(panel.columns):
        raise ValueError("stock panel requires date, code, and close")
    if not {"date", "close"}.issubset(benchmark.columns):
        raise ValueError("benchmark requires date and close")
    stocks = panel.copy()
    stocks["date"] = pd.to_datetime(stocks["date"], errors="raise")
    stocks["code"] = stocks["code"].astype(str).str.zfill(6)
    stocks["close"] = pd.to_numeric(stocks["close"], errors="coerce")
    stocks = stocks.sort_values(["code", "date"])
    stocks["future_return_20d"] = stocks.groupby("code")["close"].transform(
        lambda values: values.shift(-horizon) / values - 1.0
    )
    index = benchmark.loc[:, ["date", "close"]].copy()
    index["date"] = pd.to_datetime(index["date"], errors="raise")
    index["close"] = pd.to_numeric(index["close"], errors="coerce")
    index = index.sort_values("date").drop_duplicates("date", keep="last")
    index["benchmark_future_return_20d"] = index["close"].shift(-horizon) / index["close"] - 1.0
    output = stocks.merge(index.loc[:, ["date", "benchmark_future_return_20d"]], on="date", how="left")
    output["future_excess_return_20d"] = (
        output["future_return_20d"] - output["benchmark_future_return_20d"]
    )
    return output.sort_values(["date", "code"]).reset_index(drop=True)


class RankingModel:
    """Fit LightGBM LambdaRank, RandomForest, Ridge, or optional XGBoost."""

    def __init__(self, model_name: ModelName = "random_forest", params: dict[str, object] | None = None) -> None:
        if model_name not in {"lightgbm", "random_forest", "linear", "xgboost"}:
            raise ValueError("model_name must be lightgbm, random_forest, linear, or xgboost")
        self.model_name = model_name
        self.params = dict(params or {})
        self.model: object | None = None
        self.feature_names: list[str] = []
        self.preprocessor = ResearchPreprocessor()
        self.ranking_label_info: dict[str, int] = {}

    def fit(self, data: pd.DataFrame, features: Sequence[str], label: str) -> "RankingModel":
        required = {"date", label, *features}
        if not required.issubset(data.columns):
            raise ValueError(f"ranking data missing columns: {', '.join(sorted(required.difference(data.columns)))}")
        cleaned = self.preprocessor.process(data, features, target_columns=(label,))
        frame = cleaned.data.loc[:, ["date", *features, label]].copy().dropna()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.sort_values("date", kind="stable")
        if frame.empty or frame["date"].nunique() < 2:
            raise ValueError("ranking training requires at least two non-empty dates")
        self.feature_names = list(features)
        X, y = frame[self.feature_names], pd.to_numeric(frame[label], errors="raise")
        ResearchPreprocessor.assert_finite(frame, [*features, label])
        if self.model_name == "lightgbm":
            from lightgbm import LGBMRanker

            parameters = {"objective": "lambdarank", "random_state": 0, **self.params}
            encoding = encode_ranking_labels(frame, label)
            # LightGBM's default label-gain mapping has only 31 entries.  A
            # complete per-date rank can be wider, so use a linear mapping
            # whose size is exactly the validated relevance class range.
            parameters["label_gain"] = list(range(encoding.number_of_classes))
            model = LGBMRanker(**parameters)
            model.fit(X, encoding.labels.to_numpy(), group=encoding.groups)
            self.ranking_label_info = {
                "minimum_label": int(encoding.labels.min()),
                "maximum_label": int(encoding.labels.max()),
                "number_of_classes": encoding.number_of_classes,
            }
        elif self.model_name == "random_forest":
            parameters = {"random_state": 0, "n_estimators": 100, **self.params}
            model = RandomForestRegressor(**parameters)
            model.fit(X, y)
        elif self.model_name == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise RuntimeError("xgboost is not installed; choose another ranking model") from exc
            parameters = {"random_state": 0, "n_estimators": 100, "n_jobs": 1, **self.params}
            model = XGBRegressor(**parameters)
            model.fit(X, y)
        else:
            model = Ridge(**self.params)
            model.fit(X, y)
        self.model = model
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("ranking model must be fitted before prediction")
        if not set(self.feature_names).issubset(data.columns):
            raise ValueError("prediction data is missing fitted feature columns")
        cleaned = self.preprocessor.process(data, self.feature_names).data
        valid = cleaned.loc[:, self.feature_names].notna().all(axis=1)
        prediction = pd.Series(np.nan, index=data.index, name="prediction", dtype="float64")
        if valid.any():
            prediction.loc[valid] = self.model.predict(cleaned.loc[valid, self.feature_names])  # type: ignore[union-attr]
        return prediction

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("ranking model must be fitted before feature importance")
        if hasattr(self.model, "feature_importances_"):
            values = np.asarray(self.model.feature_importances_, dtype=float)
        elif hasattr(self.model, "coef_"):
            values = np.abs(np.asarray(self.model.coef_, dtype=float)).reshape(-1)
        else:
            raise RuntimeError("fitted model does not expose feature importance")
        return pd.DataFrame({"feature": self.feature_names, "importance": values}).sort_values(
            "importance", ascending=False, kind="stable"
        ).reset_index(drop=True)

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("ranking model must be fitted before saving")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target


def prediction_ic_metrics(
    data: pd.DataFrame,
    prediction: pd.Series,
    label: str,
    min_cross_section: int = 5,
) -> dict[str, float]:
    """Return mean daily Pearson and Spearman IC on an out-of-sample panel."""
    if "date" not in data or label not in data:
        raise ValueError("prediction evaluation requires date and label")
    if min_cross_section < 2:
        raise ValueError("min_cross_section must be at least two")
    values = data.loc[:, ["date", label]].copy()
    values["prediction"] = prediction.reindex(data.index)
    values["date"] = pd.to_datetime(values["date"], errors="raise")
    values[label] = pd.to_numeric(values[label], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values["prediction"] = pd.to_numeric(values["prediction"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    pearson: list[float] = []
    spearman: list[float] = []
    for _, group in values.groupby("date", sort=True):
        group = group.dropna()
        if len(group) < min_cross_section or group["prediction"].nunique() < 2 or group[label].nunique() < 2:
            continue
        pearson.append(float(group["prediction"].corr(group[label], method="pearson")))
        spearman.append(float(group["prediction"].corr(group[label], method="spearman")))
    return {
        "prediction_IC": float(np.mean(pearson)) if pearson else np.nan,
        "prediction_Rank_IC": float(np.mean(spearman)) if spearman else np.nan,
        "prediction_observations": float(len(pearson)),
    }
