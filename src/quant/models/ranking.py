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


ModelName = Literal["lightgbm", "random_forest", "linear", "xgboost"]


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


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

    def fit(self, data: pd.DataFrame, features: Sequence[str], label: str) -> "RankingModel":
        required = {"date", label, *features}
        if not required.issubset(data.columns):
            raise ValueError(f"ranking data missing columns: {', '.join(sorted(required.difference(data.columns)))}")
        frame = data.loc[:, ["date", *features, label]].copy().dropna()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.sort_values("date", kind="stable")
        if frame.empty or frame["date"].nunique() < 2:
            raise ValueError("ranking training requires at least two non-empty dates")
        self.feature_names = list(features)
        X, y = frame[self.feature_names], pd.to_numeric(frame[label], errors="raise")
        if self.model_name == "lightgbm":
            from lightgbm import LGBMRanker

            parameters = {"objective": "lambdarank", "random_state": 0, **self.params}
            model = LGBMRanker(**parameters)
            # LambdaRank requires integer relevance labels.  Cross-sectional
            # ranks preserve the target ordering without treating a continuous
            # return as an arbitrary relevance gain.
            relevance = (
                frame.groupby("date")[label].rank(method="first").astype(int).sub(1).to_numpy()
            )
            groups = frame.groupby("date", sort=True).size().to_list()
            model.fit(X, relevance, group=groups)
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
        return pd.Series(self.model.predict(data[self.feature_names]), index=data.index, name="prediction")  # type: ignore[union-attr]

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
    values[label] = pd.to_numeric(values[label], errors="coerce")
    values["prediction"] = pd.to_numeric(values["prediction"], errors="coerce")
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
