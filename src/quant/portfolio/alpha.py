"""Leak-safe ML ranking scores on the unchanged institutional portfolio path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..models import RankingModel
from .institutional import InstitutionalPortfolioBacktestEngine

MLModelName = Literal["linear", "random_forest", "lightgbm", "xgboost"]


class MLRankingPortfolioBacktestEngine(InstitutionalPortfolioBacktestEngine):
    """Supply rolling, matured-label ML scores to existing v2.0 accounting."""

    def __init__(
        self,
        *args: object,
        model_name: MLModelName = "random_forest",
        label_horizon: int = 20,
        train_window: int = 252,
        retrain_interval: int = 60,
        **kwargs: object,
    ) -> None:
        if label_horizon < 1 or train_window < 20 or retrain_interval < 1:
            raise ValueError("label_horizon, train_window, and retrain_interval are invalid")
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.label_horizon = label_horizon
        self.train_window = train_window
        self.retrain_interval = retrain_interval
        self.ml_feature_importance = pd.DataFrame(columns=["feature", "importance"])

    def _score_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        prepared = self._label_and_normalize(panel)
        factors = [f"{factor}_z" for factor in self.scorer.processor.factor_names]
        dates = pd.DatetimeIndex(sorted(prepared["date"].unique()))
        frames: list[pd.DataFrame] = []
        importances: list[pd.DataFrame] = []
        model: RankingModel | None = None
        last_training_index = -self.retrain_interval
        for index in range(0, len(dates) - 1, self.rebalance_interval):
            date = dates[index]
            if index < self.label_horizon:
                continue
            if model is None or index - last_training_index >= self.retrain_interval:
                model = self._fit_model(prepared, dates, index, factors)
                last_training_index = index
                if model is not None:
                    importances.append(model.feature_importance())
            if model is None:
                continue
            current = prepared.loc[prepared["date"] == date].dropna(subset=factors)
            if current.empty:
                continue
            prediction = model.predict(current)
            frames.append(
                pd.DataFrame(
                    {"date": date, "code": current["code"].to_numpy(), "composite_score": prediction.to_numpy()}
                )
            )
        if importances:
            self.ml_feature_importance = (
                pd.concat(importances, ignore_index=True)
                .groupby("feature", as_index=False)["importance"].mean()
                .sort_values("importance", ascending=False, kind="stable")
                .reset_index(drop=True)
            )
        if not frames:
            return pd.DataFrame(columns=["date", "code", "composite_score"])
        scores = pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(self.features_dir / "ml_composite_score.parquet", index=False)
        return scores

    def _label_and_normalize(self, panel: pd.DataFrame) -> pd.DataFrame:
        values = panel.sort_values(["code", "date"]).copy()
        values["future_return_20d"] = values.groupby("code")["close"].transform(
            lambda close: close.shift(-self.label_horizon) / close - 1.0
        )
        benchmark = self._load_benchmark_prices(pd.DatetimeIndex(sorted(panel["date"].unique())))
        benchmark["benchmark_future_return_20d"] = (
            benchmark["close"].shift(-self.label_horizon) / benchmark["close"] - 1.0
        )
        values = values.merge(
            benchmark.loc[:, ["date", "benchmark_future_return_20d"]], on="date", how="left"
        )
        values["future_excess_return_20d"] = (
            values["future_return_20d"] - values["benchmark_future_return_20d"]
        )
        return self.scorer.processor.process(values)

    def _fit_model(
        self, prepared: pd.DataFrame, dates: pd.DatetimeIndex, index: int, factors: list[str]
    ) -> RankingModel | None:
        mature_index = index - self.label_horizon
        if mature_index < 0:
            return None
        start_index = max(0, mature_index - self.train_window + 1)
        train_dates = dates[start_index : mature_index + 1]
        train = prepared.loc[prepared["date"].isin(train_dates)].dropna(
            subset=[*factors, "future_excess_return_20d"]
        )
        if train["date"].nunique() < 20 or len(train) < max(100, len(factors) * 10):
            return None
        model = RankingModel(self.model_name, {"n_estimators": 50} if self.model_name == "random_forest" else None)
        return model.fit(train, factors, "future_excess_return_20d")


@dataclass(frozen=True)
class AlphaBacktestComparison:
    table: pd.DataFrame


def compare_portfolio_results(handcrafted: object, ml_ranking: object) -> AlphaBacktestComparison:
    """Present comparable metrics from two backtests sharing the same accounting."""
    names = ("annual_return", "sharpe", "max_drawdown", "alpha", "beta", "turnover")
    rows = []
    for name, result in (("handcrafted", handcrafted), ("ml_ranking", ml_ranking)):
        if not hasattr(result, "metrics"):
            raise TypeError("portfolio result must expose metrics")
        rows.append({"portfolio": name, **{metric: result.metrics.get(metric, np.nan) for metric in names}})
    return AlphaBacktestComparison(pd.DataFrame(rows))
