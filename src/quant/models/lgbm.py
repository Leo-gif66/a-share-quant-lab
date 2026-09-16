from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .base import BaseAlphaModel
from .registry import register


@register("lightgbm")
class LightGBMAlphaModel(BaseAlphaModel):
    name = "lightgbm"
    def __init__(self, params: dict):
        from lightgbm import LGBMRegressor
        self.params = {k: v for k, v in dict(params).items() if k != "weights"}
        self.model = LGBMRegressor(**self.params)
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X[self.feature_names])

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, path)
