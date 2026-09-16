from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from ..utils import winsorized_zscore
from .base import BaseAlphaModel
from .registry import register


@register("rule")
class RuleAlphaModel(BaseAlphaModel):
    name = "rule"
    def __init__(self, params: dict):
        self.weights = params.get("weights", {})
        self.feature_names = list(self.weights)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame):
        score = pd.Series(0.0, index=X.index)
        for f, w in self.weights.items():
            score = score + float(w) * winsorized_zscore(X[f]).fillna(0).clip(-3, 3)
        return score.to_numpy()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, path)
