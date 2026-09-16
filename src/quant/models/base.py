from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseAlphaModel(ABC):
    name: str
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series): ...
    @abstractmethod
    def predict(self, X: pd.DataFrame): ...
    @abstractmethod
    def save(self, path: Path): ...
