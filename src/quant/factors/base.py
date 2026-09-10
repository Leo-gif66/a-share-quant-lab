from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class Factor(ABC):
    name: str

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.Series: ...
