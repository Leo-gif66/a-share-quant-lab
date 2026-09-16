from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def universe(self) -> pd.DataFrame: ...

    @abstractmethod
    def stock_daily(self, code: str, start: str, end: str) -> pd.DataFrame: ...

    @abstractmethod
    def benchmark_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...
