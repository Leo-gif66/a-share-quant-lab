"""Common interface for market-data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import ClassVar

import pandas as pd


class MarketDataProvider(ABC):
    """Provider contract for normalized A-share daily price history."""

    DAILY_HISTORY_COLUMNS: ClassVar[tuple[str, ...]] = (
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
    )

    @abstractmethod
    def get_daily_history(
        self,
        symbol: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> pd.DataFrame:
        """Return daily history using :attr:`DAILY_HISTORY_COLUMNS`."""
