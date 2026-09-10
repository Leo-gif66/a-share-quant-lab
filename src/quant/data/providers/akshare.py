"""AKShare implementation for the v0.2 daily-price data contract."""

from __future__ import annotations

from datetime import date, datetime
from typing import ClassVar

import pandas as pd


class AKShareProvider:
    """Fetch A-share daily history and normalize it to the project schema."""

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
    )

    _COLUMN_MAP: ClassVar[dict[str, str]] = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
        # Keeping the canonical names makes this adapter convenient to test and
        # tolerant of already-normalized data sources.
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
        "turnover": "turnover",
    }

    def __init__(self, ak_client=None) -> None:
        """Create a provider, optionally injecting an AKShare-compatible client."""
        if ak_client is None:
            import akshare as ak

            ak_client = ak
        self._ak = ak_client

    def get_daily_history(
        self,
        symbol: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Return daily OHLCV history with the v0.2 standard columns.

        ``stock_zh_a_hist`` expects dates in ``YYYYMMDD`` form.  The public
        method also accepts regular date/datetime values so callers do not need
        to handle that API detail themselves.
        """
        raw = self._ak.stock_zh_a_hist(
            symbol=str(symbol).zfill(6),
            period="daily",
            start_date=self._format_date(start_date),
            end_date=self._format_date(end_date),
            adjust=adjust,
        )
        if raw is None:
            raise RuntimeError(f"AKShare returned no data for {symbol}")
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("AKShare stock_zh_a_hist must return a pandas DataFrame")

        frame = raw.rename(columns=self._COLUMN_MAP)
        missing = [column for column in self.COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"AKShare history missing required columns: {', '.join(missing)}")

        result = frame.loc[:, self.COLUMNS].copy()
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        for column in self.COLUMNS[1:]:
            result[column] = pd.to_numeric(result[column], errors="coerce")

        return result.dropna(subset=["date"]).sort_values("date").drop_duplicates(
            subset=["date"], keep="last"
        ).reset_index(drop=True)

    @staticmethod
    def _format_date(value: str | date | datetime) -> str:
        return pd.Timestamp(value).strftime("%Y%m%d")
