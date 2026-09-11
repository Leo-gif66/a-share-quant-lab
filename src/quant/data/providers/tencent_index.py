"""Tencent Finance provider for the CSI 300 daily index series."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import requests


class TencentIndexProvider:
    """Fetch unadjusted daily CSI 300 bars from Tencent Finance.

    Tencent's index K-line response does not include historical traded amount.
    ``amount`` is therefore an estimated value (``close * volume``), kept so
    the stored benchmark has the same liquidity-oriented fields as price data.
    """

    ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline"
    CSI300_CODE = "000300"
    COLUMNS = ("date", "open", "high", "low", "close", "volume", "amount")
    _MAX_ROWS_PER_REQUEST = 640
    _MAX_DAYS_PER_REQUEST = 600

    def __init__(self, session: requests.Session | None = None, timeout: float = 15.0) -> None:
        self._session = session or requests.Session()
        if hasattr(self._session, "trust_env"):
            self._session.trust_env = False
        self.timeout = timeout

    def get_daily_history(
        self,
        symbol: str = CSI300_CODE,
        start_date: str | date | datetime = "20180101",
        end_date: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        """Return daily CSI 300 OHLCV bars in the normalized benchmark schema."""
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date or datetime.now(UTC).date()).normalize()
        if start > end:
            raise ValueError("start_date must not be after end_date")

        market_symbol = self._to_tencent_symbol(symbol)
        frames: list[pd.DataFrame] = []
        interval_start = start
        while interval_start <= end:
            interval_end = min(
                interval_start + timedelta(days=self._MAX_DAYS_PER_REQUEST - 1), end
            )
            frames.append(self._fetch_interval(market_symbol, interval_start, interval_end))
            interval_start = interval_end + timedelta(days=1)

        if not frames:
            return pd.DataFrame(columns=self.COLUMNS)

        result = pd.concat(frames, ignore_index=True)
        result = result[(result["date"] >= start) & (result["date"] <= end)]
        return result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    def _fetch_interval(
        self, market_symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        params = {
            "param": (
                f"{market_symbol},day,{start_date:%Y-%m-%d},{end_date:%Y-%m-%d},"
                f"{self._MAX_ROWS_PER_REQUEST}"
            )
        }
        response = self._session.get(self.ENDPOINT, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = self._parse_payload(response.text)

        if payload.get("code") not in (0, None):
            raise RuntimeError(f"Tencent Finance error: {payload.get('msg', payload['code'])}")
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise TypeError("Tencent Finance returned an invalid data payload")
        index_data = data.get(market_symbol, {})
        if not isinstance(index_data, dict):
            raise TypeError(f"Tencent Finance returned no data for {market_symbol}")
        rows = index_data.get("day") or []

        records: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            records.append(
                {
                    "date": row[0],
                    "open": row[1],
                    "high": row[3],
                    "low": row[4],
                    "close": row[2],
                    "volume": row[5],
                    "amount": self._amount(row),
                }
            )

        frame = pd.DataFrame.from_records(records, columns=self.COLUMNS)
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in self.COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["date"])

    @staticmethod
    def _parse_payload(text: str) -> dict[str, Any]:
        payload_text = text.strip().rstrip(";")
        if not payload_text.startswith("{"):
            _, separator, payload_text = payload_text.partition("=")
            if not separator:
                raise ValueError("Tencent Finance response is not valid JSON or JSONP")
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            raise TypeError("Tencent Finance response root must be a JSON object")
        return payload

    @classmethod
    def benchmark_code(cls, symbol: str = CSI300_CODE) -> str:
        """Normalize accepted CSI 300 aliases to its six-digit storage code."""
        value = str(symbol).strip().lower()
        for prefix in ("csi", "sh", "sz"):
            if value.startswith(prefix):
                value = value.removeprefix(prefix)
                break
        code = value.zfill(6)
        if code != cls.CSI300_CODE:
            raise ValueError("TencentIndexProvider supports only CSI 300 (000300)")
        return code

    @classmethod
    def _to_tencent_symbol(cls, symbol: str) -> str:
        return f"sh{cls.benchmark_code(symbol)}"

    @staticmethod
    def _amount(row: list[Any]) -> float | None:
        """Use an API amount field when supplied, otherwise estimate it."""
        # Tencent's documented index payload currently has six fields.  Keep
        # support for an extended response where historical amount is present.
        if len(row) > 6:
            try:
                return float(row[6])
            except (TypeError, ValueError):
                pass
        try:
            return float(row[2]) * float(row[5])
        except (TypeError, ValueError):
            return None
