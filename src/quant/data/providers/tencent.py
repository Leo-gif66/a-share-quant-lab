"""Tencent Finance implementation for daily A-share history."""

from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, ClassVar

import pandas as pd
import requests

from .base import MarketDataProvider


class TencentProvider(MarketDataProvider):
    """Fetch forward-adjusted A-share daily K-lines from Tencent Finance."""

    ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    # Tencent's legacy hostname occasionally returns HTTP 501 even though the
    # identical service remains available through the official finance proxy.
    PROXY_ENDPOINT = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fqkline/get"
    COLUMNS = MarketDataProvider.DAILY_HISTORY_COLUMNS
    _MAX_ROWS_PER_REQUEST = 640
    # A calendar interval of 600 days is safely below Tencent's 640 trading-bar
    # response cap, while still keeping a multi-year update reasonably small.
    _MAX_DAYS_PER_REQUEST = 600
    _REQUEST_LOCK: ClassVar[threading.Lock] = threading.Lock()
    _LAST_REQUEST_AT: ClassVar[float] = 0.0

    def __init__(
        self, session: requests.Session | None = None, timeout: float = 15.0, request_pause: float = 0.05
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if request_pause < 0:
            raise ValueError("request_pause cannot be negative")
        # ``requests.Session`` is not a safe shared mutable transport for a
        # ThreadPoolExecutor. Keep an injected session for deterministic tests,
        # while production downloads get one proxy-free session per worker.
        self._session = session
        self._sessions = threading.local() if session is None else None
        # The Tencent provider should work in the same proxy-free environment as
        # AKShare and should not inherit a user's system proxy configuration.
        if session is not None and hasattr(session, "trust_env"):
            session.trust_env = False
        self.timeout = timeout
        self.request_pause = request_pause

    def get_daily_history(
        self,
        symbol: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> pd.DataFrame:
        """Return forward-adjusted daily bars for the requested date range."""
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if start > end:
            raise ValueError("start_date must not be after end_date")

        market_symbol = self._to_tencent_symbol(symbol)
        frames = []
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
        return result.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(
            drop=True
        )

    def _fetch_interval(
        self, market_symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        variable = f"kline_dayqfq{market_symbol}"
        params = {
            "_var": variable,
            "param": (
                f"{market_symbol},day,{start_date:%Y-%m-%d},{end_date:%Y-%m-%d},"
                f"{self._MAX_ROWS_PER_REQUEST},qfq"
            ),
        }
        session = self._active_session()
        self._rate_limit()
        response = session.get(self.ENDPOINT, params=params, timeout=self.timeout)
        if getattr(response, "status_code", 200) == 501:
            self._rate_limit()
            response = session.get(self.PROXY_ENDPOINT, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = self._parse_payload(response.text)

        if payload.get("code") not in (0, None):
            raise RuntimeError(f"Tencent Finance error: {payload.get('msg', payload['code'])}")
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise TypeError("Tencent Finance returned an invalid data payload")
        stock_data = data.get(market_symbol, {})
        if not isinstance(stock_data, dict):
            raise TypeError(f"Tencent Finance returned no data for {market_symbol}")
        rows = stock_data.get("qfqday") or stock_data.get("day") or []

        records: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            records.append(
                {
                    "date": row[0],
                    "open": row[1],
                    "close": row[2],
                    "high": row[3],
                    "low": row[4],
                    "volume": row[5],
                    # The legacy fqkline endpoint normally exposes six fields.
                    # Its extended format puts amount at index 8; otherwise a
                    # price-volume estimate keeps the common liquidity field
                    # usable for the project's factor pipeline.
                    "amount": row[8] if len(row) > 8 else self._estimate_amount(row[2], row[5]),
                    # Tencent does not include a historical turnover rate in
                    # its six-field response.  Use a neutral value so the
                    # standardized column remains numerically usable.
                    "turnover": row[10] if len(row) > 10 else 0.0,
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

    def _active_session(self) -> requests.Session:
        if self._session is not None:
            return self._session
        assert self._sessions is not None
        session = getattr(self._sessions, "session", None)
        if session is None:
            session = self._new_session()
            self._sessions.session = session
        return session

    def _rate_limit(self) -> None:
        if self.request_pause <= 0:
            return
        with self._REQUEST_LOCK:
            now = time.monotonic()
            wait = self.request_pause - (now - self._LAST_REQUEST_AT)
            if wait > 0:
                time.sleep(wait)
            type(self)._LAST_REQUEST_AT = time.monotonic()

    @staticmethod
    def _new_session() -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        return session

    @staticmethod
    def _to_tencent_symbol(symbol: str) -> str:
        value = str(symbol).lower()
        if value.startswith(("sh", "sz", "bj")):
            return value
        code = value.zfill(6)
        if code.startswith(("5", "6", "9")):
            return f"sh{code}"
        if code.startswith(("4", "8")):
            return f"bj{code}"
        return f"sz{code}"

    @staticmethod
    def _estimate_amount(close: Any, volume: Any) -> float | None:
        try:
            return float(close) * float(volume)
        except (TypeError, ValueError):
            return None
