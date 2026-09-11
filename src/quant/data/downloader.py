"""Concurrent, incremental download and persistence for the A-share universe."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .providers.akshare import AKShareProvider
from .providers.base import MarketDataProvider
from .providers.tencent import TencentProvider
from .universe import Universe


class DataDownloader:
    """Download a universe without losing usable history after an API failure.

    Tencent is deliberately the default source. AKShare is used only after the
    Tencent request (including retries) has failed. Individual files are
    written through a temporary parquet file so an interrupted download never
    replaces a prior snapshot.
    """

    REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume", "amount", "turnover")
    HISTORY_MANIFEST = "download_history.json"

    def __init__(
        self,
        universe: Universe | None = None,
        provider: MarketDataProvider | str | None = "tencent",
        fallback_provider: MarketDataProvider | None = None,
        enable_fallback: bool = True,
        data_dir: str | Path = "data/raw",
        start_date: str = "20150101",
        end_date: str | None = None,
        adjust: str = "qfq",
        reporter: Callable[[str], None] = print,
        max_workers: int = 4,
        retries: int = 1,
        retry_delay: float = 0.1,
        request_pause: float = 0.05,
        progress: bool = False,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        self.universe = universe or Universe()
        self.provider = self._create_provider(provider)
        self.fallback_provider = self._create_fallback_provider(
            fallback_provider, enable_fallback
        )
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date or datetime.now(UTC).strftime("%Y%m%d")
        self.adjust = adjust
        self.reporter = reporter
        self.max_workers = max_workers
        self.retries = retries
        self.retry_delay = retry_delay
        self.request_pause = request_pause
        self.progress = progress
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0
        self._manifest_lock = threading.Lock()
        self._history_coverage: dict[str, dict[str, str]] = {}

    def update(self) -> dict[str, object]:
        """Update all symbols, merging only newly downloaded rows with history.

        The returned ``saved`` list contains files that received data. Files
        whose end date is already current are returned in ``skipped``. A
        ``download_failures.json`` snapshot is always written, including an
        empty object after a clean batch, so operators do not act on stale
        failure state.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._history_coverage = self._read_history_manifest()
        stocks = self._deduplicated_stocks()
        results: dict[str, tuple[Path | None, str | None, list[str], str]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._update_one, code): code for code in stocks}
            for completed, future in enumerate(as_completed(futures), start=1):
                code = futures[future]
                try:
                    results[code] = future.result()
                except Exception as exc:  # noqa: BLE001 - a worker must not abort the batch
                    results[code] = (None, str(exc) or exc.__class__.__name__, [], "failed")
                if self.progress:
                    self.reporter(f"PROGRESS {completed}/{len(stocks)}")

        saved: list[Path] = []
        skipped: list[Path] = []
        failed: dict[str, str] = {}
        total = len(stocks)
        # Emit messages in universe order even though network work is parallel.
        # This keeps command output and callers' reporter callbacks deterministic.
        for position, code in enumerate(stocks, start=1):
            path, error, events, status = results[code]
            for event in events:
                self.reporter(event)
            if status == "saved":
                assert path is not None
                saved.append(path)
                self._report_status("OK", code, position, total)
            elif status == "skipped":
                assert path is not None
                skipped.append(path)
                self._report_status("SKIP", code, position, total)
            else:
                reason = error or "unknown download error"
                failed[code] = reason
                self._report_status("FAILED", code, position, total, reason)

        self._write_failures(failed)
        return {"saved": saved, "skipped": skipped, "failed": failed, "total": total}

    def _update_one(self, code: str) -> tuple[Path | None, str | None, list[str], str]:
        """Fetch and atomically merge one symbol; exceptions become batch failures."""
        path = self.data_dir / f"{code}.parquet"
        try:
            existing = self._read_existing(path)
            request_start = self._request_start(code, existing)
            if request_start > self._as_timestamp(self.end_date):
                return path, None, [], "skipped"

            history, events = self._get_history_with_events(code, request_start)
            if history.empty:
                if existing.empty:
                    raise RuntimeError("provider returned no daily history")
                return path, None, events, "skipped"
            merged = self._merge(existing, history)
            saved = self._save(code, merged)
            if request_start == self._as_timestamp(self.start_date):
                self._record_history_coverage(code, request_start, history)
            return saved, None, events, "saved"
        except Exception as exc:  # noqa: BLE001 - a batch update must continue after one bad symbol
            return None, str(exc) or exc.__class__.__name__, [], "failed"

    def _read_existing(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        return self._normalise(pd.read_parquet(path))

    def _request_start(self, code: str, existing: pd.DataFrame) -> pd.Timestamp:
        configured_start = self._as_timestamp(self.start_date)
        if existing.empty:
            return configured_start
        first_date = existing["date"].min()
        # A file that starts later than the configured history requires a
        # leading backfill; otherwise only ask for the unpersisted tail.
        if first_date > configured_start and code not in self._history_coverage:
            return configured_start
        return existing["date"].max() + pd.Timedelta(days=1)

    def _save(self, code: str, history: pd.DataFrame) -> Path:
        path = self.data_dir / f"{code}.parquet"
        temporary = self.data_dir / f".{code}.download.tmp.parquet"
        history.to_parquet(temporary, index=False)
        temporary.replace(path)
        return path

    def _write_failures(self, failures: dict[str, str]) -> Path:
        path = self.data_dir / "download_failures.json"
        temporary = self.data_dir / ".download_failures.tmp.json"
        temporary.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)
        return path

    def _read_history_manifest(self) -> dict[str, dict[str, str]]:
        path = self.data_dir / self.HISTORY_MANIFEST
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A missing manifest only costs a leading backfill. Never let it
            # make otherwise usable data unavailable.
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            str(code).zfill(6): entry
            for code, entry in value.items()
            if isinstance(entry, dict)
        }

    def _record_history_coverage(
        self, code: str, requested_start: pd.Timestamp, history: pd.DataFrame
    ) -> None:
        """Persist a completed leading-history request for future tail updates."""
        with self._manifest_lock:
            self._history_coverage[code] = {
                "requested_start": requested_start.strftime("%Y%m%d"),
                "first_available": pd.Timestamp(history["date"].min()).strftime("%Y-%m-%d"),
            }
            path = self.data_dir / self.HISTORY_MANIFEST
            temporary = self.data_dir / ".download_history.tmp.json"
            temporary.write_text(
                json.dumps(self._history_coverage, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(path)

    def _merge(self, existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        combined = pd.concat([existing, self._normalise(fresh)], ignore_index=True)
        return (
            combined.drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
            .loc[:, self.REQUIRED_COLUMNS]
        )

    def _normalise(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        result = data.loc[:, self.REQUIRED_COLUMNS].copy()
        result["date"] = pd.to_datetime(result["date"], errors="raise")
        for column in self.REQUIRED_COLUMNS[1:]:
            result[column] = pd.to_numeric(result[column], errors="raise")
        return result

    def _get_history(self, code: str) -> pd.DataFrame:
        """Compatibility helper for callers that fetch a single full range."""
        history, events = self._get_history_with_events(code, self._as_timestamp(self.start_date))
        for event in events:
            self.reporter(event)
        return history

    def _get_history_with_events(
        self, code: str, start_date: pd.Timestamp
    ) -> tuple[pd.DataFrame, list[str]]:
        events: list[str] = []
        try:
            return self._fetch_with_retries(self.provider, code, start_date), events
        except Exception as primary_error:
            if self.fallback_provider is None:
                raise
            events.append(
                f"FALLBACK {code} {self._provider_name(self.provider)} failed: {primary_error}"
            )
            try:
                return self._fetch_with_retries(self.fallback_provider, code, start_date), events
            except Exception as fallback_error:
                raise RuntimeError(
                    f"{self._provider_name(self.provider)} failed: {primary_error}; "
                    f"{self._provider_name(self.fallback_provider)} failed: {fallback_error}"
                ) from fallback_error

    def _fetch_with_retries(
        self, provider: MarketDataProvider, code: str, start_date: pd.Timestamp
    ) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.retry_delay * attempt)
            try:
                self._rate_limit()
                return self._fetch(provider, code, start_date.strftime("%Y%m%d"))
            except Exception as exc:  # noqa: BLE001 - providers have independent failure types
                last_error = exc
        assert last_error is not None
        raise last_error

    def _rate_limit(self) -> None:
        """Bound batch request starts while allowing I/O work to overlap."""
        if self.request_pause <= 0:
            return
        with self._request_lock:
            now = time.monotonic()
            wait = self.request_pause - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _fetch(
        self, provider: MarketDataProvider, code: str, start_date: str | None = None
    ) -> pd.DataFrame:
        kwargs = {
            "symbol": code,
            "start_date": start_date or self.start_date,
            "end_date": self.end_date,
        }
        if isinstance(provider, AKShareProvider):
            return provider.get_daily_history(**kwargs, adjust=self.adjust)
        return provider.get_daily_history(**kwargs)

    def _deduplicated_stocks(self) -> list[str]:
        return list(dict.fromkeys(str(stock["code"]).zfill(6) for stock in self.universe.stocks()))

    def _report_status(
        self, state: str, code: str, position: int, total: int, reason: str | None = None
    ) -> None:
        prefix = f"{state} {position}/{total} {code}" if self.progress else f"{state} {code}"
        self.reporter(f"{prefix} {reason}" if reason is not None else prefix)

    @staticmethod
    def _as_timestamp(value: str) -> pd.Timestamp:
        return pd.Timestamp(value)

    @staticmethod
    def _create_provider(provider: MarketDataProvider | str | None) -> MarketDataProvider:
        if provider is None or provider == "tencent":
            # A single stock may require several 600-day Tencent intervals;
            # keep each interval bounded so a few unavailable symbols cannot
            # starve the whole concurrent batch.
            return TencentProvider(timeout=8.0)
        if provider == "akshare":
            return AKShareProvider()
        if isinstance(provider, str):
            raise ValueError(f"unknown market data provider: {provider}")  # noqa: TRY004
        return provider

    def _create_fallback_provider(
        self, fallback_provider: MarketDataProvider | None, enable_fallback: bool
    ) -> MarketDataProvider | None:
        if not enable_fallback or isinstance(self.provider, AKShareProvider):
            return None
        if fallback_provider is not None:
            return fallback_provider
        return AKShareProvider()

    @staticmethod
    def _provider_name(provider: MarketDataProvider) -> str:
        return provider.__class__.__name__
