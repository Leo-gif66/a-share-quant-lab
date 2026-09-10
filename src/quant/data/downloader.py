"""Download and persist the configured A-share universe."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .providers.akshare import AKShareProvider
from .providers.base import MarketDataProvider
from .providers.tencent import TencentProvider
from .universe import Universe


class DataDownloader:
    """Download every stock from :class:`Universe` without failing the batch."""

    def __init__(
        self,
        universe: Universe | None = None,
        provider: MarketDataProvider | str | None = "tencent",
        fallback_provider: MarketDataProvider | None = None,
        enable_fallback: bool = True,
        data_dir: str | Path = "data/raw",
        start_date: str = "20180101",
        end_date: str | None = None,
        adjust: str = "qfq",
        reporter: Callable[[str], None] = print,
    ) -> None:
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

    def update(self) -> dict[str, object]:
        """Download the universe and return paths and per-symbol failures.

        A failure for one symbol is reported as ``FAILED <code> <reason>`` and
        does not prevent later symbols from being downloaded.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        failed: dict[str, str] = {}

        for stock in self.universe.stocks():
            code = str(stock["code"]).zfill(6)
            try:
                history = self._get_history(code)
                path = self._save(code, history)
                saved.append(path)
                self.reporter(f"OK {code}")
            except Exception as exc:  # noqa: BLE001 - a batch update must continue after one bad symbol
                reason = str(exc) or exc.__class__.__name__
                failed[code] = reason
                self.reporter(f"FAILED {code} {reason}")

        return {"saved": saved, "failed": failed}

    def _save(self, code: str, history: pd.DataFrame) -> Path:
        path = self.data_dir / f"{code}.parquet"
        history.to_parquet(path, index=False)
        return path

    def _get_history(self, code: str) -> pd.DataFrame:
        try:
            return self._fetch(self.provider, code)
        except Exception as primary_error:
            if self.fallback_provider is None:
                raise
            self.reporter(
                f"FALLBACK {code} {self._provider_name(self.provider)} failed: {primary_error}"
            )
            try:
                return self._fetch(self.fallback_provider, code)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"{self._provider_name(self.provider)} failed: {primary_error}; "
                    f"{self._provider_name(self.fallback_provider)} failed: {fallback_error}"
                ) from fallback_error

    def _fetch(self, provider: MarketDataProvider, code: str) -> pd.DataFrame:
        kwargs = {
            "symbol": code,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }
        if isinstance(provider, AKShareProvider):
            return provider.get_daily_history(**kwargs, adjust=self.adjust)
        return provider.get_daily_history(**kwargs)

    @staticmethod
    def _create_provider(provider: MarketDataProvider | str | None) -> MarketDataProvider:
        if provider is None or provider == "tencent":
            return TencentProvider()
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
