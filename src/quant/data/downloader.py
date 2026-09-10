"""Download and persist the configured A-share universe."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .providers.akshare import AKShareProvider
from .universe import Universe


class DataDownloader:
    """Download every stock from :class:`Universe` without failing the batch."""

    def __init__(
        self,
        universe: Universe | None = None,
        provider: AKShareProvider | None = None,
        data_dir: str | Path = "data/raw",
        start_date: str = "20180101",
        end_date: str | None = None,
        adjust: str = "qfq",
        reporter: Callable[[str], None] = print,
    ) -> None:
        self.universe = universe or Universe()
        self.provider = provider or AKShareProvider()
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
                history = self.provider.get_daily_history(
                    symbol=code,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    adjust=self.adjust,
                )
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
