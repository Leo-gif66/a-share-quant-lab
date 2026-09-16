"""Local benchmark coverage for CSI300, CSI500, and CSI1000 validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BENCHMARK_SYMBOLS = {"CSI300": "000300", "CSI500": "000905", "CSI1000": "000852"}


class BenchmarkCoverageValidator:
    """Validate each benchmark independently; no index substitution is permitted."""

    def __init__(self, raw_dir: str | Path = "data/raw") -> None:
        self.raw_dir = Path(raw_dir)

    def coverage(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for name, symbol in BENCHMARK_SYMBOLS.items():
            path = self.raw_dir / f"{symbol}.parquet"
            if not path.exists():
                rows.append({"benchmark": name, "symbol": symbol, "available": False, "start": None, "end": None, "sessions": 0, "detail": "local history unavailable"})
                continue
            values = pd.read_parquet(path, columns=["date", "close"])
            dates = pd.to_datetime(values["date"], errors="coerce").dropna()
            valid_close = pd.to_numeric(values["close"], errors="coerce").notna()
            dates = dates.loc[valid_close]
            rows.append({"benchmark": name, "symbol": symbol, "available": not dates.empty, "start": dates.min().date().isoformat() if not dates.empty else None, "end": dates.max().date().isoformat() if not dates.empty else None, "sessions": len(dates), "detail": "local history" if not dates.empty else "no valid close observations"})
        return pd.DataFrame(rows)

    def load(self, benchmark: str) -> pd.DataFrame:
        if benchmark not in BENCHMARK_SYMBOLS:
            raise ValueError(f"unsupported benchmark: {benchmark}")
        path = self.raw_dir / f"{BENCHMARK_SYMBOLS[benchmark]}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{benchmark} history is unavailable at {path}; no substitute is used")
        return pd.read_parquet(path)
