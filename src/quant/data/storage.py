from __future__ import annotations

from pathlib import Path

import pandas as pd


class Storage:
    def __init__(self, root: str = "data"):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.legacy_prices = self.raw / "prices"
        self.features = self.root / "features"
        self.datasets = self.root / "datasets"
        for p in (self.raw, self.legacy_prices, self.raw / "index", self.features, self.datasets):
            p.mkdir(parents=True, exist_ok=True)

    def write_stock(self, code: str, df: pd.DataFrame) -> Path:
        path = self.raw / f"{code}.parquet"
        df.to_parquet(path, index=False)
        return path

    def read_stock(self, code: str) -> pd.DataFrame:
        path = self.raw / f"{code}.parquet"
        if not path.exists():
            path = self.legacy_prices / f"{code}.parquet"
        return pd.read_parquet(path)

    def stock_codes(self) -> list[str]:
        codes = {p.stem for p in self.raw.glob("*.parquet") if p.stem != "universe"}
        # Old v0.1 downloads remain readable during the storage-layout migration.
        codes.update(p.stem for p in self.legacy_prices.glob("*.parquet"))
        return sorted(codes)

    def write_benchmark(self, df: pd.DataFrame, code: str = "000300") -> Path:
        """Store a benchmark beside daily stock files under its market code."""
        path = self.raw / f"{str(code).zfill(6)}.parquet"
        df.to_parquet(path, index=False)
        return path

    def read_benchmark(self, code: str = "000300") -> pd.DataFrame:
        """Read the direct benchmark file, retaining compatibility with v0.7."""
        path = self.raw / f"{str(code).zfill(6)}.parquet"
        if not path.exists() and str(code).zfill(6) == "000300":
            path = self.raw / "index" / "benchmark.parquet"
        return pd.read_parquet(path)

    def write_universe(self, df: pd.DataFrame) -> Path:
        path = self.raw / "universe.parquet"
        df.to_parquet(path, index=False)
        return path

    def write_features(self, df: pd.DataFrame) -> Path:
        path = self.features / "panel.parquet"
        df.to_parquet(path, index=False)
        return path

    def read_features(self) -> pd.DataFrame:
        return pd.read_parquet(self.features / "panel.parquet")

    def write_dataset(self, df: pd.DataFrame) -> Path:
        path = self.datasets / "training.parquet"
        df.to_parquet(path, index=False)
        return path

    def read_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(self.datasets / "training.parquet")

    def query(self, sql: str):
        import duckdb
        return duckdb.sql(sql)
