"""Local-first storage and as-of joining for point-in-time fundamental records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...research.leakage import assert_fundamental_available
from .base import FUNDAMENTAL_COLUMNS, FUNDAMENTAL_FIELDS, FundamentalProvider


class LocalFundamentalProvider:
    """Load a locally supplied disclosure dataset; never synthesize fundamentals."""

    def __init__(self, path: str | Path = "data/fundamental/fundamentals.parquet") -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        return pd.read_parquet(self.path)


class FundamentalStore:
    """Validate, persist, and safely as-of join disclosure data."""

    def __init__(self, path: str | Path = "data/fundamental/fundamentals.parquet") -> None:
        self.path = Path(path)

    def update(self, provider: FundamentalProvider) -> pd.DataFrame:
        """Persist actual provider records, retaining no fabricated fallback data."""
        normalized = self.normalize(provider.load())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_parquet(self.path, index=False)
        return normalized

    @staticmethod
    def normalize(records: pd.DataFrame) -> pd.DataFrame:
        missing = set(FUNDAMENTAL_COLUMNS).difference(records.columns)
        if missing:
            raise ValueError(f"fundamental records missing: {', '.join(sorted(missing))}")
        result = records.loc[:, FUNDAMENTAL_COLUMNS].copy()
        result["code"] = result["code"].astype(str).str.zfill(6)
        for column in ("report_period", "announcement_date", "effective_date"):
            result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
        if (result["effective_date"] < result["announcement_date"]).any():
            raise ValueError("effective_date must not precede announcement_date")
        for column in FUNDAMENTAL_FIELDS:
            result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return result.drop_duplicates(["code", "report_period", "announcement_date"], keep="last").sort_values(
            ["code", "announcement_date", "report_period"]
        ).reset_index(drop=True)

    def asof_join(self, panel: pd.DataFrame, records: pd.DataFrame | None = None) -> pd.DataFrame:
        """Attach only statements announced on or before each price signal date."""
        if not {"date", "code"}.issubset(panel.columns):
            raise ValueError("factor panel requires date and code for fundamental joining")
        disclosures = self.normalize(records) if records is not None else self.normalize(LocalFundamentalProvider(self.path).load())
        result = panel.copy()
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
        result["code"] = result["code"].astype(str).str.zfill(6)
        if disclosures.empty:
            for column in FUNDAMENTAL_FIELDS:
                result[column] = np.nan
            result["fundamental_announcement_date"] = pd.NaT
            return result
        joined: list[pd.DataFrame] = []
        for code, prices in result.groupby("code", sort=False):
            stock = prices.sort_values("date")
            statements = disclosures.loc[disclosures["code"] == code].sort_values("announcement_date")
            if statements.empty:
                attached = stock.copy()
                for column in FUNDAMENTAL_FIELDS:
                    attached[column] = np.nan
                attached["fundamental_announcement_date"] = pd.NaT
            else:
                statements = statements.copy()
                statements["available_date"] = statements[["announcement_date", "effective_date"]].max(axis=1)
                attached = pd.merge_asof(
                    stock,
                    statements.loc[:, ["announcement_date", "available_date", *FUNDAMENTAL_FIELDS]],
                    left_on="date",
                    right_on="available_date",
                    direction="backward",
                ).rename(columns={"announcement_date": "fundamental_announcement_date"})
                attached = attached.drop(columns="available_date")
            joined.append(attached)
        output = pd.concat(joined, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
        available = output["fundamental_announcement_date"].notna()
        assert_fundamental_available(
            output.loc[available, "fundamental_announcement_date"], output.loc[available, "date"]
        )
        return output
