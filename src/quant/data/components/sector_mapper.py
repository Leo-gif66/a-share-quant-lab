"""Industry-to-sector normalization for imported index constituents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SectorMapper:
    """Resolve component sectors from code overrides or source industry fields.

    The mapper intentionally does not impose a vendor-specific taxonomy.  A
    caller can supply code or industry mappings, while unclassified records
    remain usable with the explicit ``Unknown`` fallback.
    """

    INDUSTRY_FIELDS = ("sector", "industry", "industry_classification", "industry_name")

    def __init__(
        self,
        sectors: Mapping[str, str] | None = None,
        industry_sectors: Mapping[str, str] | None = None,
        default: str = "Unknown",
    ) -> None:
        self.sectors = {
            str(code).zfill(6): str(sector)
            for code, sector in (sectors or {}).items()
            if str(sector).strip()
        }
        self.industry_sectors = {
            str(industry).strip(): str(sector)
            for industry, sector in (industry_sectors or {}).items()
            if str(industry).strip() and str(sector).strip()
        }
        self.default = default

    def resolve(
        self,
        code: str,
        supplied_sector: object | None = None,
        industry: object | None = None,
    ) -> str:
        """Prefer code mapping, then mapped industry, then supplied metadata."""
        normalized = str(code).zfill(6)
        if normalized in self.sectors:
            return self.sectors[normalized]

        candidate = industry if industry is not None else supplied_sector
        if candidate is not None and str(candidate).strip():
            value = str(candidate).strip()
            return self.industry_sectors.get(value, value)
        return self.default

    def map_record(self, record: Mapping[str, Any]) -> str:
        """Resolve a record using common provider industry-classification fields."""
        code = record.get("code", record.get("证券代码", record.get("股票代码", "")))
        supplied = next((record[field] for field in self.INDUSTRY_FIELDS if record.get(field)), None)
        return self.resolve(str(code), supplied_sector=supplied)
