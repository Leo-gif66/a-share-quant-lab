"""Point-in-time fundamental-data contracts for V5 alpha research."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

FUNDAMENTAL_FIELDS = (
    "pe_ttm", "pb", "ps_ttm", "dividend_yield", "roe", "roa", "gross_margin", "net_margin",
    "operating_cashflow_to_profit", "revenue_growth_yoy", "profit_growth_yoy", "eps_growth_yoy",
    "debt_ratio", "current_ratio", "operating_cashflow_growth",
)
FUNDAMENTAL_COLUMNS = ("code", "report_period", "announcement_date", "effective_date", *FUNDAMENTAL_FIELDS)


class FundamentalProvider(Protocol):
    """Supply public financial statements with an explicit announcement date."""

    def load(self) -> pd.DataFrame:
        """Return data conforming to :data:`FUNDAMENTAL_COLUMNS`."""
