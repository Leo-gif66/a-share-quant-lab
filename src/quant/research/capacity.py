"""Simple liquidity and participation diagnostics for V5 portfolio candidates."""

from __future__ import annotations

import pandas as pd


def capacity_diagnostics(
    holdings: pd.DataFrame, portfolio_value: float = 1_000_000.0, max_participation: float = 0.10
) -> pd.DataFrame:
    """Flag holdings whose position value is large relative to 20-day average amount."""
    required = {"date", "code", "weight", "amount_20"}
    missing = required.difference(holdings.columns)
    if missing:
        raise ValueError(f"capacity diagnostics missing: {', '.join(sorted(missing))}")
    output = holdings.loc[:, ["date", "code", "weight", "amount_20"]].copy()
    output["position_value"] = pd.to_numeric(output["weight"], errors="coerce") * portfolio_value
    output["participation_rate"] = output["position_value"] / pd.to_numeric(output["amount_20"], errors="coerce")
    output["capacity_flag"] = output["participation_rate"] > max_participation
    return output
