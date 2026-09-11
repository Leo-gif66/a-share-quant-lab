"""Portfolio diagnostics built from a strategy equity curve."""

from __future__ import annotations

import pandas as pd

from .metrics import annual_returns


def yearly_attribution(equity_curve: pd.DataFrame, benchmark_curve: pd.DataFrame) -> pd.DataFrame:
    """Return yearly strategy, benchmark, and excess-return attribution."""
    return annual_returns(equity_curve, benchmark_curve)


def drawdown_periods(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Identify every peak-to-recovery drawdown period in chronological order."""
    curve = _normalized_equity_curve(equity_curve)
    columns = (
        "peak_date",
        "trough_date",
        "recovery_date",
        "drawdown",
        "duration_days",
    )
    if curve.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, object]] = []
    peak_value = float(curve.iloc[0]["equity"])
    peak_date = pd.Timestamp(curve.iloc[0]["date"])
    peak_index = 0
    active: dict[str, object] | None = None

    for index, row in curve.iloc[1:].iterrows():
        current_value = float(row["equity"])
        current_date = pd.Timestamp(row["date"])
        if current_value >= peak_value:
            if active is not None:
                active["recovery_date"] = current_date
                active["duration_days"] = index - peak_index
                records.append(active)
                active = None
            peak_value = current_value
            peak_date = current_date
            peak_index = index
            continue

        drawdown = current_value / peak_value - 1
        if active is None:
            active = {
                "peak_date": peak_date,
                "trough_date": current_date,
                "recovery_date": pd.NaT,
                "drawdown": drawdown,
                "duration_days": index - peak_index,
            }
        elif drawdown < active["drawdown"]:
            active["trough_date"] = current_date
            active["drawdown"] = drawdown

    if active is not None:
        active["duration_days"] = len(curve) - 1 - peak_index
        records.append(active)
    return pd.DataFrame.from_records(records, columns=columns)


def worst_holding_periods(
    equity_curve: pd.DataFrame, holding_days: int = 20, limit: int = 5
) -> pd.DataFrame:
    """Return the worst rolling holding-period returns for a strategy equity curve."""
    if holding_days < 1:
        raise ValueError("holding_days must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    curve = _normalized_equity_curve(equity_curve)
    records: list[dict[str, object]] = []
    for end_index in range(holding_days, len(curve)):
        start = curve.iloc[end_index - holding_days]
        end = curve.iloc[end_index]
        records.append(
            {
                "start_date": start["date"],
                "end_date": end["date"],
                "holding_days": holding_days,
                "return": float(end["equity"] / start["equity"] - 1),
            }
        )
    return pd.DataFrame.from_records(
        records, columns=("start_date", "end_date", "holding_days", "return")
    ).sort_values("return", kind="stable").head(limit).reset_index(drop=True)


def _normalized_equity_curve(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if not {"date", "equity"}.issubset(equity_curve.columns):
        raise ValueError("equity curve requires date and equity columns")
    curve = equity_curve.loc[:, ["date", "equity"]].copy()
    curve["date"] = pd.to_datetime(curve["date"], errors="raise")
    curve["equity"] = pd.to_numeric(curve["equity"], errors="coerce")
    if curve["equity"].isna().any() or (curve["equity"] <= 0).any():
        raise ValueError("equity curve must contain positive numeric values")
    return curve.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
