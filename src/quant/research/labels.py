"""Point-in-time forward labels for V5 cross-sectional research."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .leakage import assert_label_after_signal


def add_v5_forward_labels(
    panel: pd.DataFrame, benchmark: pd.DataFrame, horizons: Iterable[int] = (5, 10, 20, 60)
) -> pd.DataFrame:
    """Add stock, benchmark-excess, maturity-date, and relevance labels by horizon."""
    requested = tuple(sorted({int(horizon) for horizon in horizons}))
    if not requested or min(requested) < 1:
        raise ValueError("at least one positive label horizon is required")
    required = {"date", "code", "close"}
    if not required.issubset(panel.columns):
        raise ValueError("label panel requires date, code, and close")
    if not {"date", "close"}.issubset(benchmark.columns):
        raise ValueError("benchmark requires date and close")
    output = panel.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize()
    output["code"] = output["code"].astype(str).str.zfill(6)
    output["close"] = pd.to_numeric(output["close"], errors="coerce")
    output = output.sort_values(["code", "date"]).reset_index(drop=True)
    index = benchmark.loc[:, ["date", "close"]].copy()
    index["date"] = pd.to_datetime(index["date"], errors="raise").dt.normalize()
    index["close"] = pd.to_numeric(index["close"], errors="coerce")
    index = index.drop_duplicates("date", keep="last").sort_values("date")
    for horizon in requested:
        label = f"future_return_{horizon}d"
        end_date = f"label_end_date_{horizon}d"
        output[label] = output.groupby("code", sort=False)["close"].transform(
            lambda values, horizon=horizon: values.shift(-horizon) / values - 1.0
        )
        output[end_date] = output.groupby("code", sort=False)["date"].shift(-horizon)
        index_return = index["close"].shift(-horizon) / index["close"] - 1.0
        index_label = f"benchmark_future_return_{horizon}d"
        indexed = index.loc[:, ["date"]].copy()
        indexed[index_label] = index_return
        output = output.merge(indexed, on="date", how="left", validate="many_to_one")
        output[f"future_excess_return_{horizon}d"] = output[label] - output[index_label]
        output[f"relevance_label_{horizon}d"] = output.groupby("date", sort=False)[label].rank(pct=True)
        valid = output[end_date].notna()
        assert_label_after_signal(output.loc[valid, "date"], output.loc[valid, end_date])
    return output.sort_values(["date", "code"]).reset_index(drop=True)
