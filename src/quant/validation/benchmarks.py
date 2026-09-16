"""Comparable, explicit benchmark statistics for validation reports."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..backtest.metrics import calculate_metrics

BENCHMARK_NAMES = ("CSI300", "CSI500", "CSI1000")


def benchmark_comparison(
    equity_curve: pd.DataFrame,
    benchmarks: Mapping[str, pd.DataFrame | None],
) -> pd.DataFrame:
    """Compare a strategy with supplied index curves without fabricating gaps.

    Missing local index history is emitted as ``status=unavailable`` rather
    than proxying one index with another.  Each available frame needs date and
    close (or benchmark) columns.
    """
    rows: list[dict[str, object]] = []
    for name in BENCHMARK_NAMES:
        values = benchmarks.get(name)
        if values is None or values.empty:
            rows.append(_unavailable(name, "local benchmark history not available"))
            continue
        try:
            curve = _curve(values, equity_curve)
            metrics = calculate_metrics(equity_curve, curve)
            combined = equity_curve.loc[:, ["date", "equity"]].merge(curve, on="date", how="inner")
            active = combined["equity"].pct_change() - combined["benchmark"].pct_change()
            active = active.dropna()
            information_ratio = (
                float(active.mean() / active.std(ddof=0) * np.sqrt(252))
                if len(active) > 1 and active.std(ddof=0) > 0
                else np.nan
            )
            rows.append(
                {
                    "benchmark": name,
                    "status": "available",
                    "return": metrics["benchmark_return"],
                    "sharpe": _sharpe(curve),
                    "drawdown": _drawdown(curve["benchmark"]),
                    "alpha": metrics["alpha"],
                    "beta": metrics["beta"],
                    "information_ratio": information_ratio,
                    "observations": len(combined),
                    "detail": "local index history",
                }
            )
        except (ValueError, KeyError) as exc:
            rows.append(_unavailable(name, str(exc)))
    return pd.DataFrame(rows)


def _curve(values: pd.DataFrame, equity_curve: pd.DataFrame) -> pd.DataFrame:
    source_column = "benchmark" if "benchmark" in values else "close" if "close" in values else None
    if source_column is None or "date" not in values:
        raise ValueError("benchmark requires date and close or benchmark")
    frame = values.loc[:, ["date", source_column]].copy().rename(columns={source_column: "benchmark"})
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["benchmark"] = pd.to_numeric(frame["benchmark"], errors="coerce")
    frame = frame.dropna().drop_duplicates("date", keep="last").sort_values("date")
    dates = pd.to_datetime(equity_curve["date"], errors="raise")
    frame = frame.loc[frame["date"].isin(dates)].copy()
    if len(frame) < 2:
        raise ValueError("benchmark has fewer than two common strategy dates")
    initial = float(equity_curve.loc[equity_curve["date"] == frame["date"].iloc[0], "equity"].iloc[0])
    frame["benchmark"] = initial * frame["benchmark"] / float(frame["benchmark"].iloc[0])
    return frame


def _sharpe(curve: pd.DataFrame) -> float:
    values = pd.to_numeric(curve["benchmark"], errors="coerce").pct_change().dropna()
    return float(values.mean() / values.std(ddof=0) * np.sqrt(252)) if len(values) > 1 and values.std(ddof=0) > 0 else np.nan


def _drawdown(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float((numeric / numeric.cummax() - 1.0).min())


def _unavailable(name: str, detail: str) -> dict[str, object]:
    return {
        "benchmark": name,
        "status": "unavailable",
        "return": np.nan,
        "sharpe": np.nan,
        "drawdown": np.nan,
        "alpha": np.nan,
        "beta": np.nan,
        "information_ratio": np.nan,
        "observations": 0,
        "detail": detail,
    }
