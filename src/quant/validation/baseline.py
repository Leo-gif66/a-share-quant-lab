"""Capture the frozen V4 baseline before V5 experiments are introduced."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.metrics import calculate_metrics

BASELINE_FIELDS = (
    "annual_return",
    "sharpe",
    "max_drawdown",
    "alpha",
    "beta",
    "turnover",
    "information_ratio",
    "walk_forward_period_count",
    "rebalance_count",
)


def capture_v4_baseline(
    results_path: str | Path = "data/validation/walk_forward_results.parquet",
    output_path: str | Path = "research/results/v4_baseline.json",
) -> Path:
    """Persist metrics calculated from the frozen V4 walk-forward artifact."""
    source = Path(results_path)
    if not source.exists():
        raise FileNotFoundError(f"V4 walk-forward artifact not found: {source}")
    results = pd.read_parquet(source)
    required = {"date", "equity", "benchmark", "turnover", "rebalance", "period"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"V4 walk-forward artifact missing: {', '.join(sorted(missing))}")
    results = results.copy()
    results["date"] = pd.to_datetime(results["date"], errors="raise")
    metrics = calculate_metrics(
        results.loc[:, ["date", "equity"]], results.loc[:, ["date", "benchmark"]]
    )
    active = results.loc[results["rebalance"].astype(bool)]
    excess = pd.to_numeric(results["portfolio_return"], errors="coerce") - pd.to_numeric(
        results["benchmark_return"], errors="coerce"
    )
    excess = excess.dropna()
    information_ratio = (
        float(excess.mean() / excess.std(ddof=0) * np.sqrt(252)) if excess.std(ddof=0) > 0 else 0.0
    )
    payload = {
        "baseline_version": "v4.0.1",
        "source": str(source.as_posix()),
        "date_range": {
            "start": results["date"].min().date().isoformat(),
            "end": results["date"].max().date().isoformat(),
        },
        "metrics": {
            "annual_return": float(metrics["annual_return"]),
            "sharpe": float(metrics["sharpe"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "alpha": float(metrics["alpha"]),
            "beta": float(metrics["beta"]),
            "turnover": float(pd.to_numeric(results["turnover"], errors="coerce").sum()),
            "information_ratio": information_ratio,
            "walk_forward_period_count": int(results["period"].nunique()),
            "rebalance_count": len(active),
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
