"""Adaptive, auditable factor-weight selection from trailing research evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

WEIGHT_METHODS = (
    "equal",
    "ic",
    "icir",
    "rolling_performance",
    "realized_contribution",
    "combined",
)
_METHOD_ALIASES = {
    "ic_weight": "ic",
    "icir_weight": "icir",
    "realized_return_contribution": "realized_contribution",
    "combined_adaptive": "combined",
}


@dataclass(frozen=True)
class AdaptiveWeightResult:
    method: str
    weights: pd.DataFrame


class AdaptiveFactorWeightEngine:
    """Transform factor research statistics into long-only factor allocations.

    Direction remains an explicit strategy property; only the non-negative
    allocation magnitude adapts.  This avoids silently inverting a factor
    after a short, noisy period of negative IC.
    """

    def __init__(self, factor_specs: Mapping[str, Any]) -> None:
        if not factor_specs:
            raise ValueError("at least one factor specification is required")
        self.factor_specs = dict(factor_specs)

    def calculate(
        self,
        method: str = "equal",
        factor_statistics: pd.DataFrame | None = None,
        rolling_performance: pd.DataFrame | None = None,
        realized_contribution: pd.DataFrame | None = None,
    ) -> AdaptiveWeightResult:
        method = _METHOD_ALIASES.get(method, method)
        if method not in WEIGHT_METHODS:
            raise ValueError(f"unknown adaptive weighting method: {method}")
        names = list(self.factor_specs)
        source = factor_statistics if method in {"ic", "icir"} else rolling_performance
        metric_column = {
            "ic": "IC",
            "icir": "ICIR",
            "rolling_performance": "average_return",
            "realized_contribution": "contribution",
        }.get(method)
        if method == "combined":
            ic = self._metrics_for(names, factor_statistics, "IC")
            icir = self._metrics_for(names, factor_statistics, "ICIR")
            realized = self._metrics_for(names, realized_contribution, "contribution")
            # Each research measure is normalized before combination because
            # IC, ICIR, and realized return have different physical scales.
            weights = 0.4 * self._allocation(ic) + 0.3 * self._allocation(icir) + 0.3 * self._allocation(realized)
            metrics = 0.4 * ic.fillna(0.0) + 0.3 * icir.fillna(0.0) + 0.3 * realized.fillna(0.0)
        else:
            source = realized_contribution if method == "realized_contribution" else source
            metrics = (
                self._metrics_for(names, source, metric_column)
                if metric_column
                else pd.Series(1.0, index=names)
            )
            weights = self._allocation(metrics)
        frame = pd.DataFrame(
            {
                "factor": names,
                "weight": [float(weights[name]) for name in names],
                "direction": [self.factor_specs[name].direction for name in names],
                "metric": [float(metrics[name]) if pd.notna(metrics[name]) else np.nan for name in names],
                "method": method,
            }
        )
        return AdaptiveWeightResult(method=method, weights=frame)

    def save(self, result: AdaptiveWeightResult, path: str | Path = "configs/adaptive_factor_weights.yaml") -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": result.method,
            "factors": {
                row.factor: {"weight": float(row.weight), "direction": int(row.direction)}
                for row in result.weights.itertuples(index=False)
            },
        }
        target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return target

    @staticmethod
    def _allocation(metrics: pd.Series) -> pd.Series:
        raw = metrics.abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if raw.sum() <= 0:
            raw[:] = 1.0
        return raw / raw.sum()

    @staticmethod
    def _metrics_for(
        names: Sequence[str], source: pd.DataFrame | None, metric: str | None
    ) -> pd.Series:
        values = pd.Series(np.nan, index=list(names), dtype="float64")
        if source is None or source.empty or metric is None or metric not in source:
            return values
        factor_column = "factor" if "factor" in source else "factor_name" if "factor_name" in source else None
        if factor_column is None:
            raise ValueError("factor statistics require factor or factor_name")
        table = source.loc[:, [factor_column, metric]].copy()
        table[metric] = pd.to_numeric(table[metric], errors="coerce")
        aliases = {
            "volatility": ("volatility", "volatility_20"),
            "liquidity": ("liquidity", "liquidity_20"),
            "turnover": ("turnover", "turnover_20"),
            "drawdown": ("drawdown", "drawdown_60", "max_drawdown_60"),
        }
        for name in names:
            candidates = (name, *aliases.get(name, ()))
            match = table.loc[table[factor_column].astype(str).isin(candidates), metric].dropna()
            if not match.empty:
                values.loc[name] = float(match.iloc[0])
        return values
