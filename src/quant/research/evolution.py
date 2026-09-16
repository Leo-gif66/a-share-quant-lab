"""Conservative evolution of factor weights from completed-trade evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .adaptive_weights import AdaptiveFactorWeightEngine


@dataclass(frozen=True)
class EvolutionResult:
    weights: pd.DataFrame
    evidence: pd.DataFrame
    output_path: Path | None = None


@dataclass(frozen=True)
class _ConfiguredFactor:
    """Minimal local factor contract, kept import-safe for model startup."""

    weight: float
    direction: int


class StrategyEvolutionEngine:
    """Turn research and realized attribution into bounded weight updates.

    The engine reuses the v3.5 combined weighting evidence (IC, ICIR, and
    realized contribution) and enforces an absolute per-factor movement cap.
    It never looks at incomplete trades.
    """

    def __init__(self, factor_specs: Mapping[str, Any], max_weight_change: float = 0.10) -> None:
        if not factor_specs:
            raise ValueError("at least one factor specification is required")
        if not 0 < max_weight_change <= 1:
            raise ValueError("max_weight_change must be in (0, 1]")
        self.factor_specs = dict(factor_specs)
        self.max_weight_change = max_weight_change

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/factor_weights.yaml", max_weight_change: float = 0.10) -> StrategyEvolutionEngine:
        with Path(path).open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        factors = payload.get("factors") if isinstance(payload, dict) else None
        if not isinstance(factors, dict) or not factors:
            raise ValueError("factor configuration requires a non-empty factors mapping")
        specs = {
            str(name): _ConfiguredFactor(float(values["weight"]), int(values["direction"]))
            for name, values in factors.items()
        }
        return cls(specs, max_weight_change=max_weight_change)

    def evolve(
        self,
        factor_statistics: pd.DataFrame | None = None,
        realized_contribution: pd.DataFrame | None = None,
        trades: pd.DataFrame | None = None,
        output_path: str | Path | None = "configs/evolved_factor_weights.yaml",
    ) -> EvolutionResult:
        base = pd.Series({name: spec.weight for name, spec in self.factor_specs.items()}, dtype=float)
        base = self._normalise(base)
        adaptive = AdaptiveFactorWeightEngine(self.factor_specs).calculate(
            "combined", factor_statistics=factor_statistics, realized_contribution=realized_contribution
        ).weights.set_index("factor")["weight"].reindex(base.index).fillna(0.0)
        proposed = self._bounded_projection(base, self._normalise(adaptive))
        stability = self._weight_stability(trades)
        evidence = self._evidence(base.index, factor_statistics, realized_contribution, stability)
        frame = pd.DataFrame(
            {
                "factor": base.index,
                "previous_weight": base.to_numpy(),
                "target_weight": adaptive.to_numpy(),
                "weight": proposed.to_numpy(),
                "weight_change": (proposed - base).to_numpy(),
                "direction": [self.factor_specs[name].direction for name in base.index],
                "stability": [stability.get(name, np.nan) for name in base.index],
            }
        )
        target = Path(output_path) if output_path is not None else None
        if target is not None:
            self.save(frame, target)
        return EvolutionResult(frame, evidence, target)

    def save(self, weights: pd.DataFrame, output_path: str | Path) -> Path:
        required = {"factor", "weight", "direction", "weight_change"}
        if not required.issubset(weights.columns):
            raise ValueError("evolved weights missing required fields")
        if (weights["weight_change"].abs() > self.max_weight_change + 1e-9).any():
            raise ValueError("evolved factor update exceeds maximum weight change")
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "bounded_combined_adaptive",
            "max_weight_change": float(self.max_weight_change),
            "factors": {
                row.factor: {"weight": float(row.weight), "direction": int(row.direction)}
                for row in weights.itertuples(index=False)
            },
        }
        target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return target

    def _bounded_projection(self, base: pd.Series, target: pd.Series) -> pd.Series:
        """Project target weights onto sum=1 and individual ± change bounds."""
        lower = (base - self.max_weight_change).clip(lower=0.0)
        upper = (base + self.max_weight_change).clip(upper=1.0)
        result = target.clip(lower=lower, upper=upper).copy()
        for _ in range(len(result) + 2):
            residual = 1.0 - float(result.sum())
            if abs(residual) < 1e-12:
                break
            if residual > 0:
                capacity = (upper - result).clip(lower=0.0)
            else:
                capacity = (result - lower).clip(lower=0.0)
            if capacity.sum() <= 1e-12:
                break
            movement = capacity / capacity.sum() * min(abs(residual), float(capacity.sum()))
            result = result + movement if residual > 0 else result - movement
        if not np.isclose(result.sum(), 1.0, atol=1e-8):
            # Starting base is feasible, so this is only reachable for invalid
            # caller input or numerical corruption; returning base is safer
            # than emitting a non-normalized allocation.
            return base.copy()
        return result

    @staticmethod
    def _normalise(values: pd.Series) -> pd.Series:
        clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        return clean / clean.sum() if clean.sum() > 0 else pd.Series(1.0 / len(clean), index=clean.index)

    @staticmethod
    def _weight_stability(trades: pd.DataFrame | None) -> pd.Series:
        if trades is None or trades.empty or "factor_scores" not in trades or "future_return" not in trades:
            return pd.Series(dtype=float)
        # Stability is the absolute monthly mean contribution; only observed,
        # completed returns take part in this diagnostic.
        values = trades.copy()
        values["future_return"] = pd.to_numeric(values["future_return"], errors="coerce")
        values["date"] = pd.to_datetime(values["date"], errors="coerce")
        values = values.dropna(subset=["future_return", "date"])
        rows: list[dict[str, object]] = []
        for row in values.itertuples(index=False):
            scores = row.factor_scores if isinstance(row.factor_scores, dict) else {}
            for factor, score in scores.items():
                rows.append({"factor": factor, "month": row.date.to_period("M"), "contribution": float(score) * float(row.future_return)})
        if not rows:
            return pd.Series(dtype=float)
        monthly = pd.DataFrame(rows).groupby(["factor", "month"])["contribution"].mean()
        return monthly.groupby("factor").mean().abs()

    @staticmethod
    def _evidence(
        names: pd.Index,
        factor_statistics: pd.DataFrame | None,
        realized: pd.DataFrame | None,
        stability: pd.Series,
    ) -> pd.DataFrame:
        result = pd.DataFrame({"factor": names})
        for source, columns in ((factor_statistics, ("IC", "ICIR")), (realized, ("contribution",))):
            if source is None or source.empty:
                continue
            key = "factor" if "factor" in source else "factor_name" if "factor_name" in source else None
            if key is None:
                continue
            available = [key, *[column for column in columns if column in source]]
            result = result.merge(source.loc[:, available].rename(columns={key: "factor"}), on="factor", how="left")
        result["stability"] = result["factor"].map(stability)
        return result
