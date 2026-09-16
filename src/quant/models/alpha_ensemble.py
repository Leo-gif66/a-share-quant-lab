"""Validation-fitted alpha ensembles that never learn weights from a test period."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from ..research.leakage import assert_training_before_test


@dataclass(frozen=True)
class AlphaEnsembleWeights:
    method: str
    weights: dict[str, float]
    validation_end: pd.Timestamp


class AlphaEnsemble:
    """Combine factor, fundamental, model, and market-context scores using validation only."""

    METHODS: ClassVar[set[str]] = {"equal", "ic_weighted", "validation_weighted"}

    def __init__(self, method: str = "validation_weighted") -> None:
        if method not in self.METHODS:
            raise ValueError(f"ensemble method must be one of: {', '.join(sorted(self.METHODS))}")
        self.method = method
        self.fitted: AlphaEnsembleWeights | None = None

    def fit(self, validation: pd.DataFrame, components: Sequence[str], label: str) -> AlphaEnsembleWeights:
        """Learn component weights from an already matured validation window."""
        required = {"date", label, *components}
        missing = required.difference(validation.columns)
        if missing:
            raise ValueError(f"ensemble validation data missing: {', '.join(sorted(missing))}")
        frame = validation.loc[:, ["date", label, *components]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        metrics = {component: self._validation_score(frame, component, label) for component in components}
        available = {name: value for name, value in metrics.items() if pd.notna(value) and value != 0}
        if not available:
            raise ValueError("no ensemble component has a finite validation score")
        if self.method == "equal":
            weights = {name: 1 / len(available) for name in available}
        else:
            total = sum(abs(value) for value in available.values())
            weights = {name: value / total for name, value in available.items()}
        self.fitted = AlphaEnsembleWeights(self.method, weights, frame["date"].max())
        return self.fitted

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Return a cross-sectional ensemble score for a later test panel."""
        if self.fitted is None:
            raise RuntimeError("ensemble must be fitted on validation data before prediction")
        if "date" not in data:
            raise ValueError("ensemble prediction data requires date")
        assert_training_before_test(pd.Series([self.fitted.validation_end]), data["date"])
        output = pd.Series(0.0, index=data.index, dtype="float64")
        active_weight = pd.Series(0.0, index=data.index, dtype="float64")
        for component, weight in self.fitted.weights.items():
            if component not in data:
                raise ValueError(f"ensemble prediction data lacks {component}")
            values = pd.to_numeric(data[component], errors="coerce")
            ranks = values.groupby(pd.to_datetime(data["date"], errors="raise"), sort=False).rank(pct=True)
            output = output.add(ranks.fillna(0.0) * weight, fill_value=0.0)
            active_weight = active_weight.add(values.notna().astype(float) * abs(weight), fill_value=0.0)
        return output.where(active_weight > 0, np.nan).rename("ensemble_score")

    def _validation_score(self, frame: pd.DataFrame, component: str, label: str) -> float:
        correlations: list[float] = []
        for _, values in frame.groupby("date", sort=True):
            current = values.loc[:, [component, label]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(current) < 5 or current[component].nunique() < 2 or current[label].nunique() < 2:
                continue
            correlations.append(float(current[component].corr(current[label], method="spearman")))
        if not correlations:
            return np.nan
        mean = float(np.mean(correlations))
        if self.method == "validation_weighted":
            standard_deviation = float(np.std(correlations, ddof=0))
            return mean / standard_deviation if standard_deviation > 0 else mean
        return mean
