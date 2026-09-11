"""Deterministic parameter-search interface for reproducible research."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product
from pathlib import Path

import pandas as pd


class ParameterOptimizer:
    """Evaluate a finite parameter grid through an explicit backtest callback."""

    def __init__(self, search_space: Mapping[str, Sequence[object]]) -> None:
        if not search_space or any(not values for values in search_space.values()):
            raise ValueError("search space must contain at least one value for every parameter")
        self.search_space = {str(name): tuple(values) for name, values in search_space.items()}

    def candidates(self) -> list[dict[str, object]]:
        names = list(self.search_space)
        return [dict(zip(names, values, strict=True)) for values in product(*(self.search_space[name] for name in names))]

    def run(
        self,
        evaluate: Callable[[dict[str, object]], Mapping[str, float]],
        output_dir: str | Path = "research/results",
        objective: str = "sharpe",
    ) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for candidate in self.candidates():
            metrics = evaluate(candidate.copy())
            if objective not in metrics:
                raise ValueError(f"parameter evaluator must return objective {objective!r}")
            records.append({**candidate, **{name: float(value) for name, value in metrics.items()}})
        result = pd.DataFrame(records).sort_values(objective, ascending=False, kind="stable").reset_index(drop=True)
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        result.to_csv(directory / "parameter_search.csv", index=False)
        return result


def factor_weight_candidates(
    factors: Sequence[str], values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0), max_candidates: int = 10_000
) -> list[dict[str, float]]:
    """Generate normalized non-negative factor weights with a bounded grid."""
    if not factors or len(set(factors)) != len(factors):
        raise ValueError("factors must be a non-empty unique sequence")
    if not values or any(value < 0 for value in values):
        raise ValueError("weight candidates must be non-negative")
    candidates: list[dict[str, float]] = []
    for combination in product(values, repeat=len(factors)):
        total = float(sum(combination))
        if total == 0:
            continue
        candidates.append({factor: float(weight / total) for factor, weight in zip(factors, combination, strict=True)})
        if len(candidates) > max_candidates:
            raise ValueError("factor-weight grid exceeds max_candidates")
    return candidates
