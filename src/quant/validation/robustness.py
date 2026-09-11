"""Scenario matrix for real, reproducible robustness experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustnessConfig:
    rebalance_periods: tuple[int, ...] = (5, 10, 20)
    transaction_cost_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    factor_weight_perturbations: tuple[float, ...] = (-0.05, 0.05)
    universe_fractions: tuple[float, ...] = (0.8, 1.0)

    def __post_init__(self) -> None:
        if any(value < 1 for value in self.rebalance_periods):
            raise ValueError("rebalance periods must be positive")
        if any(value < 0 for value in self.transaction_cost_multipliers):
            raise ValueError("cost multipliers must be non-negative")
        if any(not 0 < value <= 1 for value in self.universe_fractions):
            raise ValueError("universe fractions must be in (0, 1]")


class RobustnessTester:
    """Call an actual scenario runner; never synthesize sensitivity results.

    ``runner`` receives the scenario fields below and must return measured
    metrics (typically from :class:`WalkForwardSimulator`).  Individual
    scenario failures are retained in the result table so an unavailable test
    is visible rather than silently replaced by a baseline number.
    """

    METRICS = ("annual_return", "sharpe", "max_drawdown", "turnover")

    def __init__(self, config: RobustnessConfig | None = None) -> None:
        self.config = config or RobustnessConfig()

    def run(self, runner: Callable[[Mapping[str, float | int]], Mapping[str, float]]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for category, values, field in (
            ("rebalance_frequency", self.config.rebalance_periods, "rebalance_frequency"),
            ("transaction_cost", self.config.transaction_cost_multipliers, "transaction_cost_multiplier"),
            ("factor_weight_perturbation", self.config.factor_weight_perturbations, "factor_weight_perturbation"),
            ("universe_change", self.config.universe_fractions, "universe_fraction"),
        ):
            for value in values:
                parameters: dict[str, float | int] = {field: value}
                row: dict[str, object] = {"category": category, "scenario": f"{field}={value}", **parameters}
                try:
                    measured = runner(parameters)
                    missing = set(self.METRICS).difference(measured)
                    if missing:
                        raise ValueError(f"scenario runner missing metrics: {', '.join(sorted(missing))}")
                    row.update({metric: float(measured[metric]) for metric in self.METRICS})
                    row.update({"status": "available", "detail": "measured walk-forward scenario"})
                except Exception as exc:  # noqa: BLE001 - preserve every failed robustness cell
                    row.update({metric: np.nan for metric in self.METRICS})
                    row.update({"status": "unavailable", "detail": str(exc) or exc.__class__.__name__})
                rows.append(row)
        return pd.DataFrame(rows)
