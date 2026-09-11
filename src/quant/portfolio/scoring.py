"""Composite alpha scoring from standardized, direction-aware factors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .factor_processing import FactorProcessor


class CompositeScorer:
    """Calculate the configured weighted sum of direction-adjusted z-scores."""

    def __init__(self, processor: FactorProcessor) -> None:
        self.processor = processor

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/factor_weights.yaml"):
        return cls(FactorProcessor.from_yaml(path))

    def score(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return one composite alpha score for every code and rebalance date."""
        processed = self.processor.process(panel)
        components = pd.DataFrame(
            {
                name: processed[f"{name}_z"] * spec.weight * spec.direction
                for name, spec in self.processor.factor_specs.items()
            }
        )
        result = processed.loc[:, ["date", "code"]].copy()
        # A partial score would silently change configured factor weights during
        # warm-up periods, so require every factor to be available.
        result["composite_score"] = components.sum(axis=1, min_count=len(components.columns))
        return result.sort_values(["date", "code"]).reset_index(drop=True)

    def score_and_store(
        self, panel: pd.DataFrame, path: str | Path = "data/features/composite_score.parquet"
    ) -> pd.DataFrame:
        """Score a panel and persist the compact date/code/score contract."""
        result = self.score(panel)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(target, index=False)
        return result
