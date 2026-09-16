"""Shared finite-value and coverage contract for research, ML, and scoring."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchPreprocessResult:
    """Cleaned panel plus factor-level data-quality observations."""

    data: pd.DataFrame
    coverage: pd.DataFrame


class ResearchPreprocessor:
    """Sanitize numeric factor inputs on each cross-sectional date.

    Targets are never imputed.  Factor values are clipped to a configurable
    sigma band and missing values are filled with the contemporaneous median
    when one exists.  This avoids both temporal leakage and invalid numeric
    values reaching correlation, model, or portfolio code.
    """

    COVERAGE_COLUMNS = (
        "factor",
        "coverage_pct",
        "nan_ratio",
        "valid_samples",
        "imputed_samples",
        "post_clean_valid_samples",
    )

    def __init__(self, sigma: float = 3.0, impute_missing: bool = True) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = sigma
        self.impute_missing = impute_missing

    def process(
        self,
        panel: pd.DataFrame,
        factors: Sequence[str],
        target_columns: Sequence[str] = (),
    ) -> ResearchPreprocessResult:
        """Return finite, winsorized factor inputs and their coverage statistics."""
        factor_names = tuple(factors)
        required = {"date", *factor_names, *target_columns}
        missing = sorted(required.difference(panel.columns))
        if missing:
            raise ValueError(f"research panel missing columns: {', '.join(missing)}")

        result = panel.copy()
        result["date"] = pd.to_datetime(result["date"], errors="raise")
        records: list[dict[str, object]] = []
        for factor in factor_names:
            raw = self._finite_numeric(result[factor])
            valid_samples = int(raw.notna().sum())
            total = len(raw)
            clipped = raw.groupby(result["date"], sort=False).transform(self._clip)
            cleaned = (
                clipped.groupby(result["date"], sort=False).transform(self._impute)
                if self.impute_missing
                else clipped
            )
            result[factor] = cleaned
            records.append(
                {
                    "factor": factor,
                    "coverage_pct": 100.0 * valid_samples / total if total else 0.0,
                    "nan_ratio": 1.0 - valid_samples / total if total else 1.0,
                    "valid_samples": valid_samples,
                    "imputed_samples": int(raw.isna().sum() - cleaned.isna().sum()),
                    "post_clean_valid_samples": int(cleaned.notna().sum()),
                }
            )

        for target in target_columns:
            # Forward returns are labels, not features: silently filling them
            # would create false observations and invalidate IC/model results.
            result[target] = self._finite_numeric(result[target])

        self.assert_finite(result, factor_names)
        coverage = pd.DataFrame.from_records(records, columns=self.COVERAGE_COLUMNS)
        LOGGER.info("factor coverage statistics: %s", coverage.to_dict(orient="records"))
        return ResearchPreprocessResult(result, coverage)

    @staticmethod
    def assert_finite(panel: pd.DataFrame, columns: Sequence[str]) -> None:
        """Raise when any supplied numeric column still contains infinity."""
        for column in columns:
            values = pd.to_numeric(panel[column], errors="coerce")
            if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"non-finite values remain in {column}")

    @staticmethod
    def _finite_numeric(values: pd.Series) -> pd.Series:
        return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def _clip(self, values: pd.Series) -> pd.Series:
        valid = values.dropna()
        if len(valid) < 2:
            return values
        mean = valid.mean()
        std = valid.std(ddof=0)
        if pd.isna(std) or std == 0:
            return values
        return values.clip(lower=mean - self.sigma * std, upper=mean + self.sigma * std)

    @staticmethod
    def _impute(values: pd.Series) -> pd.Series:
        median = values.median(skipna=True)
        return values.fillna(median) if pd.notna(median) else values
