"""Cross-sectional V5 factor processing with preserved raw factor columns."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class V5FactorPreprocessor:
    """Clean, winsorize, standardize, and industry-neutralize by signal date."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("winsorization quantiles must satisfy 0 <= low < high <= 1")
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def process(self, panel: pd.DataFrame, factors: Sequence[str]) -> pd.DataFrame:
        """Append ``*_processed`` columns without mutating raw factor values."""
        if not {"date", "industry", *factors}.issubset(panel.columns):
            raise ValueError("V5 factor processing requires date, industry, and requested factors")
        output = panel.copy()
        output["date"] = pd.to_datetime(output["date"], errors="raise")
        for factor in factors:
            values = pd.to_numeric(output[factor], errors="coerce").replace([np.inf, -np.inf], np.nan)
            cleaned = values.groupby(output["date"], sort=False).transform(self._cross_sectional_zscore)
            industry_mean = cleaned.groupby([output["date"], output["industry"]], sort=False).transform("mean")
            neutral = cleaned - industry_mean
            output[f"{factor}_processed"] = neutral.groupby(output["date"], sort=False).transform(self._zscore)
        return output

    def _cross_sectional_zscore(self, values: pd.Series) -> pd.Series:
        finite = values.dropna()
        if len(finite) < 3:
            return pd.Series(np.nan, index=values.index, dtype="float64")
        low, high = finite.quantile([self.lower_quantile, self.upper_quantile])
        return self._zscore(values.clip(lower=low, upper=high))

    @staticmethod
    def _zscore(values: pd.Series) -> pd.Series:
        standard_deviation = values.std(ddof=0)
        if pd.isna(standard_deviation) or standard_deviation == 0:
            return pd.Series(np.nan, index=values.index, dtype="float64")
        return (values - values.mean()) / standard_deviation
