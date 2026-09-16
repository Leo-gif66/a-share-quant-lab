"""Leak-safe multi-factor score combination methods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

CombinationMethod = Literal["equal", "ic", "icir", "rolling_optimized"]


@dataclass(frozen=True)
class FactorCombinationResult:
    scores: pd.DataFrame
    weights: pd.DataFrame


class FactorCombiner:
    """Combine cross-sectionally standardized factors using prior IC history."""

    def __init__(self, factors: Sequence[str], directions: dict[str, int] | None = None) -> None:
        self.factors = tuple(factors)
        if not self.factors or len(set(self.factors)) != len(self.factors):
            raise ValueError("factors must be a non-empty unique sequence")
        self.directions = {factor: 1 for factor in self.factors}
        for factor, direction in (directions or {}).items():
            if factor not in self.directions or direction not in {-1, 1}:
                raise ValueError("directions must be +/-1 for configured factors")
            self.directions[factor] = direction

    def combine(
        self,
        panel: pd.DataFrame,
        method: CombinationMethod = "equal",
        ic_history: pd.DataFrame | None = None,
        rolling_window: int = 60,
    ) -> FactorCombinationResult:
        if method not in {"equal", "ic", "icir", "rolling_optimized"}:
            raise ValueError("unsupported factor combination method")
        if rolling_window < 2:
            raise ValueError("rolling_window must be at least two")
        required = {"date", "code", *self.factors}
        if not required.issubset(panel.columns):
            raise ValueError(f"factor panel missing columns: {', '.join(sorted(required.difference(panel.columns)))}")
        values = panel.loc[:, ["date", "code", *self.factors]].copy()
        values["date"] = pd.to_datetime(values["date"], errors="raise")
        values["code"] = values["code"].astype(str).str.zfill(6)
        if values.duplicated(["date", "code"]).any():
            raise ValueError("factor panel requires one row per date and code")
        for factor in self.factors:
            values[factor] = pd.to_numeric(values[factor], errors="coerce")
            mean = values.groupby("date")[factor].transform("mean")
            std = values.groupby("date")[factor].transform("std", ddof=0)
            values[f"_{factor}_z"] = ((values[factor] - mean) / std).where(std > 0, 0.0)
            values[f"_{factor}_z"] *= self.directions[factor]

        history = _prepare_ic_history(ic_history, self.factors)
        result_frames: list[pd.DataFrame] = []
        weight_records: list[dict[str, object]] = []
        for date, group in values.groupby("date", sort=True):
            weights = self._weights_for_date(date, method, history, rolling_window)
            score = sum(group[f"_{factor}_z"].fillna(0.0) * weights[factor] for factor in self.factors)
            result_frames.append(pd.DataFrame({"date": date, "code": group["code"], "factor_score": score}))
            weight_records.extend(
                {"date": date, "factor_name": factor, "weight": weights[factor]} for factor in self.factors
            )
        return FactorCombinationResult(
            scores=pd.concat(result_frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True),
            weights=pd.DataFrame(weight_records, columns=["date", "factor_name", "weight"]),
        )

    def _weights_for_date(
        self, date: pd.Timestamp, method: CombinationMethod, history: pd.DataFrame, window: int
    ) -> dict[str, float]:
        equal = {factor: 1.0 / len(self.factors) for factor in self.factors}
        if method == "equal" or history.empty:
            return equal
        prior = history.loc[history["date"] < date].pivot(index="date", columns="factor_name", values="IC")
        prior = prior.reindex(columns=self.factors).tail(window)
        if prior.empty:
            return equal
        if method == "ic":
            raw = prior.mean().fillna(0.0)
        elif method == "icir":
            std = prior.std(ddof=1).replace(0.0, np.nan)
            raw = (prior.mean() / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            if len(prior) < 2:
                return equal
            covariance = prior.fillna(0.0).cov().to_numpy() + np.eye(len(self.factors)) * 1e-6
            raw = pd.Series(
                np.linalg.pinv(covariance) @ prior.mean().fillna(0.0).to_numpy(), index=self.factors
            )
        normalizer = float(raw.abs().sum())
        if normalizer <= 0 or not np.isfinite(normalizer):
            return equal
        return {factor: float(raw[factor] / normalizer) for factor in self.factors}


def _prepare_ic_history(history: pd.DataFrame | None, factors: Sequence[str]) -> pd.DataFrame:
    if history is None:
        return pd.DataFrame(columns=["date", "factor_name", "IC"])
    required = {"date", "factor_name", "IC"}
    if not required.issubset(history.columns):
        raise ValueError("IC history requires date, factor_name, and IC")
    values = history.loc[history["factor_name"].isin(factors), ["date", "factor_name", "IC"]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="raise")
    values["IC"] = pd.to_numeric(values["IC"], errors="coerce")
    return values.dropna(subset="IC").drop_duplicates(["date", "factor_name"], keep="last")
