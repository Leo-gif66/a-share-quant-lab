"""Simple transparent portfolio construction for the first v1.0 release."""

from __future__ import annotations

import pandas as pd


class EqualWeightOptimizer:
    """Select top composite scores and assign equal target weights."""

    def __init__(self, top_n: int = 20) -> None:
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        self.top_n = top_n

    def construct(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Build equal-weight top-N targets independently for every score date."""
        required = {"date", "code", "composite_score"}
        if not required.issubset(scores.columns):
            raise ValueError("scores require date, code, and composite_score columns")
        records: list[pd.DataFrame] = []
        for date, group in scores.groupby("date", sort=True):
            selected = (
                group.dropna(subset=["composite_score"])
                .sort_values(["composite_score", "code"], ascending=[False, True], kind="stable")
                .head(self.top_n)
                .copy()
            )
            if selected.empty:
                continue
            selected["date"] = date
            selected["weight"] = 1.0 / len(selected)
            records.append(selected.loc[:, ["date", "code", "weight", "composite_score"]])
        if not records:
            return pd.DataFrame(columns=["date", "code", "weight", "composite_score"])
        return pd.concat(records, ignore_index=True)

    # ``optimize`` is kept as an intuitive synonym for optimizer clients.
    optimize = construct
