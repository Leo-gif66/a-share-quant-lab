"""Correlation-aware selection of compact V5 factor sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .v5_reporting import write_v5_report


@dataclass(frozen=True)
class FactorSelectionResult:
    selection: pd.DataFrame
    correlation: pd.DataFrame
    rank_correlation: pd.DataFrame


class FactorSelector:
    """Group highly correlated factors and retain one evidence-ranked candidate per group."""

    def __init__(self, threshold: float = 0.80, max_dates: int = 120) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("correlation threshold must be in (0, 1]")
        if max_dates < 1:
            raise ValueError("max_dates must be positive")
        self.threshold = threshold
        self.max_dates = max_dates

    def select(self, panel: pd.DataFrame, research: pd.DataFrame, horizon: int = 20) -> FactorSelectionResult:
        """Use processed-factor correlation and research evidence to produce a compact factor set."""
        summary = research.loc[research["horizon"] == horizon].copy()
        if summary.empty:
            raise ValueError(f"no factor research summary is available for horizon {horizon}")
        factors = [str(name) for name in summary["factor"] if name in panel]
        if not factors:
            raise ValueError("none of the researched factors are available in the selection panel")
        sampled = self._sample_dates(panel, factors)
        correlation = sampled[factors].corr(method="pearson", min_periods=20)
        ranks = sampled.groupby("date", sort=False)[factors].rank(pct=True)
        rank_correlation = ranks.corr(method="pearson", min_periods=20)
        groups = _correlation_groups(correlation, self.threshold)
        evidence = summary.set_index("factor")
        rows: list[dict[str, object]] = []
        for number, group in enumerate(groups, start=1):
            ranking = sorted(group, key=lambda name: _selection_score(evidence.loc[name]))
            selected = ranking[-1]
            for factor in sorted(group):
                row = evidence.loc[factor]
                rows.append(
                    {
                        "factor": factor,
                        "group": number,
                        "group_size": len(group),
                        "selected": factor == selected,
                        "rank_ic": float(row["rank_ic"]),
                        "stability": 1 - float(row["yearly_ic_std"]) if pd.notna(row["yearly_ic_std"]) else np.nan,
                        "turnover": float(row["turnover"]),
                        "regime_consistency": _regime_consistency(row),
                        "selection_score": _selection_score(row),
                    }
                )
        return FactorSelectionResult(pd.DataFrame(rows).sort_values(["group", "selected", "factor"], ascending=[True, False, True]), correlation, rank_correlation)

    def save(
        self,
        result: FactorSelectionResult,
        output_path: str | Path = "research/results/factor_selection.csv",
        report_path: str | Path = "reports/factor_selection.html",
    ) -> tuple[Path, Path]:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        result.selection.to_csv(target, index=False)
        report = write_v5_report(
            "V5 Factor Selection",
            (("Selection", result.selection), ("Pearson correlation", result.correlation.reset_index()), ("Rank correlation", result.rank_correlation.reset_index())),
            report_path,
        )
        return target, report

    def _sample_dates(self, panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
        frame = panel.loc[:, ["date", *factors]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
        if len(dates) > self.max_dates:
            dates = dates[np.linspace(0, len(dates) - 1, self.max_dates, dtype=int)]
        return frame.loc[frame["date"].isin(dates)].replace([np.inf, -np.inf], np.nan)


def _correlation_groups(correlation: pd.DataFrame, threshold: float) -> list[set[str]]:
    pending = set(correlation.columns)
    groups: list[set[str]] = []
    while pending:
        root = pending.pop()
        group = {root}
        changed = True
        while changed:
            changed = False
            for factor in list(pending):
                if any(abs(float(correlation.loc[factor, member])) >= threshold for member in group if pd.notna(correlation.loc[factor, member])):
                    pending.remove(factor)
                    group.add(factor)
                    changed = True
        groups.append(group)
    return groups


def _regime_consistency(row: pd.Series) -> float:
    values = row[["bull_regime_ic", "bear_regime_ic", "sideways_regime_ic"]].dropna()
    return float((np.sign(values) == np.sign(row["rank_ic"])).mean()) if not values.empty else np.nan


def _selection_score(row: pd.Series) -> float:
    stability = 1 - float(row["yearly_ic_std"]) if pd.notna(row["yearly_ic_std"]) else 0.0
    regime = _regime_consistency(row)
    return abs(float(row["rank_ic"])) * max(stability, 0.0) * (regime if pd.notna(regime) else 0.0) / (1 + float(row["turnover"]))
