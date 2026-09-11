"""Cross-sectional factor evaluation and long-short contribution analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pandas as pd

from .preprocessing import ResearchPreprocessor


class FactorEvaluator:
    """Evaluate configured factors against a forward-return column.

    Each daily IC is a cross-sectional Pearson correlation of a factor with
    future returns; rank IC is the equivalent Spearman correlation.  Summary
    IC is the mean of the daily IC series and ICIR is its mean divided by its
    sample standard deviation.
    """

    FACTORS: ClassVar[tuple[str, ...]] = (
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "trend_20",
        "trend_60",
        "volatility",
        "liquidity",
        "turnover",
        "drawdown",
    )
    FACTOR_COLUMNS: ClassVar[tuple[str, ...]] = FACTORS
    _ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "momentum_5": ("momentum_5", "mom_5"),
        "momentum_20": ("momentum_20", "mom_20"),
        "momentum_60": ("momentum_60", "mom_60"),
        "trend_20": ("trend_20",),
        "trend_60": ("trend_60",),
        "volatility": ("volatility", "volatility_20", "vol_20"),
        "liquidity": ("liquidity", "liquidity_20"),
        "turnover": ("turnover", "turnover_20"),
        "drawdown": ("drawdown", "drawdown_60"),
    }
    SUMMARY_COLUMNS: ClassVar[tuple[str, ...]] = (
        "factor_name",
        "IC",
        "rank_IC",
        "IC_mean",
        "IC_std",
        "ICIR",
        "coverage_pct",
        "nan_ratio",
        "valid_samples",
        "IC_sample_count",
        "status",
    )
    DAILY_IC_COLUMNS: ClassVar[tuple[str, ...]] = (
        "date",
        "factor_name",
        "IC",
        "rank_IC",
        "observations",
    )
    CONTRIBUTION_COLUMNS: ClassVar[tuple[str, ...]] = (
        "factor_name",
        "average_return",
        "win_rate",
        "IC",
        "rank_IC",
    )
    ANALYSIS_COLUMNS: ClassVar[tuple[str, ...]] = (
        "factor_name",
        "IC",
        "rank_IC",
        "IC_mean",
        "IC_std",
        "ICIR",
        "average_return",
        "win_rate",
        "coverage_pct",
        "nan_ratio",
        "valid_samples",
        "IC_sample_count",
        "status",
    )

    def __init__(self, factors: Sequence[str] | None = None, min_ic_samples: int = 2) -> None:
        requested = tuple(factors or self.FACTORS)
        if not requested:
            raise ValueError("at least one factor is required")
        if min_ic_samples < 2:
            raise ValueError("min_ic_samples must be at least two")
        self.factors = tuple(self._canonical_factor(factor) for factor in requested)
        if len(set(self.factors)) != len(self.factors):
            raise ValueError("factor names must be unique")
        self.min_ic_samples = min_ic_samples
        self.preprocessor = ResearchPreprocessor()
        self.last_coverage = pd.DataFrame(columns=ResearchPreprocessor.COVERAGE_COLUMNS)
        self._last_factor_columns: dict[str, str] = {}

    def daily_ic(
        self, panel: pd.DataFrame, target_column: str | None = None
    ) -> pd.DataFrame:
        """Return the date-by-date Pearson and Spearman IC series."""
        target_column = self._target_column(panel, target_column)
        prepared, factor_columns = self._prepare(panel, target_column)
        frames: list[pd.DataFrame] = []
        for factor_name, column in factor_columns.items():
            values = prepared.loc[:, ["date", column, target_column]].rename(
                columns={column: "factor", target_column: "target"}
            )
            values = values.dropna(subset=["factor", "target"])
            if values.empty:
                continue
            summary = pd.DataFrame(
                {
                    "IC": self._cross_sectional_correlation(values, "factor", "target"),
                    "rank_IC": self._cross_sectional_correlation(
                        self._rank_by_date(values), "factor_rank", "target_rank"
                    ),
                    "observations": values.groupby("date").size(),
                }
            ).reset_index()
            summary["factor_name"] = factor_name
            frames.append(summary.loc[:, self.DAILY_IC_COLUMNS])
        if not frames:
            return pd.DataFrame(columns=self.DAILY_IC_COLUMNS)
        return pd.concat(frames, ignore_index=True).sort_values(
            ["date", "factor_name"]
        ).reset_index(drop=True)

    def evaluate(self, panel: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
        """Summarize IC, rank IC, IC mean, IC standard deviation, and ICIR."""
        daily = self.daily_ic(panel, target_column)
        records: list[dict[str, object]] = []
        for factor_name in self.factors:
            series = daily.loc[daily["factor_name"] == factor_name, "IC"].dropna()
            rank_series = daily.loc[daily["factor_name"] == factor_name, "rank_IC"].dropna()
            factor_column = self._last_factor_columns[factor_name]
            coverage = self.last_coverage.loc[self.last_coverage["factor"] == factor_column]
            coverage_row = coverage.iloc[0] if not coverage.empty else None
            sufficient = len(series) >= self.min_ic_samples
            ic_mean = float(series.mean()) if sufficient else np.nan
            ic_std = float(series.std(ddof=1)) if sufficient and len(series) > 1 else np.nan
            icir = ic_mean / ic_std if pd.notna(ic_std) and ic_std > 0 else np.nan
            records.append(
                {
                    "factor_name": factor_name,
                    "IC": ic_mean,
                    "rank_IC": float(rank_series.mean()) if sufficient else np.nan,
                    "IC_mean": ic_mean,
                    "IC_std": ic_std,
                    "ICIR": icir,
                    "coverage_pct": float(coverage_row["coverage_pct"])
                    if coverage_row is not None
                    else 0.0,
                    "nan_ratio": float(coverage_row["nan_ratio"])
                    if coverage_row is not None
                    else 1.0,
                    "valid_samples": int(coverage_row["valid_samples"])
                    if coverage_row is not None
                    else 0,
                    "IC_sample_count": int(len(series)),
                    "status": "ok" if sufficient else "insufficient_data",
                }
            )
        return pd.DataFrame.from_records(records, columns=self.SUMMARY_COLUMNS)

    def contribution_analysis(
        self,
        panel: pd.DataFrame,
        target_column: str | None = None,
        quantile: float = 0.2,
    ) -> pd.DataFrame:
        """Return daily top-minus-bottom factor portfolio performance summaries.

        For every date, the top and bottom ``quantile`` of a factor are equally
        weighted.  ``average_return`` is the mean top-minus-bottom return and
        ``win_rate`` is the share of valid dates with a positive spread.
        """
        if not 0 < quantile <= 0.5:
            raise ValueError("quantile must be greater than 0 and at most 0.5")
        target_column = self._target_column(panel, target_column)
        prepared, factor_columns = self._prepare(panel, target_column)
        evaluation = self.evaluate(prepared, target_column).set_index("factor_name")
        records: list[dict[str, object]] = []
        for factor_name, column in factor_columns.items():
            spreads: list[float] = []
            values = prepared.loc[:, ["date", column, target_column]].rename(
                columns={column: "factor", target_column: "target"}
            )
            for _, group in values.groupby("date", sort=True):
                group = group.dropna(subset=["factor", "target"])
                count = len(group)
                if count < 2:
                    continue
                bucket_size = max(1, int(np.ceil(count * quantile)))
                ordered = group.sort_values("factor", kind="stable")
                bottom = ordered.head(bucket_size)["target"].mean()
                top = ordered.tail(bucket_size)["target"].mean()
                spreads.append(float(top - bottom))

            spread_series = pd.Series(spreads, dtype="float64")
            records.append(
                {
                    "factor_name": factor_name,
                    "average_return": float(spread_series.mean())
                    if not spread_series.empty
                    else np.nan,
                    "win_rate": float((spread_series > 0).mean())
                    if not spread_series.empty
                    else np.nan,
                    "IC": evaluation.at[factor_name, "IC"],
                    "rank_IC": evaluation.at[factor_name, "rank_IC"],
                }
            )
        return pd.DataFrame.from_records(records, columns=self.CONTRIBUTION_COLUMNS)

    def analyze(self, panel: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
        """Return the complete v0.9 factor-analysis table in one data frame."""
        target_column = self._target_column(panel, target_column)
        evaluation = self.evaluate(panel, target_column)
        contribution = self.contribution_analysis(panel, target_column)
        return evaluation.merge(
            contribution.loc[:, ["factor_name", "average_return", "win_rate"]],
            on="factor_name",
            how="left",
            validate="one_to_one",
        ).loc[:, self.ANALYSIS_COLUMNS]

    # A concise alias for callers that use the noun phrase from the CLI.
    factor_contribution = contribution_analysis

    @classmethod
    def _canonical_factor(cls, factor: str) -> str:
        for canonical, aliases in cls._ALIASES.items():
            if factor in aliases:
                return canonical
        raise ValueError(f"unsupported research factor: {factor}")

    def _prepare(
        self, panel: pd.DataFrame, target_column: str
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        if "date" not in panel:
            raise ValueError("factor panel requires a date column")
        if target_column not in panel:
            raise ValueError(f"factor panel requires a {target_column} column")

        factor_columns: dict[str, str] = {}
        missing: list[str] = []
        for factor_name in self.factors:
            column = next((name for name in self._ALIASES[factor_name] if name in panel), None)
            if column is None:
                missing.append(factor_name)
            else:
                factor_columns[factor_name] = column
        if missing:
            raise ValueError(f"factor panel missing columns: {', '.join(missing)}")

        prepared = panel.copy()
        prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
        for column in factor_columns.values():
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        self._last_factor_columns = dict(factor_columns)
        cleaned = self.preprocessor.process(
            prepared, tuple(factor_columns.values()), target_columns=(target_column,)
        )
        self.last_coverage = cleaned.coverage
        return cleaned.data, factor_columns

    @staticmethod
    def _target_column(panel: pd.DataFrame, target_column: str | None) -> str:
        if target_column is not None:
            return target_column
        if "future_return_20d" in panel:
            return "future_return_20d"
        # Retain the public evaluator's v0.9 API for callers that already pass
        # a generic forward-return column directly.
        if "future_return" in panel:
            return "future_return"
        raise ValueError("factor panel requires a future_return_20d column")

    @staticmethod
    def _rank_by_date(values: pd.DataFrame) -> pd.DataFrame:
        ranked = values.copy()
        ranked["factor_rank"] = ranked.groupby("date")["factor"].rank(method="average")
        ranked["target_rank"] = ranked.groupby("date")["target"].rank(method="average")
        return ranked

    @staticmethod
    def _cross_sectional_correlation(
        values: pd.DataFrame, left_column: str, right_column: str
    ) -> pd.Series:
        """Vectorized per-date Pearson correlation, including constant-value guards."""
        if values.empty:
            return pd.Series(dtype="float64", name="correlation")
        grouped = values.groupby("date", sort=True)
        left_centered = values[left_column] - grouped[left_column].transform("mean")
        right_centered = values[right_column] - grouped[right_column].transform("mean")
        numerators = (left_centered * right_centered).groupby(values["date"]).sum()
        left_squares = (left_centered**2).groupby(values["date"]).sum()
        right_squares = (right_centered**2).groupby(values["date"]).sum()
        denominator = (left_squares * right_squares) ** 0.5
        return (numerators / denominator).where(denominator > 0)


def factor_contribution_analysis(
    panel: pd.DataFrame,
    target_column: str | None = None,
    quantile: float = 0.2,
) -> pd.DataFrame:
    """Evaluate the standard factor set as top-minus-bottom portfolios."""
    return FactorEvaluator().contribution_analysis(panel, target_column, quantile)
