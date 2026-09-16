"""Multiple-horizon, cross-sectional factor research for V5 alpha reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from .v5_reporting import write_v5_report


@dataclass(frozen=True)
class V5FactorResearchResult:
    summary: pd.DataFrame
    daily: pd.DataFrame
    yearly: pd.DataFrame
    regimes: pd.DataFrame
    sectors: pd.DataFrame


class V5FactorResearchEngine:
    """Evaluate processed factors over non-overlapping forward-label observations."""

    SUMMARY_COLUMNS = (
        "factor", "horizon", "pearson_ic", "rank_ic", "ic_mean", "ic_std", "icir", "t_stat",
        "positive_ic_ratio", "long_short_return", "long_short_sharpe", "turnover", "coverage",
        "observations", "yearly_ic_std", "bull_regime_ic", "bear_regime_ic", "sideways_regime_ic",
        "sector_stability", "p_value", "fdr_q_value", "fdr_significant", "status",
    )

    def __init__(self, min_cross_section: int = 20, quantile: float = 0.2) -> None:
        if min_cross_section < 3:
            raise ValueError("min_cross_section must be at least three")
        if not 0 < quantile <= 0.5:
            raise ValueError("quantile must be in (0, 0.5]")
        self.min_cross_section = min_cross_section
        self.quantile = quantile

    def evaluate(
        self, panel: pd.DataFrame, factors: Sequence[str], horizons: Sequence[int] = (5, 10, 20, 60)
    ) -> V5FactorResearchResult:
        """Calculate every requested metric using one non-overlapping date sample per horizon."""
        required = {"date", "sector", "market_regime", *factors}
        missing = required.difference(panel.columns)
        if missing:
            raise ValueError(f"factor research panel missing: {', '.join(sorted(missing))}")
        frame = panel.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        daily_frames: list[pd.DataFrame] = []
        sector_frames: list[pd.DataFrame] = []
        for horizon in sorted(set(horizons)):
            label = f"future_excess_return_{horizon}d"
            if label not in frame:
                raise ValueError(f"factor research panel lacks {label}")
            dates = pd.DatetimeIndex(sorted(frame.loc[frame[label].notna(), "date"].unique()))[::horizon]
            sampled = frame.loc[frame["date"].isin(dates), ["date", "sector", "market_regime", label, *factors]].copy()
            for factor in factors:
                daily, sector = self._factor_daily(sampled, factor, label, horizon)
                daily_frames.append(daily)
                sector_frames.append(sector)
        daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
        sectors = pd.concat(sector_frames, ignore_index=True) if sector_frames else pd.DataFrame()
        summary = self._summary(daily, sectors)
        summary = _benjamini_hochberg(summary)
        yearly = self._grouped_summary(daily, "year")
        regimes = self._grouped_summary(daily, "market_regime")
        return V5FactorResearchResult(summary, daily, yearly, regimes, sectors)

    def save(
        self,
        result: V5FactorResearchResult,
        output_path: str | Path = "research/results/factor_v5_results.parquet",
        report_path: str | Path = "reports/factor_v5_research.html",
    ) -> tuple[Path, Path]:
        """Persist the requested V5 result table and its full supporting report."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        result.summary.to_parquet(target, index=False)
        report = write_v5_report(
            "V5 Factor Research",
            (
                ("Factor summary", result.summary),
                ("Yearly IC", result.yearly),
                ("Regime IC", result.regimes),
                ("Sector stability", result.sectors),
            ),
            report_path,
        )
        return target, report

    def _factor_daily(
        self, panel: pd.DataFrame, factor: str, label: str, horizon: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[dict[str, object]] = []
        sector_rows: list[dict[str, object]] = []
        previous_top: set[str] = set()
        for date, group in panel.loc[:, ["date", "sector", "market_regime", label, factor]].groupby("date", sort=True):
            values = group.rename(columns={factor: "factor", label: "label"}).replace([np.inf, -np.inf], np.nan)
            values = values.dropna(subset=["factor", "label"])
            if len(values) < self.min_cross_section or values["factor"].nunique() < 2 or values["label"].nunique() < 2:
                continue
            ranked = values["factor"].rank(method="first", pct=True)
            top = values.loc[ranked >= 1 - self.quantile]
            bottom = values.loc[ranked <= self.quantile]
            top_ids = set(top.index.astype(str))
            turnover = 1.0 if not previous_top else 1 - len(previous_top.intersection(top_ids)) / max(len(previous_top), len(top_ids))
            previous_top = top_ids
            rows.append(
                {
                    "date": date,
                    "year": int(pd.Timestamp(date).year),
                    "market_regime": str(values["market_regime"].iloc[0]),
                    "factor": factor,
                    "horizon": horizon,
                    "pearson_ic": float(values["factor"].corr(values["label"], method="pearson")),
                    "rank_ic": float(values["factor"].corr(values["label"], method="spearman")),
                    "long_short_return": float(top["label"].mean() - bottom["label"].mean()),
                    "turnover": float(turnover),
                    "coverage": float(len(values) / len(group)),
                    "observations": len(values),
                }
            )
            for sector, sector_values in values.groupby("sector", sort=False):
                if len(sector_values) >= self.min_cross_section and sector_values["factor"].nunique() > 1 and sector_values["label"].nunique() > 1:
                    sector_rows.append(
                        {
                            "date": date,
                            "factor": factor,
                            "horizon": horizon,
                            "sector": str(sector),
                            "rank_ic": float(sector_values["factor"].corr(sector_values["label"], method="spearman")),
                        }
                    )
        return pd.DataFrame(rows), pd.DataFrame(sector_rows)

    def _summary(self, daily: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        if daily.empty:
            return pd.DataFrame(columns=self.SUMMARY_COLUMNS)
        for (factor, horizon), values in daily.groupby(["factor", "horizon"], sort=True):
            ic = values["pearson_ic"].dropna()
            rank = values["rank_ic"].dropna()
            count = len(ic)
            standard_deviation = float(ic.std(ddof=1)) if count > 1 else np.nan
            ic_mean = float(ic.mean()) if count else np.nan
            t_stat = ic_mean / (standard_deviation / sqrt(count)) if count > 1 and standard_deviation > 0 else np.nan
            long_short = values["long_short_return"].dropna()
            long_short_std = long_short.std(ddof=1)
            sector_values = sectors.loc[(sectors["factor"] == factor) & (sectors["horizon"] == horizon), "rank_ic"] if not sectors.empty else pd.Series(dtype=float)
            regime = values.groupby("market_regime")["rank_ic"].mean()
            yearly = values.groupby("year")["rank_ic"].mean()
            p_value = erfc(abs(t_stat) / sqrt(2)) if pd.notna(t_stat) else np.nan
            rows.append(
                {
                    "factor": factor,
                    "horizon": int(horizon),
                    "pearson_ic": ic_mean,
                    "rank_ic": float(rank.mean()) if not rank.empty else np.nan,
                    "ic_mean": ic_mean,
                    "ic_std": standard_deviation,
                    "icir": ic_mean / standard_deviation if pd.notna(standard_deviation) and standard_deviation > 0 else np.nan,
                    "t_stat": t_stat,
                    "positive_ic_ratio": float((ic > 0).mean()) if not ic.empty else np.nan,
                    "long_short_return": float(long_short.mean()) if not long_short.empty else np.nan,
                    "long_short_sharpe": float(long_short.mean() / long_short_std * sqrt(252 / horizon)) if len(long_short) > 1 and long_short_std > 0 else np.nan,
                    "turnover": float(values["turnover"].mean()),
                    "coverage": float(values["coverage"].mean()),
                    "observations": count,
                    "yearly_ic_std": float(yearly.std(ddof=0)) if len(yearly) else np.nan,
                    "bull_regime_ic": float(regime.get("bull", np.nan)),
                    "bear_regime_ic": float(regime.get("bear", np.nan)),
                    "sideways_regime_ic": float(regime.get("sideways", np.nan)),
                    "sector_stability": float((sector_values.groupby(sectors.loc[sector_values.index, "sector"]).mean() > 0).mean()) if not sector_values.empty else np.nan,
                    "p_value": p_value,
                    "fdr_q_value": np.nan,
                    "fdr_significant": False,
                    "status": "ok" if count >= 10 else "insufficient_data",
                }
            )
        return pd.DataFrame(rows, columns=self.SUMMARY_COLUMNS)

    @staticmethod
    def _grouped_summary(daily: pd.DataFrame, group: str) -> pd.DataFrame:
        if daily.empty:
            return pd.DataFrame(columns=["factor", "horizon", group, "rank_ic", "observations"])
        return daily.groupby(["factor", "horizon", group], as_index=False).agg(
            rank_ic=("rank_ic", "mean"), pearson_ic=("pearson_ic", "mean"), observations=("observations", "sum")
        )


def _benjamini_hochberg(summary: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    if summary.empty:
        return summary
    result = summary.copy()
    valid = result["p_value"].notna()
    ordered = result.loc[valid].sort_values("p_value")
    total = len(ordered)
    if total == 0:
        return result
    raw = ordered["p_value"].to_numpy() * total / np.arange(1, total + 1)
    q_values = np.minimum.accumulate(raw[::-1])[::-1].clip(0, 1)
    result.loc[ordered.index, "fdr_q_value"] = q_values
    result["fdr_significant"] = result["fdr_q_value"] <= alpha
    return result
