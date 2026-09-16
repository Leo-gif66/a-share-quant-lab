"""Industry-aware target construction for the v1.2 portfolio mode.

This module deliberately changes only the target selection stage.  The v1.0
backtest continues to own order timing, transaction costs, accounting and risk
controls; :class:`IndustryNeutralPortfolioBacktestEngine` reuses that path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ..data.universe import Universe
from ..risk import RiskSettings
from .backtest import PortfolioBacktestEngine

TARGET_COLUMNS = (
    "date",
    "code",
    "industry",
    "weight",
    "composite_score",
    "industry_rank",
)
EXPOSURE_COLUMNS = (
    "date",
    "industry",
    "portfolio_weight",
    "benchmark_weight",
    "active_weight",
)


@dataclass(frozen=True)
class PortfolioConstraints:
    """Limits applied before v1.0 risk controls scale the target portfolio."""

    max_industry_weight: float = 0.25
    min_industry_count: int = 5
    max_stock_weight: float = 0.10

    def __post_init__(self) -> None:
        if not 0 < self.max_industry_weight <= 1:
            raise ValueError("max_industry_weight must be in (0, 1]")
        if self.min_industry_count < 1:
            raise ValueError("min_industry_count must be at least 1")
        if not 0 < self.max_stock_weight <= 1:
            raise ValueError("max_stock_weight must be in (0, 1]")

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/portfolio_constraints.yaml") -> PortfolioConstraints:
        with Path(path).open(encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}
        if not isinstance(values, Mapping):
            raise TypeError("portfolio constraints must be a mapping")
        allowed = {"max_industry_weight", "min_industry_count", "max_stock_weight"}
        unknown = set(values).difference(allowed)
        if unknown:
            raise ValueError(f"unknown portfolio constraint(s): {', '.join(sorted(unknown))}")
        return cls(**dict(values))


class IndustryNeutralBuilder:
    """Rank stocks by score inside industries and build constrained targets.

    At each rebalance the highest-ranked industries (by their leading stock's
    score) provide candidates.  The builder uses at least the configured
    number of industries where available, gives selected industries equal
    allocations, and then gives selected stocks equal weights within each
    industry.  When the available universe cannot satisfy a limit, the
    unallocatable portion is intentionally left as cash rather than breaching
    an industry or single-stock limit.
    """

    def __init__(
        self, top_n: int = 20, constraints: PortfolioConstraints | None = None
    ) -> None:
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        self.top_n = top_n
        self.constraints = constraints or PortfolioConstraints()

    @classmethod
    def from_yaml(
        cls, path: str | Path = "configs/portfolio_constraints.yaml", top_n: int = 20
    ) -> IndustryNeutralBuilder:
        return cls(top_n=top_n, constraints=PortfolioConstraints.from_yaml(path))

    def construct(
        self,
        scores: pd.DataFrame,
        industry_metadata: pd.DataFrame | Mapping[str, object],
    ) -> pd.DataFrame:
        """Return constrained weights for one or more score dates.

        ``industry_metadata`` accepts a dataframe with ``code`` and either
        ``industry`` or ``sector``, or a code-to-industry mapping.  ``sector``
        is accepted because the current universe stores its investable broad
        industry classification in that field.
        """
        required = {"date", "code", "composite_score"}
        if not required.issubset(scores.columns):
            raise ValueError("scores require date, code, and composite_score columns")
        metadata = _normalise_industry_metadata(industry_metadata)
        frame = scores.loc[:, ["date", "code", "composite_score"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["composite_score"] = pd.to_numeric(frame["composite_score"], errors="coerce")
        frame = frame.merge(metadata, on="code", how="left", validate="many_to_one")
        frame["industry"] = frame["industry"].fillna("Unknown").astype(str)

        selections = [
            self._construct_date(date, group)
            for date, group in frame.groupby("date", sort=True)
        ]
        selections = [selection for selection in selections if not selection.empty]
        if not selections:
            return pd.DataFrame(columns=TARGET_COLUMNS)
        return pd.concat(selections, ignore_index=True).loc[:, TARGET_COLUMNS]

    # ``optimize`` matches the existing optimizer interface.
    optimize = construct

    def _construct_date(self, date: pd.Timestamp, scores: pd.DataFrame) -> pd.DataFrame:
        ranked = scores.dropna(subset=["composite_score"]).copy()
        if ranked.empty:
            return pd.DataFrame(columns=TARGET_COLUMNS)
        ranked = ranked.sort_values(
            ["industry", "composite_score", "code"], ascending=[True, False, True], kind="stable"
        )
        ranked["industry_rank"] = ranked.groupby("industry").cumcount() + 1

        leaders = (
            ranked.loc[ranked["industry_rank"] == 1]
            .sort_values(["composite_score", "industry"], ascending=[False, True], kind="stable")
        )
        minimum_for_cap = math.ceil(1 / self.constraints.max_industry_weight)
        requested_industries = max(self.constraints.min_industry_count, minimum_for_cap)
        industry_count = min(len(leaders), self.top_n, requested_industries)
        selected_industries = leaders.head(industry_count)["industry"].tolist()
        # Sparse detailed industry classifications are common.  Add the next
        # strongest industries until there are enough names to honour top-N,
        # rather than concentrating an underfilled allocation in the first
        # five industries.
        for industry in leaders.iloc[industry_count:]["industry"]:
            available = ranked["industry"].isin(selected_industries).sum()
            if available >= self.top_n or len(selected_industries) >= self.top_n:
                break
            selected_industries.append(industry)
        if not selected_industries:
            return pd.DataFrame(columns=TARGET_COLUMNS)

        # Allocate the requested number of holdings as evenly as possible.
        allocations = {industry: 1 for industry in selected_industries}
        remaining = self.top_n - len(selected_industries)
        while remaining > 0:
            allocated = False
            for industry in selected_industries:
                available = int((ranked["industry"] == industry).sum())
                if allocations[industry] < available:
                    allocations[industry] += 1
                    remaining -= 1
                    allocated = True
                    if remaining == 0:
                        break
            if not allocated:
                break

        # A complete allocation is possible only when enough industries are
        # present.  Otherwise the cap preserves a cash allocation.
        industry_weight = min(1.0 / len(selected_industries), self.constraints.max_industry_weight)
        selected_frames: list[pd.DataFrame] = []
        for industry in selected_industries:
            candidates = ranked.loc[ranked["industry"] == industry].head(allocations[industry]).copy()
            if candidates.empty:
                continue
            candidates["date"] = date
            candidates["weight"] = min(
                industry_weight / len(candidates), self.constraints.max_stock_weight
            )
            selected_frames.append(candidates)
        if not selected_frames:
            return pd.DataFrame(columns=TARGET_COLUMNS)
        return pd.concat(selected_frames, ignore_index=True).loc[:, TARGET_COLUMNS]


def industry_exposure(
    holdings_history: pd.DataFrame, benchmark_weights: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Calculate portfolio, benchmark-proxy, and active industry weights.

    Benchmark constituent weights are not available from OHLC index data.  The
    benchmark column therefore represents the equal-stock-weight distribution
    of the eligible universe supplied by the caller at each rebalance date.
    """
    if holdings_history.empty:
        return pd.DataFrame(columns=EXPOSURE_COLUMNS)
    industry_column = "industry" if "industry" in holdings_history else "sector"
    required = {"date", industry_column, "weight"}
    if not required.issubset(holdings_history.columns):
        raise ValueError("holdings require date, industry (or sector), and weight")
    portfolio = (
        holdings_history.loc[:, ["date", industry_column, "weight"]]
        .rename(columns={industry_column: "industry", "weight": "portfolio_weight"})
        .groupby(["date", "industry"], as_index=False)["portfolio_weight"]
        .sum()
    )
    portfolio["date"] = pd.to_datetime(portfolio["date"], errors="raise")

    if benchmark_weights is None or benchmark_weights.empty:
        benchmark = pd.DataFrame(columns=["date", "industry", "benchmark_weight"])
    else:
        benchmark = benchmark_weights.copy()
        if "weight" in benchmark and "benchmark_weight" not in benchmark:
            benchmark = benchmark.rename(columns={"weight": "benchmark_weight"})
        expected = {"date", "industry", "benchmark_weight"}
        if not expected.issubset(benchmark.columns):
            raise ValueError("benchmark weights require date, industry, and benchmark_weight")
        benchmark = benchmark.loc[:, ["date", "industry", "benchmark_weight"]]
        benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise")
        benchmark["benchmark_weight"] = pd.to_numeric(
            benchmark["benchmark_weight"], errors="coerce"
        ).fillna(0.0)

    result = portfolio.merge(benchmark, on=["date", "industry"], how="outer")
    result["portfolio_weight"] = result["portfolio_weight"].fillna(0.0)
    result["benchmark_weight"] = result["benchmark_weight"].fillna(0.0)
    result["active_weight"] = result["portfolio_weight"] - result["benchmark_weight"]
    return result.loc[:, EXPOSURE_COLUMNS].sort_values(["date", "industry"]).reset_index(drop=True)


class IndustryNeutralPortfolioBacktestEngine(PortfolioBacktestEngine):
    """v1.2 target selector layered on top of the unchanged v1.0 backtest."""

    def __init__(
        self,
        *args: object,
        constraints_path: str | Path = "configs/portfolio_constraints.yaml",
        **kwargs: object,
    ) -> None:
        constraints = PortfolioConstraints.from_yaml(constraints_path)
        if kwargs.get("universe") is None:
            large_universe = Path("configs/universe_large.yaml")
            kwargs["universe"] = Universe(large_universe) if large_universe.exists() else Universe()
        if kwargs.get("risk_settings") is None:
            kwargs["risk_settings"] = RiskSettings(max_position=constraints.max_stock_weight)
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.constraints = constraints
        self.industry_builder = IndustryNeutralBuilder(
            top_n=self.optimizer.top_n, constraints=self.constraints
        )
        self._benchmark_weights_by_date: dict[pd.Timestamp, pd.DataFrame] = {}

    def _rebalance_targets(
        self, scores: pd.DataFrame, dates: pd.DatetimeIndex
    ) -> dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]]:
        targets: dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]] = {}
        metadata = pd.DataFrame(
            [
                {"code": code, "industry": industry}
                for code, industry in self._industries().items()
            ]
        )
        for index in range(0, len(dates) - 1, self.rebalance_interval):
            signal_date = dates[index]
            signal_scores = scores.loc[scores["date"] == signal_date]
            selected = self.industry_builder.construct(signal_scores, metadata)
            if selected.empty:
                continue
            target_date = dates[index + 1]
            targets[target_date] = (signal_date, selected)
            self._benchmark_weights_by_date[target_date] = _equal_weight_industry_benchmark(
                signal_scores, metadata, target_date
            )
        return targets

    def _sector_exposure(self, holdings_history: pd.DataFrame) -> pd.DataFrame:
        benchmark = (
            pd.concat(self._benchmark_weights_by_date.values(), ignore_index=True)
            if self._benchmark_weights_by_date
            else pd.DataFrame(columns=["date", "industry", "benchmark_weight"])
        )
        return industry_exposure(holdings_history, benchmark)

    def _sectors(self) -> dict[str, str]:
        """Provide detailed industry labels to inherited holdings recording."""
        return self._industries()

    def _industries(self) -> dict[str, str]:
        return {
            str(stock["code"]).zfill(6): str(
                stock.get("industry") or stock.get("sector") or "Unknown"
            )
            for stock in self.universe.stocks()
        }


def _normalise_industry_metadata(
    metadata: pd.DataFrame | Mapping[str, object],
) -> pd.DataFrame:
    if isinstance(metadata, Mapping):
        records = []
        for code, value in metadata.items():
            industry = value.get("industry", value.get("sector")) if isinstance(value, Mapping) else value
            records.append({"code": str(code).zfill(6), "industry": industry})
        frame = pd.DataFrame(records)
    elif isinstance(metadata, pd.DataFrame):
        if "code" not in metadata:
            raise ValueError("industry metadata requires code")
        industry_column = "industry" if "industry" in metadata else "sector"
        if industry_column not in metadata:
            raise ValueError("industry metadata requires industry or sector")
        frame = metadata.loc[:, ["code", industry_column]].rename(columns={industry_column: "industry"}).copy()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    else:
        raise TypeError("industry metadata must be a dataframe or mapping")
    if frame.empty:
        return pd.DataFrame(columns=["code", "industry"])
    frame["industry"] = frame["industry"].fillna("Unknown").astype(str)
    return frame.drop_duplicates("code", keep="last").loc[:, ["code", "industry"]]


def _equal_weight_industry_benchmark(
    scores: pd.DataFrame, metadata: pd.DataFrame, target_date: pd.Timestamp
) -> pd.DataFrame:
    eligible = scores.loc[:, ["code", "composite_score"]].copy()
    eligible["code"] = eligible["code"].astype(str).str.zfill(6)
    eligible["composite_score"] = pd.to_numeric(eligible["composite_score"], errors="coerce")
    eligible = eligible.dropna(subset=["composite_score"]).merge(metadata, on="code", how="left")
    eligible["industry"] = eligible["industry"].fillna("Unknown")
    if eligible.empty:
        return pd.DataFrame(columns=["date", "industry", "benchmark_weight"])
    weights = eligible.groupby("industry", as_index=False).size().rename(columns={"size": "count"})
    weights["benchmark_weight"] = weights["count"] / weights["count"].sum()
    weights["date"] = pd.Timestamp(target_date)
    return weights.loc[:, ["date", "industry", "benchmark_weight"]]
