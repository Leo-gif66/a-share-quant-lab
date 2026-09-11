"""Constrained, explainable position sizing for daily paper portfolios."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


ALLOCATION_COLUMNS = (
    "date",
    "symbol",
    "weight",
    "industry",
    "model_score",
    "risk_score",
    "regime",
)


@dataclass(frozen=True)
class AllocationSettings:
    """Hard portfolio constraints plus conservative state-dependent cash."""

    max_stock_weight: float = 0.10
    max_industry_weight: float = 0.25
    top_n: int = 20
    cash_by_regime: dict[str, float] = field(
        default_factory=lambda: {
            "bull": 0.0,
            "sideways": 0.30,
            "bear": 0.60,
            "high_volatility": 0.50,
        }
    )

    def __post_init__(self) -> None:
        if not 0 < self.max_stock_weight <= 1:
            raise ValueError("max_stock_weight must be in (0, 1]")
        if not 0 < self.max_industry_weight <= 1:
            raise ValueError("max_industry_weight must be in (0, 1]")
        if self.top_n < 1:
            raise ValueError("top_n must be at least one")
        if any(not 0 <= value <= 1 for value in self.cash_by_regime.values()):
            raise ValueError("cash_by_regime values must be in [0, 1]")


@dataclass(frozen=True)
class AllocationResult:
    holdings: pd.DataFrame
    cash_weight: float
    regime: str

    @property
    def invested_weight(self) -> float:
        return float(self.holdings["weight"].sum()) if not self.holdings.empty else 0.0

    @property
    def industry_weights(self) -> pd.DataFrame:
        if self.holdings.empty:
            return pd.DataFrame(columns=["industry", "weight"])
        return self.holdings.groupby("industry", as_index=False)["weight"].sum()


class PortfolioAllocator:
    """Allocate only from observable daily candidate fields.

    The ranking signal is a blend of cross-sectional score and optional model
    prediction.  Missing model predictions are intentionally neutral rather
    than imputed from a future return.  Constraints are applied by clipping;
    any capital that cannot safely be assigned remains cash.
    """

    def __init__(self, settings: AllocationSettings | None = None) -> None:
        self.settings = settings or AllocationSettings()

    def allocate(self, candidates: pd.DataFrame, as_of: pd.Timestamp | str | None = None) -> AllocationResult:
        values = self._normalise(candidates, as_of)
        if values.empty:
            return AllocationResult(pd.DataFrame(columns=ALLOCATION_COLUMNS), 1.0, "sideways")
        regime = self._single_regime(values)
        cash_target = float(self.settings.cash_by_regime.get(regime, self.settings.cash_by_regime["sideways"]))
        investable = 1.0 - cash_target
        selected = values.sort_values(["_quality", "symbol"], ascending=[False, True], kind="stable").head(self.settings.top_n).copy()
        if selected.empty or investable <= 0:
            return AllocationResult(pd.DataFrame(columns=ALLOCATION_COLUMNS), 1.0, regime)

        # Relative non-negative quality, penalised by explicit candidate risk.
        raw = selected["_quality"].clip(lower=0.0) * (1.0 - selected["risk_score"].clip(0.0, 1.0))
        if raw.sum() <= 0 or not np.isfinite(raw.sum()):
            raw = pd.Series(1.0, index=selected.index)
        weights = raw / raw.sum() * investable
        selected["weight"] = weights
        # Industry cap precedes stock cap.  This is safe even when a universe
        # has too few industries: unallocatable capital becomes cash.
        selected["weight"] = self._cap_industries(selected)
        selected["weight"] = selected["weight"].clip(upper=self.settings.max_stock_weight)
        selected = selected.loc[selected["weight"] > 0].copy()
        remaining_cash = float(np.clip(1.0 - selected["weight"].sum(), 0.0, 1.0))
        holdings = selected.loc[:, ["date", "symbol", "weight", "industry", "model_score", "risk_score", "regime"]]
        return AllocationResult(holdings.reset_index(drop=True), remaining_cash, regime)

    def _normalise(self, candidates: pd.DataFrame, as_of: pd.Timestamp | str | None) -> pd.DataFrame:
        required = {"symbol", "score", "industry", "risk", "regime"}
        missing = sorted(required.difference(candidates.columns))
        if missing:
            raise ValueError(f"candidates missing columns: {', '.join(missing)}")
        values = candidates.copy()
        if "date" not in values:
            values["date"] = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
        values["date"] = pd.to_datetime(values["date"], errors="raise").dt.normalize()
        cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else values["date"].max()
        values = values.loc[values["date"] == cutoff].copy()
        values["symbol"] = values["symbol"].astype(str).str.zfill(6)
        values["industry"] = values["industry"].fillna("Unknown").astype(str)
        values["regime"] = values["regime"].fillna("sideways").astype(str)
        values["score"] = pd.to_numeric(values["score"], errors="coerce")
        values["model_score"] = pd.to_numeric(values.get("model_prediction"), errors="coerce")
        values["risk_score"] = pd.to_numeric(values["risk"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
        values = values.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
        if values.empty:
            return values
        values["_score_rank"] = values["score"].rank(pct=True, method="average")
        if values["model_score"].notna().any():
            model_rank = values["model_score"].rank(pct=True, method="average").fillna(0.5)
            values["_quality"] = 0.5 * values["_score_rank"] + 0.5 * model_rank
        else:
            values["_quality"] = values["_score_rank"]
        return values.drop_duplicates(["date", "symbol"], keep="last")

    def _single_regime(self, values: pd.DataFrame) -> str:
        observed = values["regime"].value_counts()
        return str(observed.index[0]) if not observed.empty else "sideways"

    def _cap_industries(self, values: pd.DataFrame) -> pd.Series:
        result = values["weight"].copy()
        by_industry = values.groupby("industry", sort=False)["weight"].sum()
        for industry, total in by_industry.items():
            if total > self.settings.max_industry_weight:
                mask = values["industry"] == industry
                result.loc[mask] *= self.settings.max_industry_weight / total
        return result
