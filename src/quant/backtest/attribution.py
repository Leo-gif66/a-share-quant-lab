"""Explicit daily PnL decomposition with no inferred factor exposures."""

from __future__ import annotations

import pandas as pd


class DailyAttribution:
    """Turn supplied market and factor return effects into daily PnL effects.

    Market and factor effects must be supplied by the caller.  Selection is the
    residual needed to reconcile to the observed portfolio return, avoiding a
    misleading attempt to infer exposures from an equity curve alone.
    """

    COLUMNS = (
        "date",
        "portfolio_pnl",
        "market_effect",
        "factor_effect",
        "selection_effect",
    )

    def decompose(
        self,
        equity_curve: pd.DataFrame,
        market_return: pd.Series,
        factor_return: pd.Series,
    ) -> pd.DataFrame:
        if not {"date", "equity"}.issubset(equity_curve.columns):
            raise ValueError("equity curve requires date and equity columns")
        frame = equity_curve.loc[:, ["date", "equity"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["equity"] = pd.to_numeric(frame["equity"], errors="raise")
        frame = frame.sort_values("date").reset_index(drop=True)
        prior_equity = frame["equity"].shift(1)
        portfolio_return = frame["equity"].pct_change()
        market = _align_effect(market_return, frame["date"], "market_return")
        factor = _align_effect(factor_return, frame["date"], "factor_return")
        result = pd.DataFrame(
            {
                "date": frame["date"],
                "portfolio_pnl": (frame["equity"] - prior_equity).fillna(0.0),
                "market_effect": (prior_equity * market).fillna(0.0),
                "factor_effect": (prior_equity * factor).fillna(0.0),
            }
        )
        result["selection_effect"] = result["portfolio_pnl"] - result["market_effect"] - result["factor_effect"]
        return result.loc[:, self.COLUMNS]


def _align_effect(effect: pd.Series, dates: pd.Series, name: str) -> pd.Series:
    if not isinstance(effect.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must be indexed by date")
    normalized = pd.to_numeric(effect, errors="coerce").copy()
    normalized.index = pd.to_datetime(normalized.index, errors="raise")
    return normalized.reindex(pd.DatetimeIndex(dates)).fillna(0.0).reset_index(drop=True)
