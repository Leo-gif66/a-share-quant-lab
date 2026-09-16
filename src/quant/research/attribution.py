"""Transparent post-trade attribution based on persisted decision memory."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.diagnostics import drawdown_periods
from ..memory import TradeMemoryStore


@dataclass(frozen=True)
class TradeAttributionResult:
    """Completed-trade diagnostics; contributions are return attribution proxies."""

    summary: pd.DataFrame
    factor_contribution: pd.DataFrame
    industry_contribution: pd.DataFrame
    market_contribution: pd.DataFrame
    reviewed_trades: pd.DataFrame
    monthly_summary: pd.DataFrame
    drawdown_analysis: pd.DataFrame


class TradeAttributionEngine:
    """Review only matured decisions and distinguish model error from market return.

    A factor's contribution is the trade-weighted product of its directional
    signal component and realized return.  It is a diagnostic association,
    not a causal decomposition; the report keeps the raw factor scores in
    decision memory so the calculation remains auditable.
    """

    def review(
        self, trades: pd.DataFrame, benchmark: pd.DataFrame | None = None
    ) -> TradeAttributionResult:
        values = self._matured_trades(trades)
        reviewed = self._with_market_return(values, benchmark)
        if reviewed.empty:
            empty = pd.DataFrame()
            return TradeAttributionResult(empty, empty, empty, empty, reviewed, empty, empty)
        reviewed["effective_weight"] = self._effective_weights(reviewed)
        reviewed["weighted_return"] = reviewed["effective_weight"] * reviewed["future_return"]
        reviewed["weighted_prediction_error"] = (
            reviewed["effective_weight"] * reviewed["prediction_error"]
        )
        reviewed["market_contribution"] = (
            reviewed["effective_weight"] * reviewed["market_return"]
        )
        reviewed["excess_return"] = reviewed["future_return"] - reviewed["market_return"]

        summary = pd.DataFrame(
            [
                {
                    "trade_count": len(reviewed),
                    "total_return": float(reviewed["weighted_return"].sum()),
                    "market_return": float(reviewed["market_contribution"].sum()),
                    "excess_return": float((reviewed["effective_weight"] * reviewed["excess_return"]).sum()),
                    "prediction_error": float(reviewed["weighted_prediction_error"].sum()),
                    "model_accuracy": float(
                        (np.sign(reviewed["model_score"]) == np.sign(reviewed["future_return"])).mean()
                    ),
                    "prediction_bias": float(reviewed["prediction_error"].mean()),
                }
            ]
        )
        factor = self._factor_contribution(reviewed)
        industry = (
            reviewed.groupby("industry", as_index=False)
            .agg(
                trade_count=("symbol", "size"),
                average_return=("future_return", "mean"),
                contribution=("weighted_return", "sum"),
                prediction_error=("prediction_error", "mean"),
            )
            .sort_values("contribution", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        market = pd.DataFrame(
            [
                {
                    "market_return": float(reviewed["market_contribution"].sum()),
                    "portfolio_return": float(reviewed["weighted_return"].sum()),
                    "excess_return": float((reviewed["effective_weight"] * reviewed["excess_return"]).sum()),
                }
            ]
        )
        monthly = self._monthly_summary(reviewed)
        equity = self._trade_equity_curve(reviewed)
        return TradeAttributionResult(
            summary=summary,
            factor_contribution=factor,
            industry_contribution=industry,
            market_contribution=market,
            reviewed_trades=reviewed,
            monthly_summary=monthly,
            drawdown_analysis=drawdown_periods(equity) if len(equity) >= 1 else pd.DataFrame(),
        )

    def review_store(
        self, store: TradeMemoryStore, benchmark: pd.DataFrame | None = None
    ) -> TradeAttributionResult:
        return self.review(store.load(), benchmark)

    @staticmethod
    def _matured_trades(trades: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "symbol", "model_score", "factor_scores", "industry", "future_return"}
        if not required.issubset(trades.columns):
            raise ValueError(f"trade memory missing columns: {', '.join(sorted(required.difference(trades.columns)))}")
        values = trades.copy()
        values["date"] = pd.to_datetime(values["date"], errors="raise")
        values["exit_date"] = pd.to_datetime(values.get("exit_date"), errors="coerce")
        for column in ("model_score", "future_return", "prediction_error", "weight"):
            if column not in values:
                values[column] = np.nan
            values[column] = pd.to_numeric(values[column], errors="coerce")
        values = values.dropna(subset=["future_return", "model_score"]).copy()
        values["prediction_error"] = values["prediction_error"].fillna(
            values["future_return"] - values["model_score"]
        )
        values["industry"] = values["industry"].fillna("Unknown").astype(str)
        return values.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)

    @staticmethod
    def _effective_weights(trades: pd.DataFrame) -> pd.Series:
        raw = pd.to_numeric(trades.get("weight"), errors="coerce")
        if raw.notna().any() and raw.fillna(0).sum() > 0:
            return raw.fillna(0) / raw.fillna(0).sum()
        return pd.Series(1.0 / len(trades), index=trades.index)

    def _factor_contribution(self, trades: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for row in trades.itertuples(index=False):
            scores = _parse_scores(row.factor_scores)
            for factor, score in scores.items():
                records.append(
                    {
                        "factor_name": factor,
                        "factor_score": score,
                        "future_return": row.future_return,
                        "weight": row.effective_weight,
                        "contribution": float(score) * float(row.future_return) * float(row.effective_weight),
                    }
                )
        if not records:
            return pd.DataFrame(columns=["factor_name", "average_score", "average_return", "contribution", "observations"])
        frame = pd.DataFrame(records)
        return (
            frame.groupby("factor_name", as_index=False)
            .agg(
                average_score=("factor_score", "mean"),
                average_return=("future_return", "mean"),
                contribution=("contribution", "sum"),
                observations=("factor_score", "size"),
            )
            .sort_values("contribution", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

    @staticmethod
    def _with_market_return(trades: pd.DataFrame, benchmark: pd.DataFrame | None) -> pd.DataFrame:
        values = trades.copy()
        values["market_return"] = 0.0
        if benchmark is None or benchmark.empty or values.empty:
            return values
        if not {"date", "close"}.issubset(benchmark.columns):
            raise ValueError("benchmark requires date and close columns")
        prices = benchmark.loc[:, ["date", "close"]].copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="raise")
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
        prices = prices.dropna().drop_duplicates("date", keep="last").set_index("date")["close"]
        exits = values["exit_date"].fillna(values["date"])
        entry = values["date"].map(prices)
        exit = exits.map(prices)
        market_return = exit / entry - 1.0
        values["market_return"] = market_return.fillna(0.0)
        return values

    @staticmethod
    def _monthly_summary(trades: pd.DataFrame) -> pd.DataFrame:
        values = trades.copy()
        values["month"] = values["date"].dt.to_period("M").astype(str)
        return (
            values.groupby("month", as_index=False)
            .agg(
                trade_count=("symbol", "size"),
                average_return=("future_return", "mean"),
                total_return=("weighted_return", "sum"),
                prediction_error=("prediction_error", "mean"),
            )
            .sort_values("month")
            .reset_index(drop=True)
        )

    @staticmethod
    def _trade_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
        daily = trades.groupby("date", as_index=False)["weighted_return"].sum().sort_values("date")
        daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
        return daily.loc[:, ["date", "equity"]]


def _parse_scores(value: object) -> Mapping[str, float]:
    if isinstance(value, Mapping):
        return {str(key): float(score) for key, score in value.items() if pd.notna(score)}
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return _parse_scores(parsed)
