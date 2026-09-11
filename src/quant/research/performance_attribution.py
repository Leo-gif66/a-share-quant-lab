"""Transparent return attribution for a validated portfolio replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceAttributionResult:
    summary: pd.DataFrame
    daily: pd.DataFrame


class PerformanceAttributionEngine:
    """Decompose returns into market, industry, selection, and residual timing.

    If benchmark industry weights are not supplied, the engine uses the
    contemporaneous equal-weight local universe as the industry benchmark and
    labels that proxy in its output.  It never presents this as an index
    market-cap sector benchmark.
    """

    def decompose(
        self,
        equity_curve: pd.DataFrame,
        benchmark_curve: pd.DataFrame,
        holdings: pd.DataFrame,
        stock_prices: pd.DataFrame,
        benchmark_industry_weights: pd.DataFrame | None = None,
    ) -> PerformanceAttributionResult:
        portfolio = self._portfolio_returns(equity_curve, benchmark_curve)
        stock_returns = self._stock_returns(stock_prices)
        if portfolio.empty or stock_returns.empty or holdings.empty:
            return PerformanceAttributionResult(self._empty_summary(), pd.DataFrame())
        weights = self._daily_weights(holdings, portfolio["date"])
        daily = portfolio.merge(self._daily_components(weights, stock_returns, benchmark_industry_weights), on="date", how="left")
        daily = daily.fillna({"industry_allocation": 0.0, "stock_selection": 0.0})
        variance = daily["benchmark_return"].var(ddof=0)
        beta = float(daily["portfolio_return"].cov(daily["benchmark_return"], ddof=0) / variance) if variance > 0 else 0.0
        daily["market_beta"] = beta * daily["benchmark_return"]
        daily["factor_timing"] = daily["portfolio_return"] - daily[["market_beta", "industry_allocation", "stock_selection"]].sum(axis=1)
        rows = [
            {"component": "market_beta_contribution", "contribution": float(daily["market_beta"].sum()), "detail": f"estimated beta={beta:.4f}"},
            {"component": "industry_allocation_contribution", "contribution": float(daily["industry_allocation"].sum()), "detail": self._industry_detail(benchmark_industry_weights)},
            {"component": "stock_selection_contribution", "contribution": float(daily["stock_selection"].sum()), "detail": "held-stock return less its daily industry return"},
            {"component": "factor_timing_contribution", "contribution": float(daily["factor_timing"].sum()), "detail": "residual after market, industry allocation, and selection"},
            {"component": "portfolio_simple_return", "contribution": float(daily["portfolio_return"].sum()), "detail": "sum of daily portfolio returns; components reconcile to this value"},
        ]
        return PerformanceAttributionResult(pd.DataFrame(rows), daily)

    @staticmethod
    def _portfolio_returns(equity: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        if not {"date", "equity"}.issubset(equity.columns) or not {"date", "benchmark"}.issubset(benchmark.columns):
            raise ValueError("equity requires date/equity and benchmark requires date/benchmark")
        frame = equity.loc[:, ["date", "equity"]].merge(benchmark.loc[:, ["date", "benchmark"]], on="date", how="inner")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.sort_values("date").drop_duplicates("date", keep="last")
        frame["portfolio_return"] = pd.to_numeric(frame["equity"], errors="coerce").pct_change()
        frame["benchmark_return"] = pd.to_numeric(frame["benchmark"], errors="coerce").pct_change()
        return frame.dropna(subset=["portfolio_return", "benchmark_return"]).loc[:, ["date", "portfolio_return", "benchmark_return"]]

    @staticmethod
    def _stock_returns(prices: pd.DataFrame) -> pd.DataFrame:
        code = "code" if "code" in prices else "symbol" if "symbol" in prices else None
        if code is None or not {"date", "close"}.issubset(prices.columns):
            raise ValueError("stock prices require date, code/symbol, and close")
        frame = prices.loc[:, ["date", code, "close"]].copy()
        frame.columns = ["date", "code", "close"]
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna().sort_values(["code", "date"])
        frame["stock_return"] = frame.groupby("code", sort=False)["close"].pct_change()
        return frame.dropna(subset=["stock_return"]).loc[:, ["date", "code", "stock_return"]]

    @staticmethod
    def _daily_weights(holdings: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
        code = "code" if "code" in holdings else "symbol" if "symbol" in holdings else None
        if code is None or not {"date", "weight", "industry"}.issubset(holdings.columns):
            raise ValueError("holdings require date, code/symbol, industry, and weight")
        source = holdings.loc[:, ["date", code, "industry", "weight"]].copy()
        source.columns = ["date", "code", "industry", "weight"]
        source["date"] = pd.to_datetime(source["date"], errors="raise")
        source["code"] = source["code"].astype(str).str.zfill(6)
        source["weight"] = pd.to_numeric(source["weight"], errors="coerce")
        records: list[pd.DataFrame] = []
        for date in pd.DatetimeIndex(dates):
            eligible = source.loc[source["date"] <= date]
            if eligible.empty:
                continue
            current_date = eligible["date"].max()
            current = eligible.loc[eligible["date"] == current_date].copy()
            current["date"] = date
            records.append(current)
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["date", "code", "industry", "weight"])

    def _daily_components(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_industry_weights: pd.DataFrame | None,
    ) -> pd.DataFrame:
        merged = weights.merge(returns, on=["date", "code"], how="inner")
        if merged.empty:
            return pd.DataFrame(columns=["date", "industry_allocation", "stock_selection"])
        industry = merged.groupby(["date", "industry"], as_index=False).agg(
            portfolio_weight=("weight", "sum"),
            industry_return=("stock_return", "mean"),
        )
        market = returns.groupby("date", as_index=False)["stock_return"].mean().rename(columns={"stock_return": "market_return"})
        industry = industry.merge(market, on="date", how="left")
        if benchmark_industry_weights is None or benchmark_industry_weights.empty:
            # Equal-weight local-universe proxy, disclosed in the report.
            all_industry = returns.merge(weights.loc[:, ["code", "industry"]].drop_duplicates("code"), on="code", how="left")
            all_industry["industry"] = all_industry["industry"].fillna("Unknown")
            sizes = all_industry.groupby(["date", "industry"], as_index=False)["code"].nunique().rename(columns={"code": "count"})
            sizes["benchmark_weight"] = sizes["count"] / sizes.groupby("date")["count"].transform("sum")
            industry = industry.merge(sizes.loc[:, ["date", "industry", "benchmark_weight"]], on=["date", "industry"], how="left")
        else:
            base = benchmark_industry_weights.loc[:, ["date", "industry", "weight"]].copy().rename(columns={"weight": "benchmark_weight"})
            base["date"] = pd.to_datetime(base["date"], errors="raise")
            industry = industry.merge(base, on=["date", "industry"], how="left")
        industry["benchmark_weight"] = industry["benchmark_weight"].fillna(0.0)
        industry["industry_allocation"] = (industry["portfolio_weight"] - industry["benchmark_weight"]) * (industry["industry_return"] - industry["market_return"])
        merged = merged.merge(industry.loc[:, ["date", "industry", "industry_return"]], on=["date", "industry"], how="left")
        merged["stock_selection"] = merged["weight"] * (merged["stock_return"] - merged["industry_return"])
        allocation = industry.groupby("date", as_index=False)["industry_allocation"].sum()
        selection = merged.groupby("date", as_index=False)["stock_selection"].sum()
        return allocation.merge(selection, on="date", how="outer")

    @staticmethod
    def _industry_detail(benchmark_industry_weights: pd.DataFrame | None) -> str:
        return "provided benchmark industry weights" if benchmark_industry_weights is not None and not benchmark_industry_weights.empty else "equal-weight local-universe industry proxy"

    @staticmethod
    def _empty_summary() -> pd.DataFrame:
        return pd.DataFrame(
            [{"component": "status", "contribution": np.nan, "detail": "insufficient holdings or stock-price evidence"}]
        )
