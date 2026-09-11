"""Advanced price-derived factors and leak-safe fundamental-data integration."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class FundamentalDataProvider(Protocol):
    """Provide point-in-time financial data with an explicit availability date."""

    def load(self) -> pd.DataFrame:
        """Return code, available_date, roe, profit_growth, revenue_growth."""


class AdvancedFactorEngine:
    """Calculate technical, momentum, liquidity, and risk factors from bars."""

    REQUIRED_COLUMNS = ("date", "close", "high", "low", "volume", "amount", "turnover")

    def calculate(
        self, prices: pd.DataFrame, benchmark: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        missing = sorted(set(self.REQUIRED_COLUMNS).difference(prices.columns))
        if missing:
            raise ValueError(f"price data missing columns: {', '.join(missing)}")
        frame = prices.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        for column in self.REQUIRED_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        close, high, low = frame["close"], frame["high"], frame["low"]
        returns = close.pct_change()
        gains = returns.clip(lower=0)
        losses = -returns.clip(upper=0)
        relative_strength = gains.rolling(14).mean() / losses.rolling(14).mean()
        frame["rsi_14"] = 100 - 100 / (1 + relative_strength)
        frame.loc[(losses.rolling(14).mean() == 0) & gains.rolling(14).mean().notna(), "rsi_14"] = 100.0

        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        frame["macd"] = ema12 - ema26
        frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
        frame["macd_histogram"] = frame["macd"] - frame["macd_signal"]

        mean20 = close.rolling(20).mean()
        std20 = close.rolling(20).std(ddof=0)
        upper, lower = mean20 + 2 * std20, mean20 - 2 * std20
        frame["bollinger_position"] = ((close - lower) / (upper - lower)).where(upper > lower)

        previous_close = close.shift(1)
        true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
        frame["atr_14"] = true_range.rolling(14).mean()
        frame["volatility_breakout_20"] = close / high.rolling(20).max() - 1.0
        for window in (5, 20, 60, 120):
            frame[f"momentum_{window}"] = close.pct_change(window)
        frame["turnover"] = frame["turnover"].rolling(20).mean()
        frame["amount"] = frame["amount"].rolling(20).mean()
        frame["volume_change"] = frame["volume"].pct_change()
        frame["volatility_20"] = returns.rolling(20).std()
        frame["drawdown_60"] = close / close.rolling(60).max() - 1.0
        frame["beta_60"] = self._beta(frame.loc[:, ["date", "close"]], benchmark)
        return frame

    @staticmethod
    def _beta(prices: pd.DataFrame, benchmark: pd.DataFrame | None) -> pd.Series:
        if benchmark is None:
            return pd.Series(np.nan, index=prices.index, dtype="float64")
        if not {"date", "close"}.issubset(benchmark.columns):
            raise ValueError("benchmark requires date and close columns")
        market = benchmark.loc[:, ["date", "close"]].copy()
        market["date"] = pd.to_datetime(market["date"], errors="raise")
        market["market_return"] = pd.to_numeric(market["close"], errors="coerce").pct_change()
        stock = prices.copy()
        stock["stock_return"] = stock["close"].pct_change()
        merged = stock.merge(market.loc[:, ["date", "market_return"]], on="date", how="left")
        variance = merged["market_return"].rolling(60).var(ddof=0)
        beta = merged["stock_return"].rolling(60).cov(merged["market_return"], ddof=0) / variance
        return beta.where(variance > 0)


def merge_quality_factors(price_factors: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Attach ROE and growth values known on each stock/date, without leakage.

    Financial observations must carry ``available_date``—report period end dates
    are insufficient because they are not when the market could have known the
    value.  Values become usable only on or after their availability date.
    """
    required_prices = {"date", "code"}
    required_fundamentals = {"code", "available_date", "roe", "profit_growth", "revenue_growth"}
    if not required_prices.issubset(price_factors.columns):
        raise ValueError("price factors require date and code")
    missing = sorted(required_fundamentals.difference(fundamentals.columns))
    if missing:
        raise ValueError(f"fundamentals missing columns: {', '.join(missing)}")
    left = price_factors.copy()
    left["date"] = pd.to_datetime(left["date"], errors="raise")
    left["code"] = left["code"].astype(str).str.zfill(6)
    right = fundamentals.loc[:, list(required_fundamentals)].copy()
    right["code"] = right["code"].astype(str).str.zfill(6)
    right["available_date"] = pd.to_datetime(right["available_date"], errors="raise")
    for column in ("roe", "profit_growth", "revenue_growth"):
        right[column] = pd.to_numeric(right[column], errors="coerce")

    merged: list[pd.DataFrame] = []
    for code, stock_prices in left.groupby("code", sort=False):
        stock_fundamentals = right.loc[right["code"] == code].sort_values("available_date")
        values = stock_prices.sort_values("date")
        if stock_fundamentals.empty:
            values = values.copy()
            for column in ("roe", "profit_growth", "revenue_growth"):
                values[column] = np.nan
        else:
            values = pd.merge_asof(
                values,
                stock_fundamentals.loc[:, ["available_date", "roe", "profit_growth", "revenue_growth"]],
                left_on="date",
                right_on="available_date",
                direction="backward",
            ).drop(columns="available_date")
        merged.append(values)
    return pd.concat(merged, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
