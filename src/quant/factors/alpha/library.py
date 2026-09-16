"""Leakage-safe price and cross-sectional alpha factors for V5 research."""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA_FACTORS = (
    "momentum_5", "momentum_20", "momentum_60", "momentum_120", "momentum_20_5_skip",
    "momentum_120_20_skip", "reversal_5", "reversal_10", "overnight_reversal", "ma_distance_20",
    "ma_distance_60", "ma_distance_120", "trend_slope_20", "trend_slope_60", "realized_vol_20",
    "realized_vol_60", "downside_vol_20", "idiosyncratic_volatility", "amount_20", "amount_change_20",
    "volume_ratio_20", "turnover_20", "high_52w_distance", "low_52w_distance", "breakout_20",
    "breakout_60", "rsi_14", "macd_histogram", "atr_normalized", "bollinger_position", "beta_60",
    "residual_return_20", "industry_relative_momentum", "market_relative_momentum", "max_drawdown_60",
    "max_drawdown_120", "downside_beta",
)


class AlphaFactorEngine:
    """Calculate only factors observable at each bar's close."""

    REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume", "amount", "turnover")

    def calculate(self, prices: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return price records with V5 alpha columns; no future bar is referenced."""
        missing = set(self.REQUIRED_COLUMNS).difference(prices.columns)
        if missing:
            raise ValueError(f"price data missing: {', '.join(sorted(missing))}")
        frame = prices.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        for column in self.REQUIRED_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        close, high, low, open_ = (frame[name] for name in ("close", "high", "low", "open"))
        returns = close.pct_change()
        frame["stock_return_1d"] = returns
        for horizon in (5, 20, 60, 120):
            frame[f"momentum_{horizon}"] = close.pct_change(horizon)
        frame["momentum_20_5_skip"] = close.shift(5) / close.shift(20) - 1.0
        frame["momentum_120_20_skip"] = close.shift(20) / close.shift(120) - 1.0
        frame["reversal_5"] = -close.pct_change(5)
        frame["reversal_10"] = -close.pct_change(10)
        frame["overnight_reversal"] = -(open_ / close.shift(1) - 1.0)
        for horizon in (20, 60, 120):
            frame[f"ma_distance_{horizon}"] = close / close.rolling(horizon).mean() - 1.0
        log_close = np.log(close.where(close > 0))
        frame["trend_slope_20"] = _rolling_slope(log_close, 20)
        frame["trend_slope_60"] = _rolling_slope(log_close, 60)
        frame["realized_vol_20"] = returns.rolling(20).std(ddof=0) * np.sqrt(252)
        frame["realized_vol_60"] = returns.rolling(60).std(ddof=0) * np.sqrt(252)
        frame["downside_vol_20"] = returns.clip(upper=0).rolling(20).std(ddof=0) * np.sqrt(252)
        frame["amount_20"] = frame["amount"].rolling(20).mean()
        frame["amount_change_20"] = frame["amount"] / frame["amount"].rolling(20).mean() - 1.0
        frame["volume_ratio_20"] = frame["volume"] / frame["volume"].rolling(20).mean()
        frame["turnover_20"] = frame["turnover"].rolling(20).mean()
        frame["high_52w_distance"] = close / high.rolling(252).max() - 1.0
        frame["low_52w_distance"] = close / low.rolling(252).min() - 1.0
        frame["breakout_20"] = close / high.rolling(20).max() - 1.0
        frame["breakout_60"] = close / high.rolling(60).max() - 1.0
        frame["rsi_14"] = _rsi(returns, 14)
        ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema_fast - ema_slow
        frame["macd_histogram"] = macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()
        prior_close = close.shift(1)
        true_range = pd.concat([high - low, (high - prior_close).abs(), (low - prior_close).abs()], axis=1).max(axis=1)
        frame["atr_normalized"] = true_range.rolling(14).mean() / close
        middle = close.rolling(20).mean()
        deviation = close.rolling(20).std(ddof=0)
        upper, lower = middle + 2 * deviation, middle - 2 * deviation
        frame["bollinger_position"] = ((close - lower) / (upper - lower)).where(upper > lower)
        frame["max_drawdown_60"] = close / close.rolling(60).max() - 1.0
        frame["max_drawdown_120"] = close / close.rolling(120).max() - 1.0
        self._add_market_relative(frame, benchmark)
        return frame

    @staticmethod
    def _add_market_relative(frame: pd.DataFrame, benchmark: pd.DataFrame | None) -> None:
        empty = pd.Series(np.nan, index=frame.index, dtype="float64")
        for column in (
            "beta_60", "residual_return_20", "industry_relative_momentum", "market_relative_momentum",
            "idiosyncratic_volatility", "downside_beta",
        ):
            frame[column] = empty
        if benchmark is None or not {"date", "close"}.issubset(benchmark.columns):
            return
        market = benchmark.loc[:, ["date", "close"]].copy()
        market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
        market["market_return_1d"] = pd.to_numeric(market["close"], errors="coerce").pct_change()
        market["market_return_20d"] = pd.to_numeric(market["close"], errors="coerce").pct_change(20)
        market = market.loc[:, ["date", "market_return_1d", "market_return_20d"]]
        aligned = frame.loc[:, ["date", "stock_return_1d", "momentum_20"]].merge(market, on="date", how="left")
        variance = aligned["market_return_1d"].rolling(60).var(ddof=0)
        beta = aligned["stock_return_1d"].rolling(60).cov(aligned["market_return_1d"], ddof=0) / variance
        frame["beta_60"] = beta.where(variance > 0).to_numpy()
        frame["residual_return_20"] = frame["momentum_20"] - frame["beta_60"] * aligned["market_return_20d"].to_numpy()
        frame["market_relative_momentum"] = frame["momentum_20"] - aligned["market_return_20d"].to_numpy()
        residual = aligned["stock_return_1d"] - frame["beta_60"] * aligned["market_return_1d"]
        frame["idiosyncratic_volatility"] = residual.rolling(60).std(ddof=0).to_numpy() * np.sqrt(252)
        down_market = aligned["market_return_1d"].where(aligned["market_return_1d"] < 0)
        down_variance = down_market.rolling(60, min_periods=20).var(ddof=0)
        downside_beta = aligned["stock_return_1d"].rolling(60, min_periods=20).cov(down_market, ddof=0) / down_variance
        frame["downside_beta"] = downside_beta.where(down_variance > 0).to_numpy()


def _rsi(returns: pd.Series, window: int) -> pd.Series:
    gains = returns.clip(lower=0).rolling(window).mean()
    losses = (-returns.clip(upper=0)).rolling(window).mean()
    relative_strength = gains / losses
    rsi = 100 - 100 / (1 + relative_strength)
    return rsi.mask((losses == 0) & gains.notna(), 100.0)


def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    """Vectorized OLS slope of log price against session index in each trailing window."""
    observations = values.to_numpy(dtype=float)
    weights = np.arange(window, dtype=float)
    result = np.full(len(observations), np.nan, dtype=float)
    if len(observations) < window:
        return pd.Series(result, index=values.index)
    weighted = np.convolve(observations, weights[::-1], mode="valid")
    sums = pd.Series(observations).rolling(window).sum().to_numpy()[window - 1 :]
    denominator = window * (window**2 - 1) / 12
    result[window - 1 :] = (weighted - weights.mean() * sums) / denominator
    return pd.Series(result, index=values.index)
