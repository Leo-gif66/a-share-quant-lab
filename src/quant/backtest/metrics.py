"""Performance metrics for daily equity curves."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(
    equity_curve: pd.DataFrame, benchmark_curve: pd.DataFrame | None = None
) -> dict[str, float]:
    """Calculate strategy and benchmark metrics with zero risk-free rate."""
    if "equity" not in equity_curve:
        raise ValueError("equity curve requires an equity column")
    if len(equity_curve) < 2:
        raise ValueError("equity curve requires at least two observations")

    equity = pd.to_numeric(equity_curve["equity"], errors="coerce")
    if equity.isna().any() or (equity <= 0).any():
        raise ValueError("equity curve must contain positive numeric values")

    daily_returns = equity.pct_change().dropna()
    periods = len(daily_returns)
    annual_return = float((equity.iloc[-1] / equity.iloc[0]) ** (252 / periods) - 1)
    annual_volatility = float(daily_returns.std(ddof=0) * np.sqrt(252))
    sharpe = float(
        daily_returns.mean() / daily_returns.std(ddof=0) * np.sqrt(252)
        if daily_returns.std(ddof=0) > 0
        else 0.0
    )
    max_drawdown = float((equity / equity.cummax() - 1).min())
    win_rate = float((daily_returns > 0).mean())
    metrics = {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }
    if benchmark_curve is None:
        return metrics

    combined = _combine_curves(equity_curve, benchmark_curve)
    strategy_returns = combined["equity"].pct_change().dropna()
    benchmark_returns = combined["benchmark"].pct_change().dropna()
    benchmark_periods = len(benchmark_returns)
    benchmark_return = float(
        (combined["benchmark"].iloc[-1] / combined["benchmark"].iloc[0])
        ** (252 / benchmark_periods)
        - 1
    )
    benchmark_variance = float(benchmark_returns.var(ddof=0))
    beta = float(
        strategy_returns.cov(benchmark_returns, ddof=0) / benchmark_variance
        if benchmark_variance > 0
        else 0.0
    )
    alpha = float((strategy_returns - beta * benchmark_returns).mean() * 252)
    return {
        **metrics,
        "benchmark_return": benchmark_return,
        "excess_return": annual_return - benchmark_return,
        "alpha": alpha,
        "beta": beta,
    }


def annual_returns(equity_curve: pd.DataFrame, benchmark_curve: pd.DataFrame) -> pd.DataFrame:
    """Return strategy, benchmark, and excess returns for each calendar year."""
    return _period_returns(equity_curve, benchmark_curve, "Y", "year")


def monthly_returns(equity_curve: pd.DataFrame, benchmark_curve: pd.DataFrame) -> pd.DataFrame:
    """Return strategy, benchmark, and excess returns for each calendar month."""
    return _period_returns(equity_curve, benchmark_curve, "M", "month")


def drawdown_series(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Return the strategy's daily drawdown series."""
    if not {"date", "equity"}.issubset(equity_curve.columns):
        raise ValueError("equity curve requires date and equity columns")
    result = equity_curve.loc[:, ["date", "equity"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["equity"] = pd.to_numeric(result["equity"], errors="raise")
    result = result.sort_values("date")
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1
    return result.loc[:, ["date", "drawdown"]].reset_index(drop=True)


def _combine_curves(equity_curve: pd.DataFrame, benchmark_curve: pd.DataFrame) -> pd.DataFrame:
    required_benchmark = {"date", "benchmark"}
    if not required_benchmark.issubset(benchmark_curve.columns):
        raise ValueError("benchmark curve requires date and benchmark columns")
    strategy = equity_curve.loc[:, ["date", "equity"]].copy()
    benchmark = benchmark_curve.loc[:, ["date", "benchmark"]].copy()
    strategy["date"] = pd.to_datetime(strategy["date"], errors="raise")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise")
    combined = strategy.merge(benchmark, on="date", how="inner").sort_values("date")
    if len(combined) < 2:
        raise ValueError("strategy and benchmark require at least two shared dates")
    return combined.reset_index(drop=True)


def _period_returns(
    equity_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    frequency: str,
    label: str,
) -> pd.DataFrame:
    combined = _combine_curves(equity_curve, benchmark_curve).set_index("date")
    records = []
    for period, values in combined.groupby(combined.index.to_period(frequency)):
        strategy_return = values["equity"].iloc[-1] / values["equity"].iloc[0] - 1
        benchmark_return = values["benchmark"].iloc[-1] / values["benchmark"].iloc[0] - 1
        records.append(
            {
                label: str(period),
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": strategy_return - benchmark_return,
            }
        )
    return pd.DataFrame(records)
