from pathlib import Path

import pandas as pd
import pytest

from quant.backtest.diagnostics import drawdown_periods, worst_holding_periods, yearly_attribution
from quant.backtest.engine import BacktestEngine
from quant.backtest.metrics import (
    annual_returns,
    calculate_metrics,
    drawdown_series,
    monthly_returns,
)
from quant.backtest.portfolio import Portfolio, TradingCosts

FACTOR_COLUMNS = (
    "momentum_20",
    "trend_60",
    "volatility_20",
    "liquidity_20",
    "volume_ratio_20",
    "drawdown_60",
)


def feature_frame(code: str, daily_return: float, quality: float) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=45)
    close = 10 * (1 + daily_return) ** pd.Series(range(len(dates)))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "close": close,
            "momentum_20": quality,
            "trend_60": quality,
            "volatility_20": 1 - quality,
            "liquidity_20": quality * 1_000,
            "volume_ratio_20": quality,
            "drawdown_60": quality - 1,
        }
    )
    frame["code"] = code
    return frame


def test_portfolio_tracks_cash_positions_and_equity_curve():
    portfolio = Portfolio(initial_cash=100_000, costs=TradingCosts(0.0, 0.0, 0.0))

    portfolio.rebalance(["000001"], {"000001": 10.0})
    equity = portfolio.record(pd.Timestamp("2024-01-02"), {"000001": 11.0})

    assert portfolio.cash == pytest.approx(0.0)
    assert portfolio.positions["000001"] == pytest.approx(10_000)
    assert equity == pytest.approx(110_000)
    assert len(portfolio.equity_curve) == 1


def test_metrics_returns_required_fields():
    equity_curve = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "equity": [100_000, 110_000, 105_000, 120_000],
        }
    )

    benchmark_curve = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "benchmark": [100_000, 105_000, 102_000, 110_000],
        }
    )
    metrics = calculate_metrics(equity_curve, benchmark_curve)

    assert set(metrics) == {
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "benchmark_return",
        "excess_return",
        "alpha",
        "beta",
    }
    assert metrics["max_drawdown"] < 0
    assert 0 <= metrics["win_rate"] <= 1


def test_costs_are_deducted_from_portfolio_value():
    portfolio = Portfolio(initial_cash=100_000)

    portfolio.rebalance(["000001"], {"000001": 10.0})
    equity = portfolio.record(pd.Timestamp("2024-01-02"), {"000001": 10.0})

    assert portfolio.trading_costs > 0
    assert equity < 100_000


def test_report_tables_include_returns_and_drawdowns():
    equity_curve = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "equity": [100_000, 110_000, 105_000, 120_000],
        }
    )
    benchmark_curve = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "benchmark": [100_000, 105_000, 102_000, 110_000],
        }
    )

    annual = annual_returns(equity_curve, benchmark_curve)
    monthly = monthly_returns(equity_curve, benchmark_curve)
    drawdowns = drawdown_series(equity_curve)

    assert list(annual.columns) == ["year", "strategy_return", "benchmark_return", "excess_return"]
    assert list(monthly.columns) == ["month", "strategy_return", "benchmark_return", "excess_return"]
    assert drawdowns["drawdown"].min() < 0


def test_portfolio_diagnostics_report_attribution_drawdowns_and_worst_holds():
    equity_curve = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "equity": [100_000, 110_000, 90_000, 95_000, 115_000],
        }
    )
    benchmark_curve = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "benchmark": [100_000, 102_000, 98_000, 103_000, 108_000],
        }
    )

    attribution = yearly_attribution(equity_curve, benchmark_curve)
    periods = drawdown_periods(equity_curve)
    worst = worst_holding_periods(equity_curve, holding_days=2)

    assert list(attribution.columns) == [
        "year",
        "strategy_return",
        "benchmark_return",
        "excess_return",
    ]
    assert periods.loc[0, "peak_date"] == pd.Timestamp("2024-01-02")
    assert periods.loc[0, "trough_date"] == pd.Timestamp("2024-01-03")
    assert periods.loc[0, "recovery_date"] == pd.Timestamp("2024-01-05")
    assert periods.loc[0, "drawdown"] == pytest.approx(90_000 / 110_000 - 1)
    assert worst.loc[0, "return"] == pytest.approx(95_000 / 110_000 - 1)


def test_backtest_engine_rebalances_at_next_open_and_reports_metrics(tmp_path: Path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    feature_frame("000001", daily_return=0.01, quality=1.0).to_parquet(
        features_dir / "000001.parquet", index=False
    )
    feature_frame("000002", daily_return=0.0, quality=0.5).to_parquet(
        features_dir / "000002.parquet", index=False
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    benchmark_path = raw_dir / "000300.parquet"
    pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=45),
            "close": 100 * 1.001 ** pd.Series(range(45)),
        }
    ).to_parquet(benchmark_path, index=False)
    engine = BacktestEngine(
        features_dir=features_dir,
        initial_cash=100_000,
        rebalance_interval=20,
        top_n=1,
    )

    result = engine.run()

    assert result.equity_curve.iloc[0]["equity"] == pytest.approx(100_000)
    assert len(result.equity_curve) == 45
    assert result.equity_curve.iloc[-1]["equity"] > 100_000
    assert result.portfolio.positions.keys() == {"000001"}
    assert set(result.metrics) == {
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "benchmark_return",
        "excess_return",
        "alpha",
        "beta",
    }
    assert not result.annual_returns.empty
    assert not result.monthly_returns.empty
    assert len(result.drawdown_series) == 45
    assert engine.benchmark_code == "000300"
