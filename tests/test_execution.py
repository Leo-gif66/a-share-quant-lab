from pathlib import Path

import pandas as pd
import pytest

from quant.backtest import DailyAttribution, ExecutionSettings, ProfessionalPortfolio


def _bars(**extra) -> pd.DataFrame:
    row = {
        "code": "000001",
        "open": 10.0,
        "close": 10.0,
        "prev_close": 10.0,
        "volume": 1_000,
        "amount": 10_000_000.0,
    }
    row.update(extra)
    return pd.DataFrame([row])


def test_professional_portfolio_enforces_lots_t_plus_one_and_limit_rules(tmp_path: Path):
    portfolio = ProfessionalPortfolio(
        initial_cash=20_000,
        settings=ExecutionSettings(minimum_commission=0.0, base_slippage=0.0),
    )
    day_one = pd.Timestamp("2024-01-02")

    buys = portfolio.rebalance(day_one, {"000001": 0.5}, _bars())
    same_day_sells = portfolio.rebalance(day_one, {}, _bars())
    next_day_sells = portfolio.rebalance(day_one + pd.offsets.BDay(), {}, _bars())

    assert buys[0].side == "buy"
    assert buys[0].quantity % 100 == 0
    assert same_day_sells == []
    assert next_day_sells[0].side == "sell"
    assert portfolio.position_quantity("000001") == 0
    assert portfolio.rebalance(day_one, {"000001": 0.5}, _bars(is_limit_up=True)) == []
    assert portfolio.rebalance(day_one, {"000001": 0.5}, _bars(suspended=True)) == []

    portfolio.write_trades(tmp_path)
    stored = pd.read_parquet(next(tmp_path.glob("*.parquet")))
    assert {"date", "stock", "side", "price", "quantity", "commission", "tax"}.issubset(stored)


def test_dynamic_slippage_and_daily_attribution_reconcile_to_portfolio_pnl():
    portfolio = ProfessionalPortfolio(
        settings=ExecutionSettings(base_slippage=0.001, low_liquidity_amount=1_000_000)
    )
    assert portfolio.slippage(_bars(amount=100.0).iloc[0]) == pytest.approx(0.002)

    equity = pd.DataFrame(
        {"date": pd.bdate_range("2024-01-02", periods=3), "equity": [100.0, 110.0, 99.0]}
    )
    dates = pd.DatetimeIndex(equity["date"])
    report = DailyAttribution().decompose(
        equity,
        pd.Series([0.0, 0.05, -0.02], index=dates),
        pd.Series([0.0, 0.02, -0.01], index=dates),
    )

    assert report["portfolio_pnl"].to_numpy() == pytest.approx(
        report[["market_effect", "factor_effect", "selection_effect"]].sum(axis=1).to_numpy()
    )
