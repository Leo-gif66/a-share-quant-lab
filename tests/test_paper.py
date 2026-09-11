import json

import pandas as pd

from quant.backtest import ExecutionSettings, ProfessionalPortfolio
from quant.paper import PaperAccount, PaperTradingEngine


def test_paper_engine_plans_next_session_and_virtual_account_fills_only_at_open(tmp_path):
    scores = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 3,
            "code": ["000001", "000002", "000003"],
            "composite_score": [3.0, 2.0, 1.0],
        }
    )
    orders = PaperTradingEngine(top_n=2).plan_next_session(scores)
    account = PaperAccount(
        portfolio=ProfessionalPortfolio(
            initial_cash=20_000,
            settings=ExecutionSettings(minimum_commission=0.0, base_slippage=0.0),
        )
    )
    account.submit(orders)
    account.submit(orders)
    bars = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "open": [10.0, 20.0],
            "close": [10.0, 20.0],
            "volume": [1_000, 1_000],
            "amount": [10_000_000, 10_000_000],
        }
    )

    fills = account.process_next_open(pd.Timestamp("2024-01-03"), bars)
    account.mark_to_market(pd.Timestamp("2024-01-03"), bars)
    path = account.save(tmp_path / "account.json")
    restored = PaperAccount.load(path, ExecutionSettings(minimum_commission=0.0, base_slippage=0.0))

    assert len(account.orders) == 2
    assert {fill.stock for fill in fills} == {"000001", "000002"}
    assert all(order.status == "filled" for order in account.orders)
    assert account.daily_pnl[0]["equity"] > 0
    assert json.loads(path.read_text(encoding="utf-8"))["positions"]
    assert restored.portfolio.position_quantity("000001") == account.portfolio.position_quantity("000001")
