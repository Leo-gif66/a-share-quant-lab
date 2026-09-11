from pathlib import Path

import pandas as pd

from quant.portfolio import IntelligentPortfolioBacktestEngine


class _Universe:
    def stocks(self):
        return [
            {"code": "000001", "name": "One", "market": "SZ", "sector": "Bank"},
            {"code": "000002", "name": "Two", "market": "SZ", "sector": "Technology"},
            {"code": "000003", "name": "Three", "market": "SZ", "sector": "Consumer"},
        ]


def test_intelligent_backtest_reuses_institutional_accounting_and_writes_trade_memory(tmp_path: Path):
    raw_dir, features_dir = tmp_path / "raw", tmp_path / "features"
    raw_dir.mkdir()
    features_dir.mkdir()
    dates = pd.bdate_range("2023-01-02", periods=50)
    for position, code in enumerate(("000001", "000002", "000003"), start=1):
        close = 10 + position + pd.Series(range(len(dates))) * (0.01 + position * 0.001)
        pd.DataFrame({"date": dates, "open": close * 0.999, "close": close}).to_parquet(raw_dir / f"{code}.parquet")
        pd.DataFrame({"date": dates, "open": close * 0.999, "close": close, "momentum_5": position}).to_parquet(features_dir / f"{code}.parquet")
    pd.DataFrame({"date": dates, "close": 100 + pd.Series(range(len(dates))) * 0.02}).to_parquet(raw_dir / "000300.parquet")
    factors = tmp_path / "factors.yaml"
    factors.write_text("factors:\n  momentum_5:\n    weight: 1.0\n    direction: 1\n", encoding="utf-8")
    constraints = tmp_path / "constraints.yaml"
    constraints.write_text("max_industry_weight: 1.0\nmin_industry_count: 1\nmax_stock_weight: 0.5\n", encoding="utf-8")
    memory = tmp_path / "memory" / "trades.parquet"

    result = IntelligentPortfolioBacktestEngine(
        features_dir=features_dir,
        raw_dir=raw_dir,
        factor_config_path=factors,
        constraints_path=constraints,
        universe=_Universe(),
        top_n=2,
        rebalance_interval=5,
        memory_path=memory,
    ).run()

    assert "turnover" in result.metrics
    assert not result.regime_history.empty
    assert memory.exists()
    assert {"future_return", "prediction_error", "market_regime", "decision_reason"}.issubset(result.trades.columns)
    matured = result.trades.dropna(subset=["future_return"])
    assert (matured["model_score"].abs() < 0.2).all()
    assert result.trades["decision_reason"].str.contains("regime adjustment").all()
