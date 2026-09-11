from pathlib import Path

import pandas as pd

from quant.backtest.engine import BacktestEngine
from quant.portfolio.backtest import PortfolioBacktestEngine


def _write_v1_backtest_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    raw_dir.mkdir()
    features_dir.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=30)
    for index, code in enumerate(("000001", "000002", "000003"), start=1):
        close = 10 * (1 + index * 0.001) ** pd.Series(range(len(dates)))
        pd.DataFrame({"date": dates, "close": close}).to_parquet(raw_dir / f"{code}.parquet")
        pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.999,
                "close": close,
                "momentum_5": float(index),
            }
        ).to_parquet(features_dir / f"{code}.parquet")
    pd.DataFrame({"date": dates, "close": 100 * 1.0005 ** pd.Series(range(len(dates)))}).to_parquet(
        raw_dir / "000300.parquet"
    )
    config = tmp_path / "factor_weights.yaml"
    config.write_text(
        "factors:\n  momentum_5:\n    weight: 1.0\n    direction: 1\n", encoding="utf-8"
    )
    return raw_dir, features_dir, config


def test_portfolio_backtest_runs_v1_construction_and_writes_scores(tmp_path: Path):
    raw_dir, features_dir, config = _write_v1_backtest_data(tmp_path)

    result = PortfolioBacktestEngine(
        features_dir=features_dir,
        raw_dir=raw_dir,
        factor_config_path=config,
        top_n=2,
        rebalance_interval=5,
    ).run()

    assert (features_dir / "composite_score.parquet").exists()
    assert {
        "annual_return",
        "benchmark_return",
        "excess_return",
        "alpha",
        "beta",
        "sharpe",
        "max_drawdown",
        "volatility",
        "turnover",
    }.issubset(result.metrics)
    assert not result.holdings_history.empty
    assert result.holdings_history["weight"].max() <= 0.10
    assert not result.factor_contribution.empty
    assert not result.yearly_attribution.empty
    assert not result.sector_exposure.empty


def test_backtest_engine_keeps_legacy_default_and_exposes_v1_mode(tmp_path: Path):
    _raw_dir, features_dir, config = _write_v1_backtest_data(tmp_path)

    result = BacktestEngine(
        features_dir=features_dir,
        mode="portfolio_v1",
        factor_config_path=config,
        portfolio_top_n=2,
        rebalance_interval=5,
    ).run()

    assert "turnover" in result.metrics
