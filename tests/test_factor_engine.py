from pathlib import Path

import numpy as np
import pandas as pd

from quant.factors.engine import FactorEngine


class FakeUniverse:
    def stocks(self):
        return [
            {"code": "000001", "name": "Rising", "market": "SZ", "sector": "test"},
            {"code": "000002", "name": "Falling", "market": "SZ", "sector": "test"},
        ]


def raw_prices(days: int = 80, rising: bool = True) -> pd.DataFrame:
    close = np.linspace(10, 20, days) if rising else np.linspace(20, 10, days)
    volume = np.linspace(1_000, 2_000, days) if rising else np.linspace(2_000, 1_000, days)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=days),
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "turnover": 1.0,
        }
    )


def test_factor_engine_builds_expected_factors(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    prices = raw_prices()
    prices.to_parquet(raw_dir / "000001.parquet", index=False)
    prices.to_parquet(raw_dir / "000002.parquet", index=False)
    engine = FactorEngine(FakeUniverse(), raw_dir=raw_dir, features_dir=tmp_path / "features")

    built = engine.build_all()
    result = pd.read_parquet(built["000001"])
    expected = prices["close"].pct_change(20).iloc[-1]

    assert set(FactorEngine.FACTOR_COLUMNS).issubset(result.columns)
    assert result["momentum_5"].iloc[-1] == prices["close"].pct_change(5).iloc[-1]
    assert result["momentum_20"].iloc[-1] == expected
    assert result["momentum_60"].iloc[-1] == prices["close"].pct_change(60).iloc[-1]
    assert result["trend_20"].iloc[-1] == prices["close"].iloc[-1] / prices["close"].iloc[-20:].mean() - 1
    assert result["trend_60"].iloc[-1] == prices["close"].iloc[-1] / prices["close"].iloc[-60:].mean() - 1
    assert result["volatility_20"].iloc[-1] == prices["close"].pct_change().rolling(20).std().iloc[-1]
    assert result["liquidity_20"].iloc[-1] == prices["amount"].iloc[-20:].mean()
    assert result["volume_ratio_20"].iloc[-1] == prices["volume"].iloc[-1] / prices["volume"].iloc[-20:].mean()
    assert result["drawdown_60"].iloc[-1] == prices["close"].iloc[-1] / prices["close"].iloc[-60:].max() - 1


def test_factor_engine_ranks_latest_date(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_prices(rising=True).to_parquet(raw_dir / "000001.parquet", index=False)
    raw_prices(rising=False).to_parquet(raw_dir / "000002.parquet", index=False)
    engine = FactorEngine(FakeUniverse(), raw_dir=raw_dir, features_dir=tmp_path / "features")
    engine.build_all()

    ranking = engine.rank_latest()

    assert list(ranking.columns) == ["code", "name", "score"]
    assert ranking.iloc[0]["code"] == "000001"
    assert ranking["score"].is_monotonic_decreasing
