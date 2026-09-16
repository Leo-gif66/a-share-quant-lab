from pathlib import Path

import pandas as pd
import pytest

from quant.data.universe import Universe
from quant.portfolio.industry_neutral import (
    IndustryNeutralBuilder,
    IndustryNeutralPortfolioBacktestEngine,
)


def _scores_and_industries() -> tuple[pd.DataFrame, pd.DataFrame]:
    date = pd.Timestamp("2024-01-02")
    records = []
    industries = []
    for group_index, industry in enumerate(("Consumer", "Finance", "Healthcare", "Industrial", "Tech")):
        for rank in range(4):
            code = f"{group_index * 4 + rank + 1:06d}"
            records.append(
                {"date": date, "code": code, "composite_score": 100 - group_index * 10 - rank}
            )
            industries.append({"code": code, "industry": industry})
    return pd.DataFrame(records), pd.DataFrame(industries)


def test_industry_neutral_builder_ranks_within_industry_and_respects_limits():
    scores, industries = _scores_and_industries()
    targets = IndustryNeutralBuilder(top_n=20).construct(scores, industries)

    assert len(targets) == 20
    assert targets["weight"].sum() == pytest.approx(1.0)
    assert targets["industry"].nunique() == 5
    assert targets.groupby("industry")["weight"].sum().max() <= 0.25
    assert targets["weight"].max() <= 0.10
    assert targets.groupby("industry")["industry_rank"].max().eq(4).all()


def test_industry_neutral_builder_expands_sparse_industries_to_fill_top_n():
    date = pd.Timestamp("2024-01-02")
    scores = pd.DataFrame(
        {
            "date": [date] * 20,
            "code": [f"{number:06d}" for number in range(1, 21)],
            "composite_score": list(reversed(range(20))),
        }
    )
    metadata = pd.DataFrame(
        {
            "code": scores["code"],
            "industry": [f"Industry {number}" for number in range(20)],
        }
    )

    targets = IndustryNeutralBuilder(top_n=20).construct(scores, metadata)

    assert len(targets) == 20
    assert targets["industry"].nunique() == 20
    assert targets["weight"].sum() == pytest.approx(1.0)
    assert targets["weight"].max() <= 0.10


def test_industry_neutral_backtest_uses_existing_accounting_and_reports_exposure(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    raw_dir.mkdir()
    features_dir.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=30)
    _scores, industries = _scores_and_industries()
    stocks = []
    for number, row in enumerate(industries.itertuples(index=False), start=1):
        close = 10 * (1 + number * 0.0002) ** pd.Series(range(len(dates)))
        pd.DataFrame({"date": dates, "close": close}).to_parquet(raw_dir / f"{row.code}.parquet")
        pd.DataFrame(
            {"date": dates, "open": close * 0.999, "close": close, "momentum_5": number}
        ).to_parquet(features_dir / f"{row.code}.parquet")
        stocks.append({"code": row.code, "name": row.code, "market": "SZ", "sector": row.industry})
    pd.DataFrame({"date": dates, "close": 100 * 1.0005 ** pd.Series(range(len(dates)))}).to_parquet(
        raw_dir / "000300.parquet"
    )
    factor_config = tmp_path / "factor_weights.yaml"
    factor_config.write_text(
        "factors:\n  momentum_5:\n    weight: 1.0\n    direction: 1\n", encoding="utf-8"
    )
    universe_config = tmp_path / "universe.yaml"
    import yaml

    universe_config.write_text(yaml.safe_dump({"stocks": stocks}), encoding="utf-8")

    result = IndustryNeutralPortfolioBacktestEngine(
        features_dir=features_dir,
        raw_dir=raw_dir,
        factor_config_path=factor_config,
        rebalance_interval=5,
        top_n=20,
        universe=Universe(universe_config),
    ).run()

    assert {"annual_return", "sharpe", "max_drawdown", "alpha", "beta"}.issubset(result.metrics)
    assert {"industry", "portfolio_weight", "benchmark_weight", "active_weight"}.issubset(
        result.sector_exposure.columns
    )
    assert result.holdings_history["weight"].max() <= 0.10
