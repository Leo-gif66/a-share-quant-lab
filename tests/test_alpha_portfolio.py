from types import SimpleNamespace

import pandas as pd

from quant.portfolio import MLRankingPortfolioBacktestEngine, compare_portfolio_results


class _Universe:
    def __init__(self, codes):
        self.codes = codes

    def stocks(self):
        return [
            {"code": code, "name": code, "market": "SZ", "sector": f"Sector{index % 2}"}
            for index, code in enumerate(self.codes)
        ]


def test_ml_ranking_score_provider_uses_matured_labels_and_comparison_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    codes = [f"{index + 1:06d}" for index in range(10)]
    dates = pd.bdate_range("2023-01-02", periods=70)
    benchmark = pd.DataFrame({"date": dates, "close": 100 + pd.Series(range(len(dates))) * 0.01})
    benchmark.to_parquet(raw_dir / "000300.parquet", index=False)
    config = tmp_path / "factors.yaml"
    config.write_text("factors:\n  momentum_5:\n    weight: 1.0\n    direction: 1\n", encoding="utf-8")
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "code": code,
                    "open": 10 + index + pd.Series(range(len(dates))) * 0.01,
                    "close": 10 + index + pd.Series(range(len(dates))) * (0.01 + index * 0.0001),
                    "momentum_5": index + pd.Series(range(len(dates))) * 0.001,
                }
            )
            for index, code in enumerate(codes)
        ],
        ignore_index=True,
    )
    engine = MLRankingPortfolioBacktestEngine(
        features_dir=tmp_path / "features",
        raw_dir=raw_dir,
        factor_config_path=config,
        universe=_Universe(codes),
        label_horizon=5,
        train_window=30,
        retrain_interval=10,
        rebalance_interval=5,
        model_name="linear",
    )

    scores = engine._score_panel(panel)
    comparison = compare_portfolio_results(
        SimpleNamespace(metrics={"annual_return": 0.1, "sharpe": 1.0, "max_drawdown": -0.1, "alpha": 0.01, "beta": 0.8, "turnover": 2.0}),
        SimpleNamespace(metrics={"annual_return": 0.2, "sharpe": 1.1, "max_drawdown": -0.2, "alpha": 0.02, "beta": 0.7, "turnover": 3.0}),
    ).table

    assert not scores.empty
    assert scores["date"].min() >= dates[20]  # five-day label maturity plus training history
    assert not engine.ml_feature_importance.empty
    assert list(comparison["portfolio"]) == ["handcrafted", "ml_ranking"]
