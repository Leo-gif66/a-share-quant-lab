import json

import numpy as np
import pandas as pd
import yaml

from quant.paper import PaperTradingEngineV2, PaperTradingSettings
from quant.pipeline import DailyResearchPipeline
from quant.portfolio import AllocationSettings, PortfolioAllocator
from quant.regime import MarketRegimeDetector
from quant.research import AlphaCandidateRanker, StrategyEvolutionEngine
from quant.portfolio.factor_processing import FactorSpec


def _benchmark(periods: int = 210) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=periods)
    return pd.DataFrame({"date": dates, "close": 4_000 + np.arange(periods, dtype=float)})


def test_candidate_ranking_uses_only_returns_matured_before_signal_date(tmp_path):
    as_of = pd.Timestamp("2024-03-01")
    history = pd.DataFrame(
        {
            "model_score": np.linspace(0.01, 0.10, 10).tolist() + [0.99],
            "future_return": np.linspace(0.01, 0.10, 10).tolist() + [-0.99],
            "exit_date": list(pd.bdate_range("2024-01-02", periods=10)) + [pd.Timestamp("2024-04-01")],
        }
    )
    scores = pd.DataFrame({"date": [as_of, as_of], "code": ["1", "2"], "composite_score": [0.05, 0.02]})
    result = AlphaCandidateRanker(MarketRegimeDetector()).rank(
        scores,
        {"000001": "Bank", "000002": "Technology"},
        _benchmark(),
        trades=history,
        as_of=as_of,
        output_path=tmp_path / "daily_candidates.parquet",
    )

    assert result.output_path.exists()
    assert result.candidates.loc[0, "model_prediction"] > 0
    assert json.loads(result.candidates.loc[0, "factor_contribution"]) == {"composite_score": 0.05}


def test_allocator_respects_stock_industry_and_regime_cash_constraints():
    candidates = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 5,
            "symbol": ["1", "2", "3", "4", "5"],
            "score": [5, 4, 3, 2, 1],
            "model_prediction": [0.05, 0.04, 0.03, 0.02, 0.01],
            "industry": ["Bank", "Bank", "Bank", "Technology", "Technology"],
            "risk": [0.0] * 5,
            "regime": ["bear"] * 5,
        }
    )
    result = PortfolioAllocator(AllocationSettings(top_n=5)).allocate(candidates)

    assert (result.holdings["weight"] <= 0.10 + 1e-12).all()
    assert (result.industry_weights["weight"] <= 0.25 + 1e-12).all()
    assert result.cash_weight >= 0.60 - 1e-12
    assert result.invested_weight + result.cash_weight <= 1.0 + 1e-12


def test_paper_v2_persists_orders_fills_cash_positions_and_costs_without_future_prices(tmp_path):
    targets = pd.DataFrame(
        {"date": ["2024-01-02", "2024-01-02"], "symbol": ["1", "2"], "weight": [0.10, 0.10]}
    )
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "symbol": ["1", "2", "1"],
            "close": [10.0, 20.0, 1_000_000.0],
        }
    )
    engine = PaperTradingEngineV2(PaperTradingSettings(initial_cash=100_000, transaction_cost_rate=0.001))
    result = engine.run(targets, prices, tmp_path / "account.json", tmp_path, as_of="2024-01-02")

    assert result.account.cash > 0
    assert result.account.positions
    assert result.fills["cost"].sum() > 0
    assert (tmp_path / "orders.parquet").exists()
    assert (tmp_path / "fills.parquet").exists()
    assert (tmp_path / "performance.parquet").exists()
    assert result.performance.loc[0, "market_value"] < 30_000  # 1/3 future bar was not used
    rerun = engine.run(targets, prices, tmp_path / "account.json", tmp_path, as_of="2024-01-02")
    assert rerun.already_processed
    assert rerun.fills.empty
    assert rerun.account.cash == result.account.cash


def test_evolution_caps_each_factor_change_and_writes_a_valid_config(tmp_path):
    specs = {"momentum_5": FactorSpec(0.5, 1), "volatility": FactorSpec(0.5, -1)}
    statistics = pd.DataFrame({"factor": ["momentum_5", "volatility"], "IC": [100.0, 0.0], "ICIR": [100.0, 0.0]})
    contribution = pd.DataFrame({"factor_name": ["momentum_5", "volatility"], "contribution": [100.0, 0.0]})
    output = tmp_path / "evolved.yaml"
    result = StrategyEvolutionEngine(specs, max_weight_change=0.10).evolve(statistics, contribution, output_path=output)

    assert output.exists()
    assert np.isclose(result.weights["weight"].sum(), 1.0)
    assert (result.weights["weight_change"].abs() <= 0.10 + 1e-12).all()
    assert set(yaml.safe_load(output.read_text(encoding="utf-8"))["factors"]) == set(specs)


def test_daily_pipeline_generates_candidates_and_report_from_local_snapshots(tmp_path):
    root = tmp_path / "data"
    raw = root / "raw"
    raw.mkdir(parents=True)
    dates = pd.bdate_range("2023-01-02", periods=210)
    _benchmark().to_parquet(raw / "000300.parquet", index=False)
    for number, drift in ((1, 0.01), (2, 0.02)):
        close = 10 + np.arange(len(dates)) * drift
        pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": 1_000 + np.arange(len(dates)),
                "amount": (1_000 + np.arange(len(dates))) * close,
                "turnover": 1.0,
            }
        ).to_parquet(raw / f"{number:06d}.parquet", index=False)
    universe = tmp_path / "universe.yaml"
    universe.write_text(
        yaml.safe_dump({"stocks": [{"code": "000001", "name": "A", "market": "SZ", "sector": "Bank"}, {"code": "000002", "name": "B", "market": "SZ", "sector": "Technology"}]}),
        encoding="utf-8",
    )
    config = tmp_path / "factors.yaml"
    config.write_text(
        yaml.safe_dump({"factors": {"momentum_5": {"weight": 0.5, "direction": 1}, "volatility": {"weight": 0.5, "direction": -1}}}),
        encoding="utf-8",
    )
    pipeline = DailyResearchPipeline(
        data_root=root,
        universe_path=universe,
        factor_config_path=config,
        memory_path=tmp_path / "trades.parquet",
        candidates_path=root / "features" / "daily_candidates.parquet",
        report_path=tmp_path / "daily_alpha_report.html",
        evolved_weights_path=tmp_path / "evolved_factor_weights.yaml",
        evolution_report_path=tmp_path / "strategy_evolution.html",
        strategy_version_directory=tmp_path / "strategy_versions",
    )
    result = pipeline.run()

    assert result.candidates_path.exists()
    assert result.report_path.exists()
    assert set(result.stages["stage"]) == {"data-update", "factor-build", "research-check", "regime-check", "model prediction", "portfolio generation", "report generation"}
    assert set(pd.read_parquet(result.candidates_path)["symbol"]) == {"000001", "000002"}
