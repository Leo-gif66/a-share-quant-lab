"""A small, observable daily research pipeline built from existing engines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..data.universe import Universe
from ..data.validator import ResearchDataValidator
from ..evolution import StrategyVersionStore
from ..factors.engine import FactorEngine
from ..memory import TradeMemoryStore
from ..portfolio import CompositeScorer, FactorProcessor
from ..portfolio.allocator import AllocationResult, PortfolioAllocator
from ..regime import MarketRegimeDetector, RegimeSnapshot
from ..reporting import ResearchReportBuilder
from ..research import AlphaCandidateRanker, StrategyEvolutionEngine, TradeAttributionEngine


@dataclass(frozen=True)
class DailyPipelineResult:
    candidates_path: Path
    report_path: Path
    allocation: AllocationResult
    regime: RegimeSnapshot
    stages: pd.DataFrame
    evolved_weights_path: Path | None


class DailyResearchPipeline:
    """Run data readiness, factors, candidate ranking, and decision reporting.

    Daily operation is deliberately local-first.  It performs an incremental
    snapshot check and builds only missing or stale feature files; callers who
    want network I/O run the existing ``quant data-update`` first.  This makes
    the command repeatable and prevents a report from being delayed by an
    unavailable market-data provider.
    """

    def __init__(
        self,
        data_root: str | Path = "data",
        universe_path: str | Path = "configs/universe_large.yaml",
        factor_config_path: str | Path = "configs/factor_weights.yaml",
        memory_path: str | Path = "data/memory/trades.parquet",
        candidates_path: str | Path = "data/features/daily_candidates.parquet",
        report_path: str | Path = "reports/daily_alpha_report.html",
        evolved_weights_path: str | Path = "configs/evolved_factor_weights.yaml",
        evolution_report_path: str | Path = "reports/strategy_evolution.html",
        strategy_version_directory: str | Path = "configs/strategy_versions",
    ) -> None:
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.features_dir = self.data_root / "features"
        self.universe_path = Path(universe_path)
        self.factor_config_path = Path(factor_config_path)
        self.memory_path = Path(memory_path)
        self.candidates_path = Path(candidates_path)
        self.report_path = Path(report_path)
        self.evolved_weights_path = Path(evolved_weights_path)
        self.evolution_report_path = Path(evolution_report_path)
        self.strategy_version_directory = Path(strategy_version_directory)

    def run(self) -> DailyPipelineResult:
        stages: list[dict[str, object]] = []
        raw_files = self._raw_files()
        stages.append({"stage": "data-update", "status": "local-ready", "detail": f"{len(raw_files)} local stock histories; network refresh is explicit"})
        built = self._build_stale_factors(raw_files)
        stages.append({"stage": "factor-build", "status": "complete", "detail": f"built {built} missing or stale feature files"})
        research = self._research_summary()
        stages.append({"stage": "research-check", "status": "complete", "detail": f"valid={research['valid']}; feature_stocks={research['feature_stocks']}"})
        score_panel = self._load_or_build_scores()
        as_of = pd.Timestamp(score_panel["date"].max()).normalize()
        benchmark = self._benchmark()
        detector = MarketRegimeDetector()
        breadth = detector.market_breadth(self.raw_dir, as_of)
        industries = self._industries()
        model_predictions = self._model_predictions(as_of)
        factor_contributions = self._factor_contributions(as_of)
        candidate_result = AlphaCandidateRanker(detector).rank(
            score_panel,
            industries,
            benchmark,
            trades=TradeMemoryStore(self.memory_path).load(),
            model_predictions=model_predictions,
            factor_contributions=factor_contributions,
            as_of=as_of,
            breadth=breadth,
            output_path=self.candidates_path,
        )
        stages.append({"stage": "regime-check", "status": "complete", "detail": candidate_result.regime.state})
        prediction_count = int(candidate_result.candidates["model_prediction"].notna().sum())
        prediction_source = "persisted model" if model_predictions is not None else "past-only memory calibration"
        stages.append({"stage": "model prediction", "status": "complete", "detail": f"{prediction_source}: {prediction_count}/{len(candidate_result.candidates)} predictions"})
        allocation = PortfolioAllocator().allocate(candidate_result.candidates, as_of=as_of)
        stages.append({"stage": "portfolio generation", "status": "complete", "detail": f"holdings={len(allocation.holdings)}; cash={allocation.cash_weight:.2%}"})
        evolved_path = self._evolve_weights(benchmark)
        stages.append({"stage": "report generation", "status": "complete", "detail": str(self.report_path)})
        stage_frame = pd.DataFrame(stages)
        regime_frame = pd.DataFrame([asdict(candidate_result.regime)])
        allocation_frame = allocation.holdings.copy()
        allocation_frame["cash_weight"] = allocation.cash_weight
        ResearchReportBuilder().build_daily_alpha(
            candidate_result.candidates,
            allocation_frame,
            regime_frame,
            stage_frame,
            self.report_path,
        )
        return DailyPipelineResult(
            candidates_path=self.candidates_path,
            report_path=self.report_path,
            allocation=allocation,
            regime=candidate_result.regime,
            stages=stage_frame,
            evolved_weights_path=evolved_path,
        )

    def _raw_files(self) -> list[Path]:
        return sorted(
            path for path in self.raw_dir.glob("*.parquet") if path.stem.isdigit() and path.stem != "000300"
        )

    def _build_stale_factors(self, raw_files: list[Path]) -> int:
        engine = FactorEngine(raw_dir=self.raw_dir, features_dir=self.features_dir)
        built = 0
        for source in raw_files:
            target = self.features_dir / source.name
            if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
                engine.build_stock(source.stem)
                built += 1
        return built

    def _research_summary(self) -> Mapping[str, object]:
        stocks = Universe(self.universe_path).stocks()
        return ResearchDataValidator().summarize(stocks, raw_dir=self.raw_dir, features_dir=self.features_dir)

    def _load_or_build_scores(self) -> pd.DataFrame:
        target = self.features_dir / "composite_score.parquet"
        allowed = set(self._industries())
        if target.exists():
            values = pd.read_parquet(target)
            if {"date", "code", "composite_score"}.issubset(values.columns) and not values.empty:
                values = values.copy()
                values["code"] = values["code"].astype(str).str.zfill(6)
                return values.loc[values["code"].isin(allowed)].reset_index(drop=True)
        panel = self._latest_feature_panel()
        if panel.empty:
            raise RuntimeError("no complete feature snapshot is available for composite scoring")
        processor = FactorProcessor.from_yaml(self.factor_config_path)
        scores = CompositeScorer(processor).score_and_store(panel, target)
        return scores.loc[scores["code"].astype(str).str.zfill(6).isin(allowed)].reset_index(drop=True)

    def _latest_feature_panel(self, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        files = sorted(path for path in self.features_dir.glob("*.parquet") if path.stem.isdigit())
        frames: list[pd.DataFrame] = []
        max_dates: list[pd.Timestamp] = []
        for path in files:
            frame = pd.read_parquet(path)
            if "date" not in frame or frame.empty:
                continue
            dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
            if not dates.empty:
                max_dates.append(dates.max().normalize())
        if not max_dates:
            return pd.DataFrame()
        cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else min(max_dates)
        for path in files:
            frame = pd.read_parquet(path)
            if "date" not in frame:
                continue
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            current = frame.loc[frame["date"] == cutoff].copy()
            if current.empty:
                continue
            current["code"] = path.stem
            frames.append(current)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _processed_feature_snapshot(self, as_of: pd.Timestamp) -> pd.DataFrame:
        panel = self._latest_feature_panel(as_of)
        if panel.empty:
            return panel
        processor = FactorProcessor.from_yaml(self.factor_config_path)
        return processor.process(panel)

    def _model_predictions(self, as_of: pd.Timestamp) -> pd.DataFrame | None:
        """Use the newest compatible persisted model; do not retrain intraday."""
        panel = self._processed_feature_snapshot(as_of)
        if panel.empty:
            return None
        try:
            import joblib
        except ImportError:  # pragma: no cover - project dependency, defensive for library use
            return None
        for path in (
            Path("models/lightgbm_ranking.joblib"),
            Path("models/random_forest_ranking.joblib"),
            Path("models/linear_ranking.joblib"),
        ):
            if not path.exists():
                continue
            try:
                model = joblib.load(path)
                prediction = model.predict(panel)
            except Exception:  # noqa: BLE001, S112 - incompatible persisted artifacts are intentionally skipped
                continue
            result = panel.loc[:, ["date", "code"]].copy()
            result["model_prediction"] = pd.to_numeric(prediction, errors="coerce")
            return result
        return None

    def _factor_contributions(self, as_of: pd.Timestamp) -> pd.DataFrame | None:
        panel = self._processed_feature_snapshot(as_of)
        if panel.empty:
            return None
        processor = FactorProcessor.from_yaml(self.factor_config_path)
        result = panel.loc[:, ["date", "code"]].copy()
        for name, spec in processor.factor_specs.items():
            result[name] = panel[f"{name}_z"] * spec.weight * spec.direction
        return result

    def _benchmark(self) -> pd.DataFrame:
        path = self.raw_dir / "000300.parquet"
        if not path.exists():
            raise FileNotFoundError("daily-run requires data/raw/000300.parquet for regime detection")
        return pd.read_parquet(path)

    def _industries(self) -> dict[str, str]:
        return {
            str(stock["code"]).zfill(6): str(stock.get("industry") or stock.get("sector") or "Unknown")
            for stock in Universe(self.universe_path).stocks()
        }

    def _evolve_weights(self, benchmark: pd.DataFrame) -> Path | None:
        store = TradeMemoryStore(self.memory_path)
        trades = store.load()
        attribution = TradeAttributionEngine().review(trades, benchmark)
        stats_path = Path("research/results/factor_research_summary.csv")
        statistics = pd.read_csv(stats_path) if stats_path.exists() else None
        result = StrategyEvolutionEngine.from_yaml(self.factor_config_path).evolve(
            factor_statistics=statistics,
            realized_contribution=attribution.factor_contribution,
            trades=trades,
            output_path=self.evolved_weights_path,
        )
        ResearchReportBuilder().build_strategy_evolution(
            result.weights, result.evidence, self.evolution_report_path
        )
        StrategyVersionStore(self.strategy_version_directory).record_if_changed(
            result.weights,
            reason="Daily adaptive factor-weight update from completed-trade and factor research evidence.",
            # The daily run does not yet have a post-change holding period, so
            # unavailable performance is recorded explicitly rather than
            # attributing an imagined improvement to the new version.
            performance_before={},
            performance_after={},
        )
        return result.output_path
