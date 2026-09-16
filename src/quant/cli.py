from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml
from rich import print

from .backtest import BacktestEngine, run_backtest, score_panel
from .config import load_config
from .data.components.akshare_provider import AKShareComponentProvider
from .data.components.provider import LocalFirstComponentProvider
from .data.downloader import DataDownloader
from .data.fundamental import FundamentalStore, LocalFundamentalProvider
from .data.industry import AKShareIndustryProvider, load_industry_metadata
from .data.providers.tencent_index import TencentIndexProvider
from .data.storage import Storage
from .data.universe import Universe
from .data.universe_builder import UniverseBuilder
from .data.validator import DataValidator, ResearchDataValidator
from .demo import synthetic_prices
from .evolution import StrategyVersionStore
from .factors.engine import FactorEngine
from .factors.registry import names as factor_names
from .features import add_label, build_features
from .memory import TradeMemoryStore
from .models import (
    RankingModel,
    add_future_excess_return,
    prediction_ic_metrics,
    time_ordered_split,
)
from .models.factory import names as model_names
from .paper import PaperAccount, PaperTradingAccountV2, PaperTradingEngine, PaperTradingEngineV2
from .pipeline import DailyResearchPipeline
from .portfolio import (
    CompositeScorer,
    FactorProcessor,
    IndustryNeutralPortfolioBacktestEngine,
    InstitutionalPortfolioBacktestEngine,
    IntelligentPortfolioBacktestEngine,
    MLRankingPortfolioBacktestEngine,
    PortfolioAllocator,
    PortfolioBacktestEngine,
    compare_portfolio_results,
)
from .regime import MarketRegimeDetector
from .reporting import ResearchReportBuilder
from .research import (
    RESEARCH_FACTORS,
    AdaptiveFactorWeightEngine,
    AnnualWalkForwardResearch,
    FactorCombinationResearch,
    FactorEvaluator,
    FactorResearchDataBuilder,
    LiveSimulationEngine,
    PerformanceAttributionEngine,
    PredictionErrorAnalyzer,
    ProfessionalFactorResearchPipeline,
    StrategyDiagnosisEngine,
    TradeAttributionEngine,
)
from .training import train_model
from .validation import (
    RobustnessTester,
    ValidationCoverageAuditor,
    WalkForwardSettings,
    WalkForwardSimulator,
    capture_v4_baseline,
)

app=typer.Typer(help="Personal A-share Quant Lab")


def cfg(base, override): return load_config(base, override)


class _StockSubset:
    """Small universe adapter used when a batch has partial download coverage."""

    def __init__(self, stocks: list[dict]):
        self._stocks = stocks

    def stocks(self) -> list[dict]:
        return self._stocks


def _dataset_quality_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Compact dataset facts shared by the alpha and ML quality reports."""
    dates = pd.to_datetime(panel["date"], errors="coerce") if "date" in panel else pd.Series()
    return pd.DataFrame(
        [
            {"metric": "rows", "value": len(panel)},
            {"metric": "stocks", "value": panel["code"].nunique() if "code" in panel else 0},
            {"metric": "dates", "value": dates.nunique()},
            {"metric": "first_date", "value": str(dates.min().date()) if dates.notna().any() else ""},
            {"metric": "last_date", "value": str(dates.max().date()) if dates.notna().any() else ""},
        ]
    )


def _metrics_quality_table(metrics: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": str(name), "value": value} for name, value in metrics.items()]
    )


def _saved_factor_reliability(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _adaptive_weight_config(
    factor_config_path: str | Path,
    method: str,
    output_path: str | Path = "configs/adaptive_factor_weights.yaml",
) -> Path:
    """Build a reproducible config from saved research evidence when present."""
    processor = FactorProcessor.from_yaml(factor_config_path)
    summary_path = Path("research/results/factor_research_summary.csv")
    statistics = pd.read_csv(summary_path) if summary_path.exists() else None
    result = AdaptiveFactorWeightEngine(processor.factor_specs).calculate(
        method=method, factor_statistics=statistics, rolling_performance=statistics
    )
    return AdaptiveFactorWeightEngine(processor.factor_specs).save(result, output_path)


def _paper_price_panel(raw_dir: Path, symbols: set[str], as_of: pd.Timestamp) -> pd.DataFrame:
    """Load only bars on or before the paper decision date (no look-ahead)."""
    frames: list[pd.DataFrame] = []
    for symbol in sorted(symbols):
        path = raw_dir / f"{str(symbol).zfill(6)}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["date", "close"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.loc[frame["date"] <= as_of].copy()
        if frame.empty:
            continue
        frame["symbol"] = str(symbol).zfill(6)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol", "close"])


def _validation_inputs(
    root: Path, universe_path: str | Path, scores_path: str | Path = "data/features/composite_score.parquet"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Load the exact local snapshot used by v4 validation and perturbations."""
    scores = pd.read_parquet(scores_path)
    allowed_stocks = Universe(universe_path).stocks()
    industries = {
        str(stock["code"]).zfill(6): str(stock.get("industry") or stock.get("sector") or "Unknown")
        for stock in allowed_stocks
    }
    scores["code"] = scores["code"].astype(str).str.zfill(6)
    scores["date"] = pd.to_datetime(scores["date"], errors="raise").dt.normalize()
    scores = scores.loc[scores["code"].isin(industries)].copy()
    if scores.empty:
        raise typer.BadParameter("no composite scores match the configured universe")
    required_dates = set(scores["date"])
    frames: list[pd.DataFrame] = []
    for code in sorted(scores["code"].unique()):
        path = root / "features" / f"{code}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if not {"date", "open", "close"}.issubset(frame.columns):
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame = frame.loc[frame["date"].isin(required_dates)].copy()
        if frame.empty:
            continue
        frame["code"] = code
        frames.append(frame)
    if not frames:
        raise typer.BadParameter("no feature price rows match the composite score history")
    feature_panel = pd.concat(frames, ignore_index=True)
    prices = feature_panel.loc[:, ["date", "code", "open", "close"]].copy()
    benchmark_path = root / "raw" / "000300.parquet"
    if not benchmark_path.exists():
        raise typer.BadParameter("walk-forward requires data/raw/000300.parquet")
    benchmark = pd.read_parquet(benchmark_path, columns=["date", "close"])
    return scores, prices, benchmark, industries, feature_panel


def _perturbed_scores(feature_panel: pd.DataFrame, factor_config_path: str | Path, perturbation: float) -> pd.DataFrame:
    """Recompute composite scores after a real, normalized factor-weight shift."""
    processor = FactorProcessor.from_yaml(factor_config_path)
    names = list(processor.factor_specs)
    if not names:
        raise ValueError("factor perturbation requires configured factors")
    target = names[0]
    original = {name: spec.weight for name, spec in processor.factor_specs.items()}
    adjusted_target = float(min(1.0, max(0.0, original[target] + perturbation)))
    remainder = 1.0 - adjusted_target
    other_total = sum(weight for name, weight in original.items() if name != target)
    adjusted = {
        name: adjusted_target if name == target else (weight / other_total * remainder if other_total > 0 else 0.0)
        for name, weight in original.items()
    }
    specs = {name: replace(spec, weight=float(adjusted[name])) for name, spec in processor.factor_specs.items()}
    return CompositeScorer(FactorProcessor(specs)).score(feature_panel)


def _local_validation_benchmarks(root: Path) -> dict[str, pd.DataFrame | None]:
    paths = {"CSI500": root / "raw" / "000905.parquet", "CSI1000": root / "raw" / "000852.parquet"}
    return {name: pd.read_parquet(path, columns=["date", "close"]) if path.exists() else None for name, path in paths.items()}


def _institutional_dashboard(
    walk_results: pd.DataFrame | None = None,
    attribution: pd.DataFrame | None = None,
    robustness: pd.DataFrame | None = None,
) -> Path:
    """Build one dashboard from current evidence without assuming each artefact exists."""
    history = StrategyVersionStore().history()
    validation_root = Path("data/validation")
    if walk_results is None:
        walk_path = validation_root / "walk_forward_results.parquet"
        walk_results = pd.read_parquet(walk_path) if walk_path.exists() else pd.DataFrame()
    if attribution is None:
        attribution_path = validation_root / "performance_attribution.parquet"
        attribution = pd.read_parquet(attribution_path) if attribution_path.exists() else pd.DataFrame()
    if robustness is None:
        robustness_path = validation_root / "robustness_results.parquet"
        robustness = pd.read_parquet(robustness_path) if robustness_path.exists() else pd.DataFrame()
    paper_path = Path("data/paper/performance.parquet")
    paper = pd.read_parquet(paper_path) if paper_path.exists() else pd.DataFrame()
    sections = {
        "Walk-forward": walk_results,
        "Performance attribution": attribution,
        "Robustness": robustness,
        "Strategy history": history,
        "Paper trading": paper,
    }
    return ResearchReportBuilder().build_institutional_dashboard(sections)


@app.command("doctor")
def doctor():
    print("[bold]Registered factors[/bold]", factor_names())
    print("[bold]Registered models[/bold]", model_names())
    print("Use `quant demo` to verify the core pipeline without downloading market data.")

@app.command("data-update")
def data_update(
    base: str = "configs/base.yaml",
    override: str | None = None,
    universe_path: str | None = None,
    start_date: str = "20150101",
    end_date: str | None = None,
    max_workers: int = 4,
    only_missing: bool = False,
):
    """Incrementally download the configured A-share universe since 2015."""
    c = cfg(base, override)
    selected_universe = universe_path or c["data"].get("universe_path", "configs/universe.yaml")
    universe = Universe(selected_universe)
    if only_missing:
        raw_dir = Path(c["data"]["storage_root"]) / "raw"
        universe = _StockSubset(
            [stock for stock in universe.stocks() if not (raw_dir / f"{stock['code']}.parquet").exists()]
        )
    print(f"[bold]Downloading {len(universe.stocks())} stocks from {selected_universe}[/bold]")
    downloader = DataDownloader(
        universe=universe,
        data_dir=Path(c["data"]["storage_root"]) / "raw",
        start_date=start_date,
        end_date=end_date,
        max_workers=max_workers,
        progress=True,
    )
    result = downloader.update()
    failures_path = Path(c["data"]["storage_root"]) / "raw" / "download_failures.json"
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.write_text(
        json.dumps(result["failed"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Failures recorded in {failures_path}")
    print(
        f"[bold]Data update complete.[/bold] "
        f"saved={len(result['saved'])}, skipped={len(result.get('skipped', []))}, "
        f"failed={len(result['failed'])}"
    )


@app.command("component-update")
def component_update(component_dir: str = "data/components"):
    """Download validated CSI 300, CSI 500 and ChiNext CSV component snapshots."""
    provider = AKShareComponentProvider(component_dir=component_dir, reporter=print)
    result = provider.update_snapshots()
    for source in ("CSI300", "CSI500", "ChiNext"):
        if source in result.counts:
            print(f"[green]SAVED[/green] {source}: {result.counts[source]}")
    if result.failed:
        for source, error in result.failed.items():
            print(f"[red]FAILED[/red] {source}: {error}")
        raise typer.Exit(code=1)
    print(f"[bold]Component snapshots updated in {component_dir}[/bold]")


@app.command("industry-update")
def industry_update(
    output: str = "data/components/industry.csv",
    component_dir: str = "data/components",
):
    """Download AKShare industry metadata for use by ``universe-build``."""
    target_stocks = _component_stock_names(component_dir)
    if target_stocks:
        print(f"Reconciling industry metadata against {len(target_stocks)} component stocks")
    provider = AKShareIndustryProvider(
        output_path=output,
        reporter=print,
        target_stocks=target_stocks,
    )
    try:
        result = provider.update_snapshot()
    except Exception as exc:
        print(f"[red]FAILED[/red] {str(exc) or exc.__class__.__name__}")
        raise typer.Exit(code=1) from exc
    print(f"[green]SAVED[/green] {output}: {result.records} stocks, {result.industries} industries")
    if result.failed:
        print(f"[yellow]WARNING[/yellow] {len(result.failed)} industry boards could not be refreshed")


def _component_stock_names(component_dir: str) -> dict[str, str]:
    """Read current local component snapshots to fill provider-source coverage gaps."""
    provider = LocalFirstComponentProvider(component_dir=component_dir)
    stocks: dict[str, str] = {}
    try:
        for loader in (provider.load_csi300, provider.load_csi500, provider.load_chinext):
            for stock in loader():
                stocks[stock["code"]] = stock["name"]
    except FileNotFoundError:
        # Industry metadata is still useful without component snapshots; the
        # targeted CNInfo reconciliation simply becomes unavailable.
        return {}
    return stocks


@app.command("universe-build")
def universe_build(
    component_dir: str = "data/components",
    industry_path: str = "data/components/industry.csv",
    output: str = "configs/universe_large.yaml",
):
    """Build the large universe from local CSI 300, CSI 500 and ChiNext CSVs."""
    provider = LocalFirstComponentProvider(component_dir=component_dir)
    try:
        industry_metadata = load_industry_metadata(industry_path)
    except Exception as exc:
        print(f"[red]FAILED[/red] {str(exc) or exc.__class__.__name__}")
        raise typer.Exit(code=1) from exc
    builder = UniverseBuilder(
        component_loader=provider,
        sector_mapper=provider.sector_mapper,
        industry_metadata=industry_metadata,
    )
    try:
        stocks = builder.write_universe(output)
    except Exception as exc:
        print(f"[red]FAILED[/red] {str(exc) or exc.__class__.__name__}")
        raise typer.Exit(code=1) from exc

    counts = _universe_counts(stocks)
    print(f"[green]Universe written:[/green] {output}")
    print(f"Industry metadata records: {len(industry_metadata)}")
    _print_universe_counts(counts)


@app.command("universe-check")
def universe_check(universe_path: str = "configs/universe_large.yaml"):
    """Validate and summarize the generated real A-share universe."""
    try:
        stocks = Universe(universe_path).stocks()
    except Exception as exc:
        print(f"[red]FAILED[/red] {str(exc) or exc.__class__.__name__}")
        raise typer.Exit(code=1) from exc
    counts = _universe_counts(stocks)
    _print_universe_counts(counts)
    if counts["unknown_ratio"] >= 0.05:
        print("[red]FAILED[/red] Unknown sector ratio must be below 5%")
        raise typer.Exit(code=1)


def _universe_counts(stocks: list[dict]) -> dict[str, object]:
    """Calculate index membership and sector counts for list- or scalar-source YAML."""
    source_counts = {"CSI300": 0, "CSI500": 0, "ChiNext": 0}
    sectors = set()
    for stock in stocks:
        source = stock.get("index_source", [])
        sources = source if isinstance(source, list) else str(source).split("|")
        for name in source_counts:
            if name in sources:
                source_counts[name] += 1
        sector = stock.get("sector")
        if sector:
            sectors.add(str(sector))
    sector_distribution = {
        sector: sum(1 for stock in stocks if stock.get("sector") == sector)
        for sector in sorted(sectors)
    }
    unknown = sector_distribution.get("Unknown", 0)
    return {
        "total": len(stocks),
        **source_counts,
        "sectors": len(sectors),
        "sector_distribution": sector_distribution,
        "unknown": unknown,
        "unknown_ratio": unknown / len(stocks) if stocks else 0.0,
    }


def _print_universe_counts(counts: dict[str, object]) -> None:
    print(f"Total stocks: {counts['total']}")
    print("Index distribution:")
    print(f"CSI300: {counts['CSI300']}")
    print(f"CSI500: {counts['CSI500']}")
    print(f"ChiNext: {counts['ChiNext']}")
    print(f"Sectors: {counts['sectors']}")
    print("Sector distribution:")
    for sector, count in counts["sector_distribution"].items():
        print(f"{sector}: {count}")
    print(f"Unknown: {counts['unknown']} ({counts['unknown_ratio']:.2%})")


@app.command("benchmark-update")
def benchmark_update(base: str = "configs/base.yaml", override: str | None = None):
    """Download CSI 300 daily bars from Tencent Finance into data/raw/000300.parquet."""
    c = cfg(base, override)
    symbol = c["data"].get("benchmark_symbol", "000300")
    provider = TencentIndexProvider()
    try:
        code = provider.benchmark_code(symbol)
        benchmark = provider.get_daily_history(
            symbol=code,
            start_date=c["data"]["start_date"],
        )
        path = Storage(c["data"]["storage_root"]).write_benchmark(benchmark, code=code)
        print(f"OK {code} {path}")
    except Exception as exc:
        print(f"FAILED {symbol} {str(exc) or exc.__class__.__name__}")
        raise typer.Exit(code=1) from exc


@app.command("data-check")
def data_check(base: str = "configs/base.yaml", override: str | None = None):
    """Validate every stock parquet file in data/raw."""
    c = cfg(base, override)
    raw_dir = Path(c["data"]["storage_root"]) / "raw"
    validator = DataValidator()
    stock_files = sorted(
        path for path in raw_dir.glob("*.parquet") if path.stem.isdigit() and len(path.stem) == 6
    )

    if not stock_files:
        print(f"No stock parquet files found in {raw_dir}")
        return

    for path in stock_files:
        try:
            errors = validator.validate_parquet(path)
        except Exception as exc:  # noqa: BLE001 - corrupted parquet must not stop the scan
            errors = [str(exc) or exc.__class__.__name__]

        if errors:
            print(f"[red]FAILED[/red] {path.stem} {'; '.join(errors)}")
        else:
            print(f"[green]PASS[/green] {path.stem}")


@app.command("research-check")
def research_check(
    base: str = "configs/base.yaml",
    override: str | None = None,
    universe_path: str | None = None,
):
    """Report raw-history and feature coverage needed for cross-sectional research."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    selected_universe = universe_path or c["data"].get(
        "universe_path", "configs/universe_large.yaml"
    )
    summary = ResearchDataValidator().summarize(
        Universe(selected_universe).stocks(), raw_dir=root / "raw", features_dir=root / "features"
    )
    print(f"Stocks: {summary['stocks']}")
    print(f"Available histories: {summary['available_histories']}")
    print(f"Valid: {summary['valid']}")
    print(f"Feature stocks: {summary['feature_stocks']}")
    print(f"Average history: {summary['average_days']:.1f} days")
    print(f"Missing ratio: {summary['missing_ratio']:.2%}")
    status = "READY" if summary["research_ready"] else "NOT READY"
    print(f"Research-ready: {status}")


@app.command("validation-audit")
def validation_audit(
    base: str = "configs/base.yaml",
    override: str | None = None,
    scores_path: str = "data/features/composite_score.parquet",
):
    """Explain V4 walk-forward coverage before changing any validation setting."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    auditor = ValidationCoverageAuditor(
        raw_dir=root / "raw",
        features_dir=root / "features",
        scores_path=scores_path,
        benchmark_path=root / "raw" / "000300.parquet",
    )
    try:
        result = auditor.audit()
        report = auditor.write_report(result)
        baseline = capture_v4_baseline(root / "validation" / "walk_forward_results.parquet")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[red]FAILED[/red] {exc}")
        raise typer.Exit(code=1) from exc
    print("[bold]Validation coverage audit[/bold]")
    print(result.summary().to_string(index=False))
    print(result.rows_removed.to_string(index=False))
    print(f"Bottleneck: {result.bottleneck}")
    print(f"Coverage report: {report}")
    print(f"Frozen V4 baseline: {baseline}")


@app.command("fundamental-update")
def fundamental_update(
    source: str = "data/fundamental/fundamentals.parquet",
    output: str = "data/fundamental/fundamentals.parquet",
):
    """Validate local point-in-time fundamentals; no synthetic fallback is used."""
    provider = LocalFundamentalProvider(source)
    store = FundamentalStore(output)
    try:
        records = store.update(provider)
    except ValueError as exc:
        print(f"[red]FAILED[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if records.empty:
        print("[yellow]UNAVAILABLE[/yellow] No local fundamental disclosures were found; no data were fabricated.")
        print("Continue V5 price and market-alpha research, or supply a valid local disclosure file.")
        return
    print(
        f"[green]SAVED[/green] {output}: {len(records):,} disclosures, "
        f"{records['code'].nunique():,} securities"
    )


@app.command("factor-build")
def factor_build(base: str = "configs/base.yaml", override: str | None = None):
    """Build factors for every downloaded member of the configured universe."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    universe = Universe(c["data"].get("universe_path", "configs/universe_large.yaml"))
    downloaded = [
        stock for stock in universe.stocks() if (root / "raw" / f"{stock['code']}.parquet").exists()
    ]
    if not downloaded:
        print(f"No raw history files found for {universe.config_path}")
        return
    engine = FactorEngine(
        universe=_StockSubset(downloaded), raw_dir=root / "raw", features_dir=root / "features"
    )
    built = engine.build_all()
    for code in built:
        print(f"[green]PASS[/green] {code}")


@app.command("factor-rank")
def factor_rank(base: str = "configs/base.yaml", override: str | None = None):
    """Print latest cross-sectional v0.5 factor scores."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    ranking = FactorEngine(raw_dir=root / "raw", features_dir=root / "features").rank_latest()
    print(ranking.to_string(index=False))


@app.command("factor-analysis")
def factor_analysis(base: str = "configs/base.yaml", override: str | None = None):
    """Evaluate factors using feature files and 20-day returns from raw prices."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    builder = FactorResearchDataBuilder(features_dir=root / "features", raw_dir=root / "raw")
    analysis = FactorEvaluator().analyze(builder.build(), target_column=builder.LABEL_COLUMN)
    print("[bold]Factor analysis (future_return_20d)[/bold]")
    print(analysis.to_string(index=False))


@app.command("portfolio-backtest")
def portfolio_backtest(base: str = "configs/base.yaml", override: str | None = None):
    """Run the v1.0 risk-aware portfolio construction backtest."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    result = PortfolioBacktestEngine(features_dir=root / "features", raw_dir=root / "raw").run()
    report_keys = (
        "annual_return",
        "benchmark_return",
        "excess_return",
        "alpha",
        "beta",
        "sharpe",
        "max_drawdown",
        "volatility",
        "turnover",
    )
    print("[bold]Portfolio Performance Report[/bold]")
    print(json.dumps({key: result.metrics[key] for key in report_keys}, indent=2))
    print("[bold]Factor contribution[/bold]")
    print(result.factor_contribution.to_string(index=False))
    print("[bold]Yearly return attribution[/bold]")
    print(result.yearly_attribution.to_string(index=False))
    print("[bold]Sector exposure[/bold]")
    print(result.sector_exposure.tail(20).to_string(index=False))
    print("[bold]Top holdings history[/bold]")
    print(result.holdings_history.tail(20).to_string(index=False))


@app.command("industry-portfolio-backtest")
def industry_portfolio_backtest(base: str = "configs/base.yaml", override: str | None = None):
    """Run the v1.2 industry-neutral portfolio construction backtest."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    result = IndustryNeutralPortfolioBacktestEngine(
        features_dir=root / "features", raw_dir=root / "raw"
    ).run()
    report_keys = (
        "annual_return",
        "benchmark_return",
        "excess_return",
        "alpha",
        "beta",
        "sharpe",
        "max_drawdown",
        "volatility",
        "turnover",
    )
    print("[bold]Industry-Neutral Portfolio Performance Report[/bold]")
    print(json.dumps({key: result.metrics[key] for key in report_keys}, indent=2))
    print("[bold]Industry exposure[/bold]")
    print(result.sector_exposure.tail(30).to_string(index=False))
    print("[bold]Turnover[/bold]")
    print(result.metrics["turnover"])
    print("[bold]Holdings history[/bold]")
    print(result.holdings_history.tail(30).to_string(index=False))


@app.command("institutional-backtest")
def institutional_backtest(base: str = "configs/base.yaml", override: str | None = None):
    """Run v2.0 industry-neutral construction with the v1.3 advanced risk overlay."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    result = InstitutionalPortfolioBacktestEngine(
        features_dir=root / "features", raw_dir=root / "raw"
    ).run()
    report_keys = (
        "annual_return",
        "benchmark_return",
        "excess_return",
        "alpha",
        "beta",
        "sharpe",
        "max_drawdown",
        "volatility",
        "turnover",
    )
    print("[bold]Institutional Portfolio Performance Report[/bold]")
    print(json.dumps({key: result.metrics[key] for key in report_keys}, indent=2))
    print("[bold]Industry exposure[/bold]")
    print(result.sector_exposure.tail(30).to_string(index=False))
    print("[bold]Holdings history[/bold]")
    print(result.holdings_history.tail(30).to_string(index=False))


@app.command("regime-check")
def regime_check(base: str = "configs/base.yaml", override: str | None = None):
    """Classify the latest local CSI 300 state using trend, breadth, and volatility."""
    c = cfg(base, override)
    raw_dir = Path(c["data"]["storage_root"]) / "raw"
    benchmark = pd.read_parquet(raw_dir / "000300.parquet")
    detector = MarketRegimeDetector()
    breadth = detector.market_breadth(raw_dir, benchmark["date"].max())
    snapshot = detector.detect(benchmark, breadth)
    print("[bold]Market regime[/bold]")
    print(json.dumps({key: str(value) if key == "date" else value for key, value in asdict(snapshot).items()}, indent=2))


@app.command("intelligent-backtest")
def intelligent_backtest(
    base: str = "configs/base.yaml",
    override: str | None = None,
    adaptive_method: str = "icir",
    factor_config_path: str = "configs/factor_weights.yaml",
    memory_path: str = "data/memory/trades.parquet",
):
    """Run v3 adaptive scoring, market regimes, institutional construction, and memory."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    adaptive_path = _adaptive_weight_config(factor_config_path, adaptive_method)
    result = IntelligentPortfolioBacktestEngine(
        features_dir=root / "features",
        raw_dir=root / "raw",
        factor_config_path=adaptive_path,
        memory_path=memory_path,
    ).run()
    benchmark = pd.read_parquet(root / "raw" / "000300.parquet")
    review = TradeAttributionEngine().review_store(TradeMemoryStore(memory_path), benchmark)
    report = ResearchReportBuilder().build_strategy_review(
        review.factor_contribution,
        review.summary,
        review.industry_contribution,
        review.drawdown_analysis,
        review.monthly_summary,
    )
    print("[bold]Intelligent Portfolio Performance Report[/bold]")
    print(json.dumps(result.metrics, indent=2))
    print(f"Adaptive factor weights: {adaptive_path}")
    print(f"Trade memory: {result.memory_path}")
    print(f"Strategy review: {report}")
    print("[bold]Latest regimes[/bold]")
    print(result.regime_history.tail(20).to_string(index=False))


@app.command("trade-review")
def trade_review(
    base: str = "configs/base.yaml",
    override: str | None = None,
    memory_path: str = "data/memory/trades.parquet",
):
    """Review realized decisions in trade memory and generate the strategy review."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    benchmark_path = root / "raw" / "000300.parquet"
    benchmark = pd.read_parquet(benchmark_path) if benchmark_path.exists() else None
    review = TradeAttributionEngine().review_store(TradeMemoryStore(memory_path), benchmark)
    report = ResearchReportBuilder().build_strategy_review(
        review.factor_contribution,
        review.summary,
        review.industry_contribution,
        review.drawdown_analysis,
        review.monthly_summary,
    )
    print("[bold]Trade review[/bold]")
    print(review.summary.to_string(index=False))
    print(f"Strategy review: {report}")


@app.command("strategy-diagnosis")
def strategy_diagnosis(
    base: str = "configs/base.yaml",
    override: str | None = None,
    memory_path: str = "data/memory/trades.parquet",
    factor_config_path: str = "configs/factor_weights.yaml",
):
    """Diagnose completed trade errors and write learned factor weights."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    store = TradeMemoryStore(memory_path)
    analysis = PredictionErrorAnalyzer().analyze(store.load())
    diagnosis = StrategyDiagnosisEngine().diagnose(analysis.trades, analysis)
    report = ResearchReportBuilder().build_strategy_diagnosis(
        diagnosis.best_conditions,
        diagnosis.worst_conditions,
        diagnosis.factor_failures,
        diagnosis.prediction_biases,
        diagnosis.recommendations,
    )
    benchmark_path = root / "raw" / "000300.parquet"
    benchmark = pd.read_parquet(benchmark_path) if benchmark_path.exists() else None
    contribution = TradeAttributionEngine().review_store(store, benchmark).factor_contribution
    processor = FactorProcessor.from_yaml(factor_config_path)
    statistics_path = Path("research/results/factor_research_summary.csv")
    statistics = pd.read_csv(statistics_path) if statistics_path.exists() else None
    learned = AdaptiveFactorWeightEngine(processor.factor_specs).calculate(
        "combined", factor_statistics=statistics, realized_contribution=contribution
    )
    learned_path = AdaptiveFactorWeightEngine(processor.factor_specs).save(
        learned, "configs/learned_factor_weights.yaml"
    )
    version = StrategyVersionStore().record_if_changed(
        learned.weights,
        reason="Strategy diagnosis combined IC, ICIR, and realized contribution evidence.",
        performance_before={},
        performance_after={},
    )
    print("[bold]Strategy diagnosis[/bold]")
    print(diagnosis.recommendations.to_string(index=False))
    print("Error analysis: data/memory/error_analysis.parquet")
    print(f"Diagnosis report: {report}")
    print(f"Learned factor weights: {learned_path}")
    if version.changed:
        print(f"Strategy version: {version.path}")


@app.command("live-simulation")
def live_simulation(
    base: str = "configs/base.yaml",
    override: str | None = None,
    scores_path: str = "data/features/composite_score.parquet",
    memory_path: str = "data/memory/trades.parquet",
):
    """Run a time-ordered score calibration simulation and update decision memory."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    scores = pd.read_parquet(scores_path)
    codes = set(scores["code"].astype(str).str.zfill(6))
    price_frames: list[pd.DataFrame] = []
    for code in sorted(codes):
        path = root / "raw" / f"{code}.parquet"
        if not path.exists():
            continue
        prices = pd.read_parquet(path, columns=["date", "open"])
        prices["code"] = code
        price_frames.append(prices)
    if not price_frames:
        raise typer.BadParameter("no raw open prices matched the score universe")
    price_panel = pd.concat(price_frames, ignore_index=True)
    industries = {
        str(stock["code"]).zfill(6): str(stock.get("industry") or stock.get("sector") or "Unknown")
        for stock in Universe(c["data"].get("universe_path", "configs/universe_large.yaml")).stocks()
    }
    benchmark = pd.read_parquet(root / "raw" / "000300.parquet")
    detector = MarketRegimeDetector()

    def regime_at(date: pd.Timestamp) -> tuple[str, float, float]:
        snapshot = detector.detect(benchmark, as_of=date)
        return snapshot.state, snapshot.exposure, snapshot.volatility

    result = LiveSimulationEngine().run(
        scores,
        price_panel,
        memory=TradeMemoryStore(memory_path),
        industries=industries,
        regime_provider=regime_at,
    )
    print("[bold]Live simulation[/bold]")
    print(f"Portfolio snapshots: {result.output_path}")
    print(f"Simulated decisions: {len(result.snapshots)}")
    print(f"Trade memory rows: {len(result.trades)}")


@app.command("walk-forward")
def walk_forward(
    base: str = "configs/base.yaml",
    override: str | None = None,
    scores_path: str = "data/features/composite_score.parquet",
    train_window: int = 40,
    validation_window: int = 20,
    rebalance_frequency: int = 10,
    retrain_frequency: int = 1,
    memory_path: str = "data/validation/walk_forward_trades.parquet",
):
    """Run leakage-safe rolling training and portfolio validation on local data."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    try:
        scores, prices, benchmark, industries, _ = _validation_inputs(
            root, c["data"].get("universe_path", "configs/universe_large.yaml"), scores_path
        )
        settings = WalkForwardSettings(
            train_window=train_window,
            validation_window=validation_window,
            rebalance_frequency=rebalance_frequency,
            retrain_frequency=retrain_frequency,
        )
        result = WalkForwardSimulator(settings).run(
            scores,
            prices,
            benchmark,
            industries=industries,
            memory=TradeMemoryStore(memory_path),
            benchmarks=_local_validation_benchmarks(root),
        )
    except Exception as exc:
        print(f"[red]FAILED[/red] {str(exc) or exc.__class__.__name__}")
        raise typer.Exit(code=1) from exc
    report = ResearchReportBuilder().build_walk_forward(
        result.metrics, result.results, result.benchmark_comparison
    )
    attribution = PerformanceAttributionEngine().decompose(
        result.results.loc[:, ["date", "equity"]],
        result.results.loc[:, ["date", "benchmark"]],
        result.holdings,
        prices,
    )
    attribution_report = ResearchReportBuilder().build_performance_attribution(
        attribution.summary, attribution.daily
    )
    attribution_path = root / "validation" / "performance_attribution.parquet"
    attribution_path.parent.mkdir(parents=True, exist_ok=True)
    attribution.summary.to_parquet(attribution_path, index=False)
    dashboard = _institutional_dashboard(result.results, attribution.summary)
    print("[bold]Walk-forward validation[/bold]")
    print(json.dumps(result.metrics, indent=2, default=str))
    print(result.benchmark_comparison.to_string(index=False))
    print(f"Results: {result.output_path}")
    print(f"Trade memory: {result.memory_path}")
    print(f"Walk-forward report: {report}")
    print(f"Performance attribution: {attribution_report}")
    print(f"Institutional dashboard: {dashboard}")


@app.command("strategy-history")
def strategy_history(
    factor_config_path: str = "configs/evolved_factor_weights.yaml",
):
    """List immutable factor-weight versions and render their governance record."""
    store = StrategyVersionStore()
    if store.history().empty:
        source = Path(factor_config_path)
        if not source.exists():
            source = Path("configs/factor_weights.yaml")
        with source.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        store.record_if_changed(payload, reason="Baseline strategy configuration imported for v4 governance.")
    history = store.history()
    report = ResearchReportBuilder().build_strategy_history(history)
    dashboard = _institutional_dashboard()
    print("[bold]Strategy version history[/bold]")
    print(history.to_string(index=False))
    print(f"Strategy history report: {report}")
    print(f"Institutional dashboard: {dashboard}")


@app.command("robustness")
def robustness(
    base: str = "configs/base.yaml",
    override: str | None = None,
    scores_path: str = "data/features/composite_score.parquet",
    factor_config_path: str = "configs/factor_weights.yaml",
):
    """Measure rebalance, cost, weight, and universe sensitivity with real replays."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    try:
        scores, prices, benchmark, industries, feature_panel = _validation_inputs(
            root, c["data"].get("universe_path", "configs/universe_large.yaml"), scores_path
        )
        base_settings = WalkForwardSettings()
        scenario_number = 0

        def runner(parameters: dict[str, float | int]) -> dict[str, float]:
            nonlocal scenario_number
            scenario_number += 1
            scenario_scores = scores
            settings = base_settings
            if "factor_weight_perturbation" in parameters:
                scenario_scores = _perturbed_scores(
                    feature_panel, factor_config_path, float(parameters["factor_weight_perturbation"])
                )
            if "universe_fraction" in parameters:
                codes = sorted(scenario_scores["code"].astype(str).str.zfill(6).unique())
                keep_count = max(5, int(len(codes) * float(parameters["universe_fraction"])))
                # Evenly spaced deterministic removal is reproducible and does
                # not use return information to choose the reduced universe.
                keep = {codes[index] for index in np.linspace(0, len(codes) - 1, keep_count, dtype=int)}
                scenario_scores = scenario_scores.loc[scenario_scores["code"].astype(str).str.zfill(6).isin(keep)]
                scenario_prices = prices.loc[prices["code"].astype(str).str.zfill(6).isin(keep)]
                scenario_industries = {code: industry for code, industry in industries.items() if code in keep}
            else:
                scenario_prices = prices
                scenario_industries = industries
            if "rebalance_frequency" in parameters:
                settings = replace(settings, rebalance_frequency=int(parameters["rebalance_frequency"]))
            if "transaction_cost_multiplier" in parameters:
                multiplier = float(parameters["transaction_cost_multiplier"])
                settings = replace(
                    settings,
                    commission=settings.commission * multiplier,
                    stamp_tax=settings.stamp_tax * multiplier,
                    slippage=settings.slippage * multiplier,
                )
            result = WalkForwardSimulator(settings).run(
                scenario_scores,
                scenario_prices,
                benchmark,
                industries=scenario_industries,
                memory=TradeMemoryStore(root / "validation" / "robustness_trades.parquet"),
                output_path=root / "validation" / f"robustness_{scenario_number:02d}.parquet",
            )
            return result.metrics

        results = RobustnessTester().run(runner)
    except Exception as exc:
        print(f"[red]FAILED[/red] {str(exc) or exc.__class__.__name__}")
        raise typer.Exit(code=1) from exc
    report = ResearchReportBuilder().build_robustness(results)
    robustness_path = root / "validation" / "robustness_results.parquet"
    robustness_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(robustness_path, index=False)
    dashboard = _institutional_dashboard(robustness=results)
    print("[bold]Robustness validation[/bold]")
    print(results.to_string(index=False))
    print(f"Robustness report: {report}")
    print(f"Institutional dashboard: {dashboard}")


@app.command("paper-start")
def paper_start(scores_path: str = "data/features/composite_score.parquet", state_path: str = "data/paper/account.json"):
    """Create pending paper orders for the business day after the latest score date."""
    scores = pd.read_parquet(scores_path)
    orders = PaperTradingEngine().plan_next_session(scores)
    account = PaperAccount.load(state_path)
    account.submit(orders)
    saved = account.save(state_path)
    if not orders:
        print("No eligible scores; no paper orders created.")
        return
    print(f"Planned {len(orders)} paper orders for {orders[0].execution_date.date()}")
    print(f"Paper-account state saved to {saved}")


@app.command("daily-run")
def daily_run(
    base: str = "configs/base.yaml",
    override: str | None = None,
    factor_config_path: str = "configs/factor_weights.yaml",
    memory_path: str = "data/memory/trades.parquet",
):
    """Run the local-first v3.8 daily research and portfolio-decision pipeline."""
    c = cfg(base, override)
    pipeline = DailyResearchPipeline(
        data_root=c["data"]["storage_root"],
        universe_path=c["data"].get("universe_path", "configs/universe_large.yaml"),
        factor_config_path=factor_config_path,
        memory_path=memory_path,
    )
    try:
        result = pipeline.run()
    except Exception as exc:
        print(f"[red]FAILED[/red] {str(exc) or exc.__class__.__name__}")
        raise typer.Exit(code=1) from exc
    print("[bold]Daily research pipeline[/bold]")
    print(result.stages.to_string(index=False))
    print(f"Candidates: {result.candidates_path}")
    print(f"Daily alpha report: {result.report_path}")
    if result.evolved_weights_path is not None:
        print(f"Evolved factor weights: {result.evolved_weights_path}")


@app.command("paper-run")
def paper_run(
    base: str = "configs/base.yaml",
    override: str | None = None,
    candidates_path: str = "data/features/daily_candidates.parquet",
    state_path: str = "data/paper/v2_account.json",
    output_dir: str = "data/paper",
):
    """Execute one constrained v2 paper rebalance from daily candidates."""
    c = cfg(base, override)
    source = Path(candidates_path)
    if not source.exists():
        print(f"[red]FAILED[/red] daily candidates not found: {source}; run `quant daily-run` first")
        raise typer.Exit(code=1)
    candidates = pd.read_parquet(source)
    if candidates.empty:
        print("No candidates available; paper account was not changed.")
        return
    as_of = pd.to_datetime(candidates["date"], errors="raise").max().normalize()
    allocation = PortfolioAllocator().allocate(candidates, as_of=as_of)
    existing = PaperTradingAccountV2.load(state_path)
    symbols = set(allocation.holdings.get("symbol", pd.Series(dtype=str)).astype(str)) | set(existing.positions)
    prices = _paper_price_panel(Path(c["data"]["storage_root"]) / "raw", symbols, as_of)
    if prices.empty:
        print("[red]FAILED[/red] no local prices are available on or before the candidate date")
        raise typer.Exit(code=1)
    try:
        result = PaperTradingEngineV2().run(
            allocation.holdings,
            prices,
            state_path=state_path,
            output_dir=output_dir,
            as_of=as_of,
        )
    except Exception as exc:
        print(f"[red]FAILED[/red] {str(exc) or exc.__class__.__name__}")
        raise typer.Exit(code=1) from exc
    positions = pd.DataFrame(
        [{"symbol": symbol, "quantity": quantity} for symbol, quantity in sorted(result.account.positions.items())]
    )
    report = ResearchReportBuilder().build_paper_performance(
        result.performance,
        result.fills,
        positions,
    )
    print("[bold]Paper trading v2[/bold]")
    print(result.performance.to_string(index=False))
    if result.already_processed:
        print(f"Already processed for {as_of.date()}; no duplicate fills. cash: {result.account.cash:.2f}")
    else:
        print(f"Fills: {len(result.fills)}; cash: {result.account.cash:.2f}")
    print(f"Paper performance report: {report}")


@app.command("ml-train")
def ml_train(
    model_name: str = "random_forest",
    base: str = "configs/base.yaml",
    override: str | None = None,
    output_dir: str = "models",
):
    """Fit a chronological factor-ranking model to future 20-day excess returns."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    panel = FactorResearchDataBuilder(features_dir=root / "features", raw_dir=root / "raw").build()
    benchmark = pd.read_parquet(root / "raw" / "000300.parquet")
    labelled = add_future_excess_return(panel, benchmark)
    processor = FactorProcessor.from_yaml()
    normalized = processor.process(labelled)
    features = [f"{factor}_z" for factor in processor.factor_names]
    split = time_ordered_split(normalized.dropna(subset=["future_excess_return_20d"]), purge_dates=20)
    model = RankingModel(model_name)  # type: ignore[arg-type]
    model.fit(split.train, features, "future_excess_return_20d")
    output = Path(output_dir)
    model_path = model.save(output / f"{model_name}_ranking.joblib")
    importance = model.feature_importance()
    importance.to_csv(output / f"{model_name}_feature_importance.csv", index=False)
    metrics = prediction_ic_metrics(
        split.test, model.predict(split.test), "future_excess_return_20d"
    )
    metrics.update(
        {
            "training_rows": len(split.train),
            "validation_rows": len(split.validation),
            "test_rows": len(split.test),
            **model.ranking_label_info,
        }
    )
    (output / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    quality_report = ResearchReportBuilder().build_research_quality(
        _dataset_quality_table(normalized),
        processor.last_coverage,
        _saved_factor_reliability(Path("research/results/factor_research_summary.csv")),
        _metrics_quality_table(metrics),
    )
    print(f"Model saved: {model_path}")
    print(f"Research quality report: {quality_report}")
    print(json.dumps(metrics, indent=2))
    print(importance.to_string(index=False))


@app.command("alpha-research")
def alpha_research(
    base: str = "configs/base.yaml",
    override: str | None = None,
    output_dir: str = "research/results",
    min_cross_section: int = 20,
):
    """Run neutralized factor statistics, combinations, and annual walk-forward research."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    pipeline = ProfessionalFactorResearchPipeline(
        features_dir=root / "features", raw_dir=root / "raw", min_cross_section=min_cross_section
    )
    try:
        result = pipeline.run(output_dir)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"[red]FAILED[/red] {exc}")
        raise typer.Exit(code=1) from exc
    neutral_factors = [f"{factor}_neutralized" for factor in RESEARCH_FACTORS]
    _scores, weights = FactorCombinationResearch().compare(
        result.neutralized_panel, neutral_factors, result, collect_scores=False
    )
    directory = Path(output_dir)
    weights.to_csv(directory / "factor_weights_report.csv", index=False)
    walk_forward = AnnualWalkForwardResearch().run(
        result.neutralized_panel, neutral_factors, output_path=directory / "walk_forward_report.csv"
    )
    importance_path = Path("models/random_forest_feature_importance.csv")
    importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
    comparison_path = directory / "alpha_backtest_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    report = ResearchReportBuilder().build_alpha_research(
        result.summary.sort_values("ICIR", ascending=False),
        result.yearly_stability,
        importance,
        walk_forward,
        comparison,
    )
    quality_report = ResearchReportBuilder().build_research_quality(
        _dataset_quality_table(result.neutralized_panel),
        result.factor_coverage,
        result.summary,
        _metrics_quality_table(
            json.loads(Path("models/lightgbm_metrics.json").read_text(encoding="utf-8"))
            if Path("models/lightgbm_metrics.json").exists()
            else {}
        ),
    )
    print(f"Factor summary: {directory / 'factor_research_summary.csv'}")
    print(f"Factor weights: {directory / 'factor_weights_report.csv'}")
    print(f"Walk-forward report: {directory / 'walk_forward_report.csv'}")
    print(f"Alpha research report: {report}")
    print(f"Research quality report: {quality_report}")
    print(result.summary.sort_values("ICIR", ascending=False).to_string(index=False))


@app.command("alpha-backtest")
def alpha_backtest(
    model_name: str = "random_forest",
    base: str = "configs/base.yaml",
    override: str | None = None,
):
    """Compare handcrafted institutional targets with matured-label ML ranking."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    handcrafted = InstitutionalPortfolioBacktestEngine(features_dir=root / "features", raw_dir=root / "raw").run()
    ml_engine = MLRankingPortfolioBacktestEngine(
        features_dir=root / "features", raw_dir=root / "raw", model_name=model_name  # type: ignore[arg-type]
    )
    ml = ml_engine.run()
    comparison = compare_portfolio_results(handcrafted, ml).table
    output_dir = Path("research/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "alpha_backtest_comparison.csv", index=False)
    if not ml_engine.ml_feature_importance.empty:
        ml_engine.ml_feature_importance.to_csv("models/ml_portfolio_feature_importance.csv", index=False)
    factor_summary_path = output_dir / "factor_research_summary.csv"
    factor_stability_path = output_dir / "factor_stability.csv"
    walk_forward_path = output_dir / "walk_forward_report.csv"
    importance_path = Path("models") / f"{model_name}_feature_importance.csv"
    report = ResearchReportBuilder().build_alpha_research(
        pd.read_csv(factor_summary_path) if factor_summary_path.exists() else pd.DataFrame(),
        pd.read_csv(factor_stability_path) if factor_stability_path.exists() else pd.DataFrame(),
        ml_engine.ml_feature_importance if not ml_engine.ml_feature_importance.empty else (
            pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
        ),
        pd.read_csv(walk_forward_path) if walk_forward_path.exists() else pd.DataFrame(),
        comparison,
    )
    print("[bold]Handcrafted vs ML-ranking portfolio[/bold]")
    print(comparison.to_string(index=False))
    print(f"Alpha research report: {report}")


@app.command("report-build")
def report_build(
    base: str = "configs/base.yaml", override: str | None = None, output: str = "reports/latest.html"
):
    """Run the institutional backtest and render its self-contained HTML report."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    result = InstitutionalPortfolioBacktestEngine(
        features_dir=root / "features", raw_dir=root / "raw"
    ).run()
    importance_path = Path("models/random_forest_feature_importance.csv")
    importance = pd.read_csv(importance_path) if importance_path.exists() else None
    path = ResearchReportBuilder().from_backtest_result(result, importance, output)
    print(f"Research report written to {path}")


@app.command("features-build")
def features_build(base:str="configs/base.yaml", override:str|None=None):
    c=cfg(base,override); s=Storage(c["data"]["storage_root"])
    frames={code:s.read_stock(code) for code in s.stock_codes()}; panel=build_features(frames,c["features"]["enabled"]); s.write_features(panel)
    ds=add_label(panel,c["label"]["horizon"],c["label"]["type"]); s.write_dataset(ds)
    print(f"features rows={len(panel):,}; dataset rows={len(ds):,}")

@app.command("train")
def train(base:str="configs/base.yaml", override:str|None=None):
    c=cfg(base,override); s=Storage(c["data"]["storage_root"]); m=train_model(s.read_dataset(),c); print(json.dumps(m,indent=2))

@app.command("backtest")
def backtest(base: str = "configs/base.yaml", override: str | None = None):
    """Run the v0.6 multi-factor, next-open portfolio backtest."""
    c = cfg(base, override)
    result = BacktestEngine(features_dir=Path(c["data"]["storage_root"]) / "features").run()
    print("[bold]Multi-factor backtest report[/bold]")
    print(json.dumps(result.metrics, indent=2))
    print("[bold]Annual returns[/bold]")
    print(result.annual_returns.to_string(index=False))
    print("[bold]Monthly returns (latest 12)[/bold]")
    print(result.monthly_returns.tail(12).to_string(index=False))
    print("[bold]Drawdown series (latest 12)[/bold]")
    print(result.drawdown_series.tail(12).to_string(index=False))
    print("[bold]Yearly attribution[/bold]")
    print(result.yearly_attribution.to_string(index=False))
    print("[bold]Drawdown periods[/bold]")
    print(result.drawdown_periods.to_string(index=False))
    print("[bold]Worst holding periods[/bold]")
    print(result.worst_holding_periods.to_string(index=False))

@app.command("signal")
def signal(base:str="configs/base.yaml", override:str|None=None, model_path:str|None=None):
    c=cfg(base,override); s=Storage(c["data"]["storage_root"]); panel=s.read_features(); scored=score_panel(panel,c,model_path); d=scored.date.max(); top=scored[scored.date==d].sort_values("prediction",ascending=False).head(c["portfolio"]["top_n"]); print(top[["date","code","prediction"]].to_string(index=False))

@app.command("demo")
def demo(base:str="configs/base.yaml"):
    c=cfg(base,None); prices,b=synthetic_prices(); panel=build_features(prices,c["features"]["enabled"]); scored=score_panel(panel,c); _eq,_tr,m=run_backtest(scored,b,c); print("[bold green]Core pipeline works.[/bold green]"); print(json.dumps(m,indent=2))

if __name__=="__main__": app()
