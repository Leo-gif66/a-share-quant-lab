from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich import print

from .backtest import BacktestEngine, run_backtest, score_panel
from .config import load_config
from .data.downloader import DataDownloader
from .data.components.akshare_provider import AKShareComponentProvider
from .data.components.provider import LocalFirstComponentProvider
from .data.industry import AKShareIndustryProvider, load_industry_metadata
from .data.providers.tencent_index import TencentIndexProvider
from .data.storage import Storage
from .data.universe import Universe
from .data.universe_builder import UniverseBuilder
from .data.validator import DataValidator, ResearchDataValidator
from .demo import synthetic_prices
from .factors.engine import FactorEngine
from .factors.registry import names as factor_names
from .features import add_label, build_features
from .models.factory import names as model_names
from .models import (
    RankingModel,
    add_future_excess_return,
    prediction_ic_metrics,
    time_ordered_split,
)
from .paper import PaperAccount, PaperTradingEngine
from .portfolio import (
    IndustryNeutralPortfolioBacktestEngine,
    InstitutionalPortfolioBacktestEngine,
    MLRankingPortfolioBacktestEngine,
    FactorProcessor,
    PortfolioBacktestEngine,
    compare_portfolio_results,
)
from .reporting import ResearchReportBuilder
from .research import (
    AnnualWalkForwardResearch,
    FactorCombinationResearch,
    FactorEvaluator,
    FactorResearchDataBuilder,
    ProfessionalFactorResearchPipeline,
    RESEARCH_FACTORS,
)
from .training import train_model

app=typer.Typer(help="Personal A-share Quant Lab")


def cfg(base, override): return load_config(base, override)


class _StockSubset:
    """Small universe adapter used when a batch has partial download coverage."""

    def __init__(self, stocks: list[dict]):
        self._stocks = stocks

    def stocks(self) -> list[dict]:
        return self._stocks


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
    except Exception as exc:  # noqa: BLE001 - an old snapshot remains intact on a failed refresh
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
    except Exception as exc:  # noqa: BLE001 - invalid local metadata must not be silently ignored
        print(f"[red]FAILED[/red] {str(exc) or exc.__class__.__name__}")
        raise typer.Exit(code=1) from exc
    builder = UniverseBuilder(
        component_loader=provider,
        sector_mapper=provider.sector_mapper,
        industry_metadata=industry_metadata,
    )
    try:
        stocks = builder.write_universe(output)
    except Exception as exc:  # noqa: BLE001 - show an actionable import failure in the CLI
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
    except Exception as exc:  # noqa: BLE001 - malformed user configs should report cleanly
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
    (output / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model saved: {model_path}")
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
    _scores, weights = FactorCombinationResearch().compare(result.neutralized_panel, neutral_factors, result)
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
    print(f"Factor summary: {directory / 'factor_research_summary.csv'}")
    print(f"Factor weights: {directory / 'factor_weights_report.csv'}")
    print(f"Walk-forward report: {directory / 'walk_forward_report.csv'}")
    print(f"Alpha research report: {report}")
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
