from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from .backtest import run_backtest, score_panel
from .config import load_config
from .data.akshare_provider import AkshareProvider
from .data.downloader import DataDownloader
from .data.storage import Storage
from .data.validator import DataValidator
from .demo import synthetic_prices
from .factors.engine import FactorEngine
from .factors.registry import names as factor_names
from .features import add_label, build_features
from .models.factory import names as model_names
from .training import train_model

app=typer.Typer(help="Personal A-share Quant Lab")


def cfg(base, override): return load_config(base, override)


@app.command("doctor")
def doctor():
    print("[bold]Registered factors[/bold]", factor_names())
    print("[bold]Registered models[/bold]", model_names())
    print("Use `quant demo` to verify the core pipeline without downloading market data.")

@app.command("data-update")
def data_update(base:str="configs/base.yaml", override:str|None=None):
    """Download all stocks listed in configs/universe.yaml."""
    c = cfg(base, override)
    downloader = DataDownloader(
        data_dir=Path(c["data"]["storage_root"]) / "raw",
        start_date=c["data"]["start_date"],
    )
    result = downloader.update()
    # Keep the existing backtest workflow usable: it still reads its benchmark
    # from Storage, while stock history now comes from the v0.2 downloader.
    try:
        provider = AkshareProvider(c["data"]["universe_indexes"])
        benchmark = provider.benchmark_daily(
            c["data"]["benchmark_symbol"], c["data"]["start_date"], downloader.end_date
        )
        Storage(c["data"]["storage_root"]).write_benchmark(benchmark)
        print(f"OK {c['data']['benchmark_symbol']}")
    except Exception as exc:  # noqa: BLE001 - benchmark update is a network boundary
        print(f"FAILED {c['data']['benchmark_symbol']} {str(exc) or exc.__class__.__name__}")
    print(
        f"[bold]Data update complete.[/bold] "
        f"saved={len(result['saved'])}, failed={len(result['failed'])}"
    )


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


@app.command("factor-build")
def factor_build(base: str = "configs/base.yaml", override: str | None = None):
    """Build v0.5 factor parquet files for the configured universe."""
    c = cfg(base, override)
    root = Path(c["data"]["storage_root"])
    engine = FactorEngine(raw_dir=root / "raw", features_dir=root / "features")
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
def backtest(base:str="configs/base.yaml", override:str|None=None, model_path:str|None=None):
    c=cfg(base,override); s=Storage(c["data"]["storage_root"]); panel=s.read_features(); b=s.read_benchmark(); scored=score_panel(panel,c,model_path)
    eq,tr,m=run_backtest(scored,b,c); Path("reports").mkdir(exist_ok=True); eq.to_csv("reports/equity.csv"); tr.to_csv("reports/rebalances.csv",index=False); Path("reports/metrics.json").write_text(json.dumps(m,indent=2),encoding="utf-8"); print(json.dumps(m,indent=2))

@app.command("signal")
def signal(base:str="configs/base.yaml", override:str|None=None, model_path:str|None=None):
    c=cfg(base,override); s=Storage(c["data"]["storage_root"]); panel=s.read_features(); scored=score_panel(panel,c,model_path); d=scored.date.max(); top=scored[scored.date==d].sort_values("prediction",ascending=False).head(c["portfolio"]["top_n"]); print(top[["date","code","prediction"]].to_string(index=False))

@app.command("demo")
def demo(base:str="configs/base.yaml"):
    c=cfg(base,None); prices,b=synthetic_prices(); panel=build_features(prices,c["features"]["enabled"]); scored=score_panel(panel,c); _eq,_tr,m=run_backtest(scored,b,c); print("[bold green]Core pipeline works.[/bold green]"); print(json.dumps(m,indent=2))

if __name__=="__main__": app()
