from __future__ import annotations
from pathlib import Path
import json
import typer
from rich import print
from .config import load_config
from .data.akshare_provider import AkshareProvider
from .data.downloader import DataDownloader
from .data.storage import Storage
from .features import build_features, add_label
from .training import train_model
from .backtest import score_panel, run_backtest
from .demo import synthetic_prices
from .factors.registry import names as factor_names
from .models.factory import names as model_names

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
    except Exception as exc:
        print(f"FAILED {c['data']['benchmark_symbol']} {str(exc) or exc.__class__.__name__}")
    print(
        f"[bold]Data update complete.[/bold] "
        f"saved={len(result['saved'])}, failed={len(result['failed'])}"
    )

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
    c=cfg(base,None); prices,b=synthetic_prices(); panel=build_features(prices,c["features"]["enabled"]); scored=score_panel(panel,c); eq,tr,m=run_backtest(scored,b,c); print("[bold green]Core pipeline works.[/bold green]"); print(json.dumps(m,indent=2))

if __name__=="__main__": app()
