from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .models.factory import create


def load_or_create_model(cfg: dict, model_path: str | None):
    model_name = cfg["model"]["name"]
    if model_path and Path(model_path).exists():
        return joblib.load(model_path)
    if model_name != "rule":
        latest = Path("artifacts/latest.json")
        if latest.exists():
            import json
            mapping = json.loads(latest.read_text(encoding="utf-8"))
            p = mapping.get(model_name)
            if p and Path(p).exists():
                return joblib.load(p)
        raise RuntimeError(f"no trained {model_name} model found; run `quant train` first")
    return create(model_name, cfg["model"].get("params", {}))


def score_panel(panel: pd.DataFrame, cfg: dict, model_path: str | None = None) -> pd.DataFrame:
    feats = cfg["features"]["enabled"]
    model = load_or_create_model(cfg, model_path)
    if cfg["model"]["name"] == "rule":
        out=[]
        for d,g in panel.dropna(subset=feats).groupby("date"):
            z=g.copy(); z["prediction"]=model.predict(z[feats]); out.append(z)
        return pd.concat(out, ignore_index=True)
    x=panel.dropna(subset=feats).copy(); x["prediction"]=model.predict(x[feats]); return x


def run_backtest(scored: pd.DataFrame, benchmark: pd.DataFrame, cfg: dict):
    p=cfg["portfolio"]; c=cfg["costs"]
    scored=scored.sort_values(["date","code"]).copy()
    returns=scored.pivot(index="date", columns="code", values="ret_1").sort_index()
    b=benchmark.copy().set_index("date").sort_index(); b["ret_1"]=b["close"].pct_change(); b["ma"]=b["close"].rolling(p["market_ma_window"]).mean()
    dates=returns.index.intersection(b.index).sort_values()
    first_week=pd.Series(dates,index=dates).groupby(dates.to_period("W-FRI")).first()
    rebal=set(pd.to_datetime(first_week.values))
    w=pd.Series(0.0,index=returns.columns); nav=1.0; rec=[]; trades=[]
    for d in dates:
        nav*=1+float((w*returns.loc[d].fillna(0)).sum())
        if d in rebal:
            g=scored[scored.date==d].sort_values("prediction",ascending=False).head(p["top_n"])
            risk_on=pd.notna(b.at[d,"ma"]) and b.at[d,"close"]>b.at[d,"ma"]
            expo=p["risk_on_exposure"] if risk_on else p["risk_off_exposure"]
            nw=pd.Series(0.0,index=w.index)
            each=min(p["max_single_weight"], expo/max(len(g),1))
            for code in g.code:
                if code in nw.index: nw.loc[code]=each
            delta=nw-w; buy=float(delta.clip(lower=0).sum()); sell=float((-delta.clip(upper=0)).sum())
            cost=(buy+sell)*(c["commission"]+c["slippage"])+sell*c["stamp_duty_sell"]
            nav*=1-cost; w=nw
            trades.append({"date":d,"risk_on":risk_on,"exposure":float(w.sum()),"cost":cost,"codes":','.join(g.code.astype(str))})
        rec.append({"date":d,"strategy":nav})
    eq=pd.DataFrame(rec).set_index("date"); eq["benchmark"]=(1+b.reindex(eq.index)["ret_1"].fillna(0)).cumprod(); eq/=eq.iloc[0]
    daily=eq.strategy.pct_change().dropna(); years=max(len(daily)/252,1/252)
    ann=eq.strategy.iloc[-1]**(1/years)-1; vol=daily.std(ddof=0)*np.sqrt(252); dd=(eq.strategy/eq.strategy.cummax()-1).min()
    metrics={"annual_return":float(ann),"annual_volatility":float(vol),"sharpe_approx":float(ann/vol if vol else np.nan),"max_drawdown":float(dd)}
    return eq,pd.DataFrame(trades),metrics
