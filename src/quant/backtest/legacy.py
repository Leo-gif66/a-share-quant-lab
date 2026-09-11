"""v0.1 score-panel backtest retained for the demo and existing integrations."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..models.factory import create


def load_or_create_model(cfg: dict, model_path: str | None):
    model_name = cfg["model"]["name"]
    if model_path and Path(model_path).exists():
        return joblib.load(model_path)
    if model_name != "rule":
        latest = Path("artifacts/latest.json")
        if latest.exists():
            import json

            mapping = json.loads(latest.read_text(encoding="utf-8"))
            path = mapping.get(model_name)
            if path and Path(path).exists():
                return joblib.load(path)
        raise RuntimeError(f"no trained {model_name} model found; run `quant train` first")
    return create(model_name, cfg["model"].get("params", {}))


def score_panel(panel: pd.DataFrame, cfg: dict, model_path: str | None = None) -> pd.DataFrame:
    feats = cfg["features"]["enabled"]
    model = load_or_create_model(cfg, model_path)
    if cfg["model"]["name"] == "rule":
        outputs = []
        for _, group in panel.dropna(subset=feats).groupby("date"):
            scored = group.copy()
            scored["prediction"] = model.predict(scored[feats])
            outputs.append(scored)
        return pd.concat(outputs, ignore_index=True)
    output = panel.dropna(subset=feats).copy()
    output["prediction"] = model.predict(output[feats])
    return output


def run_backtest(scored: pd.DataFrame, benchmark: pd.DataFrame, cfg: dict):
    portfolio = cfg["portfolio"]
    costs = cfg["costs"]
    scored = scored.sort_values(["date", "code"]).copy()
    returns = scored.pivot(index="date", columns="code", values="ret_1").sort_index()
    benchmark = benchmark.copy().set_index("date").sort_index()
    benchmark["ret_1"] = benchmark["close"].pct_change()
    benchmark["ma"] = benchmark["close"].rolling(portfolio["market_ma_window"]).mean()
    dates = returns.index.intersection(benchmark.index).sort_values()
    first_week = pd.Series(dates, index=dates).groupby(dates.to_period("W-FRI")).first()
    rebalance_dates = set(pd.to_datetime(first_week.values))
    weights = pd.Series(0.0, index=returns.columns)
    nav = 1.0
    records = []
    trades = []
    for current_date in dates:
        nav *= 1 + float((weights * returns.loc[current_date].fillna(0)).sum())
        if current_date in rebalance_dates:
            group = scored[scored.date == current_date].sort_values("prediction", ascending=False).head(
                portfolio["top_n"]
            )
            risk_on = pd.notna(benchmark.at[current_date, "ma"]) and (
                benchmark.at[current_date, "close"] > benchmark.at[current_date, "ma"]
            )
            exposure = (
                portfolio["risk_on_exposure"] if risk_on else portfolio["risk_off_exposure"]
            )
            new_weights = pd.Series(0.0, index=weights.index)
            each = min(portfolio["max_single_weight"], exposure / max(len(group), 1))
            for code in group.code:
                if code in new_weights.index:
                    new_weights.loc[code] = each
            delta = new_weights - weights
            buy = float(delta.clip(lower=0).sum())
            sell = float((-delta.clip(upper=0)).sum())
            cost = (buy + sell) * (costs["commission"] + costs["slippage"]) + sell * costs[
                "stamp_duty_sell"
            ]
            nav *= 1 - cost
            weights = new_weights
            trades.append(
                {
                    "date": current_date,
                    "risk_on": risk_on,
                    "exposure": float(weights.sum()),
                    "cost": cost,
                    "codes": ",".join(group.code.astype(str)),
                }
            )
        records.append({"date": current_date, "strategy": nav})
    equity = pd.DataFrame(records).set_index("date")
    equity["benchmark"] = (1 + benchmark.reindex(equity.index)["ret_1"].fillna(0)).cumprod()
    equity /= equity.iloc[0]
    daily = equity.strategy.pct_change().dropna()
    years = max(len(daily) / 252, 1 / 252)
    annual_return = equity.strategy.iloc[-1] ** (1 / years) - 1
    volatility = daily.std(ddof=0) * np.sqrt(252)
    drawdown = (equity.strategy / equity.strategy.cummax() - 1).min()
    metrics = {
        "annual_return": float(annual_return),
        "annual_volatility": float(volatility),
        "sharpe_approx": float(annual_return / volatility if volatility else np.nan),
        "max_drawdown": float(drawdown),
    }
    return equity, pd.DataFrame(trades), metrics
