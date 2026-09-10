from __future__ import annotations
import pandas as pd
from .factors.registry import get


def build_features(price_frames: dict[str, pd.DataFrame], enabled: list[str]) -> pd.DataFrame:
    frames = []
    for code, raw in price_frames.items():
        x = raw.copy().sort_values("date")
        x["code"] = str(code).zfill(6)
        x["ret_1"] = x["close"].pct_change()
        for name in enabled:
            x[name] = get(name).calculate(x)
        frames.append(x[["date", "code", "close", "ret_1"] + enabled])
    if not frames:
        raise RuntimeError("no stock price data found")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "code"])


def add_label(panel: pd.DataFrame, horizon: int, label_type: str) -> pd.DataFrame:
    x = panel.sort_values(["code", "date"]).copy()
    x["future_return"] = x.groupby("code")["close"].shift(-horizon) / x["close"] - 1.0
    if label_type == "cross_sectional_excess_return":
        market_mean = x.groupby("date")["future_return"].transform("mean")
        x["label"] = x["future_return"] - market_mean
    elif label_type == "future_return":
        x["label"] = x["future_return"]
    else:
        raise ValueError(f"unknown label type: {label_type}")
    return x
