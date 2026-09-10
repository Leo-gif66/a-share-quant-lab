from __future__ import annotations
import numpy as np
import pandas as pd


def winsorized_zscore(s: pd.Series, q: float = 0.025) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=s.index)
    lo, hi = valid.quantile([q, 1 - q])
    x = s.clip(lo, hi)
    std = x.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (x - x.mean()) / std


def rank_ic(frame: pd.DataFrame, pred_col: str, label_col: str) -> float:
    vals = []
    for _, g in frame.groupby("date"):
        z = g[[pred_col, label_col]].dropna()
        if len(z) >= 5:
            c = z[pred_col].corr(z[label_col], method="spearman")
            if pd.notna(c):
                vals.append(c)
    return float(np.mean(vals)) if vals else float("nan")
