from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(n_stocks=30, n_days=600, seed=42):
    rng=np.random.default_rng(seed); dates=pd.bdate_range("2022-01-03",periods=n_days); out={}
    for i in range(n_stocks):
        code=f"{i+1:06d}"; drift=rng.normal(0.00025,0.00015); ret=drift+rng.normal(0,0.018,n_days)
        close=20*np.cumprod(1+ret); volume=rng.integers(1_000_000,12_000_000,n_days)
        out[code]=pd.DataFrame({"date":dates,"code":code,"close":close,"open":close*(1+rng.normal(0,0.003,n_days)),"high":close*1.01,"low":close*0.99,"volume":volume,"amount":volume*close,"turnover":rng.uniform(0.3,5.0,n_days)})
    benchmark=pd.DataFrame({"date":dates,"close":100*np.cumprod(1+rng.normal(0.0002,0.01,n_days))})
    return out,benchmark
