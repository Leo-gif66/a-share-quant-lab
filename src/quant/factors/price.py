from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Factor
from .registry import register


class Momentum(Factor):
    def __init__(self, days: int): self.days, self.name = days, f"mom_{days}"
    def calculate(self, df): return df["close"].pct_change(self.days)


class Trend(Factor):
    def __init__(self, days: int): self.days, self.name = days, f"trend_{days}"
    def calculate(self, df):
        ma = df["close"].rolling(self.days).mean()
        return df["close"] / ma - 1.0


class Volatility(Factor):
    def __init__(self, days: int): self.days, self.name = days, f"vol_{days}"
    def calculate(self, df): return df["close"].pct_change().rolling(self.days).std() * np.sqrt(252)


class Drawdown(Factor):
    def __init__(self, days: int): self.days, self.name = days, f"drawdown_{days}"
    def calculate(self, df): return 1.0 - df["close"] / df["close"].rolling(self.days).max()


class VolumeRatio(Factor):
    def __init__(self, days: int): self.days, self.name = days, f"volume_ratio_{days}"
    def calculate(self, df): return df["volume"] / df["volume"].rolling(self.days).mean()


class Turnover(Factor):
    def __init__(self, days: int): self.days, self.name = days, f"turnover_{days}"
    def calculate(self, df): return df["turnover"].rolling(self.days).mean()


class Liquidity(Factor):
    def __init__(self, days: int): self.days, self.name = days, f"liquidity_{days}"
    def calculate(self, df): return np.log1p(df["amount"].rolling(self.days).mean())


for _f in [Momentum(5), Momentum(20), Momentum(60), Trend(20), Trend(60),
           Volatility(20), Drawdown(60), VolumeRatio(20), Turnover(20), Liquidity(20)]:
    register(_f)
