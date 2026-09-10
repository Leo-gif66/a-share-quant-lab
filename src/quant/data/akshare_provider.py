from __future__ import annotations
from datetime import date
import pandas as pd
from .provider import DataProvider


class AkshareProvider(DataProvider):
    def __init__(self, index_codes: list[str]):
        import akshare as ak
        self.ak = ak
        self.index_codes = index_codes

    def universe(self) -> pd.DataFrame:
        frames = []
        for idx in self.index_codes:
            try:
                raw = self.ak.index_stock_cons_csindex(symbol=idx)
                x = raw[["成分券代码", "成分券名称"]].copy()
                x.columns = ["code", "name"]
            except Exception:
                raw = self.ak.index_stock_cons(symbol=idx)
                x = raw[["品种代码", "品种名称"]].copy()
                x.columns = ["code", "name"]
            x["index_code"] = idx
            frames.append(x)
        out = pd.concat(frames, ignore_index=True)
        out["code"] = out["code"].astype(str).str.zfill(6)
        return out.drop_duplicates("code").reset_index(drop=True)

    def stock_daily(self, code: str, start: str, end: str | None = None) -> pd.DataFrame:
        end = end or date.today().strftime("%Y%m%d")
        raw = self.ak.stock_zh_a_hist(
            symbol=str(code).zfill(6), period="daily", start_date=start,
            end_date=end, adjust="qfq"
        )
        rename = {
            "日期": "date", "股票代码": "code", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
            "换手率": "turnover", "涨跌幅": "pct_change"
        }
        x = raw.rename(columns=rename)
        keep = [c for c in rename.values() if c in x.columns]
        x = x[keep].copy()
        x["date"] = pd.to_datetime(x["date"])
        if "code" not in x:
            x["code"] = str(code).zfill(6)
        x["code"] = x["code"].astype(str).str.zfill(6)
        for c in ["open", "close", "high", "low", "volume", "amount", "turnover", "pct_change"]:
            if c in x:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        return x.sort_values("date").drop_duplicates("date")

    def benchmark_daily(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        end = end or date.today().strftime("%Y%m%d")
        x = self.ak.stock_zh_index_daily_em(symbol=symbol, start_date=start, end_date=end)
        x["date"] = pd.to_datetime(x["date"])
        return x.sort_values("date")
