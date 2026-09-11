"""Per-stock factor generation and cross-sectional ranking."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd

from ..data.universe import Universe


class FactorEngine:
    """Build v0.5 factors from raw daily-price parquet files."""

    FACTOR_COLUMNS: ClassVar[tuple[str, ...]] = (
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "trend_20",
        "trend_60",
        "volatility_20",
        "liquidity_20",
        "volume_ratio_20",
        "turnover_20",
        "drawdown_60",
    )
    _REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ("date", "close", "amount", "volume", "turnover")
    _SCORE_DIRECTIONS: ClassVar[dict[str, int]] = {
        "momentum_5": 1,
        "momentum_20": 1,
        "momentum_60": 1,
        "trend_20": 1,
        "trend_60": 1,
        "volatility_20": -1,
        "liquidity_20": 1,
        "volume_ratio_20": 1,
        "turnover_20": 1,
        "drawdown_60": 1,
    }

    def __init__(
        self,
        universe: Universe | None = None,
        raw_dir: str | Path = "data/raw",
        features_dir: str | Path = "data/features",
    ) -> None:
        self.universe = universe or Universe()
        self.raw_dir = Path(raw_dir)
        self.features_dir = Path(features_dir)

    def build_all(self) -> dict[str, Path]:
        """Build and persist factors for every stock configured in the universe."""
        return {
            self._code(stock["code"]): self.build_stock(self._code(stock["code"]))
            for stock in self._universe_stocks()
        }

    def build_stock(self, code: str) -> Path:
        """Build factors for one stock and save ``data/features/<code>.parquet``."""
        normalized_code = self._code(code)
        source = self.raw_dir / f"{normalized_code}.parquet"
        data = pd.read_parquet(source)
        features = self.calculate(data)

        self.features_dir.mkdir(parents=True, exist_ok=True)
        target = self.features_dir / f"{normalized_code}.parquet"
        features.to_parquet(target, index=False)
        return target

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return raw daily data with the v0.5 factor columns appended."""
        missing = [column for column in self._REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"missing required price columns: {', '.join(missing)}")

        result = data.copy()
        result["date"] = pd.to_datetime(result["date"], errors="raise")
        result = result.sort_values("date").reset_index(drop=True)

        close = pd.to_numeric(result["close"], errors="coerce")
        volume = pd.to_numeric(result["volume"], errors="coerce")
        amount = pd.to_numeric(result["amount"], errors="coerce")
        turnover = pd.to_numeric(result["turnover"], errors="coerce")
        daily_return = close.pct_change()

        result["momentum_5"] = close.pct_change(5)
        result["momentum_20"] = close.pct_change(20)
        result["momentum_60"] = close.pct_change(60)
        result["trend_20"] = close / close.rolling(20).mean() - 1
        result["trend_60"] = close / close.rolling(60).mean() - 1
        result["volatility_20"] = daily_return.rolling(20).std()
        result["liquidity_20"] = amount.rolling(20).mean()
        result["volume_ratio_20"] = volume / volume.rolling(20).mean()
        result["turnover_20"] = turnover.rolling(20).mean()
        result["drawdown_60"] = close / close.rolling(60).max() - 1
        return result

    def rank_latest(self) -> pd.DataFrame:
        """Rank stocks by mean cross-sectional percentile score on the latest date."""
        frames: list[pd.DataFrame] = []
        for stock in self._universe_stocks():
            code = self._code(stock["code"])
            path = self.features_dir / f"{code}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            missing = [column for column in ("date", *self.FACTOR_COLUMNS) if column not in frame]
            if missing:
                raise ValueError(f"{code} feature file missing columns: {', '.join(missing)}")
            frame = frame.loc[:, ["date", *self.FACTOR_COLUMNS]].copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            frame["code"] = code
            frame["name"] = stock["name"]
            frames.append(frame)

        if not frames:
            raise RuntimeError("no feature parquet files found for the configured universe")

        panel = pd.concat(frames, ignore_index=True)
        latest_date = panel["date"].max()
        latest = panel.loc[panel["date"] == latest_date].dropna(subset=self.FACTOR_COLUMNS).copy()
        if latest.empty:
            raise RuntimeError(f"no complete factor rows available for {latest_date.date()}")

        rank_columns = []
        for factor, direction in self._SCORE_DIRECTIONS.items():
            rank_column = f"_{factor}_rank"
            latest[rank_column] = latest[factor].rank(
                method="average", pct=True, ascending=direction > 0
            )
            rank_columns.append(rank_column)
        latest["score"] = latest[rank_columns].mean(axis=1)

        return latest.loc[:, ["code", "name", "score"]].sort_values(
            ["score", "code"], ascending=[False, True]
        ).reset_index(drop=True)

    @staticmethod
    def _code(value: str) -> str:
        return str(value).zfill(6)

    def _universe_stocks(self) -> list[dict[str, str | None]]:
        """Read the normalized v0.6 universe metadata needed by this engine."""
        stocks = self.universe.stocks()
        required = ("code", "name", "market", "sector")
        for stock in stocks:
            missing = [field for field in required if field not in stock]
            if missing:
                raise ValueError(f"universe stock missing fields: {', '.join(missing)}")
        return stocks
