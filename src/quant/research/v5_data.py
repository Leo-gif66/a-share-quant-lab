"""Full-history V5 alpha panel reconstruction from local price data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.fundamental import FUNDAMENTAL_FIELDS, FundamentalStore
from ..data.universe import Universe
from ..factors.alpha import ALPHA_FACTORS, AlphaFactorEngine
from ..factors.market import MarketBreadthEngine
from .labels import add_v5_forward_labels
from .leakage import assert_feature_available
from .v5_preprocessing import V5FactorPreprocessor


class V5ResearchDataBuilder:
    """Build a full-history, date-security research panel without V4 score truncation."""

    def __init__(
        self,
        *,
        raw_dir: str | Path = "data/raw",
        universe_path: str | Path = "configs/universe_large.yaml",
        benchmark_path: str | Path = "data/raw/000300.parquet",
        fundamental_store: FundamentalStore | None = None,
        start_date: str | None = "2018-01-01",
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.universe = Universe(universe_path)
        self.benchmark_path = Path(benchmark_path)
        self.fundamental_store = fundamental_store or FundamentalStore()
        self.start_date = pd.Timestamp(start_date) if start_date else None

    def build(self, output_path: str | Path | None = "data/features/v5_alpha_panel.parquet") -> pd.DataFrame:
        """Calculate V5 raw and processed factors for every locally available universe member."""
        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"V5 requires CSI300 benchmark history: {self.benchmark_path}")
        benchmark = pd.read_parquet(self.benchmark_path)
        engine = AlphaFactorEngine()
        metadata = {str(stock["code"]).zfill(6): stock for stock in self.universe.stocks()}
        frames: list[pd.DataFrame] = []
        for code, stock in metadata.items():
            path = self.raw_dir / f"{code}.parquet"
            if not path.exists():
                continue
            factor_frame = engine.calculate(pd.read_parquet(path), benchmark)
            factor_frame["code"] = code
            factor_frame["sector"] = str(stock.get("sector") or "Unknown")
            factor_frame["industry"] = str(stock.get("industry") or stock.get("sector") or "Unknown")
            if self.start_date is not None:
                factor_frame = factor_frame.loc[factor_frame["date"] >= self.start_date]
            frames.append(factor_frame)
        if not frames:
            raise RuntimeError("no local histories match the configured V5 universe")
        panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
        panel["industry_relative_momentum"] = panel["momentum_20"] - panel.groupby(
            ["date", "industry"], sort=False
        )["momentum_20"].transform("mean")
        breadth = MarketBreadthEngine().build_and_store(panel)
        panel = panel.merge(breadth, on="date", how="left", validate="many_to_one")
        panel = self._add_market_regime(panel, benchmark)
        panel = add_v5_forward_labels(panel, benchmark)
        panel = self.fundamental_store.asof_join(panel)
        active_fundamentals = [name for name in FUNDAMENTAL_FIELDS if panel[name].notna().any()]
        factors = [*ALPHA_FACTORS, *active_fundamentals]
        panel = V5FactorPreprocessor().process(panel, factors)
        assert_feature_available(panel["date"], panel["date"])
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            panel.to_parquet(target, index=False)
        return panel

    @staticmethod
    def _add_market_regime(panel: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        market = benchmark.loc[:, ["date", "close"]].copy()
        market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
        market["close"] = pd.to_numeric(market["close"], errors="coerce")
        average = market["close"].rolling(60).mean()
        market["market_regime"] = "sideways"
        market.loc[market["close"] > average * 1.02, "market_regime"] = "bull"
        market.loc[market["close"] < average * 0.98, "market_regime"] = "bear"
        return panel.merge(market.loc[:, ["date", "market_regime"]], on="date", how="left", validate="many_to_one")
