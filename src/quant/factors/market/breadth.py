"""Market-wide breadth features derived from the locally available stock universe."""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_FEATURES = (
    "advance_decline_ratio", "percent_above_ma20", "percent_above_ma60", "percent_above_ma120",
    "new_high_20_ratio", "new_high_60_ratio", "new_low_20_ratio", "cross_sectional_dispersion",
    "market_turnover_change", "median_stock_return_5", "median_stock_return_20", "breadth_momentum",
    "volatility_regime",
)


class MarketBreadthEngine:
    """Calculate date-level features using only same-date and earlier stock observations."""

    def calculate(self, panel: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "code", "stock_return_1d", "momentum_5", "momentum_20", "ma_distance_20", "ma_distance_60", "ma_distance_120", "breakout_20", "breakout_60", "amount"}
        missing = required.difference(panel.columns)
        if missing:
            raise ValueError(f"market breadth panel missing: {', '.join(sorted(missing))}")
        frame = panel.loc[:, list(required)].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame = frame.sort_values(["date", "code"])
        grouped = frame.groupby("date", sort=True)
        records = pd.DataFrame(
            {
                "advance_decline_ratio": grouped["stock_return_1d"].apply(_advance_decline_ratio),
                "percent_above_ma20": grouped["ma_distance_20"].apply(lambda values: (values > 0).mean()),
                "percent_above_ma60": grouped["ma_distance_60"].apply(lambda values: (values > 0).mean()),
                "percent_above_ma120": grouped["ma_distance_120"].apply(lambda values: (values > 0).mean()),
                "new_high_20_ratio": grouped["breakout_20"].apply(lambda values: (values >= -1e-12).mean()),
                "new_high_60_ratio": grouped["breakout_60"].apply(lambda values: (values >= -1e-12).mean()),
                "new_low_20_ratio": grouped["breakout_20"].apply(lambda values: (values <= -0.2).mean()),
                "cross_sectional_dispersion": grouped["stock_return_1d"].std(ddof=0),
                "market_turnover": grouped["amount"].mean(),
                "median_stock_return_5": grouped["momentum_5"].median(),
                "median_stock_return_20": grouped["momentum_20"].median(),
            }
        ).reset_index()
        records["market_turnover_change"] = records["market_turnover"] / records["market_turnover"].rolling(20).mean() - 1.0
        records["breadth_momentum"] = records["percent_above_ma20"] - records["percent_above_ma20"].shift(20)
        market_volatility = records["median_stock_return_5"].rolling(20).std(ddof=0)
        reference = market_volatility.rolling(252, min_periods=60).median()
        records["volatility_regime"] = np.select(
            [market_volatility > reference * 1.25, market_volatility < reference * 0.75],
            ["high", "low"],
            default="normal",
        )
        return records.drop(columns="market_turnover").sort_values("date").reset_index(drop=True)

    def build_and_store(self, panel: pd.DataFrame, output_path: str = "data/features/market_features.parquet") -> pd.DataFrame:
        """Calculate and persist a local daily breadth history."""
        features = self.calculate(panel)
        features.to_parquet(output_path, index=False)
        return features


def _advance_decline_ratio(values: pd.Series) -> float:
    advances = int((values > 0).sum())
    declines = int((values < 0).sum())
    return advances / declines if declines else np.nan
