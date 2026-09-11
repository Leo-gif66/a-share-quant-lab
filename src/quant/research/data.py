"""Build factor-research panels from the project's raw and feature stores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class FactorResearchDataBuilder:
    """Join per-stock factors to forward returns calculated from raw closes."""

    LABEL_COLUMN = "future_return_20d"

    def __init__(
        self,
        features_dir: str | Path = "data/features",
        raw_dir: str | Path = "data/raw",
        horizon: int = 20,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        self.features_dir = Path(features_dir)
        self.raw_dir = Path(raw_dir)
        self.horizon = horizon

    def build(self) -> pd.DataFrame:
        """Return factor rows labelled with the next 20 trading-day return.

        Feature values remain aligned to their original trading date.  The
        label uses the raw close at the same date and the close exactly
        ``horizon`` observations later, so exchange holidays do not distort the
        intended 20-trading-day holding period.
        """
        paths = sorted(
            path
            for path in self.features_dir.glob("*.parquet")
            if path.stem.isdigit() and len(path.stem) == 6
        )
        if not paths:
            raise RuntimeError(f"no per-stock feature files found in {self.features_dir}")

        frames: list[pd.DataFrame] = []
        for feature_path in paths:
            raw_path = self.raw_dir / feature_path.name
            if not raw_path.exists():
                raise FileNotFoundError(
                    f"raw price file for {feature_path.stem} not found: {raw_path}"
                )
            features = pd.read_parquet(feature_path)
            raw = pd.read_parquet(raw_path)
            frames.append(self._label_stock(feature_path.stem, features, raw))

        return pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(
            drop=True
        )

    def _label_stock(self, code: str, features: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        if "date" not in features:
            raise ValueError(f"{code} feature file requires a date column")
        if not {"date", "close"}.issubset(raw.columns):
            raise ValueError(f"{code} raw price file requires date and close columns")

        feature_frame = features.copy()
        feature_frame["date"] = pd.to_datetime(feature_frame["date"], errors="raise")
        feature_frame = feature_frame.drop_duplicates("date", keep="last")

        prices = raw.loc[:, ["date", "close"]].copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="raise")
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
        prices = prices.sort_values("date").drop_duplicates("date", keep="last")
        prices[self.LABEL_COLUMN] = prices["close"].shift(-self.horizon) / prices["close"] - 1.0

        labelled = feature_frame.merge(
            prices.loc[:, ["date", self.LABEL_COLUMN]], on="date", how="left", validate="one_to_one"
        )
        labelled["code"] = str(code).zfill(6)
        return labelled
