"""Validation rules for normalized daily stock-price parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataValidator:
    """Validate the daily-price contract before data enters the research pipeline."""

    REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")

    def __init__(self, min_rows: int = 100) -> None:
        self.min_rows = min_rows

    def validate_parquet(self, path: str | Path) -> list[str]:
        """Read a parquet file and return every validation failure found."""
        return self.validate(pd.read_parquet(path))

    def validate(self, data: pd.DataFrame) -> list[str]:
        """Return an empty list when ``data`` satisfies the daily-price contract."""
        errors: list[str] = []
        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in data.columns]
        if missing_columns:
            return [f"missing required columns: {', '.join(missing_columns)}"]

        if len(data) < self.min_rows:
            errors.append(f"only {len(data)} rows; at least {self.min_rows} trading days required")

        dates = pd.to_datetime(data["date"], errors="coerce")
        if dates.isna().any():
            errors.append("date contains null or invalid values")
        else:
            if not dates.is_monotonic_increasing:
                errors.append("dates are not in ascending order")
            if dates.duplicated().any():
                errors.append("dates contain duplicates")

        close = pd.to_numeric(data["close"], errors="coerce")
        if close.isna().any():
            errors.append("close contains null or non-numeric values")

        volume = pd.to_numeric(data["volume"], errors="coerce")
        if volume.isna().any():
            errors.append("volume contains null or non-numeric values")

        prices = data.loc[:, ["open", "high", "low"]].apply(pd.to_numeric, errors="coerce")
        if prices.isna().any().any():
            errors.append("open, high, or low contains null or non-numeric values")
        elif not close.isna().any():
            high_is_invalid = prices["high"] < pd.concat([prices["open"], close], axis=1).max(axis=1)
            low_is_invalid = prices["low"] > pd.concat([prices["open"], close], axis=1).min(axis=1)
            if high_is_invalid.any():
                errors.append("high is below open or close")
            if low_is_invalid.any():
                errors.append("low is above open or close")

        return errors
