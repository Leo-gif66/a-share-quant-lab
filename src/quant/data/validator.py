"""Validation rules for normalized daily stock-price parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataValidator:
    """Validate the daily-price contract before data enters the research pipeline."""

    REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")

    def __init__(
        self,
        min_rows: int = 501,
        max_missing_rate: float = 0.05,
        suspension_days: int = 5,
    ) -> None:
        if min_rows < 1:
            raise ValueError("min_rows must be positive")
        if not 0 <= max_missing_rate <= 1:
            raise ValueError("max_missing_rate must be between zero and one")
        if suspension_days < 1:
            raise ValueError("suspension_days must be positive")
        self.min_rows = min_rows
        self.max_missing_rate = max_missing_rate
        self.suspension_days = suspension_days

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

        missing_rates = self._missing_rates(data)
        for column, rate in missing_rates.items():
            if rate > self.max_missing_rate:
                errors.append(
                    f"{column} missing rate {rate:.2%} exceeds {self.max_missing_rate:.2%}"
                )

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
        suspended_streak = self._longest_suspension_streak(volume)
        if suspended_streak >= self.suspension_days:
            errors.append(f"suspended trading detected: {suspended_streak} consecutive zero-volume days")

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

    def _missing_rates(self, data: pd.DataFrame) -> dict[str, float]:
        rates = {"date": float(pd.to_datetime(data["date"], errors="coerce").isna().mean())}
        for column in ("open", "high", "low", "close", "volume"):
            rates[column] = float(pd.to_numeric(data[column], errors="coerce").isna().mean())
        return rates

    @staticmethod
    def _longest_suspension_streak(volume: pd.Series) -> int:
        longest = current = 0
        for value in volume:
            if pd.notna(value) and value <= 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest


class ResearchDataValidator:
    """Summarize whether a universe has enough clean raw and feature history.

    This is intentionally a read-only check. It uses the same price-quality
    rules as ``data-check`` and never treats a file's mere presence as a valid
    research history.
    """

    def __init__(
        self,
        data_validator: DataValidator | None = None,
        minimum_coverage: float = 0.95,
    ) -> None:
        if not 0 < minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be between zero and one")
        self.data_validator = data_validator or DataValidator()
        self.minimum_coverage = minimum_coverage

    def summarize(
        self,
        stocks: list[dict],
        raw_dir: str | Path = "data/raw",
        features_dir: str | Path = "data/features",
    ) -> dict[str, int | float | bool]:
        """Return coverage and readiness metrics for the supplied universe."""
        raw_root = Path(raw_dir)
        feature_root = Path(features_dir)
        codes = list(dict.fromkeys(str(stock["code"]).zfill(6) for stock in stocks))
        available = valid = feature_stocks = total_days = 0

        for code in codes:
            raw_path = raw_root / f"{code}.parquet"
            if not raw_path.exists():
                continue
            frame: pd.DataFrame | None
            try:
                frame = pd.read_parquet(raw_path)
            except Exception:  # noqa: BLE001 - a corrupt file is unavailable for research
                frame = None
            if frame is None:
                continue
            available += 1
            total_days += len(frame)
            if not self.data_validator.validate(frame):
                valid += 1
            feature_path = feature_root / f"{code}.parquet"
            if feature_path.exists():
                feature_stocks += 1

        stock_count = len(codes)
        valid_ratio = valid / stock_count if stock_count else 0.0
        feature_ratio = feature_stocks / stock_count if stock_count else 0.0
        return {
            "stocks": stock_count,
            "available_histories": available,
            "valid": valid,
            "feature_stocks": feature_stocks,
            "average_days": total_days / available if available else 0.0,
            "missing_ratio": (stock_count - available) / stock_count if stock_count else 0.0,
            "valid_ratio": valid_ratio,
            "feature_ratio": feature_ratio,
            "research_ready": bool(
                stock_count
                and valid_ratio >= self.minimum_coverage
                and feature_ratio >= self.minimum_coverage
            ),
        }
