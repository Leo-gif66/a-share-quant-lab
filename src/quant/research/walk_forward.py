"""Rolling, chronological walk-forward validation utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    train_years: int = 3
    validation_months: int = 6
    test_months: int = 6
    step_months: int = 6

    def __post_init__(self) -> None:
        if min(self.train_years, self.validation_months, self.test_months, self.step_months) < 1:
            raise ValueError("walk-forward windows must be positive")


@dataclass(frozen=True)
class WalkForwardPeriod:
    train_start: pd.Timestamp
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardValidator:
    """Run train/validate/test cycles without random shuffling or leakage."""

    OUTPUT_COLUMNS = ("period", "return", "benchmark", "alpha", "sharpe", "drawdown")

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def periods(self, dates: pd.Series | pd.DatetimeIndex) -> list[WalkForwardPeriod]:
        index = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates), errors="raise").drop_duplicates().sort_values())
        if index.empty:
            raise ValueError("walk-forward validation requires at least one date")
        first, last = index[0], index[-1]
        cursor = first
        periods: list[WalkForwardPeriod] = []
        while True:
            validation_start = cursor + pd.DateOffset(years=self.config.train_years)
            test_start = validation_start + pd.DateOffset(months=self.config.validation_months)
            test_end_exclusive = test_start + pd.DateOffset(months=self.config.test_months)
            test_dates = index[(index >= test_start) & (index < test_end_exclusive)]
            if test_dates.empty:
                break
            train_dates = index[(index >= cursor) & (index < validation_start)]
            validation_dates = index[(index >= validation_start) & (index < test_start)]
            if train_dates.empty or validation_dates.empty:
                break
            periods.append(
                WalkForwardPeriod(
                    train_start=train_dates[0],
                    validation_start=validation_dates[0],
                    test_start=test_dates[0],
                    test_end=test_dates[-1],
                )
            )
            cursor = cursor + pd.DateOffset(months=self.config.step_months)
            if cursor > last:
                break
        return periods

    def run(
        self,
        data: pd.DataFrame,
        fit: Callable[[pd.DataFrame, pd.DataFrame], object],
        predict: Callable[[object, pd.DataFrame], pd.DataFrame],
        evaluate: Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, float]],
    ) -> pd.DataFrame:
        if "date" not in data:
            raise ValueError("walk-forward data requires a date column")
        frame = data.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        results: list[dict[str, object]] = []
        for period in self.periods(frame["date"]):
            train = frame.loc[(frame["date"] >= period.train_start) & (frame["date"] < period.validation_start)].copy()
            validation = frame.loc[
                (frame["date"] >= period.validation_start) & (frame["date"] < period.test_start)
            ].copy()
            test = frame.loc[(frame["date"] >= period.test_start) & (frame["date"] <= period.test_end)].copy()
            _assert_chronological(train, validation, test)
            model = fit(train, validation)
            predictions = predict(model, test.copy())
            metrics = evaluate(predictions, test.copy())
            required = {"return", "benchmark", "alpha", "sharpe", "drawdown"}
            missing = required.difference(metrics)
            if missing:
                raise ValueError(f"walk-forward evaluator missing metrics: {', '.join(sorted(missing))}")
            results.append(
                {
                    "period": f"{period.test_start.date()} to {period.test_end.date()}",
                    **{name: float(metrics[name]) for name in required},
                }
            )
        return pd.DataFrame(results, columns=self.OUTPUT_COLUMNS)


def _assert_chronological(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    if train.empty or validation.empty or test.empty:
        raise ValueError("every walk-forward partition must be non-empty")
    if not (train["date"].max() < validation["date"].min() < test["date"].min()):
        raise RuntimeError("walk-forward partitions overlap or are not chronological")
