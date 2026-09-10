from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.data.validator import DataValidator


def valid_daily_data(days: int = 100) -> pd.DataFrame:
    close = pd.Series(range(10, 10 + days), dtype=float)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=days),
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000.0,
            "amount": close * 1_000,
            "turnover": 1.0,
        }
    )


def test_validator_accepts_valid_parquet(tmp_path: Path):
    path = tmp_path / "000001.parquet"
    valid_daily_data().to_parquet(path, index=False)

    assert DataValidator().validate_parquet(path) == []


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda frame: frame.assign(date=frame["date"].iloc[::-1].to_numpy()),
            "dates are not in ascending order",
        ),
        (
            lambda frame: frame.assign(date=[frame["date"].iloc[0]] * len(frame)),
            "dates contain duplicates",
        ),
        (
            lambda frame: frame.assign(high=frame["close"] - 1),
            "high is below open or close",
        ),
        (
            lambda frame: frame.assign(low=frame["close"] + 1),
            "low is above open or close",
        ),
        (lambda frame: frame.assign(close=np.nan), "close contains null or non-numeric values"),
        (lambda frame: frame.assign(volume=np.nan), "volume contains null or non-numeric values"),
        (lambda frame: frame.iloc[:99].copy(), "only 99 rows; at least 100 trading days required"),
    ],
)
def test_validator_rejects_invalid_daily_data(mutate, expected_error):
    errors = DataValidator().validate(mutate(valid_daily_data()))

    assert expected_error in errors
