"""Explicit assertions for the temporal contracts used by V5 research."""

from __future__ import annotations

import pandas as pd


def assert_feature_available(
    feature_timestamps: pd.Series, decision_timestamps: pd.Series
) -> None:
    """Assert every feature was observable on or before its decision timestamp."""
    _assert_aligned(feature_timestamps, decision_timestamps, "feature timestamp", "decision timestamp", strict=False)


def assert_fundamental_available(
    announcement_dates: pd.Series, signal_dates: pd.Series
) -> None:
    """Assert fundamentals are not used before public announcement."""
    _assert_aligned(announcement_dates, signal_dates, "fundamental announcement date", "signal date", strict=False)


def assert_label_after_signal(signal_dates: pd.Series, label_end_dates: pd.Series) -> None:
    """Assert every forward-return label matures strictly after its signal."""
    _assert_aligned(signal_dates, label_end_dates, "signal date", "label end date", strict=True)


def assert_training_before_test(training_dates: pd.Series, test_dates: pd.Series) -> None:
    """Assert a training cutoff is strictly earlier than a test period."""
    train = pd.to_datetime(training_dates, errors="raise").dropna()
    test = pd.to_datetime(test_dates, errors="raise").dropna()
    if train.empty or test.empty:
        raise ValueError("training and test timestamps must both be non-empty")
    if train.max() >= test.min():
        raise AssertionError("training cutoff must be strictly before the test period")


def _assert_aligned(
    earlier: pd.Series, later: pd.Series, earlier_name: str, later_name: str, *, strict: bool
) -> None:
    if len(earlier) != len(later):
        raise ValueError(f"{earlier_name} and {later_name} must have equal length")
    before = pd.to_datetime(earlier, errors="raise")
    after = pd.to_datetime(later, errors="raise")
    valid = before.notna() & after.notna()
    invalid = before[valid] >= after[valid] if strict else before[valid] > after[valid]
    if invalid.any():
        relation = "strictly before" if strict else "on or before"
        raise AssertionError(f"{earlier_name} must be {relation} {later_name}")
