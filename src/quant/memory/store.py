"""Durable, schema-checked memory for completed and pending trade decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRADE_COLUMNS = (
    "date",
    "symbol",
    "strategy",
    "model_score",
    "factor_scores",
    "industry",
    "market_regime",
    "entry_price",
    "exit_price",
    "future_return",
    "prediction_error",
    "exit_date",
    "weight",
    "market_volatility",
    "decision_reason",
)


@dataclass(frozen=True)
class SignalSnapshot:
    """Point-in-time factor signal evidence behind a trade decision."""

    date: pd.Timestamp
    symbol: str
    factor_scores: Mapping[str, float]
    industry: str = "Unknown"
    market_regime: str = "sideways"


@dataclass(frozen=True)
class ModelPredictionSnapshot:
    """Point-in-time model output, kept separate from its realized outcome."""

    date: pd.Timestamp
    symbol: str
    strategy: str
    model_score: float


@dataclass(frozen=True)
class TradeDecisionSnapshot:
    """Complete decision record, including optional later-known execution data."""

    date: pd.Timestamp
    symbol: str
    strategy: str
    model_score: float
    factor_scores: Mapping[str, float] = field(default_factory=dict)
    industry: str = "Unknown"
    market_regime: str = "sideways"
    entry_price: float | None = None
    exit_price: float | None = None
    exit_date: pd.Timestamp | None = None
    weight: float | None = None
    market_volatility: float | None = None
    decision_reason: str = "Not recorded"

    @classmethod
    def from_snapshots(
        cls,
        signal: SignalSnapshot,
        prediction: ModelPredictionSnapshot,
        **execution: Any,
    ) -> TradeDecisionSnapshot:
        if pd.Timestamp(signal.date) != pd.Timestamp(prediction.date) or signal.symbol != prediction.symbol:
            raise ValueError("signal and prediction snapshots must describe the same decision")
        return cls(
            date=signal.date,
            symbol=signal.symbol,
            strategy=prediction.strategy,
            model_score=prediction.model_score,
            factor_scores=signal.factor_scores,
            industry=signal.industry,
            market_regime=signal.market_regime,
            **execution,
        )

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["date"] = pd.Timestamp(self.date)
        record["symbol"] = str(self.symbol).zfill(6)
        record["factor_scores"] = _encode_factor_scores(self.factor_scores)
        record["exit_date"] = pd.Timestamp(self.exit_date) if self.exit_date is not None else pd.NaT
        record["future_return"] = np.nan
        record["prediction_error"] = np.nan
        return record


class TradeMemoryStore:
    """Append-safe parquet storage with idempotent decision upserts.

    Factor scores are stored as canonical JSON because factor sets can evolve
    between strategies.  :meth:`load` decodes that JSON back to dictionaries
    for research consumers.
    """

    def __init__(self, path: str | Path = "data/memory/trades.parquet") -> None:
        self.path = Path(path)

    def load(self, decode_factor_scores: bool = True) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=TRADE_COLUMNS)
        frame = pd.read_parquet(self.path)
        return self._normalise(frame, decode_factor_scores=decode_factor_scores)

    def upsert(self, snapshots: TradeDecisionSnapshot | list[TradeDecisionSnapshot]) -> pd.DataFrame:
        items = [snapshots] if isinstance(snapshots, TradeDecisionSnapshot) else list(snapshots)
        incoming = pd.DataFrame([snapshot.as_record() for snapshot in items], columns=TRADE_COLUMNS)
        existing = self.load(decode_factor_scores=False)
        merged = pd.concat([existing, incoming], ignore_index=True)
        merged = self._normalise(merged, decode_factor_scores=False)
        merged = merged.drop_duplicates(["date", "symbol", "strategy"], keep="last")
        self._write(merged)
        return self._normalise(merged, decode_factor_scores=True)

    def realize(self, prices: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """Fill matured returns from stored exits or a supplied price history.

        ``prices`` accepts ``date``, ``symbol`` (or ``code``), and ``close``.
        It is useful when live decisions were recorded before their exit price
        was known.  Existing exit prices are never overwritten.
        """
        frame = self.load(decode_factor_scores=False)
        if frame.empty:
            return self._normalise(frame, decode_factor_scores=True)
        if prices is not None:
            frame = self._fill_exit_prices(frame, prices, as_of)
        eligible = frame["entry_price"].gt(0) & frame["exit_price"].gt(0)
        if as_of is not None:
            eligible &= frame["exit_date"].notna() & (frame["exit_date"] <= pd.Timestamp(as_of))
        frame.loc[eligible, "future_return"] = (
            frame.loc[eligible, "exit_price"] / frame.loc[eligible, "entry_price"] - 1.0
        )
        frame.loc[eligible, "prediction_error"] = (
            frame.loc[eligible, "future_return"] - frame.loc[eligible, "model_score"]
        )
        self._write(frame)
        return self._normalise(frame, decode_factor_scores=True)

    def _fill_exit_prices(
        self, trades: pd.DataFrame, prices: pd.DataFrame, as_of: pd.Timestamp | None
    ) -> pd.DataFrame:
        required = {"date", "close"}
        if not required.issubset(prices.columns):
            raise ValueError("prices require date and close columns")
        symbol_column = "symbol" if "symbol" in prices else "code" if "code" in prices else None
        if symbol_column is None:
            raise ValueError("prices require symbol or code column")
        values = prices.loc[:, ["date", symbol_column, "close"]].rename(columns={symbol_column: "symbol"}).copy()
        values["date"] = pd.to_datetime(values["date"], errors="raise")
        values["symbol"] = values["symbol"].astype(str).str.zfill(6)
        values["close"] = pd.to_numeric(values["close"], errors="coerce")
        values = values.dropna().drop_duplicates(["date", "symbol"], keep="last")
        result = trades.copy()
        pending = result["exit_price"].isna() & result["exit_date"].notna()
        if as_of is not None:
            pending &= result["exit_date"] <= pd.Timestamp(as_of)
        if pending.any():
            lookup = values.rename(columns={"date": "exit_date", "close": "resolved_exit_price"})
            resolved = result.loc[pending, ["symbol", "exit_date"]].merge(
                lookup, on=["symbol", "exit_date"], how="left"
            )
            result.loc[pending, "exit_price"] = resolved["resolved_exit_price"].to_numpy()
        return result

    def _write(self, frame: pd.DataFrame) -> None:
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        self._normalise(frame, decode_factor_scores=False).to_parquet(target, index=False)

    @staticmethod
    def _normalise(frame: pd.DataFrame, decode_factor_scores: bool) -> pd.DataFrame:
        values = frame.copy()
        for column in TRADE_COLUMNS:
            if column not in values:
                values[column] = np.nan
        values = values.loc[:, TRADE_COLUMNS]
        values["date"] = pd.to_datetime(values["date"], errors="raise")
        values["exit_date"] = pd.to_datetime(values["exit_date"], errors="coerce")
        values["symbol"] = values["symbol"].astype(str).str.zfill(6)
        for column in ("strategy", "industry", "market_regime"):
            values[column] = values[column].fillna("Unknown").astype(str)
        for column in (
            "model_score",
            "entry_price",
            "exit_price",
            "future_return",
            "prediction_error",
            "weight",
            "market_volatility",
        ):
            values[column] = pd.to_numeric(values[column], errors="coerce")
        values["factor_scores"] = values["factor_scores"].map(
            _decode_factor_scores if decode_factor_scores else _encode_factor_scores
        )
        values["decision_reason"] = values["decision_reason"].astype("object")
        missing_reason = values["decision_reason"].isna() | values["decision_reason"].isin(
            ["", "Unknown", "Not recorded"]
        )
        if missing_reason.any():
            values.loc[missing_reason, "decision_reason"] = [
                _backfill_decision_reason(scores, regime, weight)
                for scores, regime, weight in zip(
                    values.loc[missing_reason, "factor_scores"],
                    values.loc[missing_reason, "market_regime"],
                    values.loc[missing_reason, "weight"],
                    strict=True,
                )
            ]
        values["decision_reason"] = values["decision_reason"].astype(str)
        return values.sort_values(["date", "symbol", "strategy"], kind="stable").reset_index(drop=True)


def _encode_factor_scores(value: object) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return "{}"
    if not isinstance(value, Mapping):
        return "{}"
    clean = {str(key): float(number) for key, number in value.items() if pd.notna(number)}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def _decode_factor_scores(value: object) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {str(key): float(number) for key, number in value.items() if pd.notna(number)}
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return _decode_factor_scores(parsed)


def _backfill_decision_reason(factor_scores: object, market_regime: object, weight: object) -> str:
    scores = _decode_factor_scores(factor_scores)
    leaders = sorted(scores.items(), key=lambda item: (-abs(item[1]), item[0]))[:3]
    factor_text = ", ".join(f"{name}={value:+.3f}" for name, value in leaders) or "unavailable"
    exposure = float(weight) if pd.notna(weight) else 0.0
    return (
        "Buy decision explanation backfilled from v3.0 snapshot; "
        f"factor contribution: {factor_text}; "
        f"regime adjustment: {market_regime}; "
        f"risk adjustment: recorded target weight {exposure:.0%}."
    )
