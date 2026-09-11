"""Point-in-time daily alpha candidates.

This module deliberately does not train a model or infer a future outcome at
candidate-generation time.  A prediction is only calibrated from trade-memory
observations whose exit date is before the signal date.  When that evidence is
not available, ``model_prediction`` is left missing rather than replaced with
an optimistic proxy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..regime import MarketRegimeDetector, RegimeSnapshot


CANDIDATE_COLUMNS = (
    "date",
    "symbol",
    "score",
    "factor_contribution",
    "model_prediction",
    "industry",
    "risk",
    "regime",
)


@dataclass(frozen=True)
class CandidateResult:
    """The ranked candidates and the market state used to produce them."""

    candidates: pd.DataFrame
    regime: RegimeSnapshot
    output_path: Path | None = None


class AlphaCandidateRanker:
    """Produce a reproducible daily ranking from an already-scored factor panel."""

    def __init__(self, regime_detector: MarketRegimeDetector | None = None) -> None:
        self.regime_detector = regime_detector or MarketRegimeDetector()

    def rank(
        self,
        scores: pd.DataFrame,
        industries: Mapping[str, str] | None,
        benchmark: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        model_predictions: pd.DataFrame | None = None,
        factor_contributions: pd.DataFrame | None = None,
        as_of: pd.Timestamp | str | None = None,
        breadth: float | None = None,
        output_path: str | Path | None = "data/features/daily_candidates.parquet",
    ) -> CandidateResult:
        """Rank a single completed score snapshot without future leakage.

        ``scores`` must have ``date``, a code/symbol column, and either
        ``composite_score`` or ``score``.  The result contains every eligible
        symbol on the chosen date, sorted by descending score.
        """
        values = self._normalise_scores(scores)
        cutoff = pd.Timestamp(as_of) if as_of is not None else values["date"].max()
        snapshot = values.loc[values["date"] == cutoff].copy()
        if snapshot.empty:
            raise ValueError("scores contain no row at the requested as_of date")

        regime = self.regime_detector.detect(benchmark, breadth=breadth, as_of=cutoff)
        predictions = self._normalise_optional_values(model_predictions, cutoff, ("model_prediction", "prediction"))
        if predictions is not None:
            snapshot = snapshot.merge(predictions, on=["date", "symbol"], how="left")
            snapshot["model_prediction"] = snapshot["model_prediction"]
        else:
            coefficients = self._past_only_calibration(trades, cutoff)
            snapshot["model_prediction"] = self._predict(snapshot["score"], coefficients)
        mapping = {str(code).zfill(6): str(industry or "Unknown") for code, industry in (industries or {}).items()}
        snapshot["industry"] = snapshot["symbol"].map(mapping).fillna("Unknown")
        contribution_frame = self._normalise_contributions(factor_contributions, cutoff)
        if contribution_frame is None:
            snapshot["factor_contribution"] = snapshot["score"].map(
                lambda value: json.dumps({"composite_score": float(value)}, separators=(",", ":"))
            )
        else:
            snapshot = snapshot.merge(contribution_frame, on=["date", "symbol"], how="left")
            snapshot["factor_contribution"] = snapshot["factor_contribution"].fillna("{}")
        # This is an explicit regime risk score, in [0, 1], not a forecasted
        # price volatility.  The allocator combines it with the model score.
        snapshot["risk"] = float(np.clip(1.0 - regime.exposure, 0.0, 1.0))
        snapshot["regime"] = regime.state
        result = (
            snapshot.loc[:, ["date", "symbol", "score", "factor_contribution", "model_prediction", "industry", "risk", "regime"]]
            .replace([np.inf, -np.inf], np.nan)
            .sort_values(["score", "symbol"], ascending=[False, True], kind="stable")
            .reset_index(drop=True)
        )
        target = Path(output_path) if output_path is not None else None
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(target, index=False)
        return CandidateResult(result, regime, target)

    @staticmethod
    def _normalise_scores(scores: pd.DataFrame) -> pd.DataFrame:
        symbol_column = "symbol" if "symbol" in scores else "code" if "code" in scores else None
        value_column = "composite_score" if "composite_score" in scores else "score" if "score" in scores else None
        if symbol_column is None or value_column is None or "date" not in scores:
            raise ValueError("scores require date, code/symbol, and composite_score/score")
        frame = scores.loc[:, ["date", symbol_column, value_column]].copy()
        frame.columns = ["date", "symbol", "score"]
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"]).drop_duplicates(
            ["date", "symbol"], keep="last"
        )

    @staticmethod
    def _past_only_calibration(
        trades: pd.DataFrame | None, cutoff: pd.Timestamp
    ) -> tuple[float, float] | None:
        if trades is None or trades.empty:
            return None
        required = {"model_score", "future_return"}
        if not required.issubset(trades.columns):
            return None
        history = trades.copy()
        date_column = "exit_date" if "exit_date" in history else "date"
        history[date_column] = pd.to_datetime(history[date_column], errors="coerce")
        history["model_score"] = pd.to_numeric(history["model_score"], errors="coerce")
        history["future_return"] = pd.to_numeric(history["future_return"], errors="coerce")
        history = history.loc[
            (history[date_column] < cutoff)
            & history["model_score"].notna()
            & history["future_return"].notna()
        ]
        # A two-point regression is mathematically defined but not useful
        # research evidence.  Keep prediction unavailable until enough past
        # observations have matured.
        if len(history) < 10 or history["model_score"].nunique() < 2:
            return None
        slope, intercept = np.polyfit(
            history["model_score"].to_numpy(dtype=float),
            history["future_return"].to_numpy(dtype=float),
            1,
        )
        return float(intercept), float(slope)

    @staticmethod
    def _normalise_optional_values(
        values: pd.DataFrame | None, cutoff: pd.Timestamp, candidates: tuple[str, ...]
    ) -> pd.DataFrame | None:
        if values is None or values.empty or "date" not in values:
            return None
        symbol_column = "symbol" if "symbol" in values else "code" if "code" in values else None
        value_column = next((name for name in candidates if name in values), None)
        if symbol_column is None or value_column is None:
            raise ValueError("model predictions require date, code/symbol, and prediction value")
        frame = values.loc[:, ["date", symbol_column, value_column]].copy()
        frame.columns = ["date", "symbol", "model_prediction"]
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        frame["model_prediction"] = pd.to_numeric(frame["model_prediction"], errors="coerce")
        return frame.loc[frame["date"] == cutoff].drop_duplicates(["date", "symbol"], keep="last")

    @staticmethod
    def _normalise_contributions(values: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame | None:
        if values is None or values.empty or "date" not in values:
            return None
        symbol_column = "symbol" if "symbol" in values else "code" if "code" in values else None
        if symbol_column is None:
            raise ValueError("factor contributions require date and code/symbol")
        frame = values.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["symbol"] = frame[symbol_column].astype(str).str.zfill(6)
        frame = frame.loc[frame["date"] == cutoff].copy()
        non_factor = {"date", "code", "symbol", "composite_score", "score"}
        factors = [column for column in frame.columns if column not in non_factor]
        if not factors:
            return None
        def encoded(row: pd.Series) -> str:
            payload = {
                name: float(value)
                for name, value in row.items()
                if name in factors and pd.notna(value) and np.isfinite(float(value))
            }
            return json.dumps(payload, sort_keys=True, separators=(",", ":"))
        result = frame.loc[:, ["date", "symbol", *factors]].copy()
        result["factor_contribution"] = result.apply(encoded, axis=1)
        return result.loc[:, ["date", "symbol", "factor_contribution"]].drop_duplicates(
            ["date", "symbol"], keep="last"
        )

    @staticmethod
    def _predict(scores: pd.Series, coefficients: tuple[float, float] | None) -> pd.Series:
        if coefficients is None:
            return pd.Series(np.nan, index=scores.index, dtype="float64")
        intercept, slope = coefficients
        return intercept + slope * pd.to_numeric(scores, errors="coerce")
