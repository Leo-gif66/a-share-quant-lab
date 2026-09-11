"""Completed-trade prediction-error classification and condition diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ERROR_ANALYSIS_COLUMNS = (
    "date",
    "symbol",
    "strategy",
    "market_regime",
    "industry",
    "model_score",
    "future_return",
    "prediction_error",
    "market_volatility",
    "volatility_bucket",
    "prediction_score_bucket",
    "classification",
    "factor_states",
)


@dataclass(frozen=True)
class ErrorAnalysisResult:
    trades: pd.DataFrame
    regime_summary: pd.DataFrame
    industry_summary: pd.DataFrame
    factor_summary: pd.DataFrame
    volatility_summary: pd.DataFrame
    score_bucket_summary: pd.DataFrame


class PredictionErrorAnalyzer:
    """Classify completed directional predictions without using pending trades."""

    def analyze(
        self,
        trades: pd.DataFrame,
        output_path: str | Path = "data/memory/error_analysis.parquet",
    ) -> ErrorAnalysisResult:
        frame = self._completed(trades)
        if frame.empty:
            empty = pd.DataFrame()
            self._save(frame, output_path)
            return ErrorAnalysisResult(frame, empty, empty, empty, empty, empty)
        frame["classification"] = np.select(
            [
                (frame["model_score"] >= 0) & (frame["future_return"] >= 0),
                (frame["model_score"] >= 0) & (frame["future_return"] < 0),
                (frame["model_score"] < 0) & (frame["future_return"] < 0),
            ],
            ["true_positive", "false_positive", "true_negative"],
            default="false_negative",
        )
        frame["volatility_bucket"] = self._volatility_bucket(frame["market_volatility"])
        frame["prediction_score_bucket"] = self._score_bucket(frame["model_score"])
        frame["factor_states"] = frame["factor_scores"].map(_factor_states)
        factor_summary = self._factor_summary(frame)
        frame = frame.loc[:, ERROR_ANALYSIS_COLUMNS]
        self._save(frame, output_path)
        return ErrorAnalysisResult(
            trades=frame,
            regime_summary=self._condition_summary(frame, "market_regime"),
            industry_summary=self._condition_summary(frame, "industry"),
            factor_summary=factor_summary,
            volatility_summary=self._condition_summary(frame, "volatility_bucket"),
            score_bucket_summary=self._condition_summary(frame, "prediction_score_bucket"),
        )

    @staticmethod
    def _completed(trades: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "symbol", "model_score", "future_return", "prediction_error", "factor_scores"}
        if not required.issubset(trades.columns):
            raise ValueError(f"trade memory missing columns: {', '.join(sorted(required.difference(trades.columns)))}")
        frame = trades.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        for column in ("model_score", "future_return", "prediction_error", "market_volatility"):
            if column not in frame:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in ("strategy", "market_regime", "industry"):
            if column not in frame:
                frame[column] = "Unknown"
            frame[column] = frame[column].fillna("Unknown").astype(str)
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        return frame.dropna(subset=["model_score", "future_return"]).sort_values(["date", "symbol"]).reset_index(drop=True)

    @staticmethod
    def _volatility_bucket(values: pd.Series) -> pd.Series:
        result = pd.Series("unknown", index=values.index, dtype="object")
        result.loc[values.notna() & (values < 0.15)] = "low"
        result.loc[values.notna() & values.between(0.15, 0.25, inclusive="left")] = "normal"
        result.loc[values.notna() & (values >= 0.25)] = "high"
        return result

    @staticmethod
    def _score_bucket(values: pd.Series) -> pd.Series:
        if values.empty:
            return pd.Series(dtype="object")
        buckets = min(5, int(values.nunique()), len(values))
        if buckets < 2:
            return pd.Series("Q1", index=values.index, dtype="object")
        ranked = values.rank(method="first")
        return pd.qcut(ranked, q=buckets, labels=[f"Q{index}" for index in range(1, buckets + 1)]).astype(str)

    @staticmethod
    def _condition_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        result = (
            frame.groupby(column, as_index=False)
            .agg(
                observations=("symbol", "size"),
                average_return=("future_return", "mean"),
                prediction_bias=("prediction_error", "mean"),
                accuracy=("classification", lambda values: values.isin(["true_positive", "true_negative"]).mean()),
                false_positive=("classification", lambda values: (values == "false_positive").sum()),
                false_negative=("classification", lambda values: (values == "false_negative").sum()),
            )
            .sort_values("average_return", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        return result.rename(columns={column: "condition"})

    @staticmethod
    def _factor_summary(frame: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for row in frame.itertuples(index=False):
            for factor, state in _factor_states_dict(row.factor_scores).items():
                records.append(
                    {
                        "factor": factor,
                        "factor_state": state,
                        "classification": row.classification,
                        "future_return": row.future_return,
                        "prediction_error": row.prediction_error,
                    }
                )
        if not records:
            return pd.DataFrame(columns=["factor", "factor_state", "observations", "average_return", "failure_rate", "prediction_bias"])
        values = pd.DataFrame(records)
        return (
            values.groupby(["factor", "factor_state"], as_index=False)
            .agg(
                observations=("classification", "size"),
                average_return=("future_return", "mean"),
                failure_rate=("classification", lambda items: items.isin(["false_positive", "false_negative"]).mean()),
                prediction_bias=("prediction_error", "mean"),
            )
            .sort_values(["failure_rate", "observations"], ascending=[False, False], kind="stable")
            .reset_index(drop=True)
        )

    @staticmethod
    def _save(frame: pd.DataFrame, output_path: str | Path) -> None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)


def _factor_states(value: object) -> str:
    return json.dumps(_factor_states_dict(value), sort_keys=True, separators=(",", ":"))


def _factor_states_dict(value: object) -> Mapping[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, Mapping):
        return {}
    return {
        str(factor): "positive" if float(score) >= 0 else "negative"
        for factor, score in value.items()
        if pd.notna(score)
    }
