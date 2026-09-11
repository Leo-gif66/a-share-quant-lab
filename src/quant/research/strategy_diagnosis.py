"""Evidence-based strategy diagnosis and conservative adjustment recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .error_analysis import ErrorAnalysisResult, PredictionErrorAnalyzer


@dataclass(frozen=True)
class StrategyDiagnosisResult:
    best_conditions: pd.DataFrame
    worst_conditions: pd.DataFrame
    factor_failures: pd.DataFrame
    prediction_biases: pd.DataFrame
    recommendations: pd.DataFrame
    error_analysis: ErrorAnalysisResult


class StrategyDiagnosisEngine:
    """Turn completed-trade error evidence into bounded, reviewable suggestions."""

    def __init__(self, minimum_observations: int = 5) -> None:
        if minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        self.minimum_observations = minimum_observations

    def diagnose(
        self, trades: pd.DataFrame, analysis: ErrorAnalysisResult | None = None
    ) -> StrategyDiagnosisResult:
        evidence = analysis or PredictionErrorAnalyzer().analyze(trades)
        regimes = evidence.regime_summary.copy()
        reliable = regimes.loc[regimes.get("observations", pd.Series(dtype=int)) >= self.minimum_observations]
        best = reliable.sort_values("average_return", ascending=False, kind="stable").head(5).reset_index(drop=True)
        worst = reliable.sort_values("average_return", kind="stable").head(5).reset_index(drop=True)
        factor_failures = evidence.factor_summary.loc[
            evidence.factor_summary.get("observations", pd.Series(dtype=int)) >= self.minimum_observations
        ].copy()
        factor_failures = factor_failures.sort_values("failure_rate", ascending=False, kind="stable").head(10)
        biases = evidence.score_bucket_summary.sort_values("condition", kind="stable").reset_index(drop=True)
        recommendations = self._recommend(worst, factor_failures, evidence.trades)
        return StrategyDiagnosisResult(best, worst, factor_failures, biases, recommendations, evidence)

    def _recommend(
        self, worst: pd.DataFrame, factor_failures: pd.DataFrame, trades: pd.DataFrame
    ) -> pd.DataFrame:
        suggestions: list[dict[str, object]] = []
        for row in worst.itertuples(index=False):
            if row.average_return < 0:
                suggestions.append(
                    {
                        "area": "market_regime",
                        "recommendation": f"Review or reduce exposure during {row.condition} conditions.",
                        "evidence": f"{row.observations} trades; average return {row.average_return:.2%}",
                    }
                )
        for row in factor_failures.itertuples(index=False):
            if row.failure_rate > 0.5:
                suggestions.append(
                    {
                        "area": "factor",
                        "recommendation": f"Review or downweight {row.factor} when its state is {row.factor_state}.",
                        "evidence": f"failure rate {row.failure_rate:.1%} across {row.observations} trades",
                    }
                )
        bias = pd.to_numeric(trades.get("prediction_error"), errors="coerce").mean()
        if pd.notna(bias) and abs(bias) >= 0.002:
            direction = "downward" if bias < 0 else "upward"
            suggestions.append(
                {
                    "area": "calibration",
                    "recommendation": f"Apply a {direction} return-calibration review.",
                    "evidence": f"mean prediction error {bias:.3%}",
                }
            )
        if not suggestions:
            suggestions.append(
                {
                    "area": "evidence",
                    "recommendation": "No adjustment recommended: available completed-trade evidence is inconclusive.",
                    "evidence": f"minimum observation threshold is {self.minimum_observations}",
                }
            )
        return pd.DataFrame(suggestions, columns=["area", "recommendation", "evidence"])
