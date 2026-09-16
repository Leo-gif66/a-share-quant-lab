"""Evidence-based explanations when V5 alpha candidates fail validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .v5_reporting import write_v5_report


class AlphaFailureAnalyzer:
    """Summarize decay, regime, cost, capacity, and overfit evidence without spin."""

    def analyze(
        self,
        factor_summary: pd.DataFrame,
        yearly: pd.DataFrame,
        regimes: pd.DataFrame,
        walk_forward: pd.DataFrame,
        matrix: pd.DataFrame,
        capacity: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return one explicit finding per failure mode, including unavailable evidence."""
        factor_decay = float(yearly.groupby(["factor", "horizon"])["rank_ic"].std().mean()) if not yearly.empty else np.nan
        regime_dependence = float(regimes.groupby(["factor", "horizon"])["rank_ic"].std().mean()) if not regimes.empty else np.nan
        weak_spread = float(factor_summary["long_short_return"].mean()) if not factor_summary.empty else np.nan
        overfit = float((walk_forward["validation_prediction_IC"] - walk_forward["prediction_IC"]).mean()) if not walk_forward.empty else np.nan
        cost_sensitivity = np.nan
        if not matrix.empty and "transaction_cost_multiplier" in matrix:
            costs = matrix.groupby("transaction_cost_multiplier")["annual_return"].mean()
            cost_sensitivity = float(costs.max() - costs.min()) if len(costs) > 1 else np.nan
        sector_concentration = float(capacity.groupby("date")["position_value"].sum().std()) if not capacity.empty else np.nan
        capacity_flag_ratio = float(capacity["capacity_flag"].mean()) if not capacity.empty else np.nan
        risk_overlay_effect = np.nan
        if not matrix.empty and "risk_overlay" in matrix:
            overlays = matrix.groupby("risk_overlay")["annual_return"].mean()
            risk_overlay_effect = float(overlays.max() - overlays.min()) if len(overlays) > 1 else np.nan
        return pd.DataFrame(
            [
                {"failure_mode": "factor_decay", "value": factor_decay, "interpretation": "Higher yearly Rank IC dispersion indicates decay or instability."},
                {"failure_mode": "regime_dependence", "value": regime_dependence, "interpretation": "Higher regime Rank IC dispersion indicates conditional behaviour."},
                {"failure_mode": "weak_cross_sectional_spread", "value": weak_spread, "interpretation": "Mean long-short return near or below zero weakens the alpha case."},
                {"failure_mode": "model_overfit", "value": overfit, "interpretation": "Positive validation-minus-test IC indicates possible model degradation."},
                {"failure_mode": "cost_sensitivity", "value": cost_sensitivity, "interpretation": "Return range across cost multipliers."},
                {"failure_mode": "sector_concentration", "value": sector_concentration, "interpretation": "Variation in aggregate position value; inspect holdings by sector separately."},
                {"failure_mode": "capacity_flags", "value": capacity_flag_ratio, "interpretation": "Share of holdings above the participation threshold."},
                {"failure_mode": "risk_overlay_effect", "value": risk_overlay_effect, "interpretation": "Return range across risk-overlay settings."},
            ]
        )

    def save(self, findings: pd.DataFrame, output_path: str | Path = "reports/alpha_failure_analysis.html") -> Path:
        return write_v5_report("V5 Alpha Failure Analysis", (("Failure modes", findings),), output_path)
