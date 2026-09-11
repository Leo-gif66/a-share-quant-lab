"""Point-in-time decision explanations for persistent trade memory."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


class DecisionExplanationEngine:
    """Create compact, evidence-only purchase rationales.

    The explanation intentionally describes the score components and applied
    exposure controls.  It does not claim that a component caused a return.
    """

    def explain(
        self,
        factor_scores: Mapping[str, float],
        market_regime: str,
        regime_exposure: float,
        risk_exposure: float,
    ) -> str:
        clean = {
            str(name): float(value)
            for name, value in factor_scores.items()
            if pd.notna(value)
        }
        leaders = sorted(clean.items(), key=lambda item: (-abs(item[1]), item[0]))[:3]
        factor_text = ", ".join(f"{name}={value:+.3f}" for name, value in leaders) or "no valid factor components"
        return (
            f"Buy because the security ranked in the selected composite-score set; "
            f"factor contribution: {factor_text}; "
            f"regime adjustment: {market_regime} exposure {regime_exposure:.0%}; "
            f"risk adjustment: final target exposure {risk_exposure:.0%}."
        )
