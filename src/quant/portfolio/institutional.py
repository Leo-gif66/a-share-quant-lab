"""v2.0 composition: v1.2 industry construction plus v1.3 risk overlay."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..risk import AdvancedRiskController, AdvancedRiskSettings
from .industry_neutral import IndustryNeutralPortfolioBacktestEngine


class InstitutionalPortfolioBacktestEngine(IndustryNeutralPortfolioBacktestEngine):
    """Use advanced risk decisions without changing backtest accounting.

    Trade timing, transaction costs, position accounting, benchmark construction,
    and portfolio diagnostics remain inherited from the existing v1.0/v1.2
    engines.  Only the risk-controller dependency is replaced.
    """

    def __init__(
        self,
        *args: object,
        advanced_risk_settings: AdvancedRiskSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        settings = advanced_risk_settings or AdvancedRiskSettings()
        if settings.max_position != self.constraints.max_stock_weight:
            settings = replace(settings, max_position=min(settings.max_position, self.constraints.max_stock_weight))
        self.risk = AdvancedRiskController(settings)
