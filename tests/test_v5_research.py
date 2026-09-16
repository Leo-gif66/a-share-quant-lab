from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.factor_selection import FactorSelector
from quant.research.factor_v5 import V5FactorResearchEngine


def _panel() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2021-01-04", periods=80)
    for day, date in enumerate(dates):
        for stock in range(30):
            factor = stock / 30 + day * 0.001
            rows.append(
                {
                    "date": date,
                    "code": f"{stock:06d}",
                    "sector": "A" if stock < 15 else "B",
                    "market_regime": ("bull", "bear", "sideways")[day % 3],
                    "factor_a_processed": factor,
                    "factor_b_processed": factor * 0.99,
                    "factor_c_processed": -factor,
                    "future_excess_return_5d": factor + np.sin(day) * 0.001,
                    "future_excess_return_10d": factor + np.cos(day) * 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_v5_factor_research_reports_multiple_horizons_and_fdr():
    result = V5FactorResearchEngine(min_cross_section=10).evaluate(
        _panel(), ["factor_a_processed", "factor_b_processed", "factor_c_processed"], horizons=(5, 10)
    )

    assert len(result.summary) == 6
    assert {"t_stat", "positive_ic_ratio", "long_short_sharpe", "fdr_q_value", "fdr_significant"}.issubset(result.summary.columns)
    assert set(result.summary["horizon"]) == {5, 10}
    assert not result.yearly.empty
    assert not result.regimes.empty


def test_factor_selector_groups_redundant_features_and_keeps_one_candidate():
    panel = _panel()
    research = V5FactorResearchEngine(min_cross_section=10).evaluate(
        panel, ["factor_a_processed", "factor_b_processed", "factor_c_processed"], horizons=(5,)
    ).summary
    result = FactorSelector(threshold=0.8, max_dates=20).select(panel, research, horizon=5)

    selected = result.selection.loc[result.selection["selected"]]
    assert len(selected) == 1
    assert result.selection.loc[result.selection["factor"] == "factor_a_processed", "group"].iloc[0] == result.selection.loc[
        result.selection["factor"] == "factor_b_processed", "group"
    ].iloc[0]
