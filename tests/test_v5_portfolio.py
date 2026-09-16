from __future__ import annotations

import pandas as pd
import pytest

from quant.models import AlphaEnsemble
from quant.research.capacity import capacity_diagnostics
from quant.research.portfolio_v5 import V5PortfolioEvaluator, run_experiment_matrix


def test_alpha_ensemble_fits_only_validation_then_scores_later_test_dates():
    validation_dates = pd.to_datetime(["2023-01-02", "2023-01-03"])
    validation = pd.DataFrame(
        {
            "date": validation_dates.repeat(6),
            "factor": list(range(6)) * 2,
            "model": list(range(6)) * 2,
            "label": list(range(6)) * 2,
        }
    )
    ensemble = AlphaEnsemble("validation_weighted")
    ensemble.fit(validation, ["factor", "model"], "label")
    test = validation.assign(date=pd.Timestamp("2023-01-04"))

    score = ensemble.predict(test)

    assert score.notna().all()
    with pytest.raises(AssertionError):
        ensemble.predict(validation)


def test_v5_portfolio_applies_constraints_costs_and_capacity_diagnostics():
    rows = []
    dates = pd.bdate_range("2023-01-02", periods=45)
    for date in dates:
        for number in range(12):
            rows.append(
                {
                    "date": date,
                    "code": f"{number:06d}",
                    "industry": f"industry-{number % 4}",
                    "ensemble_score": 12 - number,
                    "future_return_5d": 0.01 * (12 - number),
                    "benchmark_future_return_5d": 0.01,
                    "label_end_date_5d": date + pd.Timedelta(days=7),
                    "market_regime": "bull",
                    "volatility_regime": "normal",
                    "realized_vol_20": 0.2 + number * 0.01,
                    "amount_20": 5_000_000.0,
                }
            )
    result = V5PortfolioEvaluator().evaluate(
        pd.DataFrame(rows), horizon=5, top_n=8, rebalance_frequency=5,
        weighting="volatility_adjusted", industry_limit=0.25, max_stock_weight=0.2,
    )

    assert not result.results.empty
    assert result.metrics["turnover"] >= 0
    assert result.holdings.groupby("date")["weight"].sum().max() <= 1.0 + 1e-12
    capacity = capacity_diagnostics(result.holdings)
    assert {"position_value", "participation_rate", "capacity_flag"}.issubset(capacity.columns)


def test_v5_portfolio_matrix_retains_every_requested_configuration():
    rows = []
    for date in pd.bdate_range("2023-01-02", periods=2):
        for number in range(3):
            rows.append(
                {
                    "date": date, "code": f"{number:06d}", "industry": f"industry-{number}",
                    "ensemble_score": float(3 - number), "future_return_5d": 0.01,
                    "benchmark_future_return_5d": 0.005, "label_end_date_5d": date + pd.Timedelta(days=7),
                    "market_regime": "bull", "volatility_regime": "normal", "amount_20": 1_000_000.0,
                    "realized_vol_20": 0.2,
                }
            )
    matrix = run_experiment_matrix(V5PortfolioEvaluator(), pd.DataFrame(rows), horizon=5)

    assert len(matrix) == 1_728
    assert {"research_rank", "annual_return", "sharpe", "turnover"}.issubset(matrix.columns)
