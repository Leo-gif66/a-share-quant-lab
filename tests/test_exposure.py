import pandas as pd
import pytest

from quant.portfolio.industry_neutral import industry_exposure


def test_industry_exposure_reports_portfolio_benchmark_and_active_weights():
    holdings = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-03", "2024-01-03"],
            "code": ["000001", "000002", "000003"],
            "sector": ["Finance", "Finance", "Technology"],
            "weight": [0.15, 0.10, 0.20],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-03"],
            "industry": ["Finance", "Technology"],
            "benchmark_weight": [0.40, 0.60],
        }
    )

    exposure = industry_exposure(holdings, benchmark)

    finance = exposure.loc[exposure["industry"] == "Finance"].iloc[0]
    technology = exposure.loc[exposure["industry"] == "Technology"].iloc[0]
    assert finance["portfolio_weight"] == pytest.approx(0.25)
    assert finance["active_weight"] == pytest.approx(-0.15)
    assert technology["active_weight"] == pytest.approx(-0.40)
