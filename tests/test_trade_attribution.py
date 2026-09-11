import pandas as pd
import pytest

from quant.research import TradeAttributionEngine
from quant.reporting import ResearchReportBuilder


def test_trade_attribution_calculates_return_factor_industry_market_and_report(tmp_path):
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "exit_date": pd.to_datetime(["2024-01-22", "2024-01-22"]),
            "symbol": ["000001", "000002"],
            "model_score": [0.05, -0.02],
            "factor_scores": [{"momentum_5": 1.0}, {"momentum_5": -0.5, "volatility": 0.2}],
            "industry": ["Bank", "Technology"],
            "future_return": [0.10, -0.05],
            "prediction_error": [0.05, -0.03],
            "weight": [0.5, 0.5],
        }
    )
    benchmark = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02", "2024-01-22"]), "close": [100.0, 102.0]}
    )

    result = TradeAttributionEngine().review(trades, benchmark)
    report = ResearchReportBuilder().build_strategy_review(
        result.factor_contribution,
        result.summary,
        result.industry_contribution,
        result.drawdown_analysis,
        result.monthly_summary,
        tmp_path / "strategy_review.html",
    )

    assert result.summary.loc[0, "total_return"] == pytest.approx(0.025)
    assert result.summary.loc[0, "market_return"] == pytest.approx(0.02)
    assert set(result.factor_contribution["factor_name"]) == {"momentum_5", "volatility"}
    assert set(result.industry_contribution["industry"]) == {"Bank", "Technology"}
    assert "Sector mistakes" in report.read_text(encoding="utf-8")
