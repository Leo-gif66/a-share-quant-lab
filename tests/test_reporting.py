from types import SimpleNamespace

import pandas as pd

from quant.reporting import ResearchReportBuilder


def test_html_research_report_contains_required_institutional_sections(tmp_path):
    result = SimpleNamespace(
        metrics={"annual_return": 0.1, "sharpe": 1.2, "max_drawdown": -0.2, "volatility": 0.15, "beta": 0.8},
        holdings_history=pd.DataFrame({"code": ["000001"], "weight": [0.1]}),
        sector_exposure=pd.DataFrame({"industry": ["Technology"], "portfolio_weight": [0.2]}),
        factor_contribution=pd.DataFrame({"factor_name": ["momentum_20"], "IC": [0.03]}),
    )

    path = ResearchReportBuilder().from_backtest_result(
        result,
        pd.DataFrame({"feature": ["momentum_20"], "importance": [0.8]}),
        tmp_path / "latest.html",
    )

    content = path.read_text(encoding="utf-8")
    assert "Performance" in content
    assert "Sector exposure" in content
    assert "Factor contribution" in content
    assert "ML feature importance" in content
