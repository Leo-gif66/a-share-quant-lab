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


def test_alpha_research_report_includes_only_supplied_research_tables(tmp_path):
    path = ResearchReportBuilder().build_alpha_research(
        pd.DataFrame({"factor": ["momentum_20"], "IC": [0.02]}),
        pd.DataFrame({"factor": ["momentum_20"], "year": [2024], "IC_mean": [0.02]}),
        pd.DataFrame({"feature": ["momentum_20"], "importance": [0.5]}),
        pd.DataFrame({"period": ["2024"], "sharpe": [1.0]}),
        pd.DataFrame({"portfolio": ["ml_ranking"], "sharpe": [1.1]}),
        tmp_path / "alpha_research.html",
    )

    content = path.read_text(encoding="utf-8")
    assert "Factor IC ranking" in content
    assert "Walk-forward performance" in content
    assert "Portfolio comparison" in content
