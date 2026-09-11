from pathlib import Path

from quant.reporting import ResearchReportBuilder
from quant.research import PredictionErrorAnalyzer, StrategyDiagnosisEngine

from test_error_analysis import _completed_trades


def test_strategy_diagnosis_identifies_bad_conditions_and_renders_report(tmp_path: Path):
    analysis = PredictionErrorAnalyzer().analyze(_completed_trades(), tmp_path / "error.parquet")
    diagnosis = StrategyDiagnosisEngine(minimum_observations=2).diagnose(analysis.trades, analysis)
    report = ResearchReportBuilder().build_strategy_diagnosis(
        diagnosis.best_conditions,
        diagnosis.worst_conditions,
        diagnosis.factor_failures,
        diagnosis.prediction_biases,
        diagnosis.recommendations,
        tmp_path / "strategy_diagnosis.html",
    )

    assert not diagnosis.best_conditions.empty
    assert not diagnosis.worst_conditions.empty
    assert not diagnosis.recommendations.empty
    assert "Recommended adjustments" in report.read_text(encoding="utf-8")
