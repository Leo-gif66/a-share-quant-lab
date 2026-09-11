import pandas as pd

from quant.research import PredictionErrorAnalyzer


def _completed_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 8),
            "symbol": [f"{number:06d}" for number in range(1, 9)],
            "strategy": ["v3"] * 8,
            "market_regime": ["bull", "bear"] * 4,
            "industry": ["Bank"] * 4 + ["Technology"] * 4,
            "model_score": [0.02, 0.02, -0.02, -0.02, 0.03, 0.03, -0.03, -0.03],
            "future_return": [0.04, -0.04, -0.04, 0.04, 0.05, -0.05, -0.05, 0.05],
            "prediction_error": [0.02, -0.06, -0.02, 0.06, 0.02, -0.08, -0.02, 0.08],
            "market_volatility": [0.1, 0.3] * 4,
            "factor_scores": [
                {"momentum_5": 0.4, "volatility": -0.1},
                {"momentum_5": -0.4, "volatility": 0.1},
            ] * 4,
        }
    )


def test_prediction_error_analysis_classifies_all_directional_outcomes_and_persists(tmp_path):
    path = tmp_path / "error_analysis.parquet"
    result = PredictionErrorAnalyzer().analyze(_completed_trades(), path)

    assert path.exists()
    assert set(result.trades["classification"]) == {
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    }
    assert {"low", "high"}.issubset(set(result.trades["volatility_bucket"]))
    assert {"momentum_5", "volatility"}.issubset(set(result.factor_summary["factor"]))
    assert not result.regime_summary.empty
    assert not result.score_bucket_summary.empty
