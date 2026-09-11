import numpy as np
import pandas as pd
import pytest

from quant.research import (
    AnnualWalkForwardResearch,
    FactorCombinationResearch,
    FactorNeutralizer,
    ProfessionalFactorEvaluator,
)


def _cross_sectional_panel(periods: int = 40, stocks: int = 24) -> pd.DataFrame:
    records = []
    for date_index, date in enumerate(pd.bdate_range("2023-01-02", periods=periods)):
        for stock in range(stocks):
            signal = stock - stocks / 2 + date_index * 0.01
            records.append(
                {
                    "date": date,
                    "code": f"{stock:06d}",
                    "industry": f"I{stock % 3}",
                    "market_cap": 1_000_000 + stock * 10_000,
                    "factor_a": signal,
                    "factor_b": -signal,
                    "future_excess_return_20d": signal / 1_000,
                }
            )
    return pd.DataFrame(records)


def test_neutralization_clips_outliers_and_supports_industry_and_market_cap_controls():
    panel = _cross_sectional_panel(periods=1)
    panel["market_cap"] = 1_000_000 + (np.arange(len(panel)) % 7) * 10_000
    panel.loc[0, "factor_a"] = 1_000_000
    neutral = FactorNeutralizer().transform(panel, ["factor_a"], require_market_cap=True)

    score = neutral["factor_a_neutralized"].dropna()
    assert score.mean() == pytest.approx(0.0, abs=1e-12)
    assert score.std(ddof=0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="market_cap"):
        FactorNeutralizer().transform(panel.drop(columns="market_cap"), ["factor_a"], require_market_cap=True)


def test_professional_evaluator_reports_ic_t_stat_annualized_long_short_and_yearly_stability():
    panel = _cross_sectional_panel().rename(columns={"factor_a": "factor_a_neutralized"})
    result = ProfessionalFactorEvaluator(horizon=20, min_cross_section=20).evaluate(
        panel, ["factor_a_neutralized"]
    )

    summary = result.summary.iloc[0]
    assert summary["IC"] > 0.99
    assert summary["Rank_IC"] > 0.99
    assert summary["t_stat"] > 0
    assert summary["annualized_long_short_return"] > 0
    assert not result.yearly_stability.empty


def test_combination_and_annual_walk_forward_use_only_matured_training_labels(tmp_path):
    panel = _cross_sectional_panel(periods=80).rename(
        columns={"factor_a": "factor_a_neutralized", "factor_b": "factor_b_neutralized"}
    )
    evaluator = ProfessionalFactorEvaluator(horizon=5, min_cross_section=20)
    evaluation = evaluator.evaluate(panel, ["factor_a_neutralized", "factor_b_neutralized"])
    evaluation.summary["factor"] = evaluation.summary["factor"].str.removesuffix("_neutralized")
    evaluation.daily["factor"] = evaluation.daily["factor"].str.removesuffix("_neutralized")
    scores, weights = FactorCombinationResearch(horizon=5, rolling_window=30, reweight_interval=5).compare(
        panel, ["factor_a_neutralized", "factor_b_neutralized"], evaluation
    )

    assert set(weights["method"]) == {"equal", "ic", "icir", "regression", "ml"}
    assert {"factor", "weight", "IC", "stability"}.issubset(weights.columns)
    assert len(scores) == len(panel) * 5

    streamed_scores, streamed_weights = FactorCombinationResearch(
        horizon=5, rolling_window=30, reweight_interval=5
    ).compare(
        panel,
        ["factor_a_neutralized", "factor_b_neutralized"],
        evaluation,
        collect_scores=False,
    )
    assert streamed_scores.empty
    assert len(streamed_weights) == len(weights)

    # Build a calendar-year panel to exercise the 2018-2021 -> 2022 schedule.
    dates = pd.date_range("2018-01-01", "2025-12-01", freq="BMS")
    yearly = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "code": f"{stock:06d}",
                    "factor_a_neutralized": float(stock),
                    "future_excess_return_20d": float(stock) / 1_000,
                }
            )
            for stock in range(24)
        ],
        ignore_index=True,
    )
    report = AnnualWalkForwardResearch(horizon=2).run(
        yearly, ["factor_a_neutralized"], output_path=tmp_path / "walk_forward_report.csv"
    )
    assert "2022" in set(report["period"])
    assert (tmp_path / "walk_forward_report.csv").exists()
