import pandas as pd
import pytest
from typer.testing import CliRunner

from quant.cli import app
from quant.research import FactorResearchDataBuilder
from quant.research.factor_evaluation import FactorEvaluator


def research_panel() -> pd.DataFrame:
    rows = []
    returns_by_date = ([0.01, 0.02, 0.03], [0.01, 0.03, 0.02])
    for date, returns in zip(pd.to_datetime(["2024-01-02", "2024-01-03"]), returns_by_date):
        for code, (factor, future_return) in enumerate(zip([1.0, 2.0, 3.0], returns), start=1):
            rows.append(
                {
                    "date": date,
                    "code": f"{code:06d}",
                    "mom_5": factor,
                    "future_return": future_return,
                }
            )
    return pd.DataFrame(rows)


def test_factor_evaluation_calculates_ic_statistics_from_daily_cross_sections():
    result = FactorEvaluator(factors=("momentum_5",)).evaluate(research_panel())

    assert list(result.columns) == list(FactorEvaluator.SUMMARY_COLUMNS)
    assert result.loc[0, "factor_name"] == "momentum_5"
    assert result.loc[0, "IC"] == pytest.approx(0.75)
    assert result.loc[0, "rank_IC"] == pytest.approx(0.75)
    assert result.loc[0, "IC_mean"] == pytest.approx(0.75)
    assert result.loc[0, "IC_std"] == pytest.approx(2**-0.5 / 2)
    assert result.loc[0, "ICIR"] == pytest.approx(3 / 2**0.5)


def test_factor_contribution_outputs_requested_fields():
    result = FactorEvaluator(factors=("momentum_5",)).contribution_analysis(research_panel())

    assert list(result.columns) == list(FactorEvaluator.CONTRIBUTION_COLUMNS)
    assert result.loc[0, "average_return"] == pytest.approx(0.015)
    assert result.loc[0, "win_rate"] == 1.0
    assert result.loc[0, "IC"] == pytest.approx(0.75)
    assert result.loc[0, "rank_IC"] == pytest.approx(0.75)


def test_complete_factor_analysis_has_all_requested_metrics():
    panel = research_panel().rename(columns={"future_return": "future_return_20d"})

    result = FactorEvaluator(factors=("momentum_5",)).analyze(panel)

    assert list(result.columns) == list(FactorEvaluator.ANALYSIS_COLUMNS)
    assert result.loc[0, "IC"] == pytest.approx(0.75)
    assert result.loc[0, "average_return"] == pytest.approx(0.015)


def _write_factor_research_files(tmp_path):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    raw_dir.mkdir()
    features_dir.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=24)
    factor_columns = (
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "trend_20",
        "trend_60",
        "volatility_20",
        "liquidity_20",
        "turnover_20",
        "drawdown_60",
    )
    for index, code in enumerate(("000001", "000002", "000003"), start=1):
        close = 10 * (1 + index * 0.001) ** pd.Series(range(len(dates)))
        pd.DataFrame({"date": dates, "close": close}).to_parquet(raw_dir / f"{code}.parquet")
        features = pd.DataFrame({"date": dates})
        for column in factor_columns:
            features[column] = float(index)
        features.to_parquet(features_dir / f"{code}.parquet")
    return raw_dir, features_dir


def test_research_builder_calculates_future_return_20d_from_raw_prices(tmp_path):
    raw_dir, features_dir = _write_factor_research_files(tmp_path)

    panel = FactorResearchDataBuilder(features_dir=features_dir, raw_dir=raw_dir).build()

    first = panel.loc[(panel["code"] == "000003") & (panel["date"] == panel["date"].min())].iloc[0]
    expected = (1.003**20) - 1
    assert "future_return_20d" in panel
    assert first["future_return_20d"] == pytest.approx(expected)
    assert panel["future_return_20d"].tail(3).isna().all()


def test_factor_analysis_cli_builds_labels_from_raw_and_features(tmp_path):
    _write_factor_research_files(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(f'data:\n  storage_root: "{tmp_path.as_posix()}"\n', encoding="utf-8")

    result = CliRunner().invoke(app, ["factor-analysis", "--base", str(config)])

    assert result.exit_code == 0, result.stdout
    assert "Factor analysis (future_return_20d)" in result.stdout
    assert "average_return" in result.stdout
