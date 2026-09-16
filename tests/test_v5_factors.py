from __future__ import annotations

import numpy as np
import pandas as pd

from quant.factors.alpha import ALPHA_FACTORS, AlphaFactorEngine
from quant.factors.market import MARKET_FEATURES, MarketBreadthEngine
from quant.research.labels import add_v5_forward_labels
from quant.research.v5_preprocessing import V5FactorPreprocessor


def _prices(periods: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=periods)
    close = pd.Series(10 * 1.002 ** np.arange(periods))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000 + np.arange(periods),
            "amount": (1_000 + np.arange(periods)) * close,
            "turnover": 0.01,
        }
    )


def test_alpha_library_exposes_required_factors_without_future_dependency():
    prices = _prices()
    benchmark = _prices().assign(close=lambda values: values["close"] * 1.001)
    engine = AlphaFactorEngine()
    short = engine.calculate(prices.iloc[:250], benchmark.iloc[:250])
    long = engine.calculate(prices, benchmark)

    assert set(ALPHA_FACTORS).issubset(long.columns)
    pd.testing.assert_series_equal(short["momentum_120"], long.loc[:249, "momentum_120"], check_names=False)
    assert long["momentum_20_5_skip"].notna().any()
    assert long["beta_60"].notna().any()


def test_market_breadth_and_labels_use_date_available_information():
    prices = _prices(140)
    benchmark = _prices(140)
    frames = []
    for number, code in enumerate(("000001", "000002", "000003"), start=1):
        frame = AlphaFactorEngine().calculate(prices.assign(close=prices["close"] * (1 + number * 0.0001)), benchmark)
        frame["code"] = code
        frame["industry"] = "A" if number < 3 else "B"
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    market = MarketBreadthEngine().calculate(panel)
    labelled = add_v5_forward_labels(panel, benchmark, horizons=(5, 20))

    assert set(MARKET_FEATURES).issubset(market.columns)
    assert {"future_return_5d", "future_excess_return_20d", "label_end_date_20d", "relevance_label_20d"}.issubset(labelled.columns)
    valid = labelled["label_end_date_20d"].notna()
    assert (labelled.loc[valid, "label_end_date_20d"] > labelled.loc[valid, "date"]).all()


def test_v5_preprocessor_retains_raw_factors_and_neutralizes_by_industry():
    dates = pd.to_datetime(["2024-01-02"] * 6)
    panel = pd.DataFrame(
        {
            "date": dates,
            "industry": ["A", "A", "A", "B", "B", "B"],
            "factor": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        }
    )
    output = V5FactorPreprocessor().process(panel, ["factor"])

    assert output["factor"].equals(panel["factor"])
    assert output["factor_processed"].notna().all()
    assert output.groupby("industry")["factor_processed"].mean().abs().max() < 1e-12
