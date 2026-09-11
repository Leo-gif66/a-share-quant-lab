import pandas as pd

from quant.memory import TradeMemoryStore
from quant.research import LiveSimulationEngine


def test_live_simulation_retrains_only_on_returns_matured_before_each_signal(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=12)
    codes = ["000001", "000002", "000003"]
    scores = pd.concat(
        [
            pd.DataFrame(
                {"date": dates, "code": code, "composite_score": rank + pd.Series(range(len(dates))) * 0.01}
            )
            for rank, code in enumerate(codes, start=1)
        ],
        ignore_index=True,
    )
    prices = pd.concat(
        [
            pd.DataFrame(
                {"date": dates, "code": code, "open": 10 + rank + pd.Series(range(len(dates))) * rank * 0.02}
            )
            for rank, code in enumerate(codes, start=1)
        ],
        ignore_index=True,
    )
    memory = TradeMemoryStore(tmp_path / "trades.parquet")
    result = LiveSimulationEngine(rebalance_interval=2, retrain_interval=1, top_n=2).run(
        scores,
        prices,
        memory=memory,
        industries={code: "Test" for code in codes},
        output_path=tmp_path / "live_simulation.parquet",
    )

    trained = result.snapshots.dropna(subset=["training_end_date"])
    assert result.output_path.exists()
    assert not result.snapshots.empty
    assert (pd.to_datetime(trained["training_end_date"]) < pd.to_datetime(trained["date"])).all()
    assert result.snapshots["retrained"].any()
    assert memory.load()["decision_reason"].str.contains("factor contribution").all()
