import pandas as pd
import pytest

from quant.memory import (
    ModelPredictionSnapshot,
    SignalSnapshot,
    TradeDecisionSnapshot,
    TradeMemoryStore,
)


def test_trade_memory_persists_required_decision_schema_and_realizes_return(tmp_path):
    store = TradeMemoryStore(tmp_path / "trades.parquet")
    signal = SignalSnapshot(
        date=pd.Timestamp("2024-01-02"),
        symbol="1",
        factor_scores={"momentum_5": 0.4},
        industry="Bank",
        market_regime="bull",
    )
    decision = TradeDecisionSnapshot.from_snapshots(
        signal,
        ModelPredictionSnapshot(
            date=pd.Timestamp("2024-01-02"), symbol="1", strategy="v3", model_score=0.05
        ),
        entry_price=10.0,
        exit_price=11.0,
        exit_date=pd.Timestamp("2024-01-22"),
        weight=0.1,
    )

    store.upsert(decision)
    trades = store.realize()

    assert {
        "date", "symbol", "strategy", "model_score", "factor_scores", "industry",
        "market_regime", "entry_price", "exit_price", "future_return", "prediction_error",
    }.issubset(trades.columns)
    assert trades.loc[0, "symbol"] == "000001"
    assert trades.loc[0, "factor_scores"] == {"momentum_5": 0.4}
    assert trades.loc[0, "future_return"] == pytest.approx(0.1)
    assert trades.loc[0, "prediction_error"] == pytest.approx(0.05)


def test_trade_memory_can_fill_a_pending_exit_from_prices(tmp_path):
    store = TradeMemoryStore(tmp_path / "trades.parquet")
    store.upsert(
        TradeDecisionSnapshot(
            date=pd.Timestamp("2024-01-02"),
            symbol="000001",
            strategy="v3",
            model_score=0.0,
            entry_price=10.0,
            exit_date=pd.Timestamp("2024-01-22"),
        )
    )
    trades = store.realize(
        pd.DataFrame({"date": [pd.Timestamp("2024-01-22")], "code": ["000001"], "close": [12.0]})
    )

    assert trades.loc[0, "exit_price"] == 12.0
    assert trades.loc[0, "future_return"] == pytest.approx(0.2)


def test_legacy_trade_memory_gets_an_evidence_based_decision_reason(tmp_path):
    path = tmp_path / "trades.parquet"
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "symbol": ["000001"],
            "strategy": ["legacy"],
            "model_score": [0.01],
            "factor_scores": ['{"momentum_5":0.2}'],
            "industry": ["Bank"],
            "market_regime": ["bear"],
            "entry_price": [10.0],
            "exit_price": [11.0],
            "future_return": [0.1],
            "prediction_error": [0.09],
            "exit_date": [pd.Timestamp("2024-01-22")],
            "weight": [0.1],
        }
    ).to_parquet(path, index=False)

    reason = TradeMemoryStore(path).load().loc[0, "decision_reason"]

    assert "factor contribution" in reason
    assert "regime adjustment" in reason
