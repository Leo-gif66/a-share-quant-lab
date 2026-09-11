"""v3.0 orchestration around the institutional portfolio backtest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..memory import TradeDecisionSnapshot, TradeMemoryStore
from ..regime import MarketRegimeDetector, RegimeAwareRiskOverlay
from ..research.decision_explanation import DecisionExplanationEngine
from .backtest import PortfolioBacktestResult
from .institutional import InstitutionalPortfolioBacktestEngine


@dataclass(frozen=True)
class IntelligentBacktestResult:
    """Institutional result plus auditable v3 decision-memory evidence."""

    backtest: PortfolioBacktestResult
    trades: pd.DataFrame
    regime_history: pd.DataFrame
    memory_path: Path

    @property
    def metrics(self) -> dict[str, float]:
        return self.backtest.metrics


class IntelligentPortfolioBacktestEngine(InstitutionalPortfolioBacktestEngine):
    """Add adaptive scores, point-in-time regimes, and trade memory to v2.7.

    The inherited implementation still owns rebalancing, transaction costs,
    equity accounting, benchmark construction, portfolio diagnostics, and the
    advanced risk overlay.  This class only supplies another score config,
    applies a regime exposure cap, and snapshots decisions after the fact.
    """

    def __init__(
        self,
        *args: object,
        memory_path: str | Path = "data/memory/trades.parquet",
        strategy_name: str = "intelligent_v3",
        regime_detector: MarketRegimeDetector | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory = TradeMemoryStore(memory_path)
        self.strategy_name = strategy_name
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self._breadth_by_date = pd.Series(dtype="float64")
        benchmark_path = self.raw_dir / f"{self.benchmark_code}.parquet"
        self._regime_benchmark_history = (
            pd.read_parquet(benchmark_path, columns=["date", "close"])
            if benchmark_path.exists()
            else None
        )
        self.regime_overlay = RegimeAwareRiskOverlay(
            self.risk,
            self.regime_detector,
            lambda date: self._breadth_for(date),
            self._regime_benchmark_history,
        )
        self.risk = self.regime_overlay
        self._signal_components = pd.DataFrame()
        self._price_lookup = pd.Series(dtype="float64")
        self._trading_dates = pd.DatetimeIndex([])
        self.explanations = DecisionExplanationEngine()

    def _load_feature_panel(self) -> pd.DataFrame:
        panel = super()._load_feature_panel()
        ordered = panel.loc[:, ["date", "code", "close", "open"]].copy().sort_values(["code", "date"])
        moving_average = ordered.groupby("code", sort=False)["close"].transform(
            lambda values: values.rolling(self.regime_detector.settings.breadth_window).mean()
        )
        ordered["above_short_ma"] = ordered["close"] > moving_average
        valid = moving_average.notna()
        self._breadth_by_date = (
            ordered.loc[valid].groupby("date")["above_short_ma"].mean().astype("float64").sort_index()
        )
        self._price_lookup = ordered.set_index(["date", "code"])["open"].astype("float64")
        self._trading_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        return panel

    def _score_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Produce normal composite scores and retain their named components."""
        processed = self.scorer.processor.process(panel)
        components = pd.DataFrame(
            {
                factor: processed[f"{factor}_z"] * spec.weight * spec.direction
                for factor, spec in self.scorer.processor.factor_specs.items()
            }
        )
        scores = processed.loc[:, ["date", "code"]].copy()
        scores["composite_score"] = components.sum(axis=1, min_count=len(components.columns))
        signals = scores.loc[:, ["date", "code", "composite_score"]].copy()
        for factor in components:
            signals[factor] = components[factor].to_numpy()
        scores = scores.sort_values(["date", "code"]).reset_index(drop=True)
        self._signal_components = signals.sort_values(["date", "code"]).reset_index(drop=True)
        target = self.features_dir / "composite_score.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(target, index=False)
        return scores

    def run(self) -> IntelligentBacktestResult:
        self.regime_overlay.history.clear()
        result = super().run()
        trades = self._record_trade_memory(result)
        history = pd.DataFrame(
            [
                {
                    "date": item.date,
                    "market_regime": item.state,
                    "exposure": item.exposure,
                    "trend": item.trend,
                    "breadth": item.breadth,
                    "volatility": item.volatility,
                }
                for item in self.regime_overlay.history
            ]
        )
        return IntelligentBacktestResult(result, trades, history, self.memory.path)

    def _record_trade_memory(self, result: PortfolioBacktestResult) -> pd.DataFrame:
        holdings = result.holdings_history.copy()
        if holdings.empty or self._signal_components.empty:
            return self.memory.load()
        holdings["date"] = pd.to_datetime(holdings["date"], errors="raise")
        holdings["code"] = holdings["code"].astype(str).str.zfill(6)
        entries = pd.DatetimeIndex(sorted(holdings["date"].unique()))
        score_dates = pd.DatetimeIndex(sorted(self._signal_components["date"].unique()))
        signal_for_entry = {
            entry: score_dates[score_dates < entry][-1]
            for entry in entries
            if len(score_dates[score_dates < entry])
        }
        exit_for_entry = {
            entries[index]: entries[index + 1] if index + 1 < len(entries) else pd.NaT
            for index in range(len(entries))
        }
        component_index = self._signal_components.set_index(["date", "code"])
        predictions = self._walk_forward_return_predictions()
        regimes = {
            item.date: item
            for item in self.regime_overlay.history
        }
        snapshots: list[TradeDecisionSnapshot] = []
        for row in holdings.itertuples(index=False):
            entry_date = pd.Timestamp(row.date)
            signal_date = signal_for_entry.get(entry_date)
            if signal_date is None or (signal_date, row.code) not in component_index.index:
                continue
            signal = component_index.loc[(signal_date, row.code)]
            exit_date = exit_for_entry[entry_date]
            entry_price = self._price(entry_date, row.code)
            exit_price = self._price(exit_date, row.code) if pd.notna(exit_date) else np.nan
            regime_dates = [date for date in regimes if date <= signal_date]
            regime_snapshot = regimes[max(regime_dates)] if regime_dates else None
            regime = regime_snapshot.state if regime_snapshot is not None else "sideways"
            factor_scores = {
                factor: float(signal[factor])
                for factor in self.scorer.processor.factor_names
                if pd.notna(signal[factor])
            }
            snapshots.append(
                TradeDecisionSnapshot(
                    date=entry_date,
                    symbol=row.code,
                    strategy=self.strategy_name,
                    model_score=float(predictions.get((signal_date, row.code), 0.0)),
                    factor_scores=factor_scores,
                    industry=str(row.sector),
                    market_regime=regime,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    exit_date=pd.Timestamp(exit_date) if pd.notna(exit_date) else None,
                    weight=float(row.weight),
                    market_volatility=(
                        float(regime_snapshot.volatility) if regime_snapshot is not None else None
                    ),
                    decision_reason=self.explanations.explain(
                        factor_scores,
                        regime,
                        float(regime_snapshot.exposure) if regime_snapshot is not None else 0.70,
                        float(row.exposure),
                    ),
                )
            )
        if snapshots:
            self.memory.upsert(snapshots)
        return self.memory.realize()

    def _walk_forward_return_predictions(self) -> dict[tuple[pd.Timestamp, str], float]:
        """Calibrate score to holding-period return using only matured history.

        Composite scores are cross-sectional ranks and are not themselves
        return forecasts.  For each rebalance, this fits an intercept and
        slope on prior score/return pairs whose simulated exit preceded the
        current signal date.  The resulting prediction and realized return
        therefore share units, making ``prediction_error`` meaningful without
        using future observations.
        """
        if self._signal_components.empty or len(self._trading_dates) < 2:
            return {}
        labels: list[pd.DataFrame] = []
        for index in range(0, len(self._trading_dates) - 1, self.rebalance_interval):
            exit_index = index + self.rebalance_interval + 1
            if exit_index >= len(self._trading_dates):
                continue
            signal_date = self._trading_dates[index]
            entry_date = self._trading_dates[index + 1]
            exit_date = self._trading_dates[exit_index]
            scores = self._signal_components.loc[
                self._signal_components["date"] == signal_date,
                ["code", "composite_score"],
            ].copy()
            entry = self._price_lookup.xs(entry_date, level="date")
            exit = self._price_lookup.xs(exit_date, level="date")
            scores["entry_price"] = scores["code"].map(entry)
            scores["exit_price"] = scores["code"].map(exit)
            scores["future_return"] = scores["exit_price"] / scores["entry_price"] - 1.0
            scores["signal_date"] = signal_date
            scores["exit_date"] = exit_date
            labels.append(scores.loc[:, ["signal_date", "exit_date", "code", "composite_score", "future_return"]])
        if not labels:
            return {}
        history = pd.concat(labels, ignore_index=True).dropna().sort_values("signal_date")
        predictions: dict[tuple[pd.Timestamp, str], float] = {}
        for signal_date, current in self._signal_components.groupby("date", sort=True):
            matured = history.loc[history["exit_date"] <= signal_date]
            x = matured["composite_score"].to_numpy(dtype=float)
            y = matured["future_return"].to_numpy(dtype=float)
            if len(matured) >= 30 and np.unique(x).size >= 2:
                coefficients, *_ = np.linalg.lstsq(
                    np.column_stack([np.ones(len(x)), x]), y, rcond=None
                )
                values = coefficients[0] + coefficients[1] * current["composite_score"].to_numpy(dtype=float)
            else:
                values = np.zeros(len(current), dtype=float)
            predictions.update(
                {
                    (pd.Timestamp(signal_date), str(code).zfill(6)): float(value)
                    for code, value in zip(current["code"], values, strict=True)
                }
            )
        return predictions

    def _breadth_for(self, date: pd.Timestamp) -> float:
        eligible = self._breadth_by_date.loc[self._breadth_by_date.index <= pd.Timestamp(date)]
        return float(eligible.iloc[-1]) if not eligible.empty else 0.5

    def _price(self, date: pd.Timestamp, code: str) -> float | None:
        value = self._price_lookup.get((pd.Timestamp(date), str(code).zfill(6)), np.nan)
        return float(value) if pd.notna(value) and value > 0 else None
