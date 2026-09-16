"""True V5 walk-forward validation with fold-local factor selection and ensembles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import AlphaEnsemble, RankingModel, prediction_ic_metrics
from .factor_selection import FactorSelector
from .factor_v5 import V5FactorResearchEngine
from .leakage import assert_training_before_test
from .portfolio_v5 import V5PortfolioEvaluator


@dataclass(frozen=True)
class V5WalkForwardSettings:
    train_years: int = 3
    validation_months: int = 12
    test_months: int = 6
    step_months: int = 6
    horizon: int = 20
    top_n: int = 20
    rebalance_frequency: int = 20
    weighting: str = "equal_weight"
    industry_limit: float = 0.20
    max_stock_weight: float = 0.10
    risk_overlay: str = "v4_baseline"
    model: str = "linear"
    seed: int = 42
    max_factors: int = 12

    def __post_init__(self) -> None:
        if min(self.train_years, self.validation_months, self.test_months, self.step_months, self.horizon, self.top_n, self.rebalance_frequency, self.max_factors) < 1:
            raise ValueError("V5 walk-forward windows and portfolio settings must be positive")


@dataclass(frozen=True)
class V5WalkForwardResult:
    folds: pd.DataFrame
    selections: pd.DataFrame
    feature_importance: pd.DataFrame
    holdings: pd.DataFrame
    scores: pd.DataFrame


class V5WalkForwardRunner:
    """Freeze factor selection, model, and ensemble weights before each OOS test fold."""

    def __init__(self, settings: V5WalkForwardSettings | None = None) -> None:
        self.settings = settings or V5WalkForwardSettings()

    def run(
        self,
        panel: pd.DataFrame,
        factor_columns: list[str],
        precomputed_selections: pd.DataFrame | None = None,
    ) -> V5WalkForwardResult:
        """Run chronological folds using only labels matured before each subsequent window.

        ``precomputed_selections`` is limited to reusing selections previously
        frozen from the *same fold's* training window when comparing model
        classes.  It never supplies validation or test-period outcomes.
        """
        label = f"future_excess_return_{self.settings.horizon}d"
        label_end = f"label_end_date_{self.settings.horizon}d"
        required = {"date", label, label_end, "sector", "market_regime", *factor_columns}
        missing = required.difference(panel.columns)
        if missing:
            raise ValueError(f"V5 walk-forward panel missing: {', '.join(sorted(missing))}")
        frame = panel.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame[label_end] = pd.to_datetime(frame[label_end], errors="coerce").dt.normalize()
        folds: list[dict[str, object]] = []
        selections: list[pd.DataFrame] = []
        importances: list[pd.DataFrame] = []
        holdings: list[pd.DataFrame] = []
        scored_tests: list[pd.DataFrame] = []
        for fold_number, bounds in enumerate(self._periods(frame["date"]), start=1):
            train, validation, test = self._split(frame, bounds, label_end)
            assert_training_before_test(train["date"], test["date"])
            if precomputed_selections is None:
                selected, selection = self._select(train, factor_columns)
            else:
                selected, selection = self._reuse_selection(
                    precomputed_selections, fold_number, factor_columns
                )
            selection["fold"] = fold_number
            selections.append(selection)
            factor_weights = self._factor_weights(train, selected, label)
            validation_scores = _factor_composite(validation, factor_weights)
            test_scores = _factor_composite(test, factor_weights)
            model = RankingModel(self.settings.model, self._model_parameters()).fit(train, selected, label)
            validation = validation.copy()
            test = test.copy()
            validation["factor_composite"] = validation_scores
            test["factor_composite"] = test_scores
            validation["ml_prediction"] = model.predict(validation)
            test["ml_prediction"] = model.predict(test)
            ensemble = AlphaEnsemble("validation_weighted")
            ensemble_weights = ensemble.fit(validation, ["factor_composite", "ml_prediction"], label)
            test["ensemble_score"] = ensemble.predict(test)
            validation_prediction = prediction_ic_metrics(validation, validation["ml_prediction"], label)
            test_prediction = prediction_ic_metrics(test, test["ml_prediction"], label)
            portfolio = V5PortfolioEvaluator().evaluate(
                test,
                horizon=self.settings.horizon,
                top_n=self.settings.top_n,
                rebalance_frequency=self.settings.rebalance_frequency,
                weighting=self.settings.weighting,
                industry_limit=self.settings.industry_limit,
                max_stock_weight=self.settings.max_stock_weight,
                risk_overlay=self.settings.risk_overlay,
            )
            held = portfolio.holdings.copy()
            held["fold"] = fold_number
            holdings.append(held)
            importance = model.feature_importance()
            importance["fold"] = fold_number
            importances.append(importance)
            # Retain the audit fields needed to reproduce portfolio experiments
            # without keeping every raw and factor column from each test fold.
            # This sharply reduces the working set for the 1,728-trial matrix.
            score_columns = [
                "date", "code", "industry", "sector", "market_regime", "volatility_regime",
                "factor_composite", "ml_prediction", "ensemble_score", label, label_end,
                f"future_return_{self.settings.horizon}d",
                f"benchmark_future_return_{self.settings.horizon}d",
                "amount_20", "realized_vol_20",
            ]
            scored_tests.append(test.loc[:, [column for column in score_columns if column in test]].copy())
            benchmark_return = _benchmark_return(portfolio.results, self.settings.rebalance_frequency)
            folds.append(
                {
                    "fold": fold_number,
                    "train_start": bounds["train_start"], "train_end": bounds["train_end"],
                    "validation_start": bounds["validation_start"], "validation_end": bounds["validation_end"],
                    "test_start": bounds["test_start"], "test_end": bounds["test_end"],
                    "selected_factors": ",".join(selected), "model": self.settings.model,
                    "portfolio_parameters": str(asdict(self.settings)), "ensemble_weights": str(ensemble_weights.weights),
                    "return": portfolio.metrics["annual_return"], "benchmark_return": benchmark_return,
                    "alpha": portfolio.metrics["alpha"], "sharpe": portfolio.metrics["sharpe"],
                    "drawdown": portfolio.metrics["max_drawdown"], "turnover": portfolio.metrics["turnover"],
                    "information_ratio": portfolio.metrics["information_ratio"],
                    "prediction_IC": test_prediction["prediction_IC"], "prediction_Rank_IC": test_prediction["prediction_Rank_IC"],
                    "validation_prediction_IC": validation_prediction["prediction_IC"],
                    "validation_prediction_Rank_IC": validation_prediction["prediction_Rank_IC"],
                    "rebalances": len(portfolio.results),
                }
            )
        return V5WalkForwardResult(
            pd.DataFrame(folds),
            pd.concat(selections, ignore_index=True) if selections else pd.DataFrame(),
            pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(),
            pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame(),
            pd.concat(scored_tests, ignore_index=True) if scored_tests else pd.DataFrame(),
        )

    def save(self, result: V5WalkForwardResult, output_path: str | Path = "research/results/walk_forward_v5.parquet") -> Path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        result.folds.to_parquet(target, index=False)
        return target

    def _periods(self, values: pd.Series) -> list[dict[str, pd.Timestamp]]:
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(values, errors="raise").unique()))
        if dates.empty:
            return []
        cursor = dates.min()
        last = dates.max()
        periods: list[dict[str, pd.Timestamp]] = []
        while True:
            validation_start = cursor + pd.DateOffset(years=self.settings.train_years)
            test_start = validation_start + pd.DateOffset(months=self.settings.validation_months)
            test_end = test_start + pd.DateOffset(months=self.settings.test_months)
            train_dates = dates[(dates >= cursor) & (dates < validation_start)]
            validation_dates = dates[(dates >= validation_start) & (dates < test_start)]
            test_dates = dates[(dates >= test_start) & (dates < test_end)]
            if train_dates.empty or validation_dates.empty or test_dates.empty:
                break
            periods.append(
                {
                    "train_start": train_dates.min(), "train_end": train_dates.max(),
                    "validation_start": validation_dates.min(), "validation_end": validation_dates.max(),
                    "test_start": test_dates.min(), "test_end": test_dates.max(),
                }
            )
            cursor = cursor + pd.DateOffset(months=self.settings.step_months)
            if cursor >= last:
                break
        return periods

    @staticmethod
    def _split(
        frame: pd.DataFrame, bounds: dict[str, pd.Timestamp], label_end: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = frame.loc[(frame["date"] >= bounds["train_start"]) & (frame["date"] <= bounds["train_end"]) & (frame[label_end] < bounds["validation_start"])].copy()
        validation = frame.loc[(frame["date"] >= bounds["validation_start"]) & (frame["date"] <= bounds["validation_end"]) & (frame[label_end] < bounds["test_start"])].copy()
        test = frame.loc[(frame["date"] >= bounds["test_start"]) & (frame["date"] <= bounds["test_end"]) & (frame[label_end] <= bounds["test_end"])].copy()
        if train.empty or validation.empty or test.empty:
            raise ValueError("a V5 walk-forward fold has an empty matured partition")
        return train, validation, test

    def _select(self, train: pd.DataFrame, factors: list[str]) -> tuple[list[str], pd.DataFrame]:
        research = V5FactorResearchEngine(min_cross_section=20).evaluate(train, factors, horizons=(self.settings.horizon,)).summary
        selection = FactorSelector(max_dates=80).select(train, research, horizon=self.settings.horizon).selection
        selected = selection.loc[selection["selected"] & selection["selection_score"].notna()].sort_values("selection_score", ascending=False)["factor"].head(self.settings.max_factors).tolist()
        if not selected:
            raise RuntimeError("no factors survived fold-local selection")
        return selected, selection

    def _reuse_selection(
        self, selections: pd.DataFrame, fold_number: int, factors: list[str]
    ) -> tuple[list[str], pd.DataFrame]:
        required = {"fold", "factor", "selected", "selection_score"}
        missing = required.difference(selections.columns)
        if missing:
            raise ValueError(f"precomputed selections missing: {', '.join(sorted(missing))}")
        selection = selections.loc[selections["fold"] == fold_number].copy()
        if selection.empty:
            raise ValueError(f"precomputed selections have no fold {fold_number}")
        valid_factors = set(factors)
        selected = (
            selection.loc[selection["selected"].astype(bool) & selection["factor"].isin(valid_factors)]
            .sort_values("selection_score", ascending=False)["factor"]
            .head(self.settings.max_factors)
            .tolist()
        )
        if not selected:
            raise RuntimeError(f"no frozen training-only factors available for fold {fold_number}")
        selection["selection_source"] = "reused_training_only"
        return selected, selection

    def _factor_weights(self, train: pd.DataFrame, selected: list[str], label: str) -> dict[str, float]:
        weights: dict[str, float] = {}
        for factor in selected:
            correlations = [values[factor].corr(values[label], method="spearman") for _, values in train.groupby("date", sort=True) if values[factor].notna().sum() >= 20]
            weights[factor] = float(np.nanmean(correlations))
        total = sum(abs(value) for value in weights.values())
        if not np.isfinite(total) or total == 0:
            raise RuntimeError("fold-local factor weights are not finite")
        return {factor: value / total for factor, value in weights.items()}

    def _model_parameters(self) -> dict[str, object]:
        if self.settings.model == "random_forest":
            return {"random_state": self.settings.seed, "n_estimators": 100, "n_jobs": 1}
        if self.settings.model == "lightgbm":
            return {"random_state": self.settings.seed, "n_estimators": 100, "n_jobs": 1}
        return {}


def _factor_composite(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    output = pd.Series(0.0, index=frame.index, dtype="float64")
    for factor, weight in weights.items():
        values = pd.to_numeric(frame[factor], errors="coerce")
        output += values.groupby(pd.to_datetime(frame["date"], errors="raise"), sort=False).rank(pct=True).fillna(0.5) * weight
    return output.rename("factor_composite")


def _benchmark_return(results: pd.DataFrame, rebalance_frequency: int) -> float:
    values = results["benchmark_return"].dropna()
    return float((1 + values).prod() ** (252 / (rebalance_frequency * len(values))) - 1) if not values.empty else np.nan
