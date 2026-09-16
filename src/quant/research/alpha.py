"""Statistically disciplined factor discovery, neutralization, and validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..data.universe import Universe
from ..factors.advanced import AdvancedFactorEngine
from .preprocessing import ResearchPreprocessor

RESEARCH_FACTORS: tuple[str, ...] = (
    "momentum_5", "momentum_20", "momentum_60", "momentum_120",
    "trend_20", "trend_60", "trend_120",
    "volatility_20", "volatility_60", "atr_14", "rsi_14", "macd",
    "turnover_5", "turnover_20", "amount_change", "volume_ratio",
    "beta_60", "beta_120", "max_drawdown_60", "max_drawdown_120",
)


@dataclass(frozen=True)
class FactorResearchResult:
    summary: pd.DataFrame
    daily: pd.DataFrame
    yearly_stability: pd.DataFrame
    neutralized_panel: pd.DataFrame
    factor_coverage: pd.DataFrame


class FactorNeutralizer:
    """Cross-sectionally clip, neutralize, and z-score factor observations."""

    def __init__(self, sigma: float = 3.0, min_observations: int = 5) -> None:
        if sigma <= 0 or min_observations < 2:
            raise ValueError("sigma must be positive and min_observations at least two")
        self.sigma = sigma
        self.min_observations = min_observations
        self.preprocessor = ResearchPreprocessor(sigma=sigma)
        self.last_coverage = pd.DataFrame(columns=ResearchPreprocessor.COVERAGE_COLUMNS)

    def transform(
        self,
        panel: pd.DataFrame,
        factors: Sequence[str],
        industry_column: str = "industry",
        market_cap_column: str = "market_cap",
        require_market_cap: bool = False,
    ) -> pd.DataFrame:
        required = {"date", *factors}
        if not required.issubset(panel.columns):
            raise ValueError(f"factor panel missing columns: {', '.join(sorted(required.difference(panel.columns)))}")
        if require_market_cap and market_cap_column not in panel:
            raise ValueError("market-cap neutralization requires a point-in-time market_cap column")
        cleaned = self.preprocessor.process(panel, factors)
        values = cleaned.data
        self.last_coverage = cleaned.coverage
        values["date"] = pd.to_datetime(values["date"], errors="raise")
        if industry_column not in values:
            values[industry_column] = "Unknown"
        values[industry_column] = values[industry_column].fillna("Unknown").astype(str)
        if market_cap_column in values:
            values[market_cap_column] = pd.to_numeric(values[market_cap_column], errors="coerce")
        for factor in factors:
            values[factor] = pd.to_numeric(values[factor], errors="coerce")
            values[f"{factor}_neutralized"] = np.nan

        neutral_columns = [f"{factor}_neutralized" for factor in factors]
        for _, group in values.groupby("date", sort=False):
            indices = group.index
            # The design matrix is identical for every factor in one cross
            # section.  Solve all usable factor columns at once rather than
            # performing one QR decomposition per factor.  This preserves
            # the same industry/cap residualization while making full-universe
            # research practical.
            neutralized = self._neutralize_group(
                group, factors, industry_column, market_cap_column
            )
            values.loc[indices, neutral_columns] = neutralized.to_numpy()
        return values

    def _neutralize_group(
        self,
        group: pd.DataFrame,
        factors: Sequence[str],
        industry_column: str,
        market_cap_column: str,
    ) -> pd.DataFrame:
        """Return z-scored residuals for a single rebalance cross-section.

        The shared preprocessor has already imputed partial factor gaps within
        each date.  A factor that remains incomplete therefore has no usable
        observations at that date and is left as ``NaN``; all remaining factor
        columns can safely share one neutralization design matrix.
        """
        output_columns = [f"{factor}_neutralized" for factor in factors]
        output = pd.DataFrame(np.nan, index=group.index, columns=output_columns)
        usable = [
            factor
            for factor in factors
            if group[factor].notna().sum() >= self.min_observations
            and group[factor].notna().all()
        ]
        if not usable:
            return output

        columns: list[np.ndarray] = [np.ones(len(group), dtype=float)]
        industry = group[industry_column]
        dummies = pd.get_dummies(industry, dtype=float)
        if dummies.shape[1] > 1:
            columns.extend(dummies.iloc[:, 1:].to_numpy().T)
        if market_cap_column in group:
            cap = pd.to_numeric(group[market_cap_column], errors="coerce")
            valid_cap = cap > 0
            if valid_cap.all():
                columns.append(np.log(cap).to_numpy())
            elif valid_cap.any():
                columns.append(np.log(cap.where(valid_cap, cap[valid_cap].median())).to_numpy())

        design = np.column_stack(columns)
        response = group.loc[:, usable].to_numpy(dtype=float)
        if len(group) <= design.shape[1]:
            residuals = response - np.nanmean(response, axis=0, keepdims=True)
        else:
            coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
            residuals = response - design @ coefficients
        standard_deviation = residuals.std(axis=0, ddof=0)
        normalized = np.divide(
            residuals - residuals.mean(axis=0, keepdims=True),
            standard_deviation,
            out=np.zeros_like(residuals),
            where=standard_deviation > 0,
        )
        output.loc[:, [f"{factor}_neutralized" for factor in usable]] = normalized
        return output

    def _residualize(
        self,
        factor: pd.Series,
        group: pd.DataFrame,
        industry_column: str,
        market_cap_column: str,
    ) -> pd.Series:
        valid = factor.notna()
        if valid.sum() < self.min_observations:
            return pd.Series(np.nan, index=group.index, dtype="float64")
        y = factor.loc[valid].astype(float)
        columns = [np.ones(len(y))]
        industry = group.loc[valid, industry_column]
        dummies = pd.get_dummies(industry, dtype=float)
        if dummies.shape[1] > 1:
            columns.extend(dummies.iloc[:, 1:].to_numpy().T)
        if market_cap_column in group:
            cap = pd.to_numeric(group.loc[valid, market_cap_column], errors="coerce")
            usable = cap > 0
            if usable.all():
                columns.append(np.log(cap).to_numpy())
            elif usable.any():
                # Retain all rows and impute only the cross-sectional median;
                # a missing cap does not become a liquidity proxy.
                columns.append(np.log(cap.where(usable, cap[usable].median())).to_numpy())
        design = np.column_stack(columns)
        if len(y) <= design.shape[1]:
            # A saturated dummy model has no reliable residual degrees of
            # freedom.  Use the demeaned signal and preserve this limitation.
            result = y - y.mean()
        else:
            coefficients, *_ = np.linalg.lstsq(design, y.to_numpy(), rcond=None)
            result = pd.Series(y.to_numpy() - design @ coefficients, index=y.index)
        output = pd.Series(np.nan, index=group.index, dtype="float64")
        output.loc[result.index] = result
        return output


class ProfessionalFactorEvaluator:
    """Evaluate arbitrary factors against a fixed-horizon cross-sectional label."""

    SUMMARY_COLUMNS = (
        "factor", "IC", "Rank_IC", "IC_mean", "IC_std", "ICIR", "t_stat",
        "annualized_long_short_return", "stability", "observations", "coverage_pct",
        "nan_ratio", "valid_samples", "IC_sample_count", "status",
    )

    def __init__(
        self,
        horizon: int = 20,
        min_cross_section: int = 20,
        quantile: float = 0.2,
        min_ic_samples: int = 20,
    ) -> None:
        if horizon < 1 or min_cross_section < 2 or min_ic_samples < 2 or not 0 < quantile <= 0.5:
            raise ValueError("invalid factor evaluation settings")
        self.horizon = horizon
        self.min_cross_section = min_cross_section
        self.quantile = quantile
        self.min_ic_samples = min_ic_samples
        self.preprocessor = ResearchPreprocessor()

    def evaluate(
        self, panel: pd.DataFrame, factors: Sequence[str], target: str = "future_excess_return_20d"
    ) -> FactorResearchResult:
        if not {"date", target, *factors}.issubset(panel.columns):
            raise ValueError("factor evaluation panel is missing required columns")
        cleaned = self.preprocessor.process(panel, factors, target_columns=(target,))
        frame = cleaned.data
        daily_records: list[dict[str, object]] = []
        for factor in factors:
            values = pd.to_numeric(frame[factor], errors="coerce")
            for date, group in pd.DataFrame({"date": frame["date"], "factor": values, "target": frame[target]}).groupby("date"):
                group = group.dropna()
                if len(group) < self.min_cross_section or group["factor"].nunique() < 2 or group["target"].nunique() < 2:
                    continue
                ic = group["factor"].corr(group["target"], method="pearson")
                rank_ic = group["factor"].corr(group["target"], method="spearman")
                size = max(1, int(np.ceil(len(group) * self.quantile)))
                ordered = group.sort_values("factor", kind="stable")
                long_short = float(ordered.tail(size)["target"].mean() - ordered.head(size)["target"].mean())
                daily_records.append(
                    {
                        "date": date, "factor": factor, "IC": float(ic), "Rank_IC": float(rank_ic),
                        "long_short_return": long_short, "observations": len(group),
                    }
                )
        daily = pd.DataFrame(daily_records, columns=["date", "factor", "IC", "Rank_IC", "long_short_return", "observations"])
        summary, yearly = self._summarize(daily, factors, cleaned.coverage)
        return FactorResearchResult(summary, daily, yearly, frame, cleaned.coverage)

    def _summarize(
        self, daily: pd.DataFrame, factors: Sequence[str], coverage: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        records: list[dict[str, object]] = []
        yearly_records: list[dict[str, object]] = []
        for factor in factors:
            series = daily.loc[daily["factor"] == factor].copy()
            ic = series["IC"].dropna()
            coverage_row = coverage.loc[coverage["factor"] == factor]
            coverage_values = coverage_row.iloc[0] if not coverage_row.empty else None
            sufficient = len(ic) >= self.min_ic_samples
            ic_mean = float(ic.mean()) if sufficient else np.nan
            ic_std = float(ic.std(ddof=1)) if sufficient and len(ic) > 1 else np.nan
            icir = ic_mean / ic_std if pd.notna(ic_std) and ic_std > 0 else np.nan
            t_stat = ic_mean / (ic_std / np.sqrt(len(ic))) if pd.notna(ic_std) and ic_std > 0 else np.nan
            long_short = series["long_short_return"].dropna()
            mean_spread = float(long_short.mean()) if sufficient and not long_short.empty else np.nan
            annualized = (
                float((1 + mean_spread) ** (252 / self.horizon) - 1)
                if pd.notna(mean_spread) and mean_spread > -1
                else np.nan
            )
            years = series.assign(year=series["date"].dt.year).groupby("year", as_index=False).agg(
                IC_mean=("IC", "mean"), Rank_IC_mean=("Rank_IC", "mean"), observations=("IC", "size")
            )
            direction = np.sign(ic_mean)
            stability = (
                float((np.sign(years["IC_mean"].dropna()) == direction).mean())
                if direction != 0 and not years.empty
                else np.nan
            )
            for year_row in years.itertuples(index=False):
                yearly_records.append(
                    {
                        "factor": factor, "year": int(year_row.year), "IC_mean": year_row.IC_mean,
                        "Rank_IC_mean": year_row.Rank_IC_mean, "observations": year_row.observations,
                    }
                )
            records.append(
                {
                    "factor": factor, "IC": ic_mean,
                    "Rank_IC": float(series["Rank_IC"].mean()) if sufficient else np.nan,
                    "IC_mean": ic_mean, "IC_std": ic_std, "ICIR": icir, "t_stat": t_stat,
                    "annualized_long_short_return": annualized, "stability": stability,
                    "observations": len(ic),
                    "coverage_pct": float(coverage_values["coverage_pct"])
                    if coverage_values is not None else 0.0,
                    "nan_ratio": float(coverage_values["nan_ratio"])
                    if coverage_values is not None else 1.0,
                    "valid_samples": int(coverage_values["valid_samples"])
                    if coverage_values is not None else 0,
                    "IC_sample_count": len(ic),
                    "status": "ok" if sufficient else "insufficient_data",
                }
            )
        return (
            pd.DataFrame(records, columns=self.SUMMARY_COLUMNS),
            pd.DataFrame(yearly_records, columns=["factor", "year", "IC_mean", "Rank_IC_mean", "observations"]),
        )


class ProfessionalFactorResearchPipeline:
    """Build labelled factor panels directly from raw prices and feature stores."""

    def __init__(
        self,
        features_dir: str | Path = "data/features",
        raw_dir: str | Path = "data/raw",
        benchmark_code: str = "000300",
        universe: Universe | None = None,
        horizon: int = 20,
        min_cross_section: int = 20,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.raw_dir = Path(raw_dir)
        self.benchmark_code = str(benchmark_code).zfill(6)
        self.universe = universe or Universe("configs/universe_large.yaml")
        self.horizon = horizon
        self.neutralizer = FactorNeutralizer()
        self.evaluator = ProfessionalFactorEvaluator(horizon, min_cross_section)

    def build_panel(self, market_caps: pd.DataFrame | None = None) -> pd.DataFrame:
        benchmark = pd.read_parquet(self.raw_dir / f"{self.benchmark_code}.parquet")
        if not {"date", "close"}.issubset(benchmark.columns):
            raise ValueError("benchmark requires date and close")
        benchmark = benchmark.loc[:, ["date", "close"]].copy()
        benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise")
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
        benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
        benchmark["market_return"] = benchmark["close"].pct_change()
        benchmark["benchmark_future_return_20d"] = (
            benchmark["close"].shift(-self.horizon) / benchmark["close"] - 1.0
        )
        sectors = {
            str(stock["code"]).zfill(6): str(stock.get("industry") or stock.get("sector") or "Unknown")
            for stock in self.universe.stocks()
        }
        frames: list[pd.DataFrame] = []
        paths = sorted(path for path in self.raw_dir.glob("*.parquet") if path.stem.isdigit() and path.stem != self.benchmark_code)
        if len(paths) < self.evaluator.min_cross_section:
            raise RuntimeError(
                f"only {len(paths)} stock histories are available; at least "
                f"{self.evaluator.min_cross_section} are required for cross-sectional research"
            )
        for path in paths:
            raw = pd.read_parquet(path)
            required = {"date", "close", "high", "low", "volume", "amount", "turnover"}
            if not required.issubset(raw.columns):
                continue
            advanced = AdvancedFactorEngine().calculate(raw, benchmark)
            # AdvancedFactorEngine intentionally keeps its legacy liquidity
            # fields; retain raw daily values for the distinct v2.1 windows.
            raw_liquidity = raw.loc[:, ["date", "turnover", "amount"]].copy()
            raw_liquidity["date"] = pd.to_datetime(raw_liquidity["date"], errors="raise")
            raw_liquidity = raw_liquidity.sort_values("date").drop_duplicates("date", keep="last").rename(
                columns={"turnover": "raw_turnover", "amount": "raw_amount"}
            )
            advanced = advanced.merge(
                raw_liquidity, on="date", how="left"
            ).merge(
                benchmark.loc[:, ["date", "market_return"]], on="date", how="left"
            )
            frame = _research_factor_columns(advanced)
            frame["code"] = path.stem
            frame["industry"] = sectors.get(path.stem, "Unknown")
            frame["future_return_20d"] = frame["close"].shift(-self.horizon) / frame["close"] - 1.0
            frames.append(frame)
        if not frames:
            raise RuntimeError("no raw price files with the required OHLCV and turnover fields")
        panel = pd.concat(frames, ignore_index=True).merge(
            benchmark.loc[:, ["date", "benchmark_future_return_20d"]], on="date", how="left"
        )
        panel["future_excess_return_20d"] = panel["future_return_20d"] - panel["benchmark_future_return_20d"]
        if market_caps is not None:
            if not {"date", "code", "market_cap"}.issubset(market_caps.columns):
                raise ValueError("market_caps requires date, code, and market_cap")
            caps = market_caps.loc[:, ["date", "code", "market_cap"]].copy()
            caps["date"] = pd.to_datetime(caps["date"], errors="raise")
            caps["code"] = caps["code"].astype(str).str.zfill(6)
            panel = panel.merge(caps, on=["date", "code"], how="left", validate="one_to_one")
        return panel.sort_values(["date", "code"]).reset_index(drop=True)

    def run(
        self,
        output_dir: str | Path = "research/results",
        market_caps: pd.DataFrame | None = None,
        require_market_cap: bool = False,
    ) -> FactorResearchResult:
        panel = self.build_panel(market_caps)
        neutral = self.neutralizer.transform(
            panel, RESEARCH_FACTORS, require_market_cap=require_market_cap
        )
        neutral_factors = [f"{factor}_neutralized" for factor in RESEARCH_FACTORS]
        result = self.evaluator.evaluate(neutral, neutral_factors)
        if result.daily.empty:
            raise RuntimeError(
                f"no date has the required {self.evaluator.min_cross_section}-stock cross-section; "
                "download/build more universe members before treating factor results as research"
            )
        result.summary["factor"] = result.summary["factor"].str.removesuffix("_neutralized")
        result.daily["factor"] = result.daily["factor"].str.removesuffix("_neutralized")
        result.yearly_stability["factor"] = result.yearly_stability["factor"].str.removesuffix("_neutralized")
        result.factor_coverage["factor"] = result.factor_coverage["factor"].str.removesuffix(
            "_neutralized"
        )
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        result.summary.to_csv(directory / "factor_research_summary.csv", index=False)
        result.daily.to_csv(directory / "factor_daily_ic.csv", index=False)
        result.yearly_stability.to_csv(directory / "factor_stability.csv", index=False)
        result.factor_coverage.to_csv(directory / "factor_coverage.csv", index=False)
        return result


class FactorCombinationResearch:
    """Compare rolling equal, IC, ICIR, regression, and ML factor weights."""

    METHODS = ("equal", "ic", "icir", "regression", "ml")

    def __init__(
        self,
        horizon: int = 20,
        rolling_window: int = 252,
        reweight_interval: int = 20,
        ml_reweight_interval: int = 60,
        max_ml_samples: int = 20_000,
        ml_estimators: int = 20,
    ) -> None:
        if min(
            horizon,
            rolling_window,
            reweight_interval,
            ml_reweight_interval,
            max_ml_samples,
            ml_estimators,
        ) < 1:
            raise ValueError("combination windows must be positive")
        self.horizon = horizon
        self.rolling_window = rolling_window
        self.reweight_interval = reweight_interval
        self.ml_reweight_interval = ml_reweight_interval
        self.max_ml_samples = max_ml_samples
        self.ml_estimators = ml_estimators

    def compare(
        self,
        panel: pd.DataFrame,
        factors: Sequence[str],
        evaluation: FactorResearchResult,
        target: str = "future_excess_return_20d",
        collect_scores: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not {"date", "code", target, *factors}.issubset(panel.columns):
            raise ValueError("combination panel is missing required columns")
        values = panel.loc[:, ["date", "code", target, *factors]].copy()
        values["date"] = pd.to_datetime(values["date"], errors="raise")
        for factor in factors:
            values[factor] = pd.to_numeric(values[factor], errors="coerce")
        dates = pd.DatetimeIndex(sorted(values["date"].unique()))
        summary = evaluation.summary.set_index("factor")
        daily = evaluation.daily.copy()
        daily["factor"] = daily["factor"].map(lambda name: f"{name}_neutralized")
        weights_by_method = {method: np.repeat(1 / len(factors), len(factors)) for method in self.METHODS}
        score_frames: list[pd.DataFrame] = []
        weight_records: list[dict[str, object]] = []
        last_ml_reweight = -self.ml_reweight_interval
        for index, date in enumerate(dates):
            if index % self.reweight_interval == 0:
                cutoff = dates[max(0, index - self.horizon)]
                history = values.loc[(values["date"] <= cutoff) & (values["date"] < date)].tail(
                    self.rolling_window * max(1, values["code"].nunique())
                )
                weights_by_method["ic"] = self._ic_weights(daily, cutoff, factors, False)
                weights_by_method["icir"] = self._ic_weights(daily, cutoff, factors, True)
                weights_by_method["regression"] = self._regression_weights(history, factors, target)
                if index - last_ml_reweight >= self.ml_reweight_interval:
                    weights_by_method["ml"] = self._ml_weights(
                        history,
                        factors,
                        target,
                        max_samples=self.max_ml_samples,
                        estimators=self.ml_estimators,
                    )
                    last_ml_reweight = index
            day = values.loc[values["date"] == date]
            for method, weights in weights_by_method.items():
                if collect_scores:
                    factor_score = sum(day[factor].fillna(0.0) * weight for factor, weight in zip(factors, weights, strict=True))
                    score_frames.append(pd.DataFrame({"date": date, "code": day["code"], "method": method, "factor_score": factor_score}))
                for factor, weight in zip(factors, weights, strict=True):
                    base_factor = factor.removesuffix("_neutralized")
                    weight_records.append(
                        {
                            "date": date, "method": method, "factor": base_factor, "weight": float(weight),
                            "IC": summary.at[base_factor, "IC"] if base_factor in summary.index else np.nan,
                            "stability": summary.at[base_factor, "stability"] if base_factor in summary.index else np.nan,
                        }
                    )
        return (
            pd.concat(score_frames, ignore_index=True)
            if score_frames
            else pd.DataFrame(columns=["date", "code", "method", "factor_score"]),
            pd.DataFrame(weight_records, columns=["date", "method", "factor", "weight", "IC", "stability"]),
        )

    def _ic_weights(self, daily: pd.DataFrame, cutoff: pd.Timestamp, factors: Sequence[str], icir: bool) -> np.ndarray:
        history = daily.loc[daily["date"] <= cutoff].groupby("factor")["IC"]
        if icir:
            raw = pd.Series({factor: history.get_group(factor).mean() / history.get_group(factor).std(ddof=1) if factor in history.groups and history.get_group(factor).std(ddof=1) > 0 else 0.0 for factor in factors})
        else:
            raw = pd.Series({factor: history.get_group(factor).mean() if factor in history.groups else 0.0 for factor in factors})
        return _normalize_weights(raw.to_numpy())

    @staticmethod
    def _regression_weights(history: pd.DataFrame, factors: Sequence[str], target: str) -> np.ndarray:
        clean = history.dropna(subset=[*factors, target])
        if len(clean) < max(30, len(factors) * 5):
            return np.repeat(1 / len(factors), len(factors))
        design = np.column_stack([np.ones(len(clean)), clean.loc[:, factors].to_numpy(dtype=float)])
        coefficients, *_ = np.linalg.lstsq(design, clean[target].to_numpy(dtype=float), rcond=None)
        return _normalize_weights(coefficients[1:])

    @staticmethod
    def _ml_weights(
        history: pd.DataFrame,
        factors: Sequence[str],
        target: str,
        max_samples: int = 20_000,
        estimators: int = 20,
    ) -> np.ndarray:
        clean = history.dropna(subset=[*factors, target])
        if len(clean) < max(100, len(factors) * 10):
            return np.repeat(1 / len(factors), len(factors))
        if len(clean) > max_samples:
            positions = np.linspace(0, len(clean) - 1, num=max_samples, dtype=int)
            clean = clean.iloc[positions]
        model = RandomForestRegressor(
            n_estimators=estimators, random_state=0, min_samples_leaf=5, n_jobs=1
        )
        model.fit(clean.loc[:, factors], clean[target])
        return _normalize_weights(model.feature_importances_)


class AnnualWalkForwardResearch:
    """Validate rolling four-calendar-year training windows against the next year."""

    def __init__(self, horizon: int = 20, quantile: float = 0.2) -> None:
        if horizon < 1 or not 0 < quantile <= 0.5:
            raise ValueError("invalid walk-forward settings")
        self.horizon = horizon
        self.quantile = quantile

    def run(
        self, panel: pd.DataFrame, factors: Sequence[str], target: str = "future_excess_return_20d", output_path: str | Path | None = None
    ) -> pd.DataFrame:
        if not {"date", target, *factors}.issubset(panel.columns):
            raise ValueError("walk-forward panel is missing required columns")
        frame = panel.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
        records: list[dict[str, object]] = []
        years = sorted(frame["date"].dt.year.unique())
        for validation_year in years:
            train_years = range(validation_year - 4, validation_year)
            train = frame.loc[frame["date"].dt.year.isin(train_years)].copy()
            validation = frame.loc[frame["date"].dt.year == validation_year].copy()
            if train.empty or validation.empty or train["date"].dt.year.nunique() < 4:
                continue
            first_validation = validation["date"].min()
            prior_dates = dates[dates < first_validation]
            if len(prior_dates) <= self.horizon:
                continue
            # Labels at the end of training would contain validation-period
            # returns, so purge their entire holding horizon.
            cutoff = prior_dates[-self.horizon]
            weights = FactorCombinationResearch._regression_weights(
                train.loc[train["date"] <= cutoff], factors, target
            )
            validation["score"] = sum(validation[factor].fillna(0.0) * weight for factor, weight in zip(factors, weights, strict=True))
            rebal_dates = sorted(validation["date"].unique())[:: self.horizon]
            returns: list[float] = []
            for date in rebal_dates:
                day = validation.loc[validation["date"] == date].dropna(subset=["score", target])
                if len(day) < 2:
                    continue
                count = max(1, int(np.ceil(len(day) * self.quantile)))
                returns.append(float(day.nlargest(count, "score")[target].mean()))
            if not returns:
                continue
            series = pd.Series(returns, dtype="float64")
            nav = (1 + series).cumprod()
            volatility = series.std(ddof=0)
            records.append(
                {
                    "period": str(validation_year), "return": float(nav.iloc[-1] - 1), "benchmark": 0.0,
                    "alpha": float(nav.iloc[-1] - 1),
                    "sharpe": float(series.mean() / volatility * np.sqrt(252 / self.horizon)) if volatility > 0 else 0.0,
                    "drawdown": float((nav / nav.cummax() - 1).min()),
                }
            )
        report = pd.DataFrame(records, columns=["period", "return", "benchmark", "alpha", "sharpe", "drawdown"])
        if output_path is not None:
            target_path = Path(output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(target_path, index=False)
        return report


def _research_factor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    close = pd.to_numeric(values["close"], errors="coerce")
    returns = close.pct_change()
    values["trend_20"] = close / close.rolling(20).mean() - 1.0
    values["trend_60"] = close / close.rolling(60).mean() - 1.0
    values["trend_120"] = close / close.rolling(120).mean() - 1.0
    values["volatility_60"] = returns.rolling(60).std()
    turnover = pd.to_numeric(values["raw_turnover"], errors="coerce")
    amount = pd.to_numeric(values["raw_amount"], errors="coerce")
    values["turnover_5"] = turnover.rolling(5).mean()
    values["turnover_20"] = turnover.rolling(20).mean()
    values["amount_change"] = amount.pct_change()
    volume = pd.to_numeric(values["volume"], errors="coerce")
    values["volume_ratio"] = volume / volume.rolling(20).mean()
    benchmark_returns = values["close"].pct_change()
    # beta_60 is provided by AdvancedFactorEngine; calculate beta_120 against
    # its supplied benchmark feature only when the market series is available.
    values["beta_120"] = _rolling_beta(benchmark_returns, values.get("market_return"), 120)
    values["max_drawdown_60"] = close / close.rolling(60).max() - 1.0
    values["max_drawdown_120"] = close / close.rolling(120).max() - 1.0
    values["atr_14"] = values["atr_14"]
    return values.loc[:, ["date", "close", *RESEARCH_FACTORS]]


def _rolling_beta(stock_returns: pd.Series, market_returns: pd.Series | None, window: int) -> pd.Series:
    if market_returns is None:
        return pd.Series(np.nan, index=stock_returns.index, dtype="float64")
    variance = market_returns.rolling(window).var(ddof=0)
    return (stock_returns.rolling(window).cov(market_returns, ddof=0) / variance).where(variance > 0)


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    denominator = np.abs(values).sum()
    return values / denominator if np.isfinite(denominator) and denominator > 0 else np.repeat(1 / len(values), len(values))
