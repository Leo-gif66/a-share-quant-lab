"""Simple next-open, equal-weight multi-factor backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from .diagnostics import drawdown_periods, worst_holding_periods, yearly_attribution
from .metrics import annual_returns, calculate_metrics, drawdown_series, monthly_returns
from .portfolio import Portfolio, TradingCosts


@dataclass
class BacktestResult:
    """Outputs from one multi-factor backtest run."""

    portfolio: Portfolio
    equity_curve: pd.DataFrame
    benchmark_curve: pd.DataFrame
    metrics: dict[str, float]
    annual_returns: pd.DataFrame
    monthly_returns: pd.DataFrame
    drawdown_series: pd.DataFrame
    yearly_attribution: pd.DataFrame
    drawdown_periods: pd.DataFrame
    worst_holding_periods: pd.DataFrame


class BacktestEngine:
    """Backtest Factor Engine outputs with periodic next-open rebalancing."""

    FACTOR_COLUMNS: ClassVar[tuple[str, ...]] = (
        "momentum_20",
        "trend_60",
        "volatility_20",
        "liquidity_20",
        "volume_ratio_20",
        "drawdown_60",
    )
    _SCORE_DIRECTIONS: ClassVar[dict[str, int]] = {
        "momentum_20": 1,
        "trend_60": 1,
        "volatility_20": -1,
        "liquidity_20": 1,
        "volume_ratio_20": 1,
        "drawdown_60": 1,
    }

    def __init__(
        self,
        features_dir: str | Path = "data/features",
        initial_cash: float = 100_000.0,
        rebalance_interval: int = 20,
        top_n: int = 5,
        benchmark_code: str = "000300",
        benchmark_path: str | Path | None = None,
        commission: float = 0.0003,
        stamp_tax: float = 0.0005,
        slippage: float = 0.001,
        mode: str = "legacy",
        factor_config_path: str | Path = "configs/factor_weights.yaml",
        portfolio_top_n: int = 20,
    ) -> None:
        if rebalance_interval < 1:
            raise ValueError("rebalance_interval must be at least 1")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if mode not in {"legacy", "portfolio_v1"}:
            raise ValueError("mode must be either legacy or portfolio_v1")
        self.features_dir = Path(features_dir)
        self.initial_cash = initial_cash
        self.rebalance_interval = rebalance_interval
        self.top_n = top_n
        self.benchmark_code = str(benchmark_code).zfill(6)
        self.benchmark_path = Path(benchmark_path) if benchmark_path is not None else None
        self.costs = TradingCosts(commission=commission, stamp_tax=stamp_tax, slippage=slippage)
        self.mode = mode
        self.factor_config_path = Path(factor_config_path)
        self.portfolio_top_n = portfolio_top_n

    def run(self) -> BacktestResult:
        """Run the strategy using close scores and trades at the next day's open."""
        if self.mode == "portfolio_v1":
            return self._run_portfolio_v1()  # type: ignore[return-value]
        panel = self._load_feature_panel()
        scored = self._add_scores(panel)
        dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        rebalance_targets = self._rebalance_targets(scored, dates)
        portfolio = Portfolio(initial_cash=self.initial_cash, costs=self.costs)

        for current_date in dates:
            day = panel.loc[panel["date"] == current_date].set_index("code")
            open_prices = day["open"].to_dict()
            if current_date in rebalance_targets:
                portfolio.rebalance(rebalance_targets[current_date], open_prices)
            portfolio.record(current_date, day["close"].to_dict())

        equity_curve = portfolio.equity_frame()
        benchmark_curve = self._load_benchmark_curve(dates)
        return BacktestResult(
            portfolio=portfolio,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            metrics=calculate_metrics(equity_curve, benchmark_curve),
            annual_returns=annual_returns(equity_curve, benchmark_curve),
            monthly_returns=monthly_returns(equity_curve, benchmark_curve),
            drawdown_series=drawdown_series(equity_curve),
            yearly_attribution=yearly_attribution(equity_curve, benchmark_curve),
            drawdown_periods=drawdown_periods(equity_curve),
            worst_holding_periods=worst_holding_periods(equity_curve),
        )

    def _run_portfolio_v1(self):
        """Delegate to v1.0 while keeping this engine's legacy default intact."""
        from ..portfolio.backtest import PortfolioBacktestEngine

        return PortfolioBacktestEngine(
            features_dir=self.features_dir,
            initial_cash=self.initial_cash,
            rebalance_interval=self.rebalance_interval,
            top_n=self.portfolio_top_n,
            benchmark_code=self.benchmark_code,
            benchmark_path=self.benchmark_path,
            factor_config_path=self.factor_config_path,
            commission=self.costs.commission,
            stamp_tax=self.costs.stamp_tax,
            slippage=self.costs.slippage,
        ).run()

    def _load_feature_panel(self) -> pd.DataFrame:
        """Read all per-stock factor parquet files from the configured directory."""
        paths = sorted(
            path
            for path in self.features_dir.glob("*.parquet")
            if path.stem.isdigit() and len(path.stem) == 6
        )
        if not paths:
            raise RuntimeError(f"no per-stock feature files found in {self.features_dir}")

        frames = []
        required = ("date", "open", "close", *self.FACTOR_COLUMNS)
        for path in paths:
            frame = pd.read_parquet(path)
            missing = [column for column in required if column not in frame]
            if missing:
                raise ValueError(f"{path.name} missing feature columns: {', '.join(missing)}")
            frame = frame.loc[:, required].copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            frame["code"] = path.stem
            for column in required[1:]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frames.append(frame)

        panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
        common_dates = panel.groupby("date")["code"].nunique()
        common_dates = common_dates[common_dates == len(paths)].index
        if len(common_dates) < 2:
            raise RuntimeError("feature files do not share enough trading dates for backtesting")
        return panel.loc[panel["date"].isin(common_dates)].reset_index(drop=True)

    def _add_scores(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Replicate Factor Engine's daily cross-sectional percentile score."""
        scored = panel.dropna(subset=self.FACTOR_COLUMNS).copy()
        if scored.empty:
            raise RuntimeError("no complete factor rows available for backtesting")
        rank_columns = []
        for factor, direction in self._SCORE_DIRECTIONS.items():
            column = f"_{factor}_rank"
            scored[column] = scored.groupby("date")[factor].rank(
                method="average", pct=True, ascending=direction > 0
            )
            rank_columns.append(column)
        scored["score"] = scored[rank_columns].mean(axis=1)
        return scored

    def _rebalance_targets(
        self, scored: pd.DataFrame, dates: pd.DatetimeIndex
    ) -> dict[pd.Timestamp, list[str]]:
        """Map each next-open rebalance date to selections from the prior close."""
        targets: dict[pd.Timestamp, list[str]] = {}
        for index in range(0, len(dates) - 1, self.rebalance_interval):
            signal_date = dates[index]
            target_date = dates[index + 1]
            selections = scored.loc[scored["date"] == signal_date].nlargest(self.top_n, "score")
            if not selections.empty:
                targets[target_date] = selections["code"].tolist()
        return targets

    def _load_benchmark_curve(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Load default CSI 300 data and normalize it to the strategy's date range."""
        path = self._benchmark_file()
        frame = pd.read_parquet(path)
        if not {"date", "close"}.issubset(frame.columns):
            raise ValueError(f"{path.name} benchmark requires date and close columns")
        benchmark = frame.loc[:, ["date", "close"]].copy()
        benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise")
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
        benchmark = benchmark.dropna(subset=["close"]).drop_duplicates("date").set_index("date")
        benchmark = benchmark.reindex(dates)
        if benchmark["close"].isna().any():
            raise ValueError(f"{self.benchmark_code} benchmark is missing strategy trading dates")
        normalized = self.initial_cash * benchmark["close"] / benchmark["close"].iloc[0]
        return pd.DataFrame({"date": dates, "benchmark": normalized.to_numpy()})

    def _benchmark_file(self) -> Path:
        if self.benchmark_path is not None:
            return self.benchmark_path
        raw_dir = self.features_dir.parent / "raw"
        direct = raw_dir / f"{self.benchmark_code}.parquet"
        if direct.exists():
            return direct
        legacy = raw_dir / "index" / "benchmark.parquet"
        if legacy.exists():
            return legacy
        raise FileNotFoundError(
            f"benchmark {self.benchmark_code} not found; expected {direct} or {legacy}"
        )
