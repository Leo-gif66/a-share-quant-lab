"""Backtesting engines and legacy compatibility exports."""

from .engine import BacktestEngine, BacktestResult
from .diagnostics import drawdown_periods, worst_holding_periods, yearly_attribution
from .legacy import load_or_create_model, run_backtest, score_panel
from .metrics import annual_returns, calculate_metrics, drawdown_series, monthly_returns
from .portfolio import Portfolio, TradingCosts
from .execution import ExecutionSettings, ProfessionalPortfolio, TradeRecord
from .attribution import DailyAttribution

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "DailyAttribution",
    "ExecutionSettings",
    "Portfolio",
    "ProfessionalPortfolio",
    "TradeRecord",
    "TradingCosts",
    "annual_returns",
    "calculate_metrics",
    "drawdown_series",
    "drawdown_periods",
    "load_or_create_model",
    "monthly_returns",
    "run_backtest",
    "score_panel",
    "worst_holding_periods",
    "yearly_attribution",
]
