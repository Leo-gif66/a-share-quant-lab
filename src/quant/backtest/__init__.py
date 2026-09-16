"""Backtesting engines and legacy compatibility exports."""

from .attribution import DailyAttribution
from .diagnostics import drawdown_periods, worst_holding_periods, yearly_attribution
from .engine import BacktestEngine, BacktestResult
from .execution import ExecutionSettings, ProfessionalPortfolio, TradeRecord
from .legacy import load_or_create_model, run_backtest, score_panel
from .metrics import annual_returns, calculate_metrics, drawdown_series, monthly_returns
from .portfolio import Portfolio, TradingCosts

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
    "drawdown_periods",
    "drawdown_series",
    "load_or_create_model",
    "monthly_returns",
    "run_backtest",
    "score_panel",
    "worst_holding_periods",
    "yearly_attribution",
]
