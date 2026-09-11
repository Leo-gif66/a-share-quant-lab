"""Minimal portfolio accounting for the v0.6 backtest engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class TradingCosts:
    """Proportional costs applied to each next-open transaction."""

    commission: float = 0.0003
    stamp_tax: float = 0.0005
    slippage: float = 0.001


@dataclass
class Portfolio:
    """Cash, share positions, and end-of-day equity history."""

    initial_cash: float = 100_000.0
    cash: float = field(init=False)
    positions: dict[str, float] = field(default_factory=dict)
    equity_curve: list[dict[str, object]] = field(default_factory=list)
    costs: TradingCosts = field(default_factory=TradingCosts)
    trading_costs: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    def rebalance(self, target_codes: list[str], open_prices: Mapping[str, float]) -> None:
        """Sell existing holdings and buy equal-weight targets at the open."""
        self._sell_all(open_prices)

        valid_targets = [
            code for code in target_codes if code in open_prices and float(open_prices[code]) > 0
        ]
        if not valid_targets:
            return

        allocation = self.cash / len(valid_targets)
        for code in valid_targets:
            price = float(open_prices[code])
            execution_price = price * (1 + self.costs.slippage)
            trade_value = allocation / (1 + self.costs.commission)
            shares = trade_value / execution_price
            self.positions[code] = shares
            self.cash -= allocation
            self.trading_costs += allocation - shares * price

    def rebalance_weights(
        self, target_weights: Mapping[str, float], open_prices: Mapping[str, float]
    ) -> None:
        """Sell existing holdings and buy explicit cash-aware target weights.

        This extends the legacy equal-weight method for the v1.0 portfolio
        mode.  A total target below one intentionally leaves the remainder in
        cash after risk controls reduce exposure.
        """
        self._sell_all(open_prices)
        valid_targets = {
            str(code): float(weight)
            for code, weight in target_weights.items()
            if code in open_prices and float(open_prices[code]) > 0 and float(weight) > 0
        }
        if not valid_targets:
            return
        total_weight = sum(valid_targets.values())
        if total_weight > 1 + 1e-12:
            raise ValueError("target weights must sum to at most one")

        available_cash = self.cash
        for code, weight in valid_targets.items():
            allocation = available_cash * weight
            price = float(open_prices[code])
            execution_price = price * (1 + self.costs.slippage)
            trade_value = allocation / (1 + self.costs.commission)
            shares = trade_value / execution_price
            self.positions[code] = shares
            self.cash -= allocation
            self.trading_costs += allocation - shares * price

    def value(self, prices: Mapping[str, float]) -> float:
        """Return cash plus the marked-to-market value of every position."""
        missing = [code for code in self.positions if code not in prices]
        if missing:
            raise ValueError(f"missing prices for positions: {', '.join(missing)}")
        return self.cash + sum(
            shares * float(prices[code]) for code, shares in self.positions.items()
        )

    def record(self, date: pd.Timestamp, close_prices: Mapping[str, float]) -> float:
        """Append one end-of-day equity observation and return its value."""
        equity = self.value(close_prices)
        self.equity_curve.append({"date": pd.Timestamp(date), "equity": equity})
        return equity

    def equity_frame(self) -> pd.DataFrame:
        """Return the recorded equity curve as a date-sorted DataFrame."""
        return pd.DataFrame(self.equity_curve, columns=["date", "equity"]).sort_values("date")

    def _sell_all(self, open_prices: Mapping[str, float]) -> None:
        """Close current positions at the open after slippage, commission, and tax."""
        missing = [code for code in self.positions if code not in open_prices]
        if missing:
            raise ValueError(f"missing open prices for positions: {', '.join(missing)}")
        for code, shares in self.positions.items():
            market_value = shares * float(open_prices[code])
            execution_value = market_value * (1 - self.costs.slippage)
            fees = execution_value * (self.costs.commission + self.costs.stamp_tax)
            self.cash += execution_value - fees
            self.trading_costs += market_value - execution_value + fees
        self.positions.clear()
