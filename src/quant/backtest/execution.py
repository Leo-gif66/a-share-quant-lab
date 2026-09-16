"""Optional professional execution primitives; legacy accounting is untouched."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class ExecutionSettings:
    lot_size: int = 100
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    minimum_commission: float = 5.0
    base_slippage: float = 0.001
    low_liquidity_amount: float = 5_000_000.0
    low_liquidity_multiplier: float = 2.0
    limit_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.lot_size < 1 or self.low_liquidity_amount < 0:
            raise ValueError("lot size and liquidity threshold must be non-negative")
        if min(self.commission_rate, self.stamp_tax_rate, self.base_slippage) < 0:
            raise ValueError("cost rates must be non-negative")
        if self.low_liquidity_multiplier < 1 or not 0 < self.limit_ratio < 1:
            raise ValueError("liquidity multiplier and limit ratio are invalid")


@dataclass(frozen=True)
class TradeRecord:
    date: pd.Timestamp
    stock: str
    side: Literal["buy", "sell"]
    price: float
    quantity: int
    commission: float
    tax: float
    slippage: float


@dataclass
class PositionLot:
    quantity: int
    acquired_date: pd.Timestamp


@dataclass
class ProfessionalPortfolio:
    """Cash and T+1 lots for simulations using explicit daily market bars."""

    initial_cash: float = 100_000.0
    settings: ExecutionSettings = field(default_factory=ExecutionSettings)
    cash: float = field(init=False)
    lots: dict[str, list[PositionLot]] = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")
        self.cash = float(self.initial_cash)

    def position_quantity(self, stock: str) -> int:
        return sum(lot.quantity for lot in self.lots.get(str(stock), []))

    def rebalance(
        self, date: pd.Timestamp, target_weights: dict[str, float], bars: pd.DataFrame
    ) -> list[TradeRecord]:
        """Trade toward cash-aware target weights at the next session's open.

        Ineligible sells remain held (T+1, suspension, or limit-down); blocked
        buys are skipped (suspension or limit-up).  The returned records contain
        only fills, never requested-but-rejected orders.
        """
        current_date = pd.Timestamp(date).normalize()
        market = _market_index(bars)
        target_weights = {str(code).zfill(6): float(weight) for code, weight in target_weights.items()}
        if any(weight < 0 for weight in target_weights.values()) or sum(target_weights.values()) > 1 + 1e-12:
            raise ValueError("target weights must be non-negative and sum to at most one")
        valuation = self.value(market)
        desired = {
            code: self._round_lot(
                valuation * weight / self.execution_price(market.loc[code], "buy")
            )
            for code, weight in target_weights.items()
            if code in market.index and self._can_trade(market.loc[code], "buy")
        }
        records_before = len(self.trades)
        for code in sorted(set(self.lots).union(desired)):
            held = self.position_quantity(code)
            wanted = desired.get(code, 0)
            if held > wanted and code in market.index:
                self._sell(current_date, code, held - wanted, market.loc[code])
        for code in sorted(desired):
            held = self.position_quantity(code)
            if desired[code] > held:
                self._buy(current_date, code, desired[code] - held, market.loc[code])
        return self.trades[records_before:]

    def value(self, market: pd.DataFrame) -> float:
        missing = [code for code in self.lots if code not in market.index]
        if missing:
            raise ValueError(f"market bars missing held stock(s): {', '.join(sorted(missing))}")
        value = self.cash
        for code in self.lots:
            price = float(market.at[code, "open"])
            if price <= 0:
                raise ValueError(f"{code} has a non-positive open price")
            value += self.position_quantity(code) * price
        return float(value)

    def mark_to_market(self, date: pd.Timestamp, bars: pd.DataFrame) -> float:
        market = _market_index(bars)
        value = self.value(market)
        self.equity_curve.append({"date": pd.Timestamp(date), "equity": value})
        return value

    def execution_price(self, bar: pd.Series, side: Literal["buy", "sell"]) -> float:
        price = float(bar["open"])
        if price <= 0:
            raise ValueError("open price must be positive")
        slippage = self.slippage(bar)
        return price * (1 + slippage if side == "buy" else 1 - slippage)

    def slippage(self, bar: pd.Series) -> float:
        amount = pd.to_numeric(pd.Series([bar.get("amount")]), errors="coerce").iloc[0]
        multiplier = (
            self.settings.low_liquidity_multiplier
            if pd.isna(amount) or amount < self.settings.low_liquidity_amount
            else 1.0
        )
        return float(self.settings.base_slippage * multiplier)

    def write_trades(self, directory: str | Path = "data/trades") -> Path:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.trades:
            stamp = self.trades[-1].date.strftime("%Y%m%d")
        else:
            stamp = "empty"
        path = target_dir / f"trades_{stamp}.parquet"
        pd.DataFrame([asdict(record) for record in self.trades], columns=_TRADE_COLUMNS).to_parquet(
            path, index=False
        )
        return path

    def _buy(self, date: pd.Timestamp, code: str, requested: int, bar: pd.Series) -> None:
        if not self._can_trade(bar, "buy"):
            return
        price = self.execution_price(bar, "buy")
        quantity = self._round_lot(requested)
        if quantity <= 0:
            return
        affordable = self._round_lot(self._affordable_quantity(price))
        quantity = min(quantity, affordable)
        if quantity <= 0:
            return
        notional = quantity * price
        commission = max(notional * self.settings.commission_rate, self.settings.minimum_commission)
        self.cash -= notional + commission
        self.lots.setdefault(code, []).append(PositionLot(quantity=quantity, acquired_date=date))
        self.trades.append(
            TradeRecord(date, code, "buy", price, quantity, commission, 0.0, self.slippage(bar))
        )

    def _sell(self, date: pd.Timestamp, code: str, requested: int, bar: pd.Series) -> None:
        if not self._can_trade(bar, "sell"):
            return
        sellable = sum(lot.quantity for lot in self.lots.get(code, []) if lot.acquired_date < date)
        quantity = self._round_lot(min(requested, sellable))
        if quantity <= 0:
            return
        price = self.execution_price(bar, "sell")
        notional = quantity * price
        commission = max(notional * self.settings.commission_rate, self.settings.minimum_commission)
        tax = notional * self.settings.stamp_tax_rate
        self.cash += notional - commission - tax
        remaining = quantity
        new_lots: list[PositionLot] = []
        for lot in self.lots[code]:
            take = min(lot.quantity, remaining) if lot.acquired_date < date else 0
            remaining -= take
            if lot.quantity > take:
                new_lots.append(PositionLot(lot.quantity - take, lot.acquired_date))
        if new_lots:
            self.lots[code] = new_lots
        else:
            del self.lots[code]
        self.trades.append(
            TradeRecord(date, code, "sell", price, quantity, commission, tax, self.slippage(bar))
        )

    def _affordable_quantity(self, execution_price: float) -> int:
        unit_cost = execution_price * (1 + self.settings.commission_rate)
        if unit_cost <= 0:
            return 0
        # The initial minimum commission check is handled by the exact trade
        # calculation; this estimate can only overstate by a single lot.
        return int(self.cash / unit_cost)

    def _can_trade(self, bar: pd.Series, side: Literal["buy", "sell"]) -> bool:
        if bool(bar.get("suspended", False)):
            return False
        volume = bar.get("volume")
        if volume is not None and pd.notna(volume) and float(volume) <= 0:
            return False
        if side == "buy" and bool(bar.get("is_limit_up", False)):
            return False
        if side == "sell" and bool(bar.get("is_limit_down", False)):
            return False
        previous = bar.get("prev_close")
        close = bar.get("close")
        if previous is not None and close is not None and pd.notna(previous) and pd.notna(close):
            change = float(close) / float(previous) - 1 if float(previous) > 0 else 0.0
            if side == "buy" and change >= self.settings.limit_ratio - 1e-12:
                return False
            if side == "sell" and change <= -self.settings.limit_ratio + 1e-12:
                return False
        return float(bar["open"]) > 0

    def _round_lot(self, quantity: float) -> int:
        return int(float(quantity) // self.settings.lot_size * self.settings.lot_size)


_TRADE_COLUMNS = ["date", "stock", "side", "price", "quantity", "commission", "tax", "slippage"]


def _market_index(bars: pd.DataFrame) -> pd.DataFrame:
    if not {"code", "open"}.issubset(bars.columns):
        raise ValueError("market bars require code and open columns")
    market = bars.copy()
    market["code"] = market["code"].astype(str).str.zfill(6)
    if market["code"].duplicated().any():
        raise ValueError("market bars require one row per code")
    return market.set_index("code", drop=False)
