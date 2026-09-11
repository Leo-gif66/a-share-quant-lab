"""Persistent daily paper trading with explicit orders, fills, and costs.

The v2 account is additive: it does not alter the v1 T+1 order planner or its
stored account format.  It is intended for one end-of-day rebalance using
prices known on that date, and rejects a price snapshot from the future.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


ORDER_COLUMNS = ("date", "symbol", "target_weight", "status")
FILL_COLUMNS = ("date", "symbol", "side", "quantity", "price", "notional", "cost")
PERFORMANCE_COLUMNS = ("date", "equity", "cash", "market_value", "transaction_cost", "daily_return")


@dataclass(frozen=True)
class PaperTradingSettings:
    initial_cash: float = 1_000_000.0
    transaction_cost_rate: float = 0.001
    lot_size: int = 100

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 <= self.transaction_cost_rate < 1:
            raise ValueError("transaction_cost_rate must be in [0, 1)")
        if self.lot_size < 1:
            raise ValueError("lot_size must be positive")


@dataclass
class PaperTradingAccountV2:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    last_equity: float | None = None
    last_run_date: pd.Timestamp | None = None

    @classmethod
    def load(cls, path: str | Path, settings: PaperTradingSettings | None = None) -> "PaperTradingAccountV2":
        target = Path(path)
        defaults = settings or PaperTradingSettings()
        if not target.exists():
            return cls(cash=defaults.initial_cash)
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls(
            cash=float(payload["cash"]),
            positions={str(code).zfill(6): int(quantity) for code, quantity in payload.get("positions", {}).items() if int(quantity) > 0},
            last_equity=float(payload["last_equity"]) if payload.get("last_equity") is not None else None,
            last_run_date=pd.Timestamp(payload["last_run_date"]).normalize()
            if payload.get("last_run_date") is not None
            else None,
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": float(self.cash),
            "positions": self.positions,
            "last_equity": self.last_equity,
            "last_run_date": self.last_run_date.isoformat() if self.last_run_date is not None else None,
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
        return target


@dataclass(frozen=True)
class PaperRunResult:
    account: PaperTradingAccountV2
    orders: pd.DataFrame
    fills: pd.DataFrame
    performance: pd.DataFrame
    state_path: Path
    already_processed: bool = False


class PaperTradingEngineV2:
    """Rebalance a persistent virtual account to allocator target weights."""

    def __init__(self, settings: PaperTradingSettings | None = None) -> None:
        self.settings = settings or PaperTradingSettings()

    def run(
        self,
        targets: pd.DataFrame,
        prices: pd.DataFrame,
        state_path: str | Path = "data/paper/v2_account.json",
        output_dir: str | Path = "data/paper",
        as_of: pd.Timestamp | str | None = None,
    ) -> PaperRunResult:
        desired = self._normalise_targets(targets, as_of)
        date = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp(desired["date"].iloc[0])
        account = PaperTradingAccountV2.load(state_path, self.settings)
        directory = Path(output_dir)
        repeated = account.last_run_date == date
        if repeated:
            return self._existing_day_result(account, date, Path(state_path), directory)
        quote = self._price_snapshot(prices, date)
        missing_held = sorted(set(account.positions).difference(quote))
        if missing_held:
            raise ValueError(f"missing current prices for held positions: {', '.join(missing_held)}")
        equity_before = self._equity(account, quote)
        if not np.isfinite(equity_before) or equity_before <= 0:
            raise ValueError("paper account has no positive marked equity")

        orders = desired.loc[:, ["date", "symbol", "weight"]].rename(columns={"weight": "target_weight"})
        orders["status"] = "pending"
        fills: list[dict[str, object]] = []
        target_shares = self._target_shares(desired, quote, equity_before)
        # Sell first to make the rebalance cash-safe.
        for symbol in sorted(set(account.positions).union(target_shares)):
            held = int(account.positions.get(symbol, 0))
            target = int(target_shares.get(symbol, 0))
            if held > target:
                fills.append(self._fill(account, date, symbol, "sell", held - target, quote[symbol]))
        for symbol in sorted(target_shares):
            held = int(account.positions.get(symbol, 0))
            target = int(target_shares[symbol])
            if target > held:
                affordable = self._affordable_quantity(account.cash, quote[symbol], target - held)
                if affordable:
                    fills.append(self._fill(account, date, symbol, "buy", affordable, quote[symbol]))
        orders["status"] = "filled"
        fill_frame = pd.DataFrame(fills, columns=FILL_COLUMNS)
        transaction_cost = float(fill_frame["cost"].sum()) if not fill_frame.empty else 0.0
        equity = self._equity(account, quote)
        prior_equity = account.last_equity
        daily_return = float(equity / prior_equity - 1.0) if prior_equity and prior_equity > 0 else 0.0
        account.last_equity = equity
        account.last_run_date = date
        performance = pd.DataFrame(
            [{
                "date": date,
                "equity": equity,
                "cash": account.cash,
                "market_value": equity - account.cash,
                "transaction_cost": transaction_cost,
                "daily_return": daily_return,
            }],
            columns=PERFORMANCE_COLUMNS,
        )
        state = account.save(state_path)
        directory.mkdir(parents=True, exist_ok=True)
        self._append(directory / "orders.parquet", orders, ORDER_COLUMNS)
        self._append(directory / "fills.parquet", fill_frame, FILL_COLUMNS)
        self._append(directory / "performance.parquet", performance, PERFORMANCE_COLUMNS)
        return PaperRunResult(account, orders, fill_frame, performance, state)

    @staticmethod
    def _existing_day_result(
        account: PaperTradingAccountV2, date: pd.Timestamp, state_path: Path, directory: Path
    ) -> PaperRunResult:
        """Return persisted results on an intentional same-day rerun.

        Filling again would be a duplicate virtual trade and would distort the
        account.  The prior daily snapshot remains the authoritative result.
        """
        def day_values(name: str, columns: tuple[str, ...]) -> pd.DataFrame:
            path = directory / f"{name}.parquet"
            if not path.exists():
                return pd.DataFrame(columns=columns)
            values = pd.read_parquet(path)
            if "date" not in values:
                return pd.DataFrame(columns=columns)
            return values.loc[pd.to_datetime(values["date"], errors="coerce").dt.normalize() == date].reindex(columns=columns)

        orders = day_values("orders", ORDER_COLUMNS)
        # No fill is returned because this invocation did not execute one.
        # Historical fill rows remain available in the durable ledger.
        fills = pd.DataFrame(columns=FILL_COLUMNS)
        performance = day_values("performance", PERFORMANCE_COLUMNS)
        if performance.empty:
            performance = pd.DataFrame(
                [{
                    "date": date,
                    "equity": account.last_equity if account.last_equity is not None else account.cash,
                    "cash": account.cash,
                    "market_value": (account.last_equity or account.cash) - account.cash,
                    "transaction_cost": 0.0,
                    "daily_return": 0.0,
                }]
            )
        return PaperRunResult(account, orders, fills, performance, state_path, already_processed=True)

    def _normalise_targets(self, targets: pd.DataFrame, as_of: pd.Timestamp | str | None) -> pd.DataFrame:
        required = {"symbol", "weight"}
        missing = sorted(required.difference(targets.columns))
        if missing:
            raise ValueError(f"targets missing columns: {', '.join(missing)}")
        values = targets.copy()
        if "date" not in values:
            values["date"] = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
        values["date"] = pd.to_datetime(values["date"], errors="raise").dt.normalize()
        cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else values["date"].max()
        values = values.loc[values["date"] == cutoff, ["date", "symbol", "weight"]].copy()
        values["symbol"] = values["symbol"].astype(str).str.zfill(6)
        values["weight"] = pd.to_numeric(values["weight"], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna(subset=["weight"])
        if (values["weight"] < 0).any() or values["weight"].sum() > 1.0000001:
            raise ValueError("paper targets must be non-negative and sum to at most one")
        return values.groupby(["date", "symbol"], as_index=False)["weight"].sum()

    @staticmethod
    def _price_snapshot(prices: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
        symbol_column = "symbol" if "symbol" in prices else "code" if "code" in prices else None
        price_column = "close" if "close" in prices else "open" if "open" in prices else None
        if symbol_column is None or price_column is None or "date" not in prices:
            raise ValueError("prices require date, code/symbol, and close/open")
        values = prices.loc[:, ["date", symbol_column, price_column]].copy()
        values.columns = ["date", "symbol", "price"]
        values["date"] = pd.to_datetime(values["date"], errors="raise").dt.normalize()
        if (values["date"] > as_of).any():
            # Future bars may appear in the supplied panel, but they are never
            # consumed.  The cutoff makes this contract explicit.
            values = values.loc[values["date"] <= as_of]
        values["symbol"] = values["symbol"].astype(str).str.zfill(6)
        values["price"] = pd.to_numeric(values["price"], errors="coerce")
        values = values.loc[values["price"] > 0].sort_values("date").drop_duplicates("symbol", keep="last")
        if values.empty:
            raise ValueError("no valid price is available on or before as_of")
        return dict(zip(values["symbol"], values["price"], strict=True))

    def _target_shares(self, targets: pd.DataFrame, prices: dict[str, float], equity: float) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in targets.itertuples(index=False):
            if row.symbol not in prices:
                continue
            lots = int((float(row.weight) * equity) / (prices[row.symbol] * self.settings.lot_size))
            result[row.symbol] = max(0, lots * self.settings.lot_size)
        return result

    def _affordable_quantity(self, cash: float, price: float, requested: int) -> int:
        gross_per_share = price * (1.0 + self.settings.transaction_cost_rate)
        quantity = min(requested, int(cash / gross_per_share))
        return quantity // self.settings.lot_size * self.settings.lot_size

    def _fill(self, account: PaperTradingAccountV2, date: pd.Timestamp, symbol: str, side: str, quantity: int, price: float) -> dict[str, object]:
        notional = float(quantity * price)
        cost = notional * self.settings.transaction_cost_rate
        if side == "buy":
            account.cash -= notional + cost
            account.positions[symbol] = int(account.positions.get(symbol, 0)) + quantity
        else:
            account.cash += notional - cost
            remaining = int(account.positions.get(symbol, 0)) - quantity
            if remaining > 0:
                account.positions[symbol] = remaining
            else:
                account.positions.pop(symbol, None)
        return {"date": date, "symbol": symbol, "side": side, "quantity": quantity, "price": price, "notional": notional, "cost": cost}

    @staticmethod
    def _equity(account: PaperTradingAccountV2, prices: dict[str, float]) -> float:
        market_value = sum(quantity * prices.get(symbol, 0.0) for symbol, quantity in account.positions.items())
        return float(account.cash + market_value)

    @staticmethod
    def _append(path: Path, values: pd.DataFrame, columns: tuple[str, ...]) -> None:
        incoming = values.reindex(columns=columns)
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)
        combined = pd.concat([existing, incoming], ignore_index=True)
        # A rerun for the exact same date replaces that day rather than
        # recording duplicate virtual transactions.
        if "date" in combined and not combined.empty:
            combined["date"] = pd.to_datetime(combined["date"], errors="raise")
            if path.stem == "performance":
                combined = combined.drop_duplicates(["date"], keep="last")
        combined.to_parquet(path, index=False)
