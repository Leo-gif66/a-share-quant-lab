"""A deliberately local paper-trading engine with explicit next-session fills."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..backtest.execution import ExecutionSettings, PositionLot, ProfessionalPortfolio, TradeRecord


@dataclass
class PaperOrder:
    submitted_date: pd.Timestamp
    execution_date: pd.Timestamp
    code: str
    target_weight: float
    status: str = "pending"


@dataclass
class PaperAccount:
    """Persistent virtual account whose orders are filled only with supplied bars."""

    portfolio: ProfessionalPortfolio = field(default_factory=ProfessionalPortfolio)
    orders: list[PaperOrder] = field(default_factory=list)
    daily_pnl: list[dict[str, object]] = field(default_factory=list)

    def submit(self, orders: list[PaperOrder]) -> None:
        if any(order.target_weight < 0 for order in orders):
            raise ValueError("paper orders cannot contain negative weights")
        pending = {
            (order.execution_date.normalize(), order.code): order
            for order in self.orders
            if order.status == "pending"
        }
        for order in orders:
            key = (order.execution_date.normalize(), order.code)
            existing = pending.get(key)
            if existing is None:
                self.orders.append(order)
                pending[key] = order
            else:
                # Refresh a still-pending order from the latest score snapshot
                # instead of creating duplicate virtual orders.
                existing.target_weight = order.target_weight
                existing.submitted_date = order.submitted_date

    def process_next_open(self, date: pd.Timestamp, bars: pd.DataFrame) -> list[TradeRecord]:
        execution_date = pd.Timestamp(date).normalize()
        pending = [order for order in self.orders if order.status == "pending" and order.execution_date <= execution_date]
        if not pending:
            return []
        targets = {order.code: order.target_weight for order in pending}
        fills = self.portfolio.rebalance(execution_date, targets, bars)
        filled_codes = {trade.stock for trade in fills}
        for order in pending:
            # A zero target can be filled without a trade if it was already flat.
            order.status = "filled" if order.code in filled_codes or (order.target_weight == 0 and self.portfolio.position_quantity(order.code) == 0) else "pending"
        return fills

    def mark_to_market(self, date: pd.Timestamp, bars: pd.DataFrame) -> float:
        equity = self.portfolio.mark_to_market(date, bars)
        prior = self.daily_pnl[-1]["equity"] if self.daily_pnl else self.portfolio.initial_cash
        self.daily_pnl.append({"date": pd.Timestamp(date), "equity": equity, "pnl": equity - float(prior)})
        return equity

    def save(self, path: str | Path = "data/paper/account.json") -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": self.portfolio.cash,
            "positions": {code: self.portfolio.position_quantity(code) for code in self.portfolio.lots},
            "lots": {
                code: [
                    {"quantity": lot.quantity, "acquired_date": lot.acquired_date.isoformat()}
                    for lot in lots
                ]
                for code, lots in self.portfolio.lots.items()
            },
            "orders": [
                {
                    **asdict(order),
                    "submitted_date": order.submitted_date.isoformat(),
                    "execution_date": order.execution_date.isoformat(),
                }
                for order in self.orders
            ],
            "daily_pnl": [
                {**record, "date": pd.Timestamp(record["date"]).isoformat()} for record in self.daily_pnl
            ],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(
        cls, path: str | Path, settings: ExecutionSettings | None = None
    ) -> "PaperAccount":
        """Restore cash, T+1 lots, orders, and PnL from an account snapshot."""
        source = Path(path)
        if not source.exists():
            return default_paper_account(settings)
        payload = json.loads(source.read_text(encoding="utf-8"))
        portfolio = ProfessionalPortfolio(initial_cash=0.0, settings=settings or ExecutionSettings())
        portfolio.cash = float(payload["cash"])
        lots_payload = payload.get("lots")
        if lots_payload is None and payload.get("positions"):
            raise ValueError("paper-account snapshot lacks acquisition lots required for T+1 handling")
        for code, lots in (lots_payload or {}).items():
            portfolio.lots[str(code)] = [
                PositionLot(int(lot["quantity"]), pd.Timestamp(lot["acquired_date"]).normalize())
                for lot in lots
            ]
        orders = [
            PaperOrder(
                submitted_date=pd.Timestamp(order["submitted_date"]),
                execution_date=pd.Timestamp(order["execution_date"]),
                code=str(order["code"]).zfill(6),
                target_weight=float(order["target_weight"]),
                status=str(order.get("status", "pending")),
            )
            for order in payload.get("orders", [])
        ]
        pnl = [
            {**record, "date": pd.Timestamp(record["date"])} for record in payload.get("daily_pnl", [])
        ]
        return cls(portfolio=portfolio, orders=orders, daily_pnl=pnl)


class PaperTradingEngine:
    """Create pending next-trading-day orders from a composite-score snapshot."""

    def __init__(self, top_n: int = 20) -> None:
        if top_n < 1:
            raise ValueError("top_n must be at least one")
        self.top_n = top_n

    def plan_next_session(self, scores: pd.DataFrame) -> list[PaperOrder]:
        required = {"date", "code", "composite_score"}
        if not required.issubset(scores.columns):
            raise ValueError("scores require date, code, and composite_score")
        frame = scores.loc[:, ["date", "code", "composite_score"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["composite_score"] = pd.to_numeric(frame["composite_score"], errors="coerce")
        signal_date = frame["date"].max()
        selected = (
            frame.loc[frame["date"] == signal_date]
            .dropna(subset="composite_score")
            .sort_values(["composite_score", "code"], ascending=[False, True], kind="stable")
            .head(self.top_n)
        )
        if selected.empty:
            return []
        execution_date = signal_date + pd.offsets.BDay(1)
        return [
            PaperOrder(signal_date, execution_date, row.code, 1.0 / len(selected))
            for row in selected.itertuples(index=False)
        ]


def default_paper_account(settings: ExecutionSettings | None = None) -> PaperAccount:
    return PaperAccount(portfolio=ProfessionalPortfolio(settings=settings or ExecutionSettings()))
