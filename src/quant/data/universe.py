"""Universe configuration with v0.6 and legacy-format compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


class Universe:

    def __init__(self, config_path: str | Path = "configs/universe.yaml"):
        self.config_path = Path(config_path)
        self.data = self._load()

    def _load(self):
        with self.config_path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict):
            raise TypeError("universe configuration must be a mapping")
        return data

    def stocks(self) -> list[dict[str, str | None]]:
        """Return normalized stocks for v0.6 and the legacy grouped format.

        v0.6 uses a flat ``stocks`` list with ``market`` and ``sector``.  The
        old mapping of ``category -> [stocks]`` remains supported, with its
        category becoming the normalized sector.
        """
        stocks = self.data.get("stocks")
        if isinstance(stocks, list):
            return [self._normalize_stock(stock) for stock in stocks]
        if isinstance(stocks, Mapping):
            result = []
            for category, items in stocks.items():
                if not isinstance(items, list):
                    raise TypeError(f"legacy universe category {category!r} must contain a list")
                result.extend(self._normalize_stock(stock, legacy_category=str(category)) for stock in items)
            return result
        raise ValueError("universe configuration requires a stocks list or category mapping")

    @staticmethod
    def _normalize_stock(
        stock: object, legacy_category: str | None = None
    ) -> dict[str, str | None]:
        if not isinstance(stock, Mapping):
            raise TypeError("every universe stock must be a mapping")
        if "code" not in stock or "name" not in stock:
            raise ValueError("every universe stock requires code and name")

        code = str(stock["code"]).zfill(6)
        sector = stock.get("sector", stock.get("industry", legacy_category))
        category = stock.get("category", sector)
        market = stock.get("market") or Universe._infer_market(code)
        # ``category`` is a compatibility alias for old consumers.  New code
        # should use ``sector``.
        return {
            "code": code,
            "name": str(stock["name"]),
            "market": str(market).upper(),
            "sector": str(sector) if sector is not None else None,
            "category": str(category) if category is not None else None,
        }

    @staticmethod
    def _infer_market(code: str) -> str:
        if code.startswith(("5", "6", "9")):
            return "SH"
        if code.startswith(("4", "8")):
            return "BJ"
        return "SZ"


if __name__ == "__main__":

    universe = Universe()

    for stock in universe.stocks():
        print(
            stock["code"],
            stock["name"],
            stock["market"],
            stock["sector"],
        )
