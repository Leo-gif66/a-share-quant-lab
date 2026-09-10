"""Interfaces for future index-constituent universe imports."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from .universe import Universe


@dataclass(frozen=True)
class IndexImportSpec:
    """Metadata needed by a future index-constituent data adapter."""

    key: str
    name: str
    index_code: str


ComponentLoader = Callable[[IndexImportSpec], Iterable[Mapping[str, Any]]]


class UniverseBuilder:
    """Define index import targets without coupling to a market-data source.

    A future provider can be supplied as ``component_loader``.  This module
    intentionally performs no network requests itself.
    """

    INDEXES: ClassVar[dict[str, IndexImportSpec]] = {
        "hs300": IndexImportSpec("hs300", "沪深300", "000300"),
        "csi500": IndexImportSpec("csi500", "中证500", "000905"),
        "chinext": IndexImportSpec("chinext", "创业板", "399006"),
    }
    _ALIASES: ClassVar[dict[str, str]] = {
        "沪深300": "hs300",
        "中证500": "csi500",
        "创业板": "chinext",
    }

    def __init__(self, component_loader: ComponentLoader | None = None) -> None:
        self.component_loader = component_loader

    @classmethod
    def supported_indexes(cls) -> tuple[IndexImportSpec, ...]:
        """Return the index targets available for a future component loader."""
        return tuple(cls.INDEXES.values())

    @classmethod
    def get_index_spec(cls, index: str) -> IndexImportSpec:
        """Resolve a stable key or Chinese display name to an import spec."""
        key = cls._ALIASES.get(index, index)
        try:
            return cls.INDEXES[key]
        except KeyError as exc:
            raise ValueError(f"unsupported index universe: {index}") from exc

    def import_index(self, index: str) -> list[dict[str, str | None]]:
        """Normalize injected component data into the v0.6 universe contract.

        No default loader is included by design; callers must explicitly attach
        a future market-data adapter before index components can be imported.
        """
        if self.component_loader is None:
            raise NotImplementedError("configure a component_loader to import index constituents")
        spec = self.get_index_spec(index)
        return [Universe._normalize_stock(stock) for stock in self.component_loader(spec)]
