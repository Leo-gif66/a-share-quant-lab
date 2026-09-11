"""Build normalized A-share universes from injectable index component loaders."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from .components.sector import SectorMapper
from .universe import Universe


@dataclass(frozen=True)
class IndexImportSpec:
    """Metadata required to import a component index universe."""

    key: str
    name: str
    index_code: str
    index_source: str


ComponentLoader = Callable[[IndexImportSpec], Iterable[Mapping[str, Any]]]


class UniverseBuilder:
    """Combine CSI 300, CSI 500, and ChiNext constituents into one universe.

    Component retrieval is deliberately injected. Production deployments can
    connect a licensed provider while tests and offline workflows can supply an
    in-memory loader; no downloaded component file is hardcoded in the project.
    """

    INDEXES: ClassVar[dict[str, IndexImportSpec]] = {
        "hs300": IndexImportSpec("hs300", "CSI 300", "000300", "CSI300"),
        "csi500": IndexImportSpec("csi500", "CSI 500", "000905", "CSI500"),
        "chinext": IndexImportSpec("chinext", "ChiNext", "399006", "ChiNext"),
    }
    _ALIASES: ClassVar[dict[str, str]] = {
        "hs300": "hs300",
        "csi300": "hs300",
        "000300": "hs300",
        "沪深300": "hs300",
        "����300": "hs300",
        "csi500": "csi500",
        "000905": "csi500",
        "中证500": "csi500",
        "����500": "csi500",
        "chinext": "chinext",
        "399006": "chinext",
        "创业板": "chinext",
    }

    def __init__(
        self,
        component_loader: ComponentLoader | object | None = None,
        sector_mapper: SectorMapper | None = None,
        industry_metadata: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.component_loader = component_loader
        self.sector_mapper = sector_mapper or SectorMapper()
        self.industry_metadata = {
            str(code).zfill(6): {str(field): str(value) for field, value in metadata.items()}
            for code, metadata in (industry_metadata or {}).items()
        }

    @classmethod
    def supported_indexes(cls) -> tuple[IndexImportSpec, ...]:
        """Return the large-universe index targets supported by this builder."""
        return tuple(cls.INDEXES.values())

    @classmethod
    def get_index_spec(cls, index: str) -> IndexImportSpec:
        """Resolve a stable key, code, or display name to an import specification."""
        key = cls._ALIASES.get(str(index).lower(), str(index).lower())
        try:
            return cls.INDEXES[key]
        except KeyError as exc:
            raise ValueError(f"unsupported index universe: {index}") from exc

    def import_index(self, index: str) -> list[dict[str, Any]]:
        """Load and normalize one index's constituents with sector metadata."""
        if self.component_loader is None:
            raise NotImplementedError("configure a component_loader to import index constituents")
        spec = self.get_index_spec(index)
        stocks = []
        for component in self._load_components(spec):
            normalized = Universe._normalize_stock(component)
            metadata = self.industry_metadata.get(normalized["code"])
            supplied_sector = (
                metadata.get("sector")
                if metadata is not None
                else component.get("sector", component.get("industry"))
            )
            normalized["sector"] = self.sector_mapper.resolve(
                normalized["code"], supplied_sector
            )
            normalized["category"] = normalized["sector"]
            if metadata is not None:
                normalized["industry"] = metadata.get("industry", "Unknown")
            elif component.get("industry") is not None:
                normalized["industry"] = str(component["industry"])
            normalized["index_source"] = spec.index_source
            stocks.append(normalized)
        return stocks

    def build(
        self, indexes: Sequence[str] = ("hs300", "csi500", "chinext")
    ) -> list[dict[str, Any]]:
        """Return a deduplicated multi-index universe with combined index sources."""
        combined: dict[str, dict[str, Any]] = {}
        for index in indexes:
            for stock in self.import_index(index):
                code = str(stock["code"])
                if code not in combined:
                    combined[code] = stock
                    continue
                existing = combined[code]
                sources = self._source_list(existing["index_source"])
                for source in self._source_list(stock["index_source"]):
                    if source not in sources:
                        sources.append(source)
                # Single-index records retain the scalar format accepted by
                # previous universe configs.  Duplicates use the explicit
                # list required by the real-universe import format.
                existing["index_source"] = sources if len(sources) > 1 else sources[0]
                if existing["sector"] in (None, "Unknown") and stock["sector"] not in (None, "Unknown"):
                    existing["sector"] = stock["sector"]
                    existing["category"] = stock["sector"]
        return [combined[code] for code in sorted(combined)]

    def write_universe(
        self,
        path: str | Path = "configs/universe_large.yaml",
        indexes: Sequence[str] = ("hs300", "csi500", "chinext"),
    ) -> list[dict[str, Any]]:
        """Build and serialize a portable YAML universe for ``quant data-update``."""
        stocks = self.build(indexes)
        payload = {
            "name": "a_share_large",
            "stocks": [
                {
                    "code": stock["code"],
                    "name": stock["name"],
                    "market": stock["market"],
                    "sector": stock["sector"],
                    "industry": stock.get("industry", "Unknown"),
                    "index_source": stock["index_source"],
                }
                for stock in stocks
            ],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return stocks

    def _load_components(self, spec: IndexImportSpec) -> Iterable[Mapping[str, Any]]:
        loader = self.component_loader
        if hasattr(loader, "load"):
            return loader.load(spec)  # type: ignore[union-attr]
        return loader(spec)  # type: ignore[operator]

    @staticmethod
    def _source_list(value: object) -> list[str]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(item) for item in value]
        return [source for source in str(value).split("|") if source]
