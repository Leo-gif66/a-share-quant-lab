"""Local-first providers for real A-share index component files.

Component membership changes over time and is therefore supplied as data rather
than embedded in the application.  A local CSV snapshot is always preferred;
an optional injected remote provider is only consulted when it is unavailable.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from .sector_mapper import SectorMapper


@runtime_checkable
class ComponentProvider(Protocol):
    """Interface implemented by sources of CSI 300, CSI 500 and ChiNext members."""

    def load(self, spec: Any) -> Iterable[Mapping[str, Any]]:
        """Load one index specified by ``UniverseBuilder.IndexImportSpec``."""

    def load_csi300(self) -> list[dict[str, str]]:
        """Load CSI 300 constituents."""

    def load_csi500(self) -> list[dict[str, str]]:
        """Load CSI 500 constituents."""

    def load_chinext(self) -> list[dict[str, str]]:
        """Load ChiNext constituents."""


RemoteLoader = Callable[[Any], Iterable[Mapping[str, Any]]]


class LocalFirstComponentProvider:
    """Read component CSV snapshots before consulting an injected remote source.

    Files may use the conventional names ``csi300.csv``, ``csi500.csv`` and
    ``chinext.csv`` (index codes and common aliases are also recognised).  CSV
    columns can be English or commonly used Chinese names.
    """

    _FILE_ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "hs300": ("hs300", "csi300", "000300"),
        "csi500": ("csi500", "000905"),
        "chinext": ("chinext", "399006", "cyb"),
    }
    _CODE_FIELDS = ("code", "symbol", "证券代码", "成分券代码", "成分股代码", "股票代码")
    _NAME_FIELDS = ("name", "证券简称", "成分券名称", "成分股名称", "股票简称", "简称")
    _MARKET_FIELDS = ("market", "exchange", "市场", "交易所")
    _SECTOR_FIELDS = ("sector", "industry", "industry_classification", "industry_name", "行业", "所属行业")

    def __init__(
        self,
        component_dir: str | Path = "data/components",
        remote_provider: RemoteLoader | object | None = None,
        sector_mapper: SectorMapper | None = None,
    ) -> None:
        self.component_dir = Path(component_dir)
        self.remote_provider = remote_provider
        self.sector_mapper = sector_mapper or SectorMapper()

    def load(self, spec: Any) -> list[dict[str, str]]:
        """Load and normalize one index while retaining its canonical source name."""
        local_path = self._find_local_file(str(spec.key))
        if local_path is not None:
            records = self._read_csv(local_path)
        elif self.remote_provider is not None:
            records = list(self._load_remote(spec))
        else:
            aliases = ", ".join(f"{alias}.csv" for alias in self._FILE_ALIASES[str(spec.key)])
            raise FileNotFoundError(
                f"no local component file for {spec.index_source} in {self.component_dir} "
                f"(expected one of: {aliases}); configure an optional remote provider to fetch it"
            )
        return [self._normalize_record(record, str(spec.index_source)) for record in records]

    def load_csi300(self) -> list[dict[str, str]]:
        return self.load(self._spec("csi300"))

    def load_csi500(self) -> list[dict[str, str]]:
        return self.load(self._spec("csi500"))

    def load_chinext(self) -> list[dict[str, str]]:
        return self.load(self._spec("chinext"))

    @staticmethod
    def _spec(index: str) -> Any:
        # Late import avoids a module-import cycle with UniverseBuilder.
        from ..universe_builder import UniverseBuilder

        return UniverseBuilder.get_index_spec(index)

    def _find_local_file(self, key: str) -> Path | None:
        if not self.component_dir.exists():
            return None
        aliases = self._FILE_ALIASES.get(key, (key,))
        matches = []
        for path in self.component_dir.glob("*.csv"):
            stem = re.sub(r"[^a-z0-9]", "", path.stem.lower())
            if any(alias in stem for alias in aliases):
                matches.append(path)
        return min(matches) if matches else None

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        last_error: UnicodeDecodeError | None = None
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                with path.open("r", encoding=encoding, newline="") as handle:
                    return list(csv.DictReader(handle))
            except UnicodeDecodeError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _load_remote(self, spec: Any) -> Iterable[Mapping[str, Any]]:
        remote = self.remote_provider
        if hasattr(remote, "load"):
            return remote.load(spec)  # type: ignore[union-attr]
        return remote(spec)  # type: ignore[operator]

    def _normalize_record(self, record: Mapping[str, Any], index_source: str) -> dict[str, str]:
        code = self._code(self._field(record, self._CODE_FIELDS))
        name = self._field(record, self._NAME_FIELDS)
        if not code or not name:
            raise ValueError(f"component record requires code and name: {dict(record)!r}")
        market = self._field(record, self._MARKET_FIELDS) or self._infer_market(code)
        sector_value = self._field(record, self._SECTOR_FIELDS)
        sector = self.sector_mapper.resolve(code, supplied_sector=sector_value)
        return {
            "code": code,
            "name": name,
            "index_source": index_source,
            "market": market.upper(),
            "sector": sector,
        }

    @staticmethod
    def _field(record: Mapping[str, Any], fields: tuple[str, ...]) -> str:
        lower = {str(key).strip().lower(): value for key, value in record.items()}
        for field in fields:
            value = lower.get(field.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _code(value: str) -> str:
        value = value.strip()
        if value.endswith(".0") and value[:-2].isdigit():
            value = value[:-2]
        match = re.search(r"(\d{6})$", value)
        return match.group(1) if match else ""

    @staticmethod
    def _infer_market(code: str) -> str:
        if code.startswith(("5", "6", "9")):
            return "SH"
        if code.startswith(("4", "8")):
            return "BJ"
        return "SZ"
