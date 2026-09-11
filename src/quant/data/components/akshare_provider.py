"""AKShare-backed downloader for reproducible index-component snapshots."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ComponentUpdateResult:
    """Outcome of an attempted multi-index snapshot refresh."""

    counts: dict[str, int]
    failed: dict[str, str]


class AKShareComponentProvider:
    """Fetch CSI 300, CSI 500 and ChiNext components through AKShare.

    Snapshots are validated in memory and atomically replaced only after a
    complete, valid response has been received.  Therefore a network or source
    failure leaves the previous CSV untouched.
    """

    INDEXES = {
        "hs300": ("CSI300", "000300", "csi300.csv", 280),
        "csi500": ("CSI500", "000905", "csi500.csv", 480),
        "chinext": ("ChiNext", "399006", "chinext.csv", 100),
    }
    SNAPSHOT_COLUMNS = ("code", "name", "sector", "market")

    def __init__(
        self,
        component_dir: str | Path = "data/components",
        ak_client: Any | None = None,
        reporter: Callable[[str], None] | None = None,
    ) -> None:
        if ak_client is None:
            import akshare as ak

            ak_client = ak
        self.component_dir = Path(component_dir)
        self._ak = ak_client
        self._report = reporter or (lambda _message: None)

    def update_snapshots(self) -> ComponentUpdateResult:
        """Refresh all component CSVs, retaining independent prior snapshots on errors."""
        counts: dict[str, int] = {}
        failed: dict[str, str] = {}
        for key in self.INDEXES:
            source = self.INDEXES[key][0]
            self._report(f"Downloading {source} constituents...")
            try:
                count = self.download_index(key)
            except Exception as exc:  # noqa: BLE001 - network boundary, continue to other indexes
                message = str(exc) or exc.__class__.__name__
                failed[source] = message
                self._report(f"FAILED {source}: {message}; previous snapshot retained")
            else:
                counts[source] = count
                self._report(f"OK {source}: {count} stocks")
        return ComponentUpdateResult(counts=counts, failed=failed)

    # Short aliases keep the provider convenient for schedulers and notebooks.
    def download_csi300(self) -> int:
        return self.download_index("hs300")

    def download_csi500(self) -> int:
        return self.download_index("csi500")

    def download_chinext(self) -> int:
        return self.download_index("chinext")

    def download_index(self, key: str) -> int:
        """Fetch, validate, and atomically save one configured component snapshot."""
        try:
            _source, _index_code, filename, _minimum = self.INDEXES[key]
        except KeyError as exc:
            raise ValueError(f"unsupported component index: {key}") from exc
        frame = self._fetch_snapshot(key)
        self._write_snapshot_atomically(frame, self.component_dir / filename)
        return len(frame)

    def load(self, spec: Any) -> Iterable[Mapping[str, str]]:
        """Implement the component-provider interface for direct injected use."""
        frame = self._fetch_snapshot(str(spec.key))
        return [
            {
                "code": row.code,
                "name": row.name,
                "sector": row.sector,
                "market": row.market,
                "index_source": str(spec.index_source),
            }
            for row in frame.itertuples(index=False)
        ]

    def load_csi300(self) -> list[dict[str, str]]:
        return list(self.load(self._spec("csi300")))

    def load_csi500(self) -> list[dict[str, str]]:
        return list(self.load(self._spec("csi500")))

    def load_chinext(self) -> list[dict[str, str]]:
        return list(self.load(self._spec("chinext")))

    @staticmethod
    def _spec(index: str) -> Any:
        from ..universe_builder import UniverseBuilder

        return UniverseBuilder.get_index_spec(index)

    def _fetch_snapshot(self, key: str) -> pd.DataFrame:
        source, index_code, _filename, minimum = self.INDEXES[key]
        raw = self._fetch_from_akshare(index_code)
        snapshot = self._normalize(raw)
        if len(snapshot) < minimum:
            raise ValueError(
                f"{source} returned {len(snapshot)} valid rows; expected at least {minimum}"
            )
        return snapshot

    def _fetch_from_akshare(self, index_code: str) -> pd.DataFrame:
        errors: list[str] = []
        # The CSIndex endpoint is typically the most complete source for CSI
        # indexes; Sina's endpoint is a useful fallback, notably for ChiNext.
        for method_name in ("index_stock_cons_csindex", "index_stock_cons"):
            method = getattr(self._ak, method_name, None)
            if method is None:
                continue
            try:
                frame = method(symbol=index_code)
            except Exception as exc:  # noqa: BLE001 - collect fallback errors for diagnosis
                errors.append(f"{method_name}: {str(exc) or exc.__class__.__name__}")
                continue
            if not isinstance(frame, pd.DataFrame):
                errors.append(f"{method_name}: expected DataFrame, got {type(frame).__name__}")
                continue
            if frame.empty:
                errors.append(f"{method_name}: empty response")
                continue
            return frame
        detail = "; ".join(errors) or "no supported AKShare component endpoint"
        raise RuntimeError(f"AKShare component download failed for {index_code}: {detail}")

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        code_column = self._find_column(raw, "code")
        name_column = self._find_column(raw, "name")
        sector_column = self._find_column(raw, "sector", required=False)

        result = pd.DataFrame(
            {
                "code": raw[code_column].map(self._normalise_code),
                "name": raw[name_column].astype(str).str.strip(),
                "sector": (
                    raw[sector_column].astype(str).str.strip()
                    if sector_column is not None
                    else "Unknown"
                ),
            }
        )
        result["sector"] = result["sector"].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        result = result[(result["code"] != "") & (result["name"] != "")].copy()
        result = result.drop_duplicates(subset=["code"], keep="first")
        result["market"] = result["code"].map(self._infer_market)
        return result.loc[:, self.SNAPSHOT_COLUMNS].sort_values("code").reset_index(drop=True)

    @staticmethod
    def _find_column(raw: pd.DataFrame, kind: str, required: bool = True) -> object | None:
        columns = list(raw.columns)
        normalized = {column: str(column).strip().lower().replace(" ", "") for column in columns}
        if kind == "code":
            exact = {"code", "symbol"}
            priorities = (
                ("成分", "代码"),
                ("品种", "代码"),
                ("证券", "代码"),
                ("股票", "代码"),
                ("constituent", "code"),
            )
            fallback = lambda value: ("code" in value and "index" not in value) or (
                "代码" in value and "指数" not in value
            )
        elif kind == "name":
            exact = {"name"}
            priorities = (
                ("成分", "名称"),
                ("品种", "名称"),
                ("证券", "名称"),
                ("股票", "名称"),
                ("constituent", "name"),
            )
            fallback = lambda value: ("name" in value and "index" not in value) or (
                "名称" in value and "指数" not in value
            )
        else:
            exact = {"sector", "industry", "industryclassification"}
            priorities = (("行业",), ("所属行业",))
            fallback = lambda _value: False

        for column, value in normalized.items():
            if value in exact:
                return column
        for priority in priorities:
            for column, value in normalized.items():
                if all(token in value for token in priority):
                    return column
        for column, value in normalized.items():
            if fallback(value):
                return column
        if required:
            raise ValueError(f"AKShare component data has no {kind} column: {list(raw.columns)!r}")
        return None

    @staticmethod
    def _normalise_code(value: object) -> str:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        match = re.search(r"(\d{6})$", text)
        return match.group(1) if match else ""

    @staticmethod
    def _infer_market(code: str) -> str:
        if code.startswith(("5", "6", "9")):
            return "SH"
        if code.startswith(("4", "8")):
            return "BJ"
        return "SZ"

    @staticmethod
    def _write_snapshot_atomically(frame: pd.DataFrame, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=target.parent, delete=False, suffix=".csv"
            ) as handle:
                temporary_path = Path(handle.name)
                frame.to_csv(handle, index=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
