"""Industry classification providers and snapshot storage."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class IndustryProvider(Protocol):
    """Source of normalized A-share industry classifications."""

    def fetch(self) -> pd.DataFrame:
        """Return ``code``, ``name``, ``sector`` and ``industry`` columns."""


@dataclass(frozen=True)
class IndustryUpdateResult:
    """Result of a refresh, including any board-level network failures."""

    records: int
    industries: int
    failed: dict[str, str]


class AKShareIndustryProvider:
    """Build a stock-to-industry table from AKShare's Sina industry boards.

    AKShare exposes the board roster and every board's constituents separately.
    This provider joins those two datasets, maps detailed industries into a
    stable broad-sector taxonomy, and retains the detailed source industry.
    """

    OUTPUT_COLUMNS = ("code", "name", "sector", "industry")
    SNAPSHOT_COLUMNS = ("code", "sector", "industry")
    SECTOR_RULES = (
        ("Financials", ("银行", "保险", "证券", "金融", "bank", "insurance", "brokerage")),
        ("Healthcare", ("医药", "医疗", "生物", "保健", "health", "pharma", "medical")),
        ("Technology", ("半导体", "电子", "软件", "计算机", "通信", "互联网", "元件", "technology", "software", "semiconductor")),
        ("Consumer", ("食品", "饮料", "白酒", "家电", "零售", "服装", "纺织", "旅游", "酒店", "餐饮", "美容", "consumer", "food", "beverage")),
        ("Real Estate", ("房地产", "物业", "real estate")),
        ("Energy", ("煤炭", "石油", "天然气", "油气", "energy", "coal", "oil", "gas")),
        ("Utilities", ("电力", "水务", "供气", "环保", "utility", "utilities")),
        ("Materials", ("化工", "有色", "钢铁", "建材", "水泥", "玻璃", "橡胶", "塑料", "材料", "materials", "chemicals")),
        ("Industrials", ("机械", "设备", "工程", "建筑", "运输", "物流", "船舶", "航空", "军工", "工业", "industrial", "machinery")),
    )

    def __init__(
        self,
        output_path: str | Path = "data/components/industry.csv",
        ak_client: Any | None = None,
        reporter: Callable[[str], None] | None = None,
        target_stocks: Mapping[str, str] | None = None,
    ) -> None:
        if ak_client is None:
            import akshare as ak

            ak_client = ak
        self.output_path = Path(output_path)
        self._ak = ak_client
        self._report = reporter or (lambda _message: None)
        self.target_stocks = {
            self._normalise_code(code): str(name)
            for code, name in (target_stocks or {}).items()
            if self._normalise_code(code)
        }

    def fetch(self) -> pd.DataFrame:
        """Download broad-sector and detailed-industry assignments for A shares."""
        boards = self._ak.stock_sector_spot(indicator="行业")
        if not isinstance(boards, pd.DataFrame) or boards.empty:
            raise RuntimeError("AKShare returned no industry boards")
        label_column = self._find_column(boards, ("label", "板块代码", "code"))
        industry_column = self._find_column(boards, ("industry", "sector", "board", "板块", "行业"), fallback_index=1)

        records: list[pd.DataFrame] = []
        failures: dict[str, str] = {}
        total = len(boards)
        for position in range(1, total + 1):
            # Index directly because ``itertuples`` rewrites non-identifier
            # Chinese headers returned by AKShare.
            label = str(boards.iloc[position - 1][label_column]).strip()
            industry = str(boards.iloc[position - 1][industry_column]).strip()
            if not label or not industry or industry.lower() == "nan":
                continue
            self._report(f"Downloading industry {position}/{total}: {industry}")
            try:
                members = self._ak.stock_sector_detail(sector=label)
                records.append(self._normalise_members(members, industry))
            except Exception as exc:  # noqa: BLE001 - retain usable classifications from other boards
                failures[industry] = str(exc) or exc.__class__.__name__

        if not records:
            detail = "; ".join(f"{industry}: {error}" for industry, error in failures.items())
            raise RuntimeError(f"AKShare returned no industry constituents ({detail})")

        result = pd.concat(records, ignore_index=True)
        result = result.drop_duplicates(subset=["code"], keep="first")
        if self.target_stocks:
            result = self._fill_target_gaps(result, failures)
        result.attrs["failed"] = failures
        return result.loc[:, self.OUTPUT_COLUMNS].sort_values("code").reset_index(drop=True)

    def _fill_target_gaps(self, frame: pd.DataFrame, failures: dict[str, str]) -> pd.DataFrame:
        """Use CNInfo history through AKShare for target stocks absent from Sina boards."""
        known_codes = set(frame["code"])
        missing = sorted(set(self.target_stocks).difference(known_codes))
        if not missing:
            return frame
        fallback_records = []
        for position, code in enumerate(missing, start=1):
            self._report(f"Downloading industry fallback {position}/{len(missing)}: {code}")
            try:
                fallback_records.append(self._fetch_cninfo_industry(code, self.target_stocks[code]))
            except Exception as exc:  # noqa: BLE001 - leave only genuinely unavailable data unclassified
                failures[code] = str(exc) or exc.__class__.__name__
        if fallback_records:
            frame = pd.concat([frame, *fallback_records], ignore_index=True)
            frame = frame.drop_duplicates(subset=["code"], keep="first")
        return frame

    def _fetch_cninfo_industry(self, code: str, name: str) -> pd.DataFrame:
        method = getattr(self._ak, "stock_industry_change_cninfo", None)
        if method is None:
            raise RuntimeError("AKShare has no CNInfo industry-history endpoint")
        history = method(symbol=code, start_date="19900101", end_date="20991231")
        if not isinstance(history, pd.DataFrame) or history.empty:
            raise ValueError("empty CNInfo industry response")
        industry = self._latest_industry(history)
        if not industry:
            raise ValueError("CNInfo industry response has no usable classification")
        return pd.DataFrame(
            [{"code": code, "name": name, "sector": self._sector_for(industry), "industry": industry}]
        )

    def update_snapshot(self) -> IndustryUpdateResult:
        """Fetch and atomically replace ``industry.csv`` only after a valid result."""
        try:
            frame = self.fetch()
        except RuntimeError as exc:
            # Industry board endpoints can be temporarily rate-limited.  A
            # prior valid snapshot remains a useful base for targeted CNInfo
            # reconciliation, and is never replaced with an empty response.
            frame = self._existing_snapshot_frame()
            if frame.empty:
                raise
            failures = {"industry_boards": str(exc) or exc.__class__.__name__}
            if self.target_stocks:
                frame = self._fill_target_gaps(frame, failures)
            frame.attrs["failed"] = failures
        if frame.empty:
            raise RuntimeError("refusing to replace industry snapshot with no records")
        self._write_atomically(frame.loc[:, self.SNAPSHOT_COLUMNS], self.output_path)
        return IndustryUpdateResult(
            records=len(frame),
            industries=int(frame["industry"].nunique()),
            failed=dict(frame.attrs.get("failed", {})),
        )

    def _existing_snapshot_frame(self) -> pd.DataFrame:
        if not self.output_path.exists():
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)
        snapshot = pd.read_csv(self.output_path, dtype={"code": str})
        if not set(self.SNAPSHOT_COLUMNS).issubset(snapshot.columns):
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)
        result = snapshot.loc[:, self.SNAPSHOT_COLUMNS].copy()
        result["code"] = result["code"].map(self._normalise_code)
        result["name"] = result["code"].map(self.target_stocks).fillna("")
        return result.loc[:, self.OUTPUT_COLUMNS]

    def _normalise_members(self, raw: pd.DataFrame, industry: str) -> pd.DataFrame:
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise ValueError("empty constituent response")
        code_column = self._find_column(raw, ("code", "symbol", "代码", "证券代码", "股票代码"))
        name_column = self._find_column(raw, ("name", "名称", "股票名称"))
        result = pd.DataFrame(
            {
                "code": raw[code_column].map(self._normalise_code),
                "name": raw[name_column].astype(str).str.strip(),
                "sector": self._sector_for(industry),
                "industry": industry,
            }
        )
        return result[(result["code"] != "") & (result["name"] != "")].copy()

    @staticmethod
    def _find_column(
        frame: pd.DataFrame, candidates: tuple[str, ...], fallback_index: int | None = None
    ) -> object:
        normalized = {column: str(column).strip().lower().replace(" ", "") for column in frame.columns}
        for candidate in candidates:
            candidate_normalized = candidate.lower().replace(" ", "")
            for column, value in normalized.items():
                if value == candidate_normalized:
                    return column
        for candidate in candidates:
            candidate_normalized = candidate.lower().replace(" ", "")
            for column, value in normalized.items():
                if candidate_normalized in value:
                    return column
        if fallback_index is not None and len(frame.columns) > fallback_index:
            return frame.columns[fallback_index]
        raise ValueError(f"industry data missing expected columns: {list(frame.columns)!r}")

    @classmethod
    def _sector_for(cls, industry: str) -> str:
        value = industry.lower()
        for sector, keywords in cls.SECTOR_RULES:
            if any(keyword.lower() in value for keyword in keywords):
                return sector
        # The detailed board itself is known, even when it does not map to a
        # predefined broad group.  ``Other`` is more informative than Unknown.
        return "Other"

    @staticmethod
    def _latest_industry(history: pd.DataFrame) -> str:
        """Extract the most detailed nonempty industry from the latest CNInfo row."""
        priorities = ("industry", "行业中类", "行业大类", "行业次类", "行业门类")
        normalized = {column: str(column).strip().lower().replace(" ", "") for column in history.columns}
        columns: list[object] = []
        for priority in priorities:
            value = priority.lower().replace(" ", "")
            columns.extend(column for column, label in normalized.items() if label == value)
        # A classification code is not an industry name, so only use explicit
        # name fields above; spelling variants are handled by this fallback.
        if not columns:
            columns = [column for column, label in normalized.items() if "industry" in label or "行业" in label]
        for _, row in history.iloc[::-1].iterrows():
            for column in columns:
                value = str(row[column]).strip()
                if value and value.lower() not in {"nan", "none"} and not value.isdigit():
                    return value
        return ""

    @staticmethod
    def _normalise_code(value: object) -> str:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        match = re.search(r"(\d{6})$", text)
        return match.group(1) if match else ""

    @staticmethod
    def _write_atomically(frame: pd.DataFrame, target: Path) -> None:
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


def load_industry_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    """Read the saved metadata into a code-keyed mapping for universe construction."""
    source = Path(path)
    if not source.exists():
        return {}
    frame = pd.read_csv(source, dtype={"code": str})
    required = {"code", "sector", "industry"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"industry snapshot missing columns: {', '.join(sorted(missing))}")
    result: dict[str, dict[str, str]] = {}
    for row in frame.loc[:, ["code", "sector", "industry"]].itertuples(index=False):
        code = AKShareIndustryProvider._normalise_code(row.code)
        if not code:
            continue
        sector = str(row.sector).strip()
        industry = str(row.industry).strip()
        if sector and sector.lower() != "nan":
            result[code] = {
                "sector": sector,
                "industry": industry if industry and industry.lower() != "nan" else "Unknown",
            }
    return result
