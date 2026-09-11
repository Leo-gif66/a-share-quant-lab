import json
from pathlib import Path

import pandas as pd

from quant.data.downloader import DataDownloader


def _history(code: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [1000, 1100],
            "amount": [10100, 11220],
            "turnover": [1.0, 1.1],
        }
    )


class _Universe:
    def stocks(self):
        return [
            {"code": "000001", "name": "One"},
            {"code": "000002", "name": "Two"},
            {"code": "000003", "name": "Three"},
        ]


class _Primary:
    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def get_daily_history(self, symbol, start_date, end_date):
        self.calls.append((symbol, start_date, end_date))
        if symbol == "000002":
            raise RuntimeError("Tencent unavailable")
        if symbol == "000003":
            raise RuntimeError("delisted")
        return _history(symbol)


class _Fallback:
    def __init__(self):
        self.calls: list[str] = []

    def get_daily_history(self, symbol, start_date, end_date):
        self.calls.append(symbol)
        if symbol == "000003":
            raise RuntimeError("AKShare unavailable")
        return _history(symbol)


def test_full_pipeline_uses_fallback_keeps_going_and_writes_failure_snapshot(tmp_path: Path):
    primary = _Primary()
    fallback = _Fallback()
    result = DataDownloader(
        universe=_Universe(),
        provider=primary,
        fallback_provider=fallback,
        data_dir=tmp_path / "raw",
        start_date="20150101",
        end_date="20240131",
        retries=0,
        request_pause=0,
        max_workers=2,
        reporter=lambda _: None,
    ).update()

    assert {path.stem for path in result["saved"]} == {"000001", "000002"}
    assert result["failed"] == {
        "000003": "_Primary failed: delisted; _Fallback failed: AKShare unavailable"
    }
    assert {call[1] for call in primary.calls} == {"20150101"}
    assert set(fallback.calls) == {"000002", "000003"}
    assert list(pd.read_parquet(tmp_path / "raw" / "000002.parquet").columns) == list(
        DataDownloader.REQUIRED_COLUMNS
    )
    assert json.loads((tmp_path / "raw" / "download_failures.json").read_text()) == result["failed"]
    assert set(json.loads((tmp_path / "raw" / "download_history.json").read_text())) == {
        "000001",
        "000002",
    }
