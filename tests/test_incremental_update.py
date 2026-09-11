from pathlib import Path

import pandas as pd

from quant.data.downloader import DataDownloader


class _Universe:
    def stocks(self):
        return [{"code": "000001", "name": "One"}]


def _rows(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": closes,
            "high": [value + 0.1 for value in closes],
            "low": [value - 0.1 for value in closes],
            "close": closes,
            "volume": [1000] * len(closes),
            "amount": [10000] * len(closes),
            "turnover": [1.0] * len(closes),
        }
    )


class _IncrementalProvider:
    def __init__(self):
        self.starts: list[str] = []

    def get_daily_history(self, symbol, start_date, end_date):
        self.starts.append(start_date)
        return _rows(["2024-01-04", "2024-01-05"], [10.3, 10.4])


def test_incremental_update_requests_only_tail_and_deduplicates_dates(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _rows(["2024-01-02", "2024-01-03"], [10.1, 10.2]).to_parquet(raw_dir / "000001.parquet")
    provider = _IncrementalProvider()
    downloader = DataDownloader(
        universe=_Universe(),
        provider=provider,
        enable_fallback=False,
        data_dir=raw_dir,
        start_date="20240102",
        end_date="20240105",
        retries=0,
        request_pause=0,
        reporter=lambda _: None,
    )

    result = downloader.update()
    saved = pd.read_parquet(raw_dir / "000001.parquet")

    assert provider.starts == ["20240104"]
    assert result["failed"] == {}
    assert saved["date"].tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))

    repeat = downloader.update()

    assert provider.starts == ["20240104"]
    assert repeat["saved"] == []
    assert [path.stem for path in repeat["skipped"]] == ["000001"]
