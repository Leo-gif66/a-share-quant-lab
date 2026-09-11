import json
import os
from pathlib import Path

import pandas as pd

from quant.data.downloader import DataDownloader
from quant.data.providers.akshare import AKShareProvider
from quant.data.providers.tencent import TencentProvider
from quant.data.universe import Universe


def _akshare_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2024-01-03", "2024-01-02"],
            "开盘": [10.2, 10.0],
            "最高": [10.4, 10.3],
            "最低": [10.1, 9.9],
            "收盘": [10.3, 10.2],
            "成交量": [1000, 900],
            "成交额": [10200, 9200],
            "换手率": [1.2, 1.1],
        }
    )


class FakeAKShare:
    def stock_zh_a_hist(self, **kwargs):
        self.kwargs = kwargs
        return _akshare_history()


class ProxyAwareFakeAKShare:
    def stock_zh_a_hist(self, **kwargs):
        self.proxy_values = {
            "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
            "NO_PROXY": os.environ.get("NO_PROXY"),
        }
        return _akshare_history()


class FakeTencentResponse:
    def __init__(self, payload):
        self.text = f"kline_dayqfqsh600000={json.dumps(payload)}"

    def raise_for_status(self):
        return None


class FakeTencentSession:
    def get(self, url, params, timeout):
        self.url = url
        self.params = params
        self.timeout = timeout
        return FakeTencentResponse(
            {
                "code": 0,
                "data": {
                    "sh600000": {
                        "qfqday": [
                            ["2024-01-03", "10.2", "10.3", "10.4", "10.1", "1000", "10200"],
                            ["2024-01-02", "10.0", "10.2", "10.3", "9.9", "900", "9200"],
                        ]
                    }
                },
            }
        )


class FallbackTencentSession(FakeTencentSession):
    def __init__(self):
        self.urls: list[str] = []

    def get(self, url, params, timeout):
        self.urls.append(url)
        response = super().get(url, params, timeout)
        response.status_code = 501 if len(self.urls) == 1 else 200
        return response


class FakeUniverse:
    def stocks(self):
        return [{"code": "000001", "name": "Test stock", "category": "test"}]


class TwoStockUniverse:
    def stocks(self):
        return [
            {"code": "000001", "name": "Broken stock", "category": "test"},
            {"code": "000002", "name": "Working stock", "category": "test"},
        ]


class PartlyFailingProvider:
    def get_daily_history(self, symbol, start_date, end_date, adjust="qfq"):
        if symbol == "000001":
            raise RuntimeError("service unavailable")
        return AKShareProvider(ak_client=FakeAKShare()).get_daily_history(
            symbol, start_date, end_date, adjust
        )


def test_universe_can_be_read():
    stocks = Universe("configs/universe.yaml").stocks()

    assert stocks
    assert {"code", "name", "category"}.issubset(stocks[0])


def test_provider_returns_standard_fields():
    provider = AKShareProvider(ak_client=FakeAKShare())

    history = provider.get_daily_history("1", "20240101", "20240131")

    assert list(history.columns) == list(AKShareProvider.COLUMNS)
    assert history["date"].is_monotonic_increasing
    assert provider._ak.kwargs["symbol"] == "000001"
    assert provider._ak.kwargs["adjust"] == "qfq"


def test_provider_can_be_created_with_proxy_disabled_by_default():
    provider = AKShareProvider(ak_client=FakeAKShare())

    assert provider.disable_proxy is True


def test_provider_disables_proxy_for_akshare_requests(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    client = ProxyAwareFakeAKShare()
    provider = AKShareProvider(ak_client=client)

    provider.get_daily_history("000001", "20240101", "20240131")

    assert client.proxy_values == {
        "HTTP_PROXY": None,
        "HTTPS_PROXY": None,
        "NO_PROXY": "*",
    }
    assert os.environ["HTTP_PROXY"] == "http://proxy.example:8080"


def test_tencent_provider_returns_standard_fields():
    session = FakeTencentSession()
    provider = TencentProvider(session=session)

    history = provider.get_daily_history("600000", "20240101", "20240131")

    assert list(history.columns) == list(TencentProvider.COLUMNS)
    assert history["date"].is_monotonic_increasing
    assert history["amount"].notna().all()
    assert session.params["param"] == "sh600000,day,2024-01-01,2024-01-31,640,qfq"


def test_tencent_provider_retries_legacy_501_through_tencent_proxy():
    session = FallbackTencentSession()

    history = TencentProvider(session=session).get_daily_history("600000", "20240101", "20240131")

    assert not history.empty
    assert session.urls == [TencentProvider.ENDPOINT, TencentProvider.PROXY_ENDPOINT]


def test_downloader_defaults_to_tencent_with_akshare_fallback():
    downloader = DataDownloader(universe=FakeUniverse(), reporter=lambda _: None)

    assert isinstance(downloader.provider, TencentProvider)
    assert isinstance(downloader.fallback_provider, AKShareProvider)


def test_downloader_saves_parquet(tmp_path: Path):
    downloader = DataDownloader(
        universe=FakeUniverse(),
        provider=AKShareProvider(ak_client=FakeAKShare()),
        data_dir=tmp_path / "raw",
        start_date="20240101",
        end_date="20240131",
        reporter=lambda _: None,
    )

    result = downloader.update()
    path = tmp_path / "raw" / "000001.parquet"

    assert result["failed"] == {}
    assert path in result["saved"]
    assert list(pd.read_parquet(path).columns) == list(AKShareProvider.COLUMNS)


def test_downloader_falls_back_to_akshare(tmp_path: Path):
    messages: list[str] = []
    downloader = DataDownloader(
        universe=FakeUniverse(),
        provider=PartlyFailingProvider(),
        fallback_provider=AKShareProvider(ak_client=FakeAKShare()),
        data_dir=tmp_path / "raw",
        start_date="20240101",
        end_date="20240131",
        reporter=messages.append,
    )

    result = downloader.update()

    assert result["failed"] == {}
    assert (tmp_path / "raw" / "000001.parquet").exists()
    assert messages[0].startswith("FALLBACK 000001 PartlyFailingProvider failed:")


def test_downloader_continues_after_a_stock_failure(tmp_path: Path):
    messages: list[str] = []
    downloader = DataDownloader(
        universe=TwoStockUniverse(),
        provider=PartlyFailingProvider(),
        data_dir=tmp_path / "raw",
        start_date="20240101",
        end_date="20240131",
        reporter=messages.append,
        enable_fallback=False,
    )

    result = downloader.update()

    assert result["failed"] == {"000001": "service unavailable"}
    assert (tmp_path / "raw" / "000002.parquet").exists()
    assert messages[0] == "FAILED 000001 service unavailable"
