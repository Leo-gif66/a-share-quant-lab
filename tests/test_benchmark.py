import json

import pandas as pd

from quant.data.providers.tencent_index import TencentIndexProvider
from quant.data.storage import Storage


class FakeTencentResponse:
    def __init__(self, payload):
        self.text = f"kline_day_sh000300={json.dumps(payload)}"

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
                    "sh000300": {
                        "day": [
                            ["2024-01-03", "3400", "3390", "3410", "3380", "1200"],
                            ["2024-01-02", "3380", "3400", "3420", "3370", "1000"],
                        ]
                    }
                },
            }
        )


def test_tencent_index_provider_returns_csi300_daily_schema():
    session = FakeTencentSession()
    provider = TencentIndexProvider(session=session)

    history = provider.get_daily_history("csi000300", "20240101", "20240131")

    assert list(history.columns) == list(TencentIndexProvider.COLUMNS)
    assert history["date"].is_monotonic_increasing
    assert history["amount"].tolist() == [3_400_000.0, 4_068_000.0]
    assert session.params["param"] == "sh000300,day,2024-01-01,2024-01-31,640"


def test_benchmark_storage_writes_csi300_to_raw_root(tmp_path):
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "open": [3380.0],
            "high": [3420.0],
            "low": [3370.0],
            "close": [3400.0],
            "volume": [1000.0],
            "amount": [3_400_000.0],
        }
    )

    storage = Storage(tmp_path)
    path = storage.write_benchmark(benchmark)

    assert path == tmp_path / "raw" / "000300.parquet"
    pd.testing.assert_frame_equal(storage.read_benchmark(), benchmark)
