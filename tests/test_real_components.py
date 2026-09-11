from __future__ import annotations

from quant.data.components.provider import LocalFirstComponentProvider


def test_local_component_csvs_are_preferred_and_normalized(tmp_path):
    (tmp_path / "csi300.csv").write_text(
        "code,name,industry_classification\n000001,Ping An Bank,Bank\n",
        encoding="utf-8",
    )

    class Remote:
        def load(self, _spec):
            raise AssertionError("the remote provider must not be called when CSV exists")

    provider = LocalFirstComponentProvider(tmp_path, remote_provider=Remote())

    assert provider.load_csi300() == [
        {
            "code": "000001",
            "name": "Ping An Bank",
            "index_source": "CSI300",
            "market": "SZ",
            "sector": "Bank",
        }
    ]


def test_remote_provider_is_an_optional_fallback_when_no_local_csv_exists(tmp_path):
    class Remote:
        def load(self, _spec):
            return [{"code": "600001", "name": "Remote Co", "industry": "Industrial"}]

    provider = LocalFirstComponentProvider(tmp_path, remote_provider=Remote())

    assert provider.load_csi500()[0] == {
        "code": "600001",
        "name": "Remote Co",
        "index_source": "CSI500",
        "market": "SH",
        "sector": "Industrial",
    }
