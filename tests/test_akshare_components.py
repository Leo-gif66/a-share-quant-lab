from __future__ import annotations

import pandas as pd

from quant.data.components.akshare_provider import AKShareComponentProvider


def _components(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": [f"{number:06d}" for number in range(count)],
            "name": [f"Stock {number}" for number in range(count)],
            "industry": ["Financials" if number % 2 else "Technology" for number in range(count)],
        }
    )


class FakeAKShare:
    def __init__(self, failing_code: str | None = None) -> None:
        self.failing_code = failing_code
        self.frames = {
            "000300": _components(300),
            "000905": _components(500),
            "399006": _components(100),
        }

    def index_stock_cons_csindex(self, symbol: str) -> pd.DataFrame:
        if symbol == self.failing_code:
            raise RuntimeError("network unavailable")
        return self.frames[symbol]


def test_akshare_component_provider_writes_validated_snapshot_schema(tmp_path):
    provider = AKShareComponentProvider(tmp_path, ak_client=FakeAKShare())

    result = provider.update_snapshots()

    assert result.failed == {}
    assert result.counts == {"CSI300": 300, "CSI500": 500, "ChiNext": 100}
    for filename, expected_count in (("csi300.csv", 300), ("csi500.csv", 500), ("chinext.csv", 100)):
        snapshot = pd.read_csv(tmp_path / filename, dtype={"code": str})
        assert list(snapshot.columns) == ["code", "name", "sector", "market"]
        assert len(snapshot) == expected_count
        assert snapshot["code"].str.len().eq(6).all()


def test_akshare_failure_keeps_existing_snapshot_and_continues_other_indexes(tmp_path):
    previous = tmp_path / "csi500.csv"
    previous.write_text("code,name,sector,market\n999999,Previous,Unknown,SZ\n", encoding="utf-8")
    messages: list[str] = []
    provider = AKShareComponentProvider(
        tmp_path,
        ak_client=FakeAKShare(failing_code="000905"),
        reporter=messages.append,
    )

    result = provider.update_snapshots()

    assert result.counts == {"CSI300": 300, "ChiNext": 100}
    assert "CSI500" in result.failed
    assert previous.read_text(encoding="utf-8") == "code,name,sector,market\n999999,Previous,Unknown,SZ\n"
    assert (tmp_path / "csi300.csv").exists()
    assert (tmp_path / "chinext.csv").exists()
    assert any(message.startswith("Downloading CSI500") for message in messages)
