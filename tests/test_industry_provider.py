from __future__ import annotations

import pandas as pd

from quant.data.industry import AKShareIndustryProvider, IndustryProvider


class FakeIndustryAKShare:
    def stock_sector_spot(self, indicator: str) -> pd.DataFrame:
        assert indicator == "行业"
        return pd.DataFrame(
            {
                "label": ["food", "semiconductors"],
                "board": ["Food&Beverage", "Semiconductors"],
            }
        )

    def stock_sector_detail(self, sector: str) -> pd.DataFrame:
        rows = {
            "food": {"code": ["600519"], "name": ["Kweichow Moutai"]},
            "semiconductors": {"code": ["300750"], "name": ["CATL"]},
        }
        return pd.DataFrame(rows[sector])


def test_akshare_industry_provider_returns_and_saves_normalized_metadata(tmp_path):
    output = tmp_path / "industry.csv"
    provider = AKShareIndustryProvider(output, ak_client=FakeIndustryAKShare())

    frame = provider.fetch()
    result = provider.update_snapshot()

    assert isinstance(provider, IndustryProvider)
    assert list(frame.columns) == ["code", "name", "sector", "industry"]
    assert frame.set_index("code").loc["600519", "sector"] == "Consumer"
    assert frame.set_index("code").loc["300750", "sector"] == "Technology"
    assert result.records == 2
    assert result.industries == 2
    assert list(pd.read_csv(output, dtype={"code": str}).columns) == ["code", "sector", "industry"]
