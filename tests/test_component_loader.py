from quant.data.components import InMemoryComponentLoader
from quant.data.universe_builder import UniverseBuilder


def test_injectable_component_loader_combines_overlapping_index_memberships():
    loader = InMemoryComponentLoader(
        {
            "hs300": [{"code": "000001", "name": "Ping An Bank", "market": "SZ"}],
            "csi500": [
                {"code": "000001", "name": "Ping An Bank", "market": "SZ"},
                {"code": "600001", "name": "Example Co", "market": "SH"},
            ],
            "chinext": [{"code": "300001", "name": "Example Growth", "market": "SZ"}],
        }
    )

    stocks = UniverseBuilder(component_loader=loader).build()

    assert [stock["code"] for stock in stocks] == ["000001", "300001", "600001"]
    assert stocks[0]["index_source"] == ["CSI300", "CSI500"]
    assert stocks[1]["index_source"] == "ChiNext"
