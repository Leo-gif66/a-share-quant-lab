from quant.data.components import InMemoryComponentLoader, SectorMapper
from quant.data.universe_builder import UniverseBuilder


def test_sector_mapper_prefers_code_mapping_then_component_metadata_then_default():
    mapper = SectorMapper({"000001": "Bank"})

    assert mapper.resolve("1", "Insurance") == "Bank"
    assert mapper.resolve("600001", "Industrial") == "Industrial"
    assert mapper.resolve("300001") == "Unknown"


def test_universe_builder_applies_sector_mapping_to_component_records():
    loader = InMemoryComponentLoader(
        {"hs300": [{"code": "000001", "name": "Ping An Bank", "market": "SZ"}]}
    )
    builder = UniverseBuilder(component_loader=loader, sector_mapper=SectorMapper({"000001": "Bank"}))

    stock = builder.import_index("CSI300")[0]

    assert stock["sector"] == "Bank"
    assert stock["category"] == "Bank"
    assert stock["index_source"] == "CSI300"
