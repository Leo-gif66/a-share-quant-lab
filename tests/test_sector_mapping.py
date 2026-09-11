from quant.data.components.sector_mapper import SectorMapper


def test_industry_classification_is_mapped_to_sector():
    mapper = SectorMapper(industry_sectors={"Commercial Bank": "Financials"})

    assert mapper.map_record(
        {"code": "000001", "name": "Ping An Bank", "industry_classification": "Commercial Bank"}
    ) == "Financials"


def test_sector_mapper_falls_back_to_unknown_when_no_industry_is_available():
    assert SectorMapper().map_record({"code": "300001", "name": "Unknown Co"}) == "Unknown"
