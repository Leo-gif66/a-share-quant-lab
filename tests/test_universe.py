from pathlib import Path

import pytest

from quant.data.universe import Universe
from quant.data.universe_builder import UniverseBuilder


def test_universe_reads_v06_flat_stock_list(tmp_path: Path):
    config = tmp_path / "universe.yaml"
    config.write_text(
        """stocks:
  - code: \"300124\"
    name: \"汇川技术\"
    market: SZ
    sector: automation
""",
        encoding="utf-8",
    )

    stocks = Universe(config).stocks()

    assert stocks == [
        {
            "code": "300124",
            "name": "汇川技术",
            "market": "SZ",
            "sector": "automation",
            "category": "automation",
        }
    ]


def test_universe_keeps_legacy_grouped_format_compatible(tmp_path: Path):
    config = tmp_path / "universe.yaml"
    config.write_text(
        """stocks:
  finance:
    - code: \"600030\"
      name: \"中信证券\"
""",
        encoding="utf-8",
    )

    stock = Universe(config).stocks()[0]

    assert stock["market"] == "SH"
    assert stock["sector"] == "finance"
    assert stock["category"] == "finance"


def test_universe_builder_exposes_injected_index_import_interface():
    received = []

    def component_loader(spec):
        received.append(spec)
        return [{"code": "300124", "name": "汇川技术", "market": "SZ", "sector": "automation"}]

    builder = UniverseBuilder(component_loader=component_loader)
    stocks = builder.import_index("沪深300")

    assert received[0].index_code == "000300"
    assert stocks[0]["code"] == "300124"
    assert {spec.key for spec in builder.supported_indexes()} == {"hs300", "csi500", "chinext"}


def test_universe_builder_requires_an_explicit_component_loader():
    with pytest.raises(NotImplementedError, match="component_loader"):
        UniverseBuilder().import_index("csi500")
