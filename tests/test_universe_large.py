from pathlib import Path

from typer.testing import CliRunner

from quant.cli import app
from quant.data.universe import Universe
from quant.data.universe_builder import UniverseBuilder


def test_large_universe_config_includes_downloadable_metadata():
    stocks = Universe("configs/universe_large.yaml").stocks()

    assert len(stocks) >= 800
    assert {"code", "name", "market", "sector", "index_source"}.issubset(stocks[0])
    sources = {
        source
        for stock in stocks
        for source in (
            stock["index_source"]
            if isinstance(stock["index_source"], list)
            else [stock["index_source"]]
        )
    }
    assert sources.issuperset({"CSI300", "CSI500", "ChiNext"})
    assert all(stock["sector"] for stock in stocks)


def test_builder_writes_loader_output_as_large_universe_yaml(tmp_path: Path):
    def loader(spec):
        return [{"code": "000001", "name": "Ping An Bank", "market": "SZ", "sector": "Bank"}]

    output = tmp_path / "universe_large.yaml"
    stocks = UniverseBuilder(component_loader=loader).write_universe(output, indexes=("csi300",))

    persisted = Universe(output).stocks()

    assert stocks[0]["index_source"] == "CSI300"
    assert persisted[0]["index_source"] == "CSI300"


def test_data_update_uses_selected_large_universe_and_records_failures(monkeypatch, tmp_path: Path):
    captured = {}

    class FakeDownloader:
        def __init__(self, universe, **kwargs):
            captured["stocks"] = universe.stocks()

        def update(self):
            return {"saved": [tmp_path / "raw" / "000001.parquet"], "failed": {"000002": "offline"}}

    monkeypatch.setattr("quant.cli.DataDownloader", FakeDownloader)
    config = tmp_path / "config.yaml"
    config.write_text(f'data:\n  storage_root: "{tmp_path.as_posix()}"\n  start_date: "20180101"\n')

    result = CliRunner().invoke(
        app,
        ["data-update", "--base", str(config), "--universe-path", "configs/universe_large.yaml"],
    )

    assert result.exit_code == 0, result.output
    assert len(captured["stocks"]) == len(Universe("configs/universe_large.yaml").stocks())
    assert (tmp_path / "raw" / "download_failures.json").read_text(encoding="utf-8") == '{\n  "000002": "offline"\n}'
