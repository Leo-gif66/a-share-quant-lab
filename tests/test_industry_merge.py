from __future__ import annotations

from typer.testing import CliRunner

from quant.cli import app
from quant.data.universe import Universe


def test_universe_build_merges_industry_snapshot_over_unknown_component_sectors(tmp_path):
    components = tmp_path / "components"
    components.mkdir()
    for filename, code, name in (
        ("csi300.csv", "000001", "Ping An Bank"),
        ("csi500.csv", "600519", "Kweichow Moutai"),
        ("chinext.csv", "300750", "CATL"),
    ):
        (components / filename).write_text(
            f"code,name,sector,market\n{code},{name},Unknown,SZ\n", encoding="utf-8"
        )
    industry = tmp_path / "industry.csv"
    industry.write_text(
        "code,sector,industry\n"
        "000001,Financials,Commercial Banks\n"
        "600519,Consumer,Food&Beverage\n"
        "300750,Technology,Batteries\n",
        encoding="utf-8",
    )
    output = tmp_path / "universe_large.yaml"

    result = CliRunner().invoke(
        app,
        [
            "universe-build",
            "--component-dir",
            str(components),
            "--industry-path",
            str(industry),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    stocks = {stock["code"]: stock for stock in Universe(output).stocks()}
    assert stocks["000001"]["sector"] == "Financials"
    assert stocks["600519"]["industry"] == "Food&Beverage"
    assert all(stock["sector"] != "Unknown" for stock in stocks.values())
