from __future__ import annotations

from typer.testing import CliRunner

from quant.cli import app
from quant.data.universe import Universe


def _write_components(component_dir):
    component_dir.mkdir()
    (component_dir / "csi300.csv").write_text(
        "code,name,industry\n000001,Ping An Bank,Bank\n600001,Example SH,Industrial\n",
        encoding="utf-8",
    )
    (component_dir / "csi500.csv").write_text(
        "code,name,industry\n000001,Ping An Bank,Bank\n000002,Example SZ,Consumer\n",
        encoding="utf-8",
    )
    (component_dir / "chinext.csv").write_text(
        "code,name,industry\n300001,Example Growth,Technology\n",
        encoding="utf-8",
    )


def test_universe_build_merges_overlapping_index_membership_and_writes_yaml(tmp_path):
    component_dir = tmp_path / "components"
    output = tmp_path / "universe_large.yaml"
    _write_components(component_dir)

    result = CliRunner().invoke(
        app,
        [
            "universe-build",
            "--component-dir",
            str(component_dir),
            "--industry-path",
            str(tmp_path / "missing-industry.csv"),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Total stocks: 4" in result.output
    stocks = {stock["code"]: stock for stock in Universe(output).stocks()}
    assert stocks["000001"]["index_source"] == ["CSI300", "CSI500"]
    assert stocks["300001"]["index_source"] == "ChiNext"


def test_universe_check_reports_index_and_sector_counts(tmp_path):
    component_dir = tmp_path / "components"
    output = tmp_path / "universe_large.yaml"
    _write_components(component_dir)
    runner = CliRunner()
    assert runner.invoke(
        app,
        [
            "universe-build",
            "--component-dir",
            str(component_dir),
            "--industry-path",
            str(tmp_path / "missing-industry.csv"),
            "--output",
            str(output),
        ],
    ).exit_code == 0

    result = runner.invoke(app, ["universe-check", "--universe-path", str(output)])

    assert result.exit_code == 0, result.output
    assert "CSI300: 2" in result.output
    assert "CSI500: 2" in result.output
    assert "ChiNext: 1" in result.output
    assert "Sectors: 4" in result.output
