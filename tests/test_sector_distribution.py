from __future__ import annotations

import yaml
from typer.testing import CliRunner

from quant.cli import app


def test_universe_check_reports_sector_distribution_and_unknown_ratio_below_limit(tmp_path):
    stocks = [
        {
            "code": f"{code:06d}",
            "name": f"Stock {code}",
            "market": "SZ",
            "sector": "Technology" if code < 10 else ("Consumer" if code < 20 else "Unknown"),
            "index_source": "CSI300",
        }
        for code in range(21)
    ]
    universe = tmp_path / "universe.yaml"
    universe.write_text(yaml.safe_dump({"name": "test", "stocks": stocks}), encoding="utf-8")

    result = CliRunner().invoke(app, ["universe-check", "--universe-path", str(universe)])

    assert result.exit_code == 0, result.output
    assert "Technology: 10" in result.output
    assert "Consumer: 10" in result.output
    assert "Unknown: 1 (4.76%)" in result.output
