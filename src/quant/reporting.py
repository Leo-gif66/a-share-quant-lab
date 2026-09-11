"""Self-contained HTML research reports with no browser or chart dependency."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Mapping

import pandas as pd


class ResearchReportBuilder:
    """Render performance, risk, portfolio, factor, and ML research tables."""

    def build(
        self,
        metrics: Mapping[str, float],
        holdings: pd.DataFrame,
        sector_exposure: pd.DataFrame,
        factor_contribution: pd.DataFrame,
        feature_importance: pd.DataFrame | None = None,
        output: str | Path = "reports/latest.html",
    ) -> Path:
        performance = _metric_table(metrics, ("annual_return", "sharpe", "max_drawdown"))
        risk = _metric_table(metrics, ("volatility", "annual_volatility", "beta", "turnover"))
        sections = [
            ("Performance", performance),
            ("Risk", risk),
            ("Holdings", holdings),
            ("Sector exposure", sector_exposure),
            ("Factor contribution", factor_contribution),
            ("ML feature importance", feature_importance),
        ]
        body = "\n".join(
            f"<section><h2>{escape(title)}</h2>{_table(table)}</section>"
            for title, table in sections
            if table is not None
        )
        document = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Quant Research Report</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse;margin-bottom:1.5rem}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1{{margin-bottom:2rem}}</style>
</head><body><h1>Quant Research Report</h1>{body}</body></html>"""
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document, encoding="utf-8")
        return target

    def from_backtest_result(
        self, result: object, feature_importance: pd.DataFrame | None = None, output: str | Path = "reports/latest.html"
    ) -> Path:
        required = ("metrics", "holdings_history", "sector_exposure", "factor_contribution")
        if any(not hasattr(result, name) for name in required):
            raise TypeError("result must expose portfolio backtest report fields")
        return self.build(
            result.metrics,
            result.holdings_history,
            result.sector_exposure,
            result.factor_contribution,
            feature_importance,
            output,
        )

    def build_alpha_research(
        self,
        factor_ic: pd.DataFrame,
        factor_stability: pd.DataFrame,
        feature_importance: pd.DataFrame,
        walk_forward: pd.DataFrame,
        portfolio_comparison: pd.DataFrame,
        output: str | Path = "reports/alpha_research.html",
    ) -> Path:
        """Render the v2.1-v2.5 research artifacts without inventing sections."""
        sections = (
            ("Factor IC ranking", factor_ic),
            ("Factor stability", factor_stability),
            ("ML feature importance", feature_importance),
            ("Walk-forward performance", walk_forward),
            ("Portfolio comparison", portfolio_comparison),
        )
        body = "\n".join(
            f"<section><h2>{escape(title)}</h2>{_table(table)}</section>" for title, table in sections
        )
        document = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Alpha Research Report</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse;margin-bottom:1.5rem}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1{{margin-bottom:2rem}}</style>
</head><body><h1>Alpha Research Report</h1>{body}</body></html>"""
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document, encoding="utf-8")
        return target


def _metric_table(metrics: Mapping[str, float], names: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": name, "value": metrics[name]} for name in names if name in metrics]
    )


def _table(values: pd.DataFrame) -> str:
    if values.empty:
        return "<p>No data available.</p>"
    return values.to_html(index=False, escape=True, border=0, classes="report-table")
