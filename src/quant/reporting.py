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

    def build_research_quality(
        self,
        dataset_size: pd.DataFrame,
        factor_coverage: pd.DataFrame,
        factor_reliability: pd.DataFrame,
        ml_training_quality: pd.DataFrame,
        output: str | Path = "reports/research_quality.html",
    ) -> Path:
        """Render the data-quality evidence required before using research output."""
        sections = (
            ("Dataset size", dataset_size),
            ("Factor coverage and missingness", factor_coverage),
            ("Factor IC reliability", factor_reliability),
            ("ML training quality", ml_training_quality),
        )
        body = "\n".join(
            f"<section><h2>{escape(title)}</h2>{_table(table)}</section>" for title, table in sections
        )
        document = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Research Quality Report</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse;margin-bottom:1.5rem}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1{{margin-bottom:2rem}}</style>
</head><body><h1>Research Quality Report</h1>{body}</body></html>"""
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document, encoding="utf-8")
        return target

    def build_strategy_review(
        self,
        factor_contribution: pd.DataFrame,
        attribution_summary: pd.DataFrame,
        industry_contribution: pd.DataFrame,
        drawdown_analysis: pd.DataFrame,
        monthly_summary: pd.DataFrame,
        output: str | Path = "reports/strategy_review.html",
    ) -> Path:
        """Render the completed-trade review without inventing performance data."""
        best = factor_contribution.head(5) if not factor_contribution.empty else factor_contribution
        worst = (
            factor_contribution.sort_values("contribution", kind="stable").head(5)
            if "contribution" in factor_contribution
            else factor_contribution
        )
        sector_mistakes = (
            industry_contribution.sort_values("contribution", kind="stable").head(10)
            if "contribution" in industry_contribution
            else industry_contribution
        )
        sections = (
            ("Best factors", best),
            ("Worst factors", worst),
            ("Model accuracy and prediction bias", attribution_summary),
            ("Sector mistakes", sector_mistakes),
            ("Drawdown analysis", drawdown_analysis),
            ("Monthly summary", monthly_summary),
        )
        body = "\n".join(
            f"<section><h2>{escape(title)}</h2>{_table(table)}</section>" for title, table in sections
        )
        document = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Strategy Review</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse;margin-bottom:1.5rem}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1{{margin-bottom:2rem}}</style>
</head><body><h1>Strategy Review</h1>{body}</body></html>"""
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document, encoding="utf-8")
        return target

    def build_strategy_diagnosis(
        self,
        best_conditions: pd.DataFrame,
        worst_conditions: pd.DataFrame,
        factor_failures: pd.DataFrame,
        prediction_biases: pd.DataFrame,
        recommendations: pd.DataFrame,
        output: str | Path = "reports/strategy_diagnosis.html",
    ) -> Path:
        """Render v3.5 diagnosis evidence and conservative recommendations."""
        sections = (
            ("Best market conditions", best_conditions),
            ("Worst market conditions", worst_conditions),
            ("Factor failures", factor_failures),
            ("Prediction biases", prediction_biases),
            ("Recommended adjustments", recommendations),
        )
        body = "\n".join(
            f"<section><h2>{escape(title)}</h2>{_table(table)}</section>" for title, table in sections
        )
        document = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Strategy Diagnosis</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse;margin-bottom:1.5rem}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1{{margin-bottom:2rem}}</style>
</head><body><h1>Strategy Diagnosis</h1>{body}</body></html>"""
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document, encoding="utf-8")
        return target

    def build_daily_alpha(
        self,
        candidates: pd.DataFrame,
        allocation: pd.DataFrame,
        regime: pd.DataFrame,
        pipeline_status: pd.DataFrame,
        output: str | Path = "reports/daily_alpha_report.html",
    ) -> Path:
        """Render the point-in-time daily decision artefacts."""
        return self._build_sections(
            "Daily Alpha Report",
            (
                ("Pipeline status", pipeline_status),
                ("Market regime", regime),
                ("Top alpha candidates", candidates.head(50)),
                ("Proposed portfolio", allocation),
            ),
            output,
        )

    def build_paper_performance(
        self,
        performance: pd.DataFrame,
        fills: pd.DataFrame,
        positions: pd.DataFrame,
        output: str | Path = "reports/paper_performance.html",
    ) -> Path:
        """Render virtual-account performance and the day’s audited fills."""
        return self._build_sections(
            "Paper Trading Performance",
            (("Performance", performance), ("Fills", fills), ("Positions", positions)),
            output,
        )

    def build_strategy_evolution(
        self,
        weights: pd.DataFrame,
        evidence: pd.DataFrame,
        output: str | Path = "reports/strategy_evolution.html",
    ) -> Path:
        """Render bounded factor-weight changes with their supporting evidence."""
        return self._build_sections(
            "Strategy Evolution",
            (("Evolved factor weights", weights), ("Research evidence", evidence)),
            output,
        )

    def build_walk_forward(
        self,
        metrics: Mapping[str, float],
        results: pd.DataFrame,
        benchmark_comparison: pd.DataFrame,
        output: str | Path = "reports/walk_forward_report.html",
    ) -> Path:
        """Render chronological simulation evidence and benchmark availability."""
        return self._build_sections(
            "Walk-Forward Validation Report",
            (
                ("Performance metrics", _metric_table(metrics, tuple(metrics))),
                ("Benchmark comparison", benchmark_comparison),
                ("Chronological simulation", results),
            ),
            output,
        )

    def build_performance_attribution(
        self,
        summary: pd.DataFrame,
        daily: pd.DataFrame,
        output: str | Path = "reports/performance_attribution.html",
    ) -> Path:
        """Render return decomposition along with its daily reconciliation."""
        return self._build_sections(
            "Performance Attribution",
            (("Return decomposition", summary), ("Daily attribution", daily)),
            output,
        )

    def build_robustness(
        self, results: pd.DataFrame, output: str | Path = "reports/robustness_report.html"
    ) -> Path:
        """Render every measured robustness scenario, including unavailable ones."""
        return self._build_sections("Robustness Report", (("Scenario matrix", results),), output)

    def build_strategy_history(
        self, history: pd.DataFrame, output: str | Path = "reports/strategy_history.html"
    ) -> Path:
        """Render immutable factor-weight strategy versions."""
        return self._build_sections("Strategy Version History", (("Versions", history),), output)

    def build_institutional_dashboard(
        self,
        sections: Mapping[str, pd.DataFrame],
        output: str | Path = "reports/institutional_dashboard.html",
    ) -> Path:
        """Compose available validation, governance, and paper-trading evidence."""
        return self._build_sections(
            "Institutional Validation Dashboard",
            tuple((str(title), values) for title, values in sections.items()),
            output,
        )

    @staticmethod
    def _build_sections(
        title: str,
        sections: tuple[tuple[str, pd.DataFrame], ...],
        output: str | Path,
    ) -> Path:
        body = "\n".join(
            f"<section><h2>{escape(section_title)}</h2>{_table(values)}</section>"
            for section_title, values in sections
        )
        document = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>{escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse;margin-bottom:1.5rem}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1{{margin-bottom:2rem}}</style>
</head><body><h1>{escape(title)}</h1>{body}</body></html>"""
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
