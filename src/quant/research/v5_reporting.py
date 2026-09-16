"""Small self-contained HTML report helper for V5 research artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from pathlib import Path

import pandas as pd


def write_v5_report(
    title: str, sections: Iterable[tuple[str, pd.DataFrame]], output_path: str | Path
) -> Path:
    """Render supplied evidence tables without creating synthetic report content."""
    body = "\n".join(
        f"<section><h2>{escape(section_title)}</h2>{_table(values)}</section>"
        for section_title, values in sections
    )
    document = f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse;margin-bottom:1.5rem}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1{{margin-bottom:2rem}}</style>
</head><body><h1>{escape(title)}</h1>{body}</body></html>"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target


def _table(values: pd.DataFrame) -> str:
    return values.to_html(index=False, escape=True, border=0) if not values.empty else "<p>No data available.</p>"
