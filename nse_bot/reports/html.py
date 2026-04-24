"""Minimal HTML report wrapping generated PNGs + summary tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; }}
h1 {{ margin-bottom: 4px; }}
h2 {{ margin-top: 28px; color: #444; }}
img {{ max-width: 100%; height: auto; margin: 8px 0 20px; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 13px; }}
th {{ background: #f2f2f2; text-align: left; }}
.meta {{ color: #777; font-size: 13px; margin-bottom: 16px; }}
.caveat {{ background: #fff8e1; border-left: 4px solid #f6c200; padding: 10px 14px; margin: 18px 0;
          font-size: 13px; }}
</style>
</head><body>
<h1>{title}</h1>
<div class="meta">Generated {generated_at}</div>
{body}
<div class="caveat"><strong>Reminder:</strong> backtest metrics are historical fits,
not forecasts. Monte Carlo percentiles widen — not narrow — the range of plausible
live outcomes. Paper-trade before sizing live.</div>
</body></html>
"""


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=True, border=0, classes="report-table", float_format=lambda x: f"{x:.3f}")


def build_html_report(
    title: str,
    out_path: Path,
    *,
    metrics: pd.Series | None = None,
    mc_summary: pd.Series | None = None,
    image_paths: list[tuple[str, Path]] | None = None,
    extra_tables: list[tuple[str, pd.DataFrame]] | None = None,
) -> None:
    from datetime import datetime

    body_parts: list[str] = []

    if metrics is not None and not metrics.empty:
        body_parts.append("<h2>Backtest metrics</h2>")
        body_parts.append(_table_html(metrics.to_frame("value")))

    if mc_summary is not None and not mc_summary.empty:
        body_parts.append("<h2>Monte Carlo summary</h2>")
        body_parts.append(_table_html(mc_summary.to_frame("value")))

    if image_paths:
        for caption, path in image_paths:
            if path is None:
                continue
            body_parts.append(f"<h2>{caption}</h2>")
            rel = Path(path).name
            body_parts.append(f'<img src="{rel}" alt="{caption}">')

    if extra_tables:
        for caption, df in extra_tables:
            if df is None or df.empty:
                continue
            body_parts.append(f"<h2>{caption}</h2>")
            body_parts.append(_table_html(df))

    html = _TEMPLATE.format(
        title=title,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        body="\n".join(body_parts),
    )
    out_path.write_text(html, encoding="utf-8")
