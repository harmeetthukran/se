"""Backfill historical Indian-market headlines from the Wayback Machine.

Examples:
    # Last 2 years, all default sources (moneycontrol, ET, livemint):
    python scripts/fetch_historical_news.py --years 2

    # Custom range, single source:
    python scripts/fetch_historical_news.py --from 2021-01-01 --to 2022-12-31 \\
        --sources moneycontrol_markets

    # Quick sanity run — at most 5 snapshots per source:
    python scripts/fetch_historical_news.py --years 1 --max-snapshots 5

Notes:
    * Slow by design — 1 req/sec to the Wayback Machine is polite. A 2-year
      backfill of 3 sources is roughly 3 × 730 ≈ 2200 requests ≈ 35-40 min.
    * Archive coverage is not perfect; days where Wayback didn't crawl are
      silently skipped.
    * Headlines with obviously broken dates are clamped to the snapshot date.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.config import ensure_dirs
from nse_bot.data.news import RSS_FEEDS
from nse_bot.data.wayback import scrape_all


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@click.command()
@click.option("--years", type=float, default=0.0)
@click.option("--from", "from_date", type=str, default="")
@click.option("--to", "to_date", type=str, default="")
@click.option("--sources", type=str, default="",
              help=f"Comma-separated keys from: {','.join(RSS_FEEDS)}")
@click.option("--sleep", type=float, default=1.0, help="Seconds between Wayback requests.")
@click.option("--max-snapshots", type=int, default=0,
              help="0 = unlimited; cap per source for quick tests.")
def main(years: float, from_date: str, to_date: str, sources: str,
         sleep: float, max_snapshots: int) -> None:
    ensure_dirs()
    if from_date and to_date:
        start = _parse_date(from_date)
        end = _parse_date(to_date)
    elif years > 0:
        end = date.today()
        start = end - timedelta(days=int(years * 365.25))
    else:
        raise SystemExit("Specify --years or both --from and --to.")

    src_list = [s.strip() for s in sources.split(",") if s.strip()] or None
    max_snap = max_snapshots if max_snapshots > 0 else None

    print(f"Wayback scrape {start} → {end}")
    print(f"Sources: {src_list or list(RSS_FEEDS)}")
    print(f"Sleep: {sleep}s   Cap per source: {max_snap or 'unlimited'}")

    pbars: dict[str, tqdm] = {}

    def progress(source: str, i: int, total: int, items_so_far: int) -> None:
        if source not in pbars:
            pbars[source] = tqdm(total=total, desc=source, unit="snap")
        pbars[source].update(1)
        pbars[source].set_postfix(items=items_so_far)

    results = scrape_all(
        start, end,
        sources=src_list,
        sleep_s=sleep,
        max_snapshots_per_source=max_snap,
        on_progress=progress,
    )

    for bar in pbars.values():
        bar.close()

    print("\n=== Results ===")
    for r in results:
        print(
            f"  {r['source']:<24}  snapshots={r.get('snapshots',0):>4}  "
            f"ok={r.get('snapshots_ok',0):>4}  items={r.get('items',0):>5}  "
            f"written_new={r.get('written',0):>5}"
        )


if __name__ == "__main__":
    main()
