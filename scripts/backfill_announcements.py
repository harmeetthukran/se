"""Backfill historical NSE corporate announcements into the news store.

Examples:
    # Last 5 years (be patient — NSE rate-limits; will chunk into 30d windows):
    python scripts/backfill_announcements.py --years 5

    # Explicit range:
    python scripts/backfill_announcements.py --from 2020-01-01 --to 2024-12-31

NSE frequently blocks non-Indian IPs. If this returns 0 rows it's almost
always the IP block, not your code. Try a Mumbai / Indian VPN.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.config import ensure_dirs
from nse_bot.data import news_store
from nse_bot.data.news import fetch_nse_announcements_range


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@click.command()
@click.option("--years", type=float, default=0.0)
@click.option("--from", "from_date", type=str, default="")
@click.option("--to", "to_date", type=str, default="")
@click.option("--chunk-days", type=int, default=30)
@click.option("--pause", type=float, default=0.7, help="Seconds between chunked requests.")
def main(years: float, from_date: str, to_date: str, chunk_days: int, pause: float) -> None:
    ensure_dirs()
    if from_date and to_date:
        start = _parse_date(from_date)
        end = _parse_date(to_date)
    elif years > 0:
        end = date.today()
        start = end - timedelta(days=int(years * 365.25))
    else:
        raise SystemExit("Specify --years or both --from and --to.")

    print(f"Backfilling NSE announcements {start} → {end} (chunked {chunk_days}d, pause {pause}s)")
    items = fetch_nse_announcements_range(start, end, chunk_days=chunk_days, pause_s=pause)
    print(f"Fetched {len(items)} announcements.")
    if not items:
        print("Empty result. Most likely cause: NSE IP-blocked your network.")
        return
    added = news_store.write(items)
    print(f"Wrote {added} net-new rows to the news store.")


if __name__ == "__main__":
    main()
