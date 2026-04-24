"""Download NSE F&O bhavcopies for a date range.

Examples:
    # Last 5 years:
    python scripts/fetch_bhavcopy.py --years 5

    # Explicit range:
    python scripts/fetch_bhavcopy.py --from 2024-01-01 --to 2024-12-31

    # Inspect one day's cached chain:
    python scripts/fetch_bhavcopy.py --inspect 2024-10-15 --symbol NIFTY
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.config import ensure_dirs
from nse_bot.data import bhavcopy


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@click.command()
@click.option("--years", type=float, default=0.0, help="Back from today.")
@click.option("--from", "from_date", type=str, default="", help="YYYY-MM-DD")
@click.option("--to", "to_date", type=str, default="", help="YYYY-MM-DD")
@click.option("--sleep", type=float, default=0.5, help="Seconds between requests.")
@click.option("--inspect", type=str, default="", help="Print the option chain for one date (YYYY-MM-DD).")
@click.option("--symbol", type=str, default="NIFTY", help="Symbol for --inspect.")
def main(years: float, from_date: str, to_date: str, sleep: float, inspect: str, symbol: str) -> None:
    ensure_dirs()

    if inspect:
        d = _parse_date(inspect)
        df = bhavcopy.option_chain(symbol, d)
        if df.empty:
            print(f"No cached bhavcopy for {d}. Run without --inspect to download first.")
            return
        print(f"\n=== {symbol} option chain on {d} ===")
        expiries = sorted(df["expiry"].dropna().unique())
        print(f"Expiries available: {[e.isoformat() for e in expiries]}")
        near = bhavcopy.nearest_expiry(symbol, d)
        if near:
            sub = df.loc[df["expiry"] == near].copy()
            sub = sub[["strike", "option_type", "close", "settle", "volume", "open_interest"]].round(2)
            print(f"\nNearest expiry {near}:")
            print(tabulate(sub.head(40), headers="keys", tablefmt="github", showindex=False))
            print(f"... ({len(sub)} rows total)")
        return

    if from_date and to_date:
        start = _parse_date(from_date)
        end = _parse_date(to_date)
    elif years > 0:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=int(years * 365.25))
    else:
        raise SystemExit("Specify --years or both --from and --to.")

    print(f"Fetching F&O bhavcopies from {start} to {end}...")
    pbar = tqdm(total=((end - start).days + 1), unit="day")
    counters = {"fetched": 0, "skipped": 0, "failed": 0}

    def progress(d: date, status: str) -> None:
        counters[status] = counters.get(status, 0) + 1
        pbar.set_postfix(**counters)
        pbar.update(1)

    res = bhavcopy.fetch_range(start, end, sleep_s=sleep, on_progress=progress)
    pbar.close()
    print(f"\nDone: fetched={res['fetched']}  skipped={res['skipped']}  failed={res['failed']}")


if __name__ == "__main__":
    main()
