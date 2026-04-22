"""Run the end-of-day scanner and print the ranked list.

Examples:
    python scripts/nightly_scan.py
    python scripts/nightly_scan.py --min-adv-cr 10 --top 30 --save
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import click
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.config import REPORTS_DIR, ensure_dirs
from nse_bot.scanner.daily_scan import scan


@click.command()
@click.option("--min-adv-cr", type=float, default=5.0, help="Min 50-day avg daily turnover (₹ crore).")
@click.option("--top", type=int, default=50)
@click.option("--save/--no-save", default=False)
def main(min_adv_cr: float, top: int, save: bool) -> None:
    ensure_dirs()
    df = scan(min_adv_cr=min_adv_cr, top_n=top)
    if df.empty:
        print("Scanner produced no rows — did you fetch daily candles first?")
        return
    print(tabulate(df.round(3), headers="keys", tablefmt="github", showindex=False))
    if save:
        path = REPORTS_DIR / f"scan_{date.today().isoformat()}.csv"
        df.to_csv(path, index=False)
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
