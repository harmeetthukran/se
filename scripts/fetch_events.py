"""Refresh the corporate-action + earnings calendar from NSE.

NSE's public JSON endpoints require a cookie-warmed session and commonly
return 401/403 from non-Indian IPs. If this fails, the calendar stays
empty and the `--skip-events` filter becomes a no-op.

Run daily (Task Scheduler):
    python scripts/fetch_events.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.config import ensure_dirs
from nse_bot.data.events import refresh_calendar, load_calendar


if __name__ == "__main__":
    ensure_dirs()
    df = refresh_calendar()
    if df.empty:
        print("Calendar empty. NSE endpoints may be blocking this network.")
    else:
        print(f"Calendar has {len(df)} rows.")
        by_type = df["event_type"].value_counts().to_dict()
        print(f"Event types: {by_type}")
        upcoming = df.loc[df["event_date"] >= df["event_date"].max() - __import__('datetime').timedelta(days=0)]
        print(f"Earliest: {df['event_date'].min()}   Latest: {df['event_date'].max()}")
