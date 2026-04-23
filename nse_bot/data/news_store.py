"""Parquet-backed store for news events.

Schema:
    ts          datetime64[ns, Asia/Kolkata]  -- normalized to IST
    source      string                         -- rss feed key or 'nse_announcements'
    title       string
    summary     string
    url         string
    sentiment   float64                        -- VADER compound [-1, 1]

File: data/news/events.parquet

`write` appends and deduplicates on (source, title, ts_date). Intended to be
called once per day by scripts/snapshot_news.py (Windows Task Scheduler).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nse_bot.config import DATA_DIR
from nse_bot.data.news import NewsItem

NEWS_DIR = DATA_DIR / "news"
EVENTS_PATH = NEWS_DIR / "events.parquet"
IST = "Asia/Kolkata"


def _to_ist(dt: datetime | None) -> pd.Timestamp:
    if dt is None:
        return pd.Timestamp.now(tz=IST)
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.tz_convert(IST)


def _items_to_frame(items: list[NewsItem]) -> pd.DataFrame:
    rows = [
        {
            "ts": _to_ist(it.published),
            "source": it.source,
            "title": it.title or "",
            "summary": it.summary or "",
            "url": it.url or "",
            "sentiment": float(it.sentiment),
        }
        for it in items
    ]
    if not rows:
        return pd.DataFrame(columns=["ts", "source", "title", "summary", "url", "sentiment"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def read_all() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        return pd.DataFrame(columns=["ts", "source", "title", "summary", "url", "sentiment"])
    return pd.read_parquet(EVENTS_PATH)


def write(items: list[NewsItem]) -> int:
    """Append items to the store. Returns the number of net-new rows written."""
    if not items:
        return 0
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    new = _items_to_frame(items)
    if EVENTS_PATH.exists():
        existing = pd.read_parquet(EVENTS_PATH)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined["_date"] = combined["ts"].dt.tz_convert(IST).dt.date
    combined = combined.drop_duplicates(subset=["source", "title", "_date"], keep="last")
    combined = combined.sort_values("ts").drop(columns=["_date"]).reset_index(drop=True)

    added = len(combined) - (len(existing) if EVENTS_PATH.exists() else 0)
    combined.to_parquet(EVENTS_PATH, index=False)
    return max(0, added)


def query(
    start_date: date | None = None,
    end_date: date | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Return stored news filtered by date range and (optionally) ticker match."""
    df = read_all()
    if df.empty:
        return df
    d = df["ts"].dt.tz_convert(IST).dt.date
    if start_date is not None:
        df = df.loc[d >= start_date]
    if end_date is not None:
        df = df.loc[d <= end_date]
    if symbol:
        key = symbol.upper()
        mask = (
            df["title"].astype(str).str.upper().str.contains(rf"\b{key}\b", regex=True, na=False)
            | df["summary"].astype(str).str.upper().str.contains(rf"\b{key}\b", regex=True, na=False)
        )
        df = df.loc[mask]
    return df.reset_index(drop=True)


def daily_summary(start_date: date, end_date: date) -> pd.DataFrame:
    """One row per date: news count + average sentiment + top headline."""
    df = query(start_date, end_date)
    if df.empty:
        return pd.DataFrame(columns=["date", "n", "avg_sentiment", "top_headline", "top_sentiment"])
    df = df.copy()
    df["date"] = df["ts"].dt.tz_convert(IST).dt.date

    def _agg(grp: pd.DataFrame) -> pd.Series:
        strongest = grp.loc[grp["sentiment"].abs().idxmax()] if len(grp) else None
        return pd.Series(
            {
                "n": len(grp),
                "avg_sentiment": float(grp["sentiment"].mean()),
                "top_headline": strongest["title"] if strongest is not None else "",
                "top_sentiment": float(strongest["sentiment"]) if strongest is not None else 0.0,
            }
        )

    out = df.groupby("date", as_index=False).apply(_agg, include_groups=False)
    return out


def cover_gaps(start_date: date, end_date: date) -> list[date]:
    """Return the list of dates between [start, end] with zero stored news.

    Used by the backtest narrative to tell the user honestly which dates
    had no coverage (vs. genuinely quiet days).
    """
    df = query(start_date, end_date)
    have = set(df["ts"].dt.tz_convert(IST).dt.date.unique()) if not df.empty else set()
    gaps: list[date] = []
    cur = start_date
    while cur <= end_date:
        if cur not in have and cur.weekday() < 5:  # weekdays only
            gaps.append(cur)
        cur += timedelta(days=1)
    return gaps
