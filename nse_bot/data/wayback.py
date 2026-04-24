"""Fetch historical RSS snapshots from the Wayback Machine.

Strategy: archive.org keeps dated snapshots of stable RSS URLs. We list the
available snapshots for each source between (from_date, to_date) via the CDX
API, download one snapshot per day, and parse each with the stdlib RSS
parser in nse_bot/data/news.py.

Upsides: free, no auth, works for 2-3 years of headlines from major Indian
financial sites (Moneycontrol, Economic Times, LiveMint).

Downsides: slow (rate-limited, ~1 req/sec to be polite), some snapshots
are incomplete or redirect to the live site, the set of sources covered is
whatever Wayback chose to crawl.

Output: NewsItem rows written to the local news store (same store as
snapshot_news.py), so the backtest narrative layer sees them automatically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from typing import Iterable

import httpx
import pandas as pd

from nse_bot.data import news_store
from nse_bot.data.news import NewsItem, RSS_FEEDS, _parse_rss

CDX_URL = "http://web.archive.org/cdx/search/cdx"
SNAPSHOT_URL = "http://web.archive.org/web/{timestamp}id_/{original}"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class _Snapshot:
    timestamp: str  # YYYYMMDDHHMMSS
    original: str

    @property
    def snap_date(self) -> date:
        return datetime.strptime(self.timestamp[:8], "%Y%m%d").date()

    @property
    def fetch_url(self) -> str:
        return SNAPSHOT_URL.format(timestamp=self.timestamp, original=self.original)


def _cdx_list(
    client: httpx.Client,
    source_url: str,
    from_date: date,
    to_date: date,
    timeout: float = 30.0,
) -> list[_Snapshot]:
    """Query the Wayback CDX API for snapshots of source_url in the date range."""
    params = {
        "url": source_url,
        "from": from_date.strftime("%Y%m%d"),
        "to": to_date.strftime("%Y%m%d"),
        "output": "json",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",  # one snapshot per day
    }
    try:
        r = client.get(CDX_URL, params=params, timeout=timeout)
        r.raise_for_status()
        rows = r.json()
    except Exception:
        return []
    if not rows or len(rows) < 2:
        return []
    header = rows[0]
    ts_idx = header.index("timestamp")
    orig_idx = header.index("original")
    return [_Snapshot(timestamp=row[ts_idx], original=row[orig_idx]) for row in rows[1:]]


def _download(client: httpx.Client, snap: _Snapshot, timeout: float = 30.0) -> str | None:
    try:
        r = client.get(snap.fetch_url, timeout=timeout, follow_redirects=True)
        if r.status_code != 200 or not r.text:
            return None
        return r.text
    except Exception:
        return None


def _items_from_snapshot(
    snap: _Snapshot,
    xml_text: str,
    source_key: str,
) -> list[NewsItem]:
    parsed = _parse_rss(xml_text)
    if not parsed:
        return []
    out: list[NewsItem] = []
    snap_dt = datetime.strptime(snap.timestamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    for title, summary, link, published in parsed:
        ts = published or snap_dt
        # Clamp absurd dates that sometimes leak through bad RSS data.
        if ts.year < 2000 or ts > snap_dt + timedelta(days=1):
            ts = snap_dt
        score = analyzer.polarity_scores(f"{title}. {summary}")["compound"]
        out.append(
            NewsItem(
                source=f"wayback_{source_key}",
                title=title,
                summary=summary,
                url=link,
                published=ts,
                sentiment=score,
            )
        )
    return out


def scrape_source(
    source_key: str,
    from_date: date,
    to_date: date,
    *,
    sleep_s: float = 1.0,
    max_snapshots: int | None = None,
    client: httpx.Client | None = None,
    on_progress=None,
) -> dict:
    """Download and parse Wayback snapshots of one RSS source.

    source_key must exist in nse_bot.data.news.RSS_FEEDS.
    """
    url = RSS_FEEDS.get(source_key)
    if not url:
        raise ValueError(f"Unknown source_key '{source_key}'. Known: {list(RSS_FEEDS)}")

    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": _UA}, timeout=30.0, follow_redirects=True)

    try:
        snaps = _cdx_list(client, url, from_date, to_date)
        if max_snapshots is not None:
            snaps = snaps[:max_snapshots]
        if not snaps:
            return {"source": source_key, "snapshots": 0, "items": 0, "written": 0}

        total_items: list[NewsItem] = []
        ok = 0
        for idx, snap in enumerate(snaps):
            xml = _download(client, snap)
            if xml:
                items = _items_from_snapshot(snap, xml, source_key)
                total_items.extend(items)
                ok += 1
            if on_progress:
                on_progress(source_key, idx + 1, len(snaps), len(total_items))
            time.sleep(sleep_s)

        written = news_store.write(total_items) if total_items else 0
        return {
            "source": source_key,
            "snapshots": len(snaps),
            "snapshots_ok": ok,
            "items": len(total_items),
            "written": written,
        }
    finally:
        if own_client:
            client.close()


def scrape_all(
    from_date: date,
    to_date: date,
    *,
    sources: Iterable[str] | None = None,
    sleep_s: float = 1.0,
    max_snapshots_per_source: int | None = None,
    on_progress=None,
) -> list[dict]:
    keys = list(sources) if sources else list(RSS_FEEDS.keys())
    results: list[dict] = []
    with httpx.Client(headers={"User-Agent": _UA}, timeout=30.0, follow_redirects=True) as client:
        for k in keys:
            res = scrape_source(
                k, from_date, to_date,
                sleep_s=sleep_s,
                max_snapshots=max_snapshots_per_source,
                client=client,
                on_progress=on_progress,
            )
            results.append(res)
    return results
