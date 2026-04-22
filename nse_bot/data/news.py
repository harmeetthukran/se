"""News fetcher with lightweight sentiment scoring.

Sources:
    * Free RSS: Moneycontrol markets, Economic Times markets, LiveMint markets.
    * NSE corporate announcements JSON endpoint (bhavcopy-style daily filings).

Sentiment: VADER (rule-based, fast, no ML download). For per-stock sentiment,
we match tickers/company names against headline text. This is coarse — FinBERT
would be more accurate but much slower and heavier. Start with VADER; swap in
FinBERT later if signal quality warrants the cost.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime

import httpx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RSS_FEEDS = {
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
}

NSE_ANNOUNCEMENTS_URL = (
    "https://www.nseindia.com/api/corporate-announcements?index=equities"
)
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
}

_analyzer = SentimentIntensityAnalyzer()


@dataclass
class NewsItem:
    source: str
    title: str
    summary: str
    url: str
    published: datetime | None
    sentiment: float  # VADER compound in [-1, 1]

    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()


def fetch_rss(sources: list[str] | None = None, limit_per_source: int = 50, timeout: float = 15.0) -> list[NewsItem]:
    out: list[NewsItem] = []
    feeds = sources or list(RSS_FEEDS.keys())
    with httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": _NSE_HEADERS["User-Agent"]}) as c:
        for name in feeds:
            url = RSS_FEEDS.get(name)
            if not url:
                continue
            try:
                r = c.get(url)
                r.raise_for_status()
                items = _parse_rss(r.text)[:limit_per_source]
            except Exception:
                continue
            for title, summary, link, published in items:
                score = _analyzer.polarity_scores(f"{title}. {summary}")["compound"]
                out.append(
                    NewsItem(
                        source=name,
                        title=title,
                        summary=summary,
                        url=link,
                        published=published,
                        sentiment=score,
                    )
                )
    return out


def _parse_rss(xml_text: str) -> list[tuple[str, str, str, datetime | None]]:
    """Minimal RSS 2.0 / Atom parser. Returns (title, summary, link, published)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out: list[tuple[str, str, str, datetime | None]] = []

    for item in root.iter():
        tag = item.tag.split("}", 1)[-1].lower()
        if tag not in ("item", "entry"):
            continue
        title = _text(item, ("title",))
        summary = _strip_html(_text(item, ("description", "summary", "content")))
        link = _text(item, ("link",))
        if not link:
            for child in item:
                if child.tag.split("}", 1)[-1].lower() == "link":
                    link = (child.get("href") or child.text or "").strip()
                    if link:
                        break
        pub = _text(item, ("pubdate", "published", "updated"))
        dt = _parse_http_date(pub)
        out.append((title, summary, link, dt))
    return out


def _text(elem: ET.Element, names: tuple[str, ...]) -> str:
    for child in elem:
        tag = child.tag.split("}", 1)[-1].lower()
        if tag in names and (child.text or "").strip():
            return child.text.strip()
    return ""


def _parse_http_date(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None


def fetch_nse_announcements(timeout: float = 15.0) -> list[NewsItem]:
    """NSE's corporate announcements feed. Requires a cookie-warmed session.

    On failure (NSE frequently 401s clients from outside India), returns [].
    """
    try:
        with httpx.Client(timeout=timeout, headers=_NSE_HEADERS) as c:
            c.get("https://www.nseindia.com", timeout=timeout)
            r = c.get(NSE_ANNOUNCEMENTS_URL, timeout=timeout)
            r.raise_for_status()
            rows = r.json() or []
    except Exception:
        return []

    items: list[NewsItem] = []
    for row in rows:
        title = (row.get("desc") or row.get("subject") or "").strip()
        summary = (row.get("sm_name") or row.get("symbol") or "").strip()
        link = row.get("attchmntFile") or ""
        dt = _parse_nse_time(row.get("an_dt") or row.get("exchdisstime"))
        score = _analyzer.polarity_scores(f"{title}. {summary}")["compound"]
        items.append(
            NewsItem(
                source="nse_announcements",
                title=title,
                summary=summary,
                url=link,
                published=dt,
                sentiment=score,
            )
        )
    return items


def score_for_symbols(items: list[NewsItem], symbols: list[str]) -> dict[str, dict]:
    """Aggregate sentiment per symbol by case-insensitive match in title/summary."""
    by_sym: dict[str, list[float]] = {s.upper(): [] for s in symbols}
    patterns = {s.upper(): re.compile(rf"\b{re.escape(s.upper())}\b") for s in symbols}
    for item in items:
        text = item.text().upper()
        for sym, pat in patterns.items():
            if pat.search(text):
                by_sym[sym].append(item.sentiment)
    out: dict[str, dict] = {}
    for sym, scores in by_sym.items():
        if not scores:
            out[sym] = {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        else:
            out[sym] = {
                "n": len(scores),
                "mean": float(sum(scores) / len(scores)),
                "min": float(min(scores)),
                "max": float(max(scores)),
            }
    return out


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _parse_nse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
