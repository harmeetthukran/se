"""Corporate-action and earnings-event calendar.

Sources (fetched once per day; NSE frequently blocks non-Indian IPs):
    * https://www.nseindia.com/api/corporates-upcoming-events?index=equities
    * https://www.nseindia.com/api/corporates-corporateActions?index=equities
      (splits, bonuses, dividends, mergers, rights issues)

Cached as:
    data/events/calendar.parquet  -- {symbol, event_date, event_type, notes}

Filter helper `skip_around_events(signal, ts, symbol, window_days)` zeroes
the signal within ±window_days of any event for the symbol — keeps you
out of the overnight gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from nse_bot.config import DATA_DIR

EVENTS_DIR = DATA_DIR / "events"
CALENDAR_PATH = EVENTS_DIR / "calendar.parquet"

NSE_CA_URL = "https://www.nseindia.com/api/corporates-corporateActions?index=equities"
NSE_EV_URL = "https://www.nseindia.com/api/corporates-upcoming-events?index=equities"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
}


def refresh_calendar(timeout: float = 20.0) -> pd.DataFrame:
    """Fetch upcoming events + corporate actions from NSE and write parquet."""
    rows: list[dict] = []
    try:
        with httpx.Client(timeout=timeout, headers=_HEADERS) as c:
            c.get("https://www.nseindia.com", timeout=timeout)
            for url, kind in ((NSE_EV_URL, "event"), (NSE_CA_URL, "corporate_action")):
                try:
                    r = c.get(url)
                    r.raise_for_status()
                    payload = r.json() or []
                except Exception:
                    continue
                for item in payload:
                    sym = (item.get("symbol") or item.get("sm_symbol") or "").upper().strip()
                    d = _parse_date(
                        item.get("date") or item.get("bcDate") or item.get("exDate") or item.get("recDate")
                    )
                    purpose = (
                        item.get("purpose")
                        or item.get("subject")
                        or item.get("ca_type")
                        or item.get("event")
                        or ""
                    )
                    if not sym or d is None:
                        continue
                    rows.append(
                        {
                            "symbol": sym,
                            "event_date": d,
                            "event_type": _classify(kind, purpose),
                            "notes": str(purpose)[:200],
                        }
                    )
    except Exception:
        return load_calendar()

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        return load_calendar()

    if CALENDAR_PATH.exists():
        existing = pd.read_parquet(CALENDAR_PATH)
        df = pd.concat([existing, df], ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "event_date", "event_type", "notes"], keep="last")
    df = df.sort_values(["event_date", "symbol"]).reset_index(drop=True)
    df.to_parquet(CALENDAR_PATH, index=False)
    return df


def load_calendar() -> pd.DataFrame:
    if not CALENDAR_PATH.exists():
        return pd.DataFrame(columns=["symbol", "event_date", "event_type", "notes"])
    return pd.read_parquet(CALENDAR_PATH)


def events_for(symbol: str, d: date | None = None, window_days: int = 0) -> pd.DataFrame:
    cal = load_calendar()
    if cal.empty:
        return cal
    sym = symbol.upper()
    cal = cal.loc[cal["symbol"].astype(str).str.upper() == sym]
    if d is not None:
        lo = d - timedelta(days=window_days)
        hi = d + timedelta(days=window_days)
        cal = cal.loc[(cal["event_date"] >= lo) & (cal["event_date"] <= hi)]
    return cal.reset_index(drop=True)


def skip_around_events(
    signal: pd.Series,
    ts_series: pd.Series,
    symbol: str,
    window_days: int = 1,
    event_types: tuple[str, ...] | None = None,
) -> pd.Series:
    """Zero the signal on ±window_days of any event for `symbol`.

    `event_types=None` means "any event". Otherwise restrict to the listed
    types (e.g. ('earnings','results') to only skip earnings-adjacent bars).
    """
    cal = load_calendar()
    if cal.empty:
        return signal
    sym = symbol.upper()
    sub = cal.loc[cal["symbol"].astype(str).str.upper() == sym]
    if event_types:
        sub = sub.loc[sub["event_type"].astype(str).isin(event_types)]
    if sub.empty:
        return signal

    blocked = set()
    for d in sub["event_date"]:
        for delta in range(-window_days, window_days + 1):
            blocked.add(d + timedelta(days=delta))

    dates = ts_series.dt.tz_convert("Asia/Kolkata").dt.date if hasattr(ts_series.dt, "tz_convert") else ts_series.dt.date
    mask = dates.map(lambda x: x in blocked)
    sig = signal.copy().fillna(0).astype(int)
    sig = sig.where(~mask.values, 0)
    return sig


def _parse_date(s) -> date | None:
    if not s:
        return None
    if isinstance(s, date):
        return s
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(s), fmt).date()
        except ValueError:
            continue
    return None


def _classify(kind: str, text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("result", "earning", "quarterly", "q1", "q2", "q3", "q4")):
        return "earnings"
    if "split" in t:
        return "split"
    if "bonus" in t:
        return "bonus"
    if "dividend" in t:
        return "dividend"
    if "rights" in t:
        return "rights"
    if "merger" in t or "amalgamation" in t:
        return "merger"
    if "agm" in t or "egm" in t or "meeting" in t:
        return "meeting"
    return kind
