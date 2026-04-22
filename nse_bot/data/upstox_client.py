"""Upstox v2 REST client — focused on historical candles.

Endpoint reference:
    GET /v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}
    Intervals: 1minute, 30minute, day, week, month
    Dates: YYYY-MM-DD

Rate limiting is handled client-side with a simple token bucket (25 req/sec),
and 429 responses are retried with exponential backoff.

Upstox caps the per-request date range for intraday candles (1minute is ~1
month per call). The `fetch_candles` wrapper transparently chunks longer
ranges and concatenates results.
"""
from __future__ import annotations

import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

BASE_URL = "https://api.upstox.com/v2"

# Upstox published limits (as of 2024): 25/s, 250/min, 1000/30min. We
# conservatively throttle the per-second bucket only; minute/30-min limits
# are rarely the binding constraint for a single-machine caller.
_REQ_PER_SEC = 20


class UpstoxError(RuntimeError):
    pass


class RateLimitError(UpstoxError):
    pass


@dataclass
class _Bucket:
    capacity: int
    tokens: float
    last_refill: float
    lock: threading.Lock


def _new_bucket(capacity: int) -> _Bucket:
    return _Bucket(capacity=capacity, tokens=float(capacity), last_refill=time.monotonic(), lock=threading.Lock())


def _take(bucket: _Bucket) -> None:
    while True:
        with bucket.lock:
            now = time.monotonic()
            elapsed = now - bucket.last_refill
            bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.capacity)
            bucket.last_refill = now
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return
            wait = (1 - bucket.tokens) / bucket.capacity
        time.sleep(wait)


class UpstoxClient:
    def __init__(self, access_token: str, *, timeout: float = 30.0) -> None:
        if not access_token:
            raise UpstoxError(
                "No access token. Run `python scripts/auth.py` first "
                "(or set UPSTOX_ACCESS_TOKEN in .env)."
            )
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        self._bucket = _new_bucket(_REQ_PER_SEC)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "UpstoxClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _get(self, path: str) -> dict:
        _take(self._bucket)
        r = self._client.get(path)
        if r.status_code == 429:
            raise RateLimitError(f"429 on {path}")
        if r.status_code >= 400:
            raise UpstoxError(f"{r.status_code} on {path}: {r.text[:300]}")
        return r.json()

    # --- Historical candles ---

    def _historical(
        self,
        instrument_key: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> pd.DataFrame:
        """Single-request historical call. Caller ensures range fits Upstox limits."""
        ik = urllib.parse.quote(instrument_key, safe="")
        path = f"/historical-candle/{ik}/{interval}/{to_date.isoformat()}/{from_date.isoformat()}"
        data = self._get(path)
        candles = (data.get("data") or {}).get("candles") or []
        if not candles:
            return _empty_candles()
        df = pd.DataFrame(
            candles,
            columns=["ts", "open", "high", "low", "close", "volume", "oi"],
        )
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
        df = df.sort_values("ts").reset_index(drop=True)
        return df

    def fetch_candles(
        self,
        instrument_key: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> pd.DataFrame:
        """Fetch candles for arbitrary date range, chunking per Upstox limits."""
        if from_date > to_date:
            raise ValueError("from_date must be <= to_date")
        chunks: list[pd.DataFrame] = []
        for start, end in _chunk_range(interval, from_date, to_date):
            df = self._historical(instrument_key, interval, start, end)
            if not df.empty:
                chunks.append(df)
        if not chunks:
            return _empty_candles()
        out = pd.concat(chunks, ignore_index=True)
        out = out.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        return out


def _empty_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.Series(dtype="datetime64[ns, Asia/Kolkata]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="int64"),
            "oi": pd.Series(dtype="int64"),
        }
    )


# Approximate max window per request, per Upstox docs.
_MAX_DAYS = {
    "1minute": 30,
    "30minute": 365,
    "day": 365 * 5,
    "week": 365 * 10,
    "month": 365 * 20,
}


def _chunk_range(interval: str, from_date: date, to_date: date) -> Iterable[tuple[date, date]]:
    max_days = _MAX_DAYS.get(interval, 365)
    cur = from_date
    while cur <= to_date:
        end = min(to_date, cur + timedelta(days=max_days - 1))
        yield cur, end
        cur = end + timedelta(days=1)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a 1-minute OHLCV frame to e.g. '5min' or '15min' bars."""
    if df.empty:
        return df
    x = df.set_index("ts")
    agg = x.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "oi": "last",
        }
    ).dropna(subset=["open"]).reset_index()
    return agg
