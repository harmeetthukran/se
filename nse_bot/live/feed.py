"""Real-time market data — Upstox v2 quote polling.

Upstox's official streaming feed is a WebSocket protocol with protobuf
messages. Implementing it well requires their .proto file and message
parsing, which is beyond the scope of this stack right now. This module
uses the simpler `/v2/market-quote/quotes` REST endpoint at a configurable
interval. For 1-5 second polling on tens of instruments it's perfectly
adequate during paper-trading and most live-trading workflows; only
high-frequency intraday strategies actually need the WebSocket.

Output: caller-supplied `on_tick(quote_dict)` callback per fetched quote,
plus optional persistence to data/live/ticks_{date}.parquet.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable

import httpx
import pandas as pd

from nse_bot.config import DATA_DIR

LIVE_DIR = DATA_DIR / "live"
BASE_URL = "https://api.upstox.com/v2"


@dataclass
class FeedConfig:
    poll_interval_s: float = 1.0
    persist_ticks: bool = True
    request_timeout_s: float = 5.0
    max_runtime_s: float | None = None  # None = until interrupted


def _ticks_path(d: date) -> Path:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    return LIVE_DIR / f"ticks_{d.isoformat()}.parquet"


def _persist(rows: list[dict]) -> None:
    if not rows:
        return
    today = date.today()
    p = _ticks_path(today)
    df = pd.DataFrame(rows)
    if p.exists():
        existing = pd.read_parquet(p)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(p, index=False)


def fetch_quotes(
    client: httpx.Client,
    instrument_keys: Iterable[str],
    timeout: float = 5.0,
) -> dict[str, dict]:
    """One request → quotes for all listed instruments.

    Upstox's /market-quote/quotes accepts up to ~250 instrument_keys in a
    single comma-separated `symbol` param. Returns a dict keyed by the
    upstox-side identifier.
    """
    keys = list(instrument_keys)
    if not keys:
        return {}
    r = client.get(
        "/market-quote/quotes",
        params={"symbol": ",".join(keys)},
        timeout=timeout,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"quotes {r.status_code}: {r.text[:300]}")
    payload = r.json() or {}
    return payload.get("data") or {}


def stream(
    access_token: str,
    instrument_keys: list[str],
    on_tick: Callable[[dict], None] | None = None,
    cfg: FeedConfig = FeedConfig(),
) -> int:
    """Poll quotes every cfg.poll_interval_s seconds. Returns total ticks emitted.

    `on_tick` is called for each quote on each poll. Persistence to parquet
    happens once per poll (a small batch write).
    """
    if not access_token:
        raise RuntimeError("Missing access_token. Run scripts/auth.py first.")
    if not instrument_keys:
        raise ValueError("Provide at least one instrument_key.")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    started = time.monotonic()
    total_ticks = 0
    rows_buffer: list[dict] = []
    flush_every = 50

    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=cfg.request_timeout_s) as client:
        while True:
            t0 = time.monotonic()
            try:
                quotes = fetch_quotes(client, instrument_keys, timeout=cfg.request_timeout_s)
            except Exception as e:
                if cfg.max_runtime_s and (time.monotonic() - started) > cfg.max_runtime_s:
                    break
                time.sleep(min(5.0, cfg.poll_interval_s * 2))
                continue

            now = datetime.now()
            for key, q in quotes.items():
                tick = {
                    "ts": now,
                    "instrument_key": key,
                    "ltp": float(q.get("last_price") or 0),
                    "open": float((q.get("ohlc") or {}).get("open") or 0),
                    "high": float((q.get("ohlc") or {}).get("high") or 0),
                    "low": float((q.get("ohlc") or {}).get("low") or 0),
                    "close": float((q.get("ohlc") or {}).get("close") or 0),
                    "volume": int(q.get("volume") or 0),
                    "bid": float(q.get("depth", {}).get("buy", [{}])[0].get("price") or 0)
                           if q.get("depth") else 0.0,
                    "ask": float(q.get("depth", {}).get("sell", [{}])[0].get("price") or 0)
                           if q.get("depth") else 0.0,
                }
                if on_tick:
                    try:
                        on_tick(tick)
                    except Exception:
                        pass
                if cfg.persist_ticks:
                    rows_buffer.append(tick)
                total_ticks += 1

            if cfg.persist_ticks and len(rows_buffer) >= flush_every:
                _persist(rows_buffer)
                rows_buffer = []

            if cfg.max_runtime_s and (time.monotonic() - started) > cfg.max_runtime_s:
                break

            sleep_s = max(0.0, cfg.poll_interval_s - (time.monotonic() - t0))
            if sleep_s > 0:
                time.sleep(sleep_s)

    if cfg.persist_ticks and rows_buffer:
        _persist(rows_buffer)

    return total_ticks
