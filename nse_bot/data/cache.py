"""Parquet-backed cache for OHLCV candles.

File layout:
    data/candles/{interval}/{safe_symbol}.parquet

Keyed by (interval, tradingsymbol). `safe_symbol` sanitizes characters that
confuse filesystems (mostly '&' and '/'). We merge incoming rows with the
existing file on write — so fetches are naturally incremental.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from nse_bot.config import DATA_DIR

CANDLES_DIR = DATA_DIR / "candles"
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _path(symbol: str, interval: str) -> Path:
    safe = _UNSAFE.sub("_", symbol.upper())
    d = CANDLES_DIR / interval
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{safe}.parquet"


def read(symbol: str, interval: str) -> pd.DataFrame:
    p = _path(symbol, interval)
    if not p.exists():
        return _empty()
    return pd.read_parquet(p)


def write(symbol: str, interval: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    p = _path(symbol, interval)
    if p.exists():
        existing = pd.read_parquet(p)
        merged = pd.concat([existing, df], ignore_index=True)
    else:
        merged = df
    merged = (
        merged.drop_duplicates(subset=["ts"], keep="last")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    merged.to_parquet(p, index=False)


def last_ts(symbol: str, interval: str) -> pd.Timestamp | None:
    df = read(symbol, interval)
    if df.empty:
        return None
    return df["ts"].iloc[-1]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ts", "open", "high", "low", "close", "volume", "oi"],
    )
