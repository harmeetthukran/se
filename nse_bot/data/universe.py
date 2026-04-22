"""NSE instrument universe loader.

Upstox publishes a daily instruments dump listing every tradable instrument
with its `instrument_key` — the ID required by the historical-candle API.

    https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz

We cache it locally under data/instruments.parquet and provide helpers to
filter by segment (NSE equity, NSE F&O, etc.).
"""
from __future__ import annotations

import gzip
import io
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from nse_bot.config import DATA_DIR

INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
CACHE_PATH = DATA_DIR / "instruments.parquet"
STAMP_PATH = DATA_DIR / "instruments.stamp"


def refresh_instruments(force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not force and _is_fresh_today():
        return load_instruments()

    r = httpx.get(INSTRUMENTS_URL, timeout=120, follow_redirects=True)
    r.raise_for_status()
    raw = gzip.decompress(r.content) if r.content[:2] == b"\x1f\x8b" else r.content
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)

    df.columns = [c.strip().lower() for c in df.columns]
    df.to_parquet(CACHE_PATH, index=False)
    STAMP_PATH.write_text(date.today().isoformat(), encoding="utf-8")
    return df


def load_instruments() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return refresh_instruments(force=True)
    return pd.read_parquet(CACHE_PATH)


def _is_fresh_today() -> bool:
    if not STAMP_PATH.exists() or not CACHE_PATH.exists():
        return False
    try:
        return STAMP_PATH.read_text(encoding="utf-8").strip() == date.today().isoformat()
    except OSError:
        return False


def nse_equity(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return NSE cash-market equity rows only."""
    df = df if df is not None else load_instruments()
    seg = df.get("segment")
    ex = df.get("exchange")
    instr_type = df.get("instrument_type")
    mask = pd.Series(True, index=df.index)
    if seg is not None:
        mask &= seg.astype(str).str.upper().eq("NSE_EQ")
    if ex is not None:
        mask &= ex.astype(str).str.upper().eq("NSE")
    if instr_type is not None:
        mask &= instr_type.astype(str).str.upper().isin(["EQ", "EQUITY"])
    return df.loc[mask].reset_index(drop=True)


def nse_fno(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return NSE futures & options rows."""
    df = df if df is not None else load_instruments()
    seg = df.get("segment")
    if seg is None:
        return df.head(0)
    return df.loc[seg.astype(str).str.upper().eq("NSE_FO")].reset_index(drop=True)


def find_symbol(symbol: str, df: pd.DataFrame | None = None) -> pd.Series | None:
    """Lookup by trading symbol (e.g. 'RELIANCE')."""
    df = df if df is not None else nse_equity()
    key = symbol.strip().upper()
    for col in ("tradingsymbol", "trading_symbol", "symbol", "name"):
        if col in df.columns:
            hit = df.loc[df[col].astype(str).str.upper() == key]
            if not hit.empty:
                return hit.iloc[0]
    return None
