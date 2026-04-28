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


# Curated Nifty 50 constituents (late 2024). Used by `--symbols nifty50` shortcuts.
# Constituents change quarterly; this list is "good enough" — minor deviations
# don't change backtest conclusions on a 5-year window.
NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "BPCL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM",
    "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ZOMATO",
]


# Compact list of additional very-liquid names beyond Nifty 50.
NIFTY_NEXT_50_PARTIAL = [
    "DLF", "GAIL", "GODREJCP", "HAVELLS", "HINDPETRO",
    "ICICIPRULI", "IOC", "PIDILITIND", "PNB", "SIEMENS",
    "VEDL", "AMBUJACEM", "DABUR", "DMART", "INDIGO",
    "IRCTC", "MUTHOOTFIN", "NAUKRI", "PFC", "SBICARD",
    "SRF", "TVSMOTOR", "UPL", "ZYDUSLIFE", "BERGEPAINT",
    "BIOCON", "CHOLAFIN", "COLPAL", "GODREJPROP", "HAL",
    "ICICIGI", "IDEA", "JINDALSTEL", "LICI", "LUPIN",
    "MOTHERSON", "MPHASIS", "PEL", "PIIND", "RECLTD",
    "TATAPOWER", "TORNTPHARM", "TVSMOTOR", "ABCAPITAL", "BANDHANBNK",
    "BANKBARODA", "GMRINFRA", "INDHOTEL", "IRFC", "PAYTM",
]


def expand_universe_keyword(symbol_or_keyword: str) -> list[str] | None:
    """If symbol_or_keyword is a magic universe alias, return the expanded list.

    Only matches the compact, space-free forms — `nifty50`, `NIFTY50`, `N50` —
    so a literal `"Nifty 50"` (the index trading symbol) still resolves to the
    actual index in the instruments dump.
    """
    s = symbol_or_keyword.strip().upper().replace("_", "").replace("-", "")
    if s in ("NIFTY50", "N50"):
        return list(NIFTY_50)
    if s in ("NIFTY100", "N100"):
        return list(NIFTY_50) + list(NIFTY_NEXT_50_PARTIAL)
    return None


def _isin_from_key(instrument_key: str) -> str:
    """instrument_key looks like 'NSE_EQ|INE002A01018'. Return the ISIN portion."""
    s = str(instrument_key)
    return s.split("|", 1)[1] if "|" in s else s


def nse_equity(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return NSE cash-market equity rows only.

    Upstox's `exchange` column on the instruments dump uses values like
    `NSE_EQ` / `BSE_EQ` (not `NSE`). The `instrument_type` column tags every
    NSE_EQ row as `EQUITY` even for government securities, SDLs, and T-bills.
    Real equities have ISINs that start with `INE`; bonds/SDLs/T-bills start
    with `IN0`/`IN1`/`IN2`/etc.
    """
    df = df if df is not None else load_instruments()
    ex = df.get("exchange")
    instr_type = df.get("instrument_type")
    seg = df.get("segment")
    key = df.get("instrument_key")

    mask = pd.Series(True, index=df.index)
    if ex is not None:
        mask &= ex.astype(str).str.upper().eq("NSE_EQ")
    elif seg is not None:
        mask &= seg.astype(str).str.upper().eq("NSE_EQ")
    if instr_type is not None:
        mask &= instr_type.astype(str).str.upper().isin(["EQ", "EQUITY"])
    if key is not None:
        # Keep only real equities — ISIN starts with INE.
        mask &= key.astype(str).map(_isin_from_key).str.upper().str.startswith("INE")
    return df.loc[mask].reset_index(drop=True)


def nse_fno(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return NSE futures & options rows."""
    df = df if df is not None else load_instruments()
    ex = df.get("exchange")
    seg = df.get("segment")
    if ex is not None:
        return df.loc[ex.astype(str).str.upper().eq("NSE_FO")].reset_index(drop=True)
    if seg is not None:
        return df.loc[seg.astype(str).str.upper().eq("NSE_FO")].reset_index(drop=True)
    return df.head(0)


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
