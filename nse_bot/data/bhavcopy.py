"""NSE F&O bhavcopy fetcher and loader.

Source of truth for EOD option chain and futures data, going back 15+ years.
Free, no auth, no rate limits beyond reasonable pacing.

Two URL/format generations are handled transparently:

    Legacy  (pre-2024-07-08):
        https://nsearchives.nseindia.com/content/historical/DERIVATIVES/
            {YYYY}/{MON}/fo{DDMONYYYY}bhav.csv.zip
        CSV columns: INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP,
                     OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH,
                     OPEN_INT, CHG_IN_OI, TIMESTAMP

    UDiFF   (2024-07-08 onwards, multiple patterns — NSE has changed this
             endpoint several times):
        https://nsearchives.nseindia.com/content/fo/
            BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
        Wider schema, column names prefixed (TckrSymb, StrkPric, OptnTp, ...).

The fetcher writes one parquet per trading date under
`data/bhavcopy/{YYYY-MM-DD}.parquet` with a unified schema:

    date, symbol, instrument_type, expiry, strike, option_type,
    open, high, low, close, settle, volume, open_interest, chg_in_oi

`load_range(symbols, from_date, to_date)` returns a long frame filtered
to the requested universe and date range.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd

from nse_bot.config import DATA_DIR

BHAVCOPY_DIR = DATA_DIR / "bhavcopy"

_MON_UPPER = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/zip, text/csv, */*",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}

UNIFIED_COLS = [
    "date", "symbol", "instrument_type", "expiry", "strike", "option_type",
    "open", "high", "low", "close", "settle", "volume", "open_interest", "chg_in_oi",
]


def _legacy_url(d: date) -> str:
    return (
        f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
        f"{d.year}/{_MON_UPPER[d.month - 1]}/fo{d.day:02d}{_MON_UPPER[d.month - 1]}{d.year}bhav.csv.zip"
    )


def _udiff_urls(d: date) -> list[str]:
    """All known UDiFF URL patterns. Try in order."""
    ymd = d.strftime("%Y%m%d")
    return [
        f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip",
        f"https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip",
    ]


def _download_zip(client: httpx.Client, url: str, timeout: float = 30.0) -> bytes | None:
    try:
        r = client.get(url, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return None
        return r.content
    except Exception:
        return None


def _parse_legacy(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = [c.strip().upper() for c in df.columns]
    needed = {"INSTRUMENT", "SYMBOL", "EXPIRY_DT", "OPEN", "HIGH", "LOW", "CLOSE"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["TIMESTAMP"]).dt.date if "TIMESTAMP" in df.columns else pd.NaT
    out["symbol"] = df["SYMBOL"].astype(str).str.upper()
    out["instrument_type"] = df["INSTRUMENT"].astype(str)
    out["expiry"] = pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce").dt.date
    out["strike"] = pd.to_numeric(df.get("STRIKE_PR"), errors="coerce")
    out["option_type"] = df.get("OPTION_TYP", "").astype(str).str.upper().replace({"XX": ""})
    out["open"] = pd.to_numeric(df["OPEN"], errors="coerce")
    out["high"] = pd.to_numeric(df["HIGH"], errors="coerce")
    out["low"] = pd.to_numeric(df["LOW"], errors="coerce")
    out["close"] = pd.to_numeric(df["CLOSE"], errors="coerce")
    out["settle"] = pd.to_numeric(df.get("SETTLE_PR"), errors="coerce")
    out["volume"] = pd.to_numeric(df.get("CONTRACTS"), errors="coerce").fillna(0).astype("int64")
    out["open_interest"] = pd.to_numeric(df.get("OPEN_INT"), errors="coerce").fillna(0).astype("int64")
    out["chg_in_oi"] = pd.to_numeric(df.get("CHG_IN_OI"), errors="coerce").fillna(0).astype("int64")
    return out[UNIFIED_COLS]


def _parse_udiff(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # Column aliases across NSE revisions.
    col = {c.lower(): c for c in df.columns}

    def g(*names):
        for n in names:
            if n.lower() in col:
                return df[col[n.lower()]]
        return pd.Series([pd.NA] * len(df))

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(g("TradDt", "BizDt"), errors="coerce").dt.date
    out["symbol"] = g("TckrSymb", "Symbol").astype(str).str.upper()
    out["instrument_type"] = g("FinInstrmTp").astype(str)
    out["expiry"] = pd.to_datetime(g("XpryDt"), errors="coerce").dt.date
    out["strike"] = pd.to_numeric(g("StrkPric"), errors="coerce")
    out["option_type"] = g("OptnTp").astype(str).str.upper().replace({"NAN": ""})
    out["open"] = pd.to_numeric(g("OpnPric"), errors="coerce")
    out["high"] = pd.to_numeric(g("HghPric"), errors="coerce")
    out["low"] = pd.to_numeric(g("LwPric"), errors="coerce")
    out["close"] = pd.to_numeric(g("ClsPric"), errors="coerce")
    out["settle"] = pd.to_numeric(g("SttlmPric"), errors="coerce")
    out["volume"] = pd.to_numeric(g("TtlTradgVol"), errors="coerce").fillna(0).astype("int64")
    out["open_interest"] = pd.to_numeric(g("OpnIntrst"), errors="coerce").fillna(0).astype("int64")
    out["chg_in_oi"] = pd.to_numeric(g("ChngInOpnIntrst"), errors="coerce").fillna(0).astype("int64")
    return out[UNIFIED_COLS]


def _extract_csv(zip_bytes: bytes) -> bytes | None:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                return None
            with zf.open(names[0]) as f:
                return f.read()
    except zipfile.BadZipFile:
        return None


def fetch_day(d: date, client: httpx.Client | None = None, *, sleep_s: float = 0.5) -> pd.DataFrame:
    """Download and parse one trading day's F&O bhavcopy. Returns empty on failure."""
    own_client = client is None
    if own_client:
        client = httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=30.0)
        # Cookie warm-up.
        try:
            client.get("https://www.nseindia.com", timeout=15.0)
        except Exception:
            pass

    try:
        # Try UDiFF first (newer data is more likely requested), then legacy.
        for url in _udiff_urls(d):
            raw = _download_zip(client, url)
            if raw:
                csv = _extract_csv(raw)
                if csv:
                    df = _parse_udiff(csv)
                    if not df.empty:
                        return df
        time.sleep(sleep_s)

        legacy = _download_zip(client, _legacy_url(d))
        if legacy:
            csv = _extract_csv(legacy)
            if csv:
                return _parse_legacy(csv)
        return pd.DataFrame(columns=UNIFIED_COLS)
    finally:
        if own_client:
            client.close()


def _path_for(d: date) -> Path:
    BHAVCOPY_DIR.mkdir(parents=True, exist_ok=True)
    return BHAVCOPY_DIR / f"{d.isoformat()}.parquet"


def save_day(d: date, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.to_parquet(_path_for(d), index=False)


def fetch_range(
    from_date: date,
    to_date: date,
    *,
    resume: bool = True,
    sleep_s: float = 0.5,
    on_progress=None,
) -> dict:
    """Fetch and cache bhavcopies for every weekday in [from_date, to_date].

    Returns dict with counts (fetched, skipped, failed).
    """
    BHAVCOPY_DIR.mkdir(parents=True, exist_ok=True)
    dates = []
    cur = from_date
    while cur <= to_date:
        if cur.weekday() < 5:  # Mon-Fri
            dates.append(cur)
        cur += timedelta(days=1)

    fetched, skipped, failed = 0, 0, 0
    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=30.0) as client:
        try:
            client.get("https://www.nseindia.com", timeout=15.0)
        except Exception:
            pass
        for d in dates:
            if resume and _path_for(d).exists():
                skipped += 1
                if on_progress:
                    on_progress(d, "skipped")
                continue
            df = fetch_day(d, client=client, sleep_s=sleep_s)
            if df.empty:
                failed += 1
                if on_progress:
                    on_progress(d, "failed")
            else:
                save_day(d, df)
                fetched += 1
                if on_progress:
                    on_progress(d, "fetched")
            time.sleep(sleep_s)
    return {"fetched": fetched, "skipped": skipped, "failed": failed}


# ---------- Query / loader ----------

def load_day(d: date) -> pd.DataFrame:
    p = _path_for(d)
    if not p.exists():
        return pd.DataFrame(columns=UNIFIED_COLS)
    return pd.read_parquet(p)


def load_range(
    from_date: date,
    to_date: date,
    symbols: Iterable[str] | None = None,
    instrument_types: Iterable[str] | None = None,
    expiry: date | None = None,
    option_type: str | None = None,
) -> pd.DataFrame:
    """Concatenate cached bhavcopies in [from_date, to_date] with optional filters."""
    frames: list[pd.DataFrame] = []
    cur = from_date
    syms = {s.upper() for s in symbols} if symbols else None
    insts = {s.upper() for s in instrument_types} if instrument_types else None
    opt = option_type.upper() if option_type else None
    while cur <= to_date:
        df = load_day(cur)
        if not df.empty:
            if syms is not None:
                df = df.loc[df["symbol"].isin(syms)]
            if insts is not None:
                df = df.loc[df["instrument_type"].astype(str).str.upper().isin(insts)]
            if expiry is not None:
                df = df.loc[df["expiry"] == expiry]
            if opt is not None:
                df = df.loc[df["option_type"].astype(str).str.upper() == opt]
            if not df.empty:
                frames.append(df)
        cur += timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLS)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol", "strike"]).reset_index(drop=True)


def option_chain(
    symbol: str,
    as_of: date,
    expiry: date | None = None,
) -> pd.DataFrame:
    """Return the option chain (CE + PE) for `symbol` on the given trading day.

    If `expiry` is None, returns all expiries.
    """
    df = load_day(as_of)
    if df.empty:
        return df
    df = df.loc[
        (df["symbol"].astype(str).str.upper() == symbol.upper())
        & (df["instrument_type"].astype(str).str.upper().isin(["OPTIDX", "OPTSTK", "STO", "IDO"]))
    ]
    if expiry is not None:
        df = df.loc[df["expiry"] == expiry]
    return df.sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)


def nearest_expiry(
    symbol: str,
    as_of: date,
    weekly: bool = True,
) -> date | None:
    """Find the nearest (future) expiry for symbol in the bhavcopy of `as_of`."""
    df = load_day(as_of)
    if df.empty:
        return None
    sub = df.loc[
        (df["symbol"].astype(str).str.upper() == symbol.upper())
        & (df["instrument_type"].astype(str).str.upper().isin(["OPTIDX", "OPTSTK", "STO", "IDO"]))
        & (df["expiry"] > as_of)
    ]
    if sub.empty:
        return None
    expiries = sorted(sub["expiry"].dropna().unique())
    return expiries[0] if expiries else None


def underlying_price(symbol: str, as_of: date) -> float | None:
    """Approximate spot price from the nearest-expiry futures close.

    For Indian index/stock futures the basis is small near expiry and
    converges to zero at expiry, so the nearest-expiry futures close is a
    standard proxy when bhavcopy doesn't carry the spot directly.
    """
    df = load_day(as_of)
    if df.empty:
        return None
    fut = df.loc[
        (df["symbol"].astype(str).str.upper() == symbol.upper())
        & df["instrument_type"].astype(str).str.upper().isin(["FUTIDX", "FUTSTK", "STF", "IDF"])
        & (df["expiry"] >= as_of)
    ]
    if fut.empty:
        return None
    fut = fut.sort_values("expiry")
    px = fut.iloc[0]["close"]
    try:
        return float(px) if pd.notna(px) else None
    except (TypeError, ValueError):
        return None


def expiries_for(symbol: str, as_of: date) -> list[date]:
    """All option expiries listed for `symbol` on `as_of`."""
    df = load_day(as_of)
    if df.empty:
        return []
    sub = df.loc[
        (df["symbol"].astype(str).str.upper() == symbol.upper())
        & df["instrument_type"].astype(str).str.upper().isin(["OPTIDX", "OPTSTK", "STO", "IDO"])
    ]
    return sorted({d for d in sub["expiry"].dropna().unique() if d >= as_of})
