"""ATM implied volatility & IV rank from cached bhavcopy.

For each trading day in the cache:
    1. Look up the nearest-expiry option chain for the symbol.
    2. Find the strike closest to spot (ATM).
    3. Solve IV from the ATM CE price and the ATM PE price using Black-Scholes.
    4. Take the average of the two as the day's "ATM IV".

`iv_history(symbol, from_date, to_date)` returns a series of ATM IVs, one per
trading day with cached data.

`iv_rank(history, current)` returns where today's IV sits between its
trailing min and max (0 = at low, 1 = at high). The classic "IV rank" used
to gate options entries — buy premium when IV rank is low, sell premium
when IV rank is high.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from nse_bot.data import bhavcopy
from nse_bot.options.pricing import implied_vol


def atm_iv_for_day(symbol: str, d: date, *, risk_free_rate: float = 0.065) -> float | None:
    """ATM IV from average of CE and PE on the nearest expiry."""
    chain = bhavcopy.load_day(d)
    spot = bhavcopy.underlying_price(symbol, d)
    if chain.empty or spot is None:
        return None
    expiry = bhavcopy.nearest_expiry(symbol, d)
    if expiry is None:
        return None
    dte_days = (expiry - d).days
    if dte_days <= 0:
        return None
    T = dte_days / 365.0

    sub = chain.loc[
        (chain["symbol"].astype(str).str.upper() == symbol.upper())
        & (chain["expiry"] == expiry)
        & chain["instrument_type"].astype(str).str.upper().isin(["OPTIDX", "OPTSTK", "STO", "IDO"])
    ]
    if sub.empty:
        return None

    strikes = sorted(sub["strike"].dropna().unique())
    atm = min(strikes, key=lambda k: abs(k - spot))
    atm_rows = sub.loc[sub["strike"] == atm]
    ce = atm_rows.loc[atm_rows["option_type"].str.upper() == "CE"]
    pe = atm_rows.loc[atm_rows["option_type"].str.upper() == "PE"]
    if ce.empty or pe.empty:
        return None
    ce_px = float(ce.iloc[0]["close"])
    pe_px = float(pe.iloc[0]["close"])
    if ce_px <= 0 or pe_px <= 0:
        return None
    iv_ce = implied_vol(ce_px, spot, atm, T, risk_free_rate, "CE")
    iv_pe = implied_vol(pe_px, spot, atm, T, risk_free_rate, "PE")
    if iv_ce <= 0 and iv_pe <= 0:
        return None
    if iv_ce <= 0:
        return float(iv_pe)
    if iv_pe <= 0:
        return float(iv_ce)
    return float((iv_ce + iv_pe) / 2.0)


def iv_history(
    symbol: str,
    from_date: date,
    to_date: date,
    *,
    risk_free_rate: float = 0.065,
) -> pd.Series:
    """Series of daily ATM IVs over [from_date, to_date]."""
    out: dict[date, float] = {}
    cur = from_date
    while cur <= to_date:
        if cur.weekday() < 5:
            v = atm_iv_for_day(symbol, cur, risk_free_rate=risk_free_rate)
            if v is not None and v > 0:
                out[cur] = v
        cur += timedelta(days=1)
    return pd.Series(out, name="atm_iv")


def iv_rank(history: pd.Series, current_value: float, lookback: int = 252) -> float:
    """Where current_value sits between trailing min/max of `history`.

    Returns a number in [0, 1]. Returns NaN if too little data.
    """
    if history is None or history.empty:
        return float("nan")
    window = history.iloc[-lookback:]
    if len(window) < 30:
        return float("nan")
    lo = float(window.min())
    hi = float(window.max())
    if hi <= lo:
        return 0.5
    return float((current_value - lo) / (hi - lo))


def iv_percentile(history: pd.Series, current_value: float, lookback: int = 252) -> float:
    """Fraction of past values <= current. More robust than rank in skewed regimes."""
    if history is None or history.empty:
        return float("nan")
    window = history.iloc[-lookback:]
    if len(window) < 30:
        return float("nan")
    return float((window.values <= current_value).mean())
