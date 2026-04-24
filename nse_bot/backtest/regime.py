"""Market-regime gates for strategy signals.

A regime mask is a boolean Series (same length as the bars). Where it is
False, the engine treats the signal as 0 (no position). Where True, the
signal is taken as-is.

Three gate builders:

    trend_above_sma(df, period=200)
        Close is above an N-period SMA — a long-bias filter for momentum
        strategies. Set direction='short' to invert (for short bias).

    volatility_floor(df, atr_period=14, min_atr_pct=0.005)
        Enough intraday range to justify the trade. Useful for intraday
        mean-reversion and ORB strategies that bleed on dead days.

    nifty_regime(bars_ts, nifty_daily_df, period=200)
        Applies Nifty's 200-SMA trend filter to an arbitrary bar index.
        Used to apply an index-level gate to individual stocks or sectors.

Combine masks with bitwise `&` / `|` as needed.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from nse_bot.backtest.risk import atr


def trend_above_sma(df: pd.DataFrame, period: int = 200, direction: Literal["long", "short"] = "long") -> pd.Series:
    close = df["close"].astype(float).reset_index(drop=True)
    sma = close.rolling(period, min_periods=period).mean()
    mask = close > sma if direction == "long" else close < sma
    return mask.fillna(False).astype(bool)


def volatility_floor(df: pd.DataFrame, atr_period: int = 14, min_atr_pct: float = 0.005) -> pd.Series:
    a = atr(df, atr_period).reset_index(drop=True)
    close = df["close"].astype(float).reset_index(drop=True)
    pct = (a / close).fillna(0.0)
    return (pct >= min_atr_pct).astype(bool)


def nifty_regime(
    bars_ts: pd.Series,
    nifty_daily_df: pd.DataFrame,
    period: int = 200,
    direction: Literal["long", "short"] = "long",
) -> pd.Series:
    """Apply Nifty's daily 200-SMA trend to arbitrary (possibly intraday) bars.

    nifty_daily_df: OHLCV frame of Nifty 50 daily candles with a `ts` column.
    bars_ts:        pd.Series of IST timestamps from the primary instrument.

    Returns a boolean mask aligned to bars_ts index.
    """
    n = nifty_daily_df.copy().sort_values("ts").reset_index(drop=True)
    n["date"] = n["ts"].dt.tz_convert("Asia/Kolkata").dt.date if hasattr(n["ts"].dt, "tz_convert") else n["ts"].dt.date
    n["sma"] = n["close"].astype(float).rolling(period, min_periods=period).mean()
    n["mask"] = (n["close"] > n["sma"]) if direction == "long" else (n["close"] < n["sma"])
    daily_mask = n.set_index("date")["mask"].fillna(False)

    dates = bars_ts.dt.tz_convert("Asia/Kolkata").dt.date if hasattr(bars_ts.dt, "tz_convert") else bars_ts.dt.date
    out = dates.map(lambda d: bool(daily_mask.get(d, False)))
    return out.astype(bool).reset_index(drop=True)


def apply_mask(signal: pd.Series, mask: pd.Series) -> pd.Series:
    """Mask a signal Series with a boolean Series. Shapes must match."""
    sig = signal.reset_index(drop=True).fillna(0).astype(int)
    m = mask.reset_index(drop=True).astype(bool)
    if len(sig) != len(m):
        raise ValueError(f"signal/mask length mismatch: {len(sig)} vs {len(m)}")
    return sig.where(m, 0)
