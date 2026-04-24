"""Mask strategy signals by news sentiment from the local news store.

Two policies:

    require_confirmation
        Long signals pass only if the day's avg sentiment >= min_long
        (default 0.0); short signals only if avg <= max_short (default 0.0).
        Skips trades that fight the day's news tone.

    block_contradiction
        Long signals pass unless avg sentiment <= min_long (default -0.3).
        Short signals pass unless avg sentiment >= max_short (default 0.3).
        More permissive — only blocks trades with strong opposing news.

Days without any news in the store default to *pass through* (no filter),
since we can't fairly block based on absent data. Toggle with
`on_missing='block'` for the stricter behavior.

Symbol-level matching is attempted first (headline / summary contains the
symbol token); falls back to day-level aggregate when no ticker match.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Literal

import pandas as pd

from nse_bot.data import news_store

IST = "Asia/Kolkata"


def sentiment_mask(
    ts_series: pd.Series,
    symbol: str,
    policy: Literal["require_confirmation", "block_contradiction"] = "block_contradiction",
    min_long: float | None = None,
    max_short: float | None = None,
    on_missing: Literal["pass", "block"] = "pass",
) -> tuple[pd.Series, pd.Series]:
    """Return (long_mask, short_mask) booleans aligned to ts_series.

    Where long_mask is False, long signals should be dropped (masked to 0).
    Where short_mask is False, short signals should be dropped.
    """
    if min_long is None:
        min_long = 0.0 if policy == "require_confirmation" else -0.3
    if max_short is None:
        max_short = 0.0 if policy == "require_confirmation" else 0.3

    df = news_store.read_all()
    if df.empty:
        pass_all = pd.Series(on_missing == "pass", index=ts_series.index)
        return pass_all.copy(), pass_all.copy()

    df = df.copy()
    df["date"] = df["ts"].dt.tz_convert(IST).dt.date

    key = symbol.upper()
    pat = re.compile(rf"\b{re.escape(key)}\b")
    title_u = df["title"].astype(str).str.upper()
    summary_u = df["summary"].astype(str).str.upper()
    sym_match = title_u.str.contains(pat, regex=True, na=False) | summary_u.str.contains(pat, regex=True, na=False)
    df_sym = df.loc[sym_match]

    day_aggr_sym = df_sym.groupby("date")["sentiment"].mean() if not df_sym.empty else pd.Series(dtype=float)
    day_aggr_all = df.groupby("date")["sentiment"].mean()

    def _score_for(d: date) -> tuple[float | None, bool]:
        """Return (sentiment, has_symbol_match). sentiment is None when missing entirely."""
        if d in day_aggr_sym.index:
            return float(day_aggr_sym.loc[d]), True
        if d in day_aggr_all.index:
            return float(day_aggr_all.loc[d]), False
        return None, False

    dates = ts_series.dt.tz_convert(IST).dt.date if hasattr(ts_series.dt, "tz_convert") else ts_series.dt.date
    long_pass = []
    short_pass = []
    for d in dates:
        score, _has_sym = _score_for(d)
        if score is None:
            long_pass.append(on_missing == "pass")
            short_pass.append(on_missing == "pass")
            continue
        if policy == "require_confirmation":
            long_pass.append(score >= min_long)
            short_pass.append(score <= max_short)
        else:
            long_pass.append(score > min_long)
            short_pass.append(score < max_short)
    idx = ts_series.index
    return pd.Series(long_pass, index=idx), pd.Series(short_pass, index=idx)


def apply_news_filter(
    signal: pd.Series,
    ts_series: pd.Series,
    symbol: str,
    policy: Literal["require_confirmation", "block_contradiction"] = "block_contradiction",
    **kwargs,
) -> pd.Series:
    """Convenience: return the signal with longs/shorts masked by sentiment."""
    long_mask, short_mask = sentiment_mask(ts_series, symbol, policy=policy, **kwargs)
    sig = signal.copy().fillna(0).astype(int)
    sig = sig.where(~((sig > 0) & ~long_mask.values), 0)
    sig = sig.where(~((sig < 0) & ~short_mask.values), 0)
    return sig
