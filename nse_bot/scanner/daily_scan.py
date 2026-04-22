"""End-of-day market scanner.

Ranks the NSE equity universe on a composite score combining:
    * 20-day price momentum
    * 50-day liquidity (avg daily turnover)
    * Distance from 52-week high
    * RSI(14) regime
    * News sentiment (last 24h, if news module returned anything)

Scanner reads daily candles from the parquet cache (so you must run
`fetch_history.py --interval day` first).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nse_bot.data import cache, news, universe


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _features_for(symbol: str) -> dict | None:
    df = cache.read(symbol, "day")
    if df.empty or len(df) < 60:
        return None
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    turnover = close * volume
    high_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()

    ret20 = close.iloc[-1] / close.iloc[-21] - 1.0 if len(close) > 21 else 0.0
    adv50 = turnover.iloc[-50:].mean() if len(turnover) >= 50 else turnover.mean()
    dist_52w = close.iloc[-1] / high_52w - 1.0 if high_52w else 0.0
    rsi = float(_rsi(close).iloc[-1]) if len(close) > 20 else 50.0

    return {
        "symbol": symbol,
        "close": float(close.iloc[-1]),
        "ret_20d_pct": float(ret20 * 100),
        "adv50_cr": float(adv50 / 1e7),
        "dist_52w_pct": float(dist_52w * 100),
        "rsi14": rsi,
    }


def scan(min_adv_cr: float = 5.0, top_n: int = 50) -> pd.DataFrame:
    """Run the scanner across everything in the NSE equity universe cache."""
    inst = universe.nse_equity()
    symbol_col = next(
        (c for c in ("tradingsymbol", "trading_symbol", "symbol") if c in inst.columns),
        None,
    )
    if symbol_col is None:
        raise RuntimeError("Could not locate trading-symbol column in instruments frame")

    rows: list[dict] = []
    for sym in inst[symbol_col].astype(str).unique():
        feats = _features_for(sym)
        if feats is None:
            continue
        rows.append(feats)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.loc[df["adv50_cr"] >= min_adv_cr].copy()

    items = news.fetch_rss()
    sent = news.score_for_symbols(items, df["symbol"].tolist())
    df["news_n"] = df["symbol"].map(lambda s: sent.get(s, {}).get("n", 0))
    df["news_sentiment"] = df["symbol"].map(lambda s: sent.get(s, {}).get("mean", 0.0))

    def _z(x: pd.Series) -> pd.Series:
        std = x.std()
        return (x - x.mean()) / std if std and std > 0 else pd.Series(0.0, index=x.index)

    df["score"] = (
        _z(df["ret_20d_pct"])
        + _z(df["dist_52w_pct"])
        + _z(df["news_sentiment"]) * 0.5
        - _z((df["rsi14"] - 50).abs()) * 0.25
    )
    df = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return df
