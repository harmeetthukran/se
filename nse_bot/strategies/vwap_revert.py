"""VWAP mean-reversion — intraday.

Rules:
    * Compute session VWAP (intraday, reset daily).
    * Compute rolling standard deviation of price around VWAP (20-bar).
    * Long when close < VWAP - k*sd; flat when close crosses back up through VWAP.
    * Short when close > VWAP + k*sd; flat when close crosses back down through VWAP.
    * Engine force-flat at session close.

Works best on liquid, range-bound names. Burns capital on trending days —
the backtest will surface this in drawdown and win-rate metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nse_bot.strategies.base import Strategy, StrategyResult


class VWAPRevertStrategy(Strategy):
    name = "vwap_revert"
    interval = "5minute"

    def __init__(self, k: float = 2.0, lookback: int = 20) -> None:
        self.k = k
        self.lookback = lookback

    @classmethod
    def param_grid(cls) -> dict[str, list]:
        return {"k": [1.5, 2.0, 2.5, 3.0], "lookback": [10, 20, 40]}

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        if df.empty:
            return StrategyResult(signal=pd.Series(dtype=int), intraday=True)

        d = df.copy()
        d["date"] = d["ts"].dt.date
        tp = (d["high"] + d["low"] + d["close"]) / 3.0
        pv = tp * d["volume"].astype(float)
        vwap_num = pv.groupby(d["date"]).cumsum()
        vwap_den = d["volume"].astype(float).groupby(d["date"]).cumsum().replace(0, np.nan)
        vwap = vwap_num / vwap_den
        dev = (d["close"] - vwap).groupby(d["date"]).rolling(self.lookback).std().reset_index(level=0, drop=True)

        upper = vwap + self.k * dev
        lower = vwap - self.k * dev

        sig = pd.Series(0, index=df.index, dtype=int)
        sig = sig.mask(d["close"] < lower, 1)
        sig = sig.mask(d["close"] > upper, -1)

        # Forward-fill within day until close crosses VWAP.
        parts = []
        for day, grp in sig.groupby(d["date"]):
            vw_grp = vwap.loc[grp.index]
            cl_grp = d["close"].loc[grp.index]
            s = grp.copy()
            current = 0
            for i in grp.index:
                raw = int(grp.loc[i])
                if current == 0 and raw != 0:
                    current = raw
                elif current == 1 and cl_grp.loc[i] >= vw_grp.loc[i]:
                    current = 0
                elif current == -1 and cl_grp.loc[i] <= vw_grp.loc[i]:
                    current = 0
                s.loc[i] = current
            parts.append(s)
        sig = pd.concat(parts).sort_index().astype(int)
        return StrategyResult(signal=sig, intraday=True)
