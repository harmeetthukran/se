"""Opening Range Breakout (ORB) — intraday.

Rules:
    * Define the opening range as the first N minutes of the session (default 15).
    * Long if price breaks above the OR high. Short if it breaks below OR low.
    * Exit on opposite breakout or session close (engine handles the close).

Works on intraday bars (5min or 15min recommended). For 1-min bars, set
`or_minutes` accordingly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nse_bot.strategies.base import Strategy, StrategyResult

SESSION_START = (9, 15)
SESSION_END = (15, 30)


class ORBStrategy(Strategy):
    name = "orb"
    interval = "15minute"

    def __init__(self, or_minutes: int = 15) -> None:
        self.or_minutes = or_minutes

    @classmethod
    def param_grid(cls) -> dict[str, list]:
        return {"or_minutes": [15, 30, 45, 60]}

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        if df.empty:
            return StrategyResult(signal=pd.Series(dtype=int), intraday=True)

        d = df.copy()
        d["date"] = d["ts"].dt.date
        minutes = d["ts"].dt.hour * 60 + d["ts"].dt.minute
        start_m = SESSION_START[0] * 60 + SESSION_START[1]
        or_end_m = start_m + self.or_minutes
        end_m = SESSION_END[0] * 60 + SESSION_END[1]
        in_session = (minutes >= start_m) & (minutes < end_m)
        in_or = in_session & (minutes < or_end_m)

        d["_or"] = np.where(in_or, 1, 0)
        or_groups = d.groupby("date")["_or"].cumsum()
        or_high = d.loc[d["_or"] == 1].groupby("date")["high"].max()
        or_low = d.loc[d["_or"] == 1].groupby("date")["low"].min()
        d = d.merge(or_high.rename("or_high"), left_on="date", right_index=True, how="left")
        d = d.merge(or_low.rename("or_low"), left_on="date", right_index=True, how="left")

        past_or = in_session & (minutes >= or_end_m)
        long_entry = past_or & (d["close"] > d["or_high"])
        short_entry = past_or & (d["close"] < d["or_low"])

        raw = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        sig = pd.Series(raw, index=df.index, dtype=int)

        # Hold the position until the opposite breakout (i.e. forward-fill
        # within the session, reset daily).
        session_key = d["date"].astype(str)
        filled_parts = []
        for _, grp in sig.groupby(session_key):
            s = grp.replace(0, np.nan).ffill().fillna(0).astype(int)
            filled_parts.append(s)
        sig = pd.concat(filled_parts).sort_index()
        sig = sig.where(in_session, 0).astype(int)
        return StrategyResult(signal=sig, intraday=True)
