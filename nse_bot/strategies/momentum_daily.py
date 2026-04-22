"""Daily momentum — swing / positional (CNC delivery).

Rules:
    * Long when 20-day EMA crosses above 50-day EMA AND close is above 200-day SMA.
    * Exit when 20-EMA crosses back below 50-EMA OR close falls below 200-SMA.
    * No shorts (cash-market delivery can't short).

Classic long-only trend-following. Works in bull markets; suffers in sideways
regimes. Useful baseline against which intraday strategies must justify costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nse_bot.strategies.base import Strategy, StrategyResult


class DailyMomentumStrategy(Strategy):
    name = "momentum_daily"
    interval = "day"

    def __init__(self, fast: int = 20, slow: int = 50, trend: int = 200) -> None:
        self.fast = fast
        self.slow = slow
        self.trend = trend

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        if df.empty:
            return StrategyResult(signal=pd.Series(dtype=int), intraday=False)

        close = df["close"].astype(float)
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        sma_trend = close.rolling(self.trend).mean()

        long_cond = (ema_fast > ema_slow) & (close > sma_trend)
        sig = pd.Series(np.where(long_cond, 1, 0), index=df.index, dtype=int)
        return StrategyResult(signal=sig, intraday=False)
