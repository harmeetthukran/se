"""Risk-management configuration for the backtest engine.

Three layers:

1. Per-trade stops and targets (ATR-based).
2. Volatility-targeted position sizing — qty derived from capital, stop
   distance, and a per-trade risk budget (default 1%% of capital).
3. Max-drawdown circuit-breaker — halts new entries once running drawdown
   exceeds the configured threshold; resumes after drawdown recovers to
   half the threshold (hysteresis prevents flip-flop).

Set `enabled=False` to reproduce the old behavior (no stops, `size_pct`
sizing, no DD halt).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    enabled: bool = True

    atr_period: int = 14
    stop_atr_mult: float = 2.0
    target_atr_mult: float = 3.0

    risk_per_trade_pct: float = 0.01      # 1%% of capital risked per trade
    max_position_pct: float = 1.0         # cap on notional/capital (for MIS this can exceed 1)

    max_drawdown_pct: float = 0.15        # halt when DD <= -15%%
    dd_resume_pct: float = 0.075          # resume when DD recovers above -7.5%%


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def size_from_risk(
    capital: float,
    entry_px: float,
    stop_px: float,
    risk_pct: float,
    max_position_pct: float,
) -> float:
    """Qty such that loss at stop == capital * risk_pct, capped by max-position notional."""
    if entry_px <= 0:
        return 0.0
    risk_per_share = abs(entry_px - stop_px)
    if risk_per_share <= 0:
        return 0.0
    risk_qty = (capital * risk_pct) / risk_per_share
    notional_cap_qty = (capital * max_position_pct) / entry_px
    return float(max(0.0, min(risk_qty, notional_cap_qty)))
