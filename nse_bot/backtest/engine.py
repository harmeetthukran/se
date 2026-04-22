"""Vectorized backtest engine.

The engine is deliberately simple: a single-instrument, long/short,
one-position-at-a-time backtester. Strategies emit a `signal` series
(+1 long, -1 short, 0 flat) aligned to the input OHLCV bars; the engine
converts state changes into round-trip trades, applies realistic NSE costs,
and produces a trade log + equity curve.

Intraday strategies should include a force-flat at session close (15:15 IST)
in their own signal logic, OR set `intraday_squareoff=True` to have the
engine handle it.

Caveats (honest):
    * Entry/exit fill at bar close => slight optimism vs real fills. Slippage
      accounts for most of this; on 1-min bars the effect is small.
    * Single position, no partial fills, no pyramiding.
    * No margin/SPAN modelling for F&O — quantity is user-provided.
    * No short-selling restrictions for delivery (don't short-sell CNC).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nse_bot.backtest.costs import CostConfig, round_trip_cost


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 50_000.0
    cost: CostConfig = CostConfig()
    intraday_squareoff: bool = False
    squareoff_hhmm: tuple[int, int] = (15, 15)
    size_pct: float = 1.0  # fraction of capital to deploy per trade (MIS leverage handled externally)


def _apply_squareoff(signal: pd.Series, ts: pd.Series, hhmm: tuple[int, int]) -> pd.Series:
    hh, mm = hhmm
    minutes = ts.dt.hour * 60 + ts.dt.minute
    cutoff = hh * 60 + mm
    return signal.where(minutes < cutoff, 0)


def run(
    df: pd.DataFrame,
    signal: pd.Series,
    cfg: BacktestConfig = BacktestConfig(),
) -> tuple[pd.Series, pd.DataFrame]:
    """Run a backtest on a single instrument.

    Args:
        df:     OHLCV frame with columns ts, open, high, low, close, volume.
        signal: Integer series aligned to df index: +1 long, -1 short, 0 flat.
                The signal at bar i is the *desired* position during bar i+1
                (the engine shifts internally to avoid look-ahead).
        cfg:    Backtest configuration.

    Returns:
        equity: Equity-curve series indexed by ts.
        trades: DataFrame of round-trip trades.
    """
    if df.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    if len(df) != len(signal):
        raise ValueError("signal length must match df length")

    data = df.reset_index(drop=True).copy()
    sig = signal.reset_index(drop=True).fillna(0).astype(int).clip(-1, 1)

    if cfg.intraday_squareoff:
        sig = _apply_squareoff(sig, data["ts"], cfg.squareoff_hhmm)

    pos = sig.shift(1).fillna(0).astype(int)
    prev_pos = pos.shift(1).fillna(0).astype(int)
    close = data["close"].astype(float).values

    equity = np.empty(len(data), dtype=float)
    capital = cfg.initial_capital
    equity[0] = capital

    entry_i: int | None = None
    entry_px: float = 0.0
    entry_pos: int = 0
    qty: float = 0.0

    trades: list[dict] = []
    position_pnl_running = 0.0

    pos_values = pos.values
    for i in range(1, len(data)):
        p_now = int(pos_values[i])
        p_prev = int(pos_values[i - 1])

        if p_prev != 0:
            bar_ret = (close[i] - close[i - 1]) / close[i - 1]
            position_pnl_running += p_prev * bar_ret * (qty * entry_px)

        if p_now != p_prev:
            # Close existing position
            if p_prev != 0 and entry_i is not None:
                exit_px = close[i]
                gross = p_prev * (exit_px - entry_px) * qty
                buy_val = entry_px * qty if p_prev > 0 else exit_px * qty
                sell_val = exit_px * qty if p_prev > 0 else entry_px * qty
                cost = round_trip_cost(buy_val, sell_val, cfg.cost)
                net = gross - cost
                capital += net
                trades.append(
                    {
                        "entry_ts": data["ts"].iloc[entry_i],
                        "exit_ts": data["ts"].iloc[i],
                        "side": "LONG" if p_prev > 0 else "SHORT",
                        "entry_px": entry_px,
                        "exit_px": exit_px,
                        "qty": qty,
                        "gross_inr": gross,
                        "cost_inr": cost,
                        "pnl_inr": net,
                        "return_pct": (exit_px / entry_px - 1.0) * 100.0 * p_prev,
                    }
                )
                position_pnl_running = 0.0
                entry_i = None
                entry_pos = 0
                qty = 0.0
                entry_px = 0.0

            # Open new position
            if p_now != 0:
                entry_i = i
                entry_px = close[i]
                entry_pos = p_now
                qty = (capital * cfg.size_pct) / entry_px if entry_px > 0 else 0.0

        equity[i] = capital + position_pnl_running

    eq = pd.Series(equity, index=data["ts"], name="equity")
    tr = pd.DataFrame(trades)
    return eq, tr
