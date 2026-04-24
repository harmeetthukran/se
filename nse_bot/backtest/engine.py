"""Vectorized-ish backtest engine with risk management.

Single instrument, long/short, one position at a time. Enhancements over the
naive signal-flip backtester:

    * ATR-based stops and targets (checked against each bar's high/low).
    * Volatility-targeted position sizing.
    * Max-drawdown circuit-breaker with hysteresis.
    * Intraday squareoff at session close.
    * `exit_reason` column on every trade: signal / stop / target / squareoff /
      dd_breaker / eod.

Conservative assumptions:
    * If a bar's range covers both stop and target, we assume the stop hit first.
    * Entries fill at bar close (strategies emit signals after bar close).
    * Slippage is already modelled in the cost layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nse_bot.backtest.costs import CostConfig, round_trip_cost
from nse_bot.backtest.risk import RiskConfig, atr, size_from_risk


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    cost: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    intraday_squareoff: bool = False
    squareoff_hhmm: tuple[int, int] = (15, 15)
    size_pct: float = 1.0  # fallback when risk.enabled=False


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
                The engine shifts internally to avoid look-ahead.

    Returns:
        equity: Equity curve series indexed by ts.
        trades: Round-trip trade log with exit_reason.
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
    close = data["close"].astype(float).values
    high = data["high"].astype(float).values
    low = data["low"].astype(float).values
    ts = data["ts"]

    atr_series = atr(data, cfg.risk.atr_period).bfill().ffill()
    atr_vals = atr_series.values

    # Squareoff detection — bars that forced the position to 0 due to session end.
    squareoff_now = np.zeros(len(data), dtype=bool)
    if cfg.intraday_squareoff:
        hh, mm = cfg.squareoff_hhmm
        minutes = ts.dt.hour * 60 + ts.dt.minute
        squareoff_now = (minutes >= (hh * 60 + mm)).values

    equity = np.empty(len(data), dtype=float)
    capital = cfg.initial_capital
    equity[0] = capital
    peak_equity = capital
    halted = False

    entry_i: int | None = None
    entry_px = 0.0
    qty = 0.0
    initial_qty = 0.0  # for partial-TP bookkeeping
    stop_px = np.nan
    target_px = np.nan
    best_favorable_px = np.nan  # highest high for longs / lowest low for shorts
    partial_done = False
    current_pos = 0
    last_exit_bar = -1  # no re-entry on the same bar as a stop/target/squareoff

    trades: list[dict] = []
    pos_values = pos.values

    def _close(exit_i: int, exit_px_override: float | None, reason: str, partial_qty: float | None = None) -> None:
        nonlocal capital, entry_i, entry_px, qty, current_pos, stop_px, target_px, peak_equity, last_exit_bar, best_favorable_px, partial_done, initial_qty
        exit_px = float(exit_px_override if exit_px_override is not None else close[exit_i])
        close_qty = float(partial_qty) if partial_qty is not None else qty
        if close_qty <= 0:
            return
        gross = current_pos * (exit_px - entry_px) * close_qty
        buy_val = entry_px * close_qty if current_pos > 0 else exit_px * close_qty
        sell_val = exit_px * close_qty if current_pos > 0 else entry_px * close_qty
        cost = round_trip_cost(buy_val, sell_val, cfg.cost)
        net = gross - cost
        capital += net
        peak_equity = max(peak_equity, capital)
        trades.append(
            {
                "entry_ts": ts.iloc[entry_i],
                "exit_ts": ts.iloc[exit_i],
                "side": "LONG" if current_pos > 0 else "SHORT",
                "entry_px": entry_px,
                "exit_px": exit_px,
                "qty": close_qty,
                "gross_inr": gross,
                "cost_inr": cost,
                "pnl_inr": net,
                "return_pct": (exit_px / entry_px - 1.0) * 100.0 * current_pos,
                "exit_reason": reason,
            }
        )
        if partial_qty is not None and close_qty < qty:
            qty -= close_qty
            last_exit_bar = exit_i
            return
        last_exit_bar = exit_i
        entry_i = None
        entry_px = 0.0
        qty = 0.0
        initial_qty = 0.0
        current_pos = 0
        stop_px = np.nan
        target_px = np.nan
        best_favorable_px = np.nan
        partial_done = False

    def _open(i: int, direction: int) -> None:
        nonlocal entry_i, entry_px, qty, initial_qty, current_pos, stop_px, target_px, best_favorable_px, partial_done
        entry_i = i
        entry_px = float(close[i])
        current_pos = direction
        best_favorable_px = entry_px
        partial_done = False
        if cfg.risk.enabled and not np.isnan(atr_vals[i]) and atr_vals[i] > 0:
            stop_px = entry_px - direction * cfg.risk.stop_atr_mult * atr_vals[i]
            target_px = entry_px + direction * cfg.risk.target_atr_mult * atr_vals[i]
            qty = size_from_risk(
                capital,
                entry_px,
                stop_px,
                cfg.risk.risk_per_trade_pct,
                cfg.risk.max_position_pct,
            )
            initial_qty = qty
        else:
            stop_px = np.nan
            target_px = np.nan
            qty = (capital * cfg.size_pct) / entry_px if entry_px > 0 else 0.0

    for i in range(1, len(data)):
        # Update trailing stop / best-favorable price on the existing position.
        if current_pos != 0 and cfg.risk.enabled:
            # Advance best-favorable based on bar's high/low.
            if current_pos > 0:
                best_favorable_px = max(best_favorable_px, high[i])
            else:
                best_favorable_px = min(best_favorable_px, low[i])

            # Partial TP first (at first touch of partial_tp_atr_mult × ATR in favor).
            if (
                not partial_done
                and cfg.risk.partial_tp_atr_mult is not None
                and cfg.risk.partial_tp_ratio is not None
                and not np.isnan(atr_vals[i])
            ):
                partial_level = entry_px + current_pos * cfg.risk.partial_tp_atr_mult * atr_vals[i]
                reached = (current_pos > 0 and high[i] >= partial_level) or (current_pos < 0 and low[i] <= partial_level)
                if reached and initial_qty > 0:
                    partial_qty = initial_qty * cfg.risk.partial_tp_ratio
                    _close(i, partial_level, "partial_tp", partial_qty=min(partial_qty, qty))
                    partial_done = True
                    if cfg.risk.move_stop_to_breakeven_after_partial and current_pos != 0:
                        stop_px = entry_px

            # Trailing stop supersedes the static stop once engaged.
            if cfg.risk.trailing_stop_atr_mult is not None and not np.isnan(atr_vals[i]) and atr_vals[i] > 0:
                trail = best_favorable_px - current_pos * cfg.risk.trailing_stop_atr_mult * atr_vals[i]
                if current_pos > 0:
                    stop_px = max(stop_px, trail) if not np.isnan(stop_px) else trail
                else:
                    stop_px = min(stop_px, trail) if not np.isnan(stop_px) else trail

            # Static stop / target check.
            if current_pos != 0 and not np.isnan(stop_px):
                hit_stop = (current_pos > 0 and low[i] <= stop_px) or (current_pos < 0 and high[i] >= stop_px)
                if hit_stop:
                    _close(i, stop_px, "stop")

            if current_pos != 0 and not np.isnan(target_px):
                hit_target = (current_pos > 0 and high[i] >= target_px) or (current_pos < 0 and low[i] <= target_px)
                if hit_target:
                    _close(i, target_px, "target")

            # Time-based exit.
            if (
                current_pos != 0
                and cfg.risk.max_bars_in_trade is not None
                and entry_i is not None
                and (i - entry_i) >= cfg.risk.max_bars_in_trade
            ):
                _close(i, None, "time_exit")

        p_now = int(pos_values[i])

        if current_pos != 0 and p_now != current_pos:
            reason = "squareoff" if (cfg.intraday_squareoff and squareoff_now[i]) else "signal"
            _close(i, None, reason)

        # Mark-to-market equity after any close.
        if current_pos != 0:
            unreal = current_pos * (close[i] - entry_px) * qty
            equity[i] = capital + unreal
        else:
            equity[i] = capital

        # DD circuit-breaker
        peak_equity = max(peak_equity, equity[i])
        dd = (equity[i] - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        if cfg.risk.enabled and not halted and dd <= -cfg.risk.max_drawdown_pct:
            halted = True
            if current_pos != 0:
                _close(i, None, "dd_breaker")
                equity[i] = capital
        elif halted and dd >= -cfg.risk.dd_resume_pct:
            halted = False

        # New entry this bar — respect halt and the same-bar cooldown.
        if current_pos == 0 and p_now != 0 and not halted and i != last_exit_bar:
            _open(i, 1 if p_now > 0 else -1)

    # Force-close anything still open at the final bar.
    if current_pos != 0:
        _close(len(data) - 1, None, "eod")
        equity[-1] = capital

    eq = pd.Series(equity, index=ts, name="equity")
    tr = pd.DataFrame(trades)
    return eq, tr
