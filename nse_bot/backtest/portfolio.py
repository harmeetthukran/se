"""Portfolio-level backtest: one strategy across many symbols, shared capital.

Loop:
    1. Generate per-symbol signals independently (same strategy + params).
    2. Build a unified timeline (union of all symbol bar times).
    3. At each timestamp, check each open position for exit (signal flipped
       or opposite signal appeared). Close and release capital.
    4. For each active candidate entry signal at this timestamp, if we have
       room (open_positions < max_positions), allocate capital per the
       chosen rule (equal-weight of the cap, or vol-targeted risk).
    5. Sector cap: optionally limit positions per sector.

Limitations (honest):
    * Signals are generated per-symbol in isolation; no cross-sectional
      ranking (first-come-first-served at each timestamp).
    * No margin/SPAN modelling — futures/options left as future work.
    * No intrabar stop/target — exits happen at bar close on signal flip.
      (The per-symbol `engine.run` applies stops/targets; the portfolio
      layer consumes the signals as-is. If you want portfolio-level stops,
      we can route individual trades through engine.run piecewise.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from nse_bot.backtest.costs import CostConfig, round_trip_cost
from nse_bot.backtest.risk import RiskConfig, atr, size_from_risk

Allocation = Literal["equal", "risk"]


@dataclass
class PortfolioConfig:
    initial_capital: float = 100_000.0
    max_positions: int = 5
    allocation: Allocation = "equal"
    cost: CostConfig = field(default_factory=lambda: CostConfig(segment="equity_delivery"))
    risk: RiskConfig = field(default_factory=RiskConfig)
    sector_cap: int | None = None  # max concurrent positions per sector


@dataclass
class _Position:
    symbol: str
    side: int  # +1 long, -1 short
    qty: float
    entry_ts: pd.Timestamp
    entry_px: float
    sector: str = ""


def run_portfolio(
    per_symbol: dict[str, tuple[pd.DataFrame, pd.Series]],
    cfg: PortfolioConfig,
    sectors: dict[str, str] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run a portfolio backtest.

    Args:
        per_symbol: {symbol: (ohlcv_df, signal_series)}. df and signal must
                    be aligned (same length) per symbol.
        cfg:        Portfolio config.
        sectors:    Optional {symbol: sector_name} for the sector cap.

    Returns:
        equity_curve: pd.Series indexed by ts.
        trades:       trade log (entry/exit, PnL, cost).
    """
    sectors = sectors or {}
    if not per_symbol:
        return pd.Series(dtype=float), pd.DataFrame()

    # Build the unified timeline and precompute per-symbol lookup tables.
    frames = []
    for sym, (df, sig) in per_symbol.items():
        if len(df) != len(sig):
            raise ValueError(f"{sym}: signal/df length mismatch")
        if df.empty:
            continue
        f = df[["ts", "open", "high", "low", "close"]].copy().reset_index(drop=True)
        f["symbol"] = sym
        f["signal"] = sig.reset_index(drop=True).fillna(0).astype(int).clip(-1, 1)
        f["pos"] = f["signal"].shift(1).fillna(0).astype(int)
        f["atr"] = atr(f, cfg.risk.atr_period).bfill().ffill()
        frames.append(f)
    if not frames:
        return pd.Series(dtype=float), pd.DataFrame()

    all_bars = pd.concat(frames, ignore_index=True)
    timeline = sorted(all_bars["ts"].unique())

    by_ts_sym = all_bars.set_index(["ts", "symbol"])

    capital = cfg.initial_capital
    positions: dict[str, _Position] = {}
    trades: list[dict] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []

    for t in timeline:
        try:
            slice_now = by_ts_sym.loc[t]
        except KeyError:
            continue
        if isinstance(slice_now, pd.Series):
            slice_now = slice_now.to_frame().T

        # 1) Close positions whose signal flipped this bar.
        for sym in list(positions.keys()):
            if sym not in slice_now.index:
                continue
            row = slice_now.loc[sym]
            desired = int(row["pos"])
            if desired != positions[sym].side:
                pos = positions.pop(sym)
                exit_px = float(row["close"])
                gross = pos.side * (exit_px - pos.entry_px) * pos.qty
                buy_val = pos.entry_px * pos.qty if pos.side > 0 else exit_px * pos.qty
                sell_val = exit_px * pos.qty if pos.side > 0 else pos.entry_px * pos.qty
                cost = round_trip_cost(buy_val, sell_val, cfg.cost)
                net = gross - cost
                capital += net
                trades.append(
                    {
                        "symbol": sym,
                        "side": "LONG" if pos.side > 0 else "SHORT",
                        "entry_ts": pos.entry_ts,
                        "exit_ts": t,
                        "entry_px": pos.entry_px,
                        "exit_px": exit_px,
                        "qty": pos.qty,
                        "gross_inr": gross,
                        "cost_inr": cost,
                        "pnl_inr": net,
                        "return_pct": (exit_px / pos.entry_px - 1.0) * 100.0 * pos.side,
                        "exit_reason": "signal",
                        "sector": pos.sector,
                    }
                )

        # 2) Sector counters.
        sector_count: dict[str, int] = {}
        for pos in positions.values():
            sector_count[pos.sector] = sector_count.get(pos.sector, 0) + 1

        # 3) Try to open new positions. Candidates: symbols whose pos != 0 and we don't hold.
        candidates = []
        for sym, row in slice_now.iterrows():
            if sym in positions:
                continue
            desired = int(row["pos"])
            if desired == 0:
                continue
            candidates.append((sym, row, desired))

        for sym, row, desired in candidates:
            if len(positions) >= cfg.max_positions:
                break
            sector = sectors.get(sym, "")
            if cfg.sector_cap is not None and sector_count.get(sector, 0) >= cfg.sector_cap:
                continue
            entry_px = float(row["close"])
            atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
            slots_left = cfg.max_positions - len(positions)
            capital_per_slot = max(0.0, capital / max(1, slots_left))

            if cfg.allocation == "risk" and cfg.risk.enabled and atr_val > 0:
                stop_px = entry_px - desired * cfg.risk.stop_atr_mult * atr_val
                qty = size_from_risk(
                    capital_per_slot,
                    entry_px,
                    stop_px,
                    cfg.risk.risk_per_trade_pct,
                    cfg.risk.max_position_pct,
                )
            else:
                qty = capital_per_slot / entry_px if entry_px > 0 else 0.0

            if qty <= 0:
                continue
            positions[sym] = _Position(
                symbol=sym,
                side=desired,
                qty=qty,
                entry_ts=t,
                entry_px=entry_px,
                sector=sector,
            )
            sector_count[sector] = sector_count.get(sector, 0) + 1

        # 4) Mark-to-market portfolio equity at this timestamp.
        unreal = 0.0
        for pos in positions.values():
            if pos.symbol in slice_now.index:
                px = float(slice_now.loc[pos.symbol, "close"])
                unreal += pos.side * (px - pos.entry_px) * pos.qty
        equity_rows.append((t, capital + unreal))

    # 5) Force-close anything still open at the final bar.
    if positions and timeline:
        t_final = timeline[-1]
        try:
            slice_final = by_ts_sym.loc[t_final]
        except KeyError:
            slice_final = None
        for sym in list(positions.keys()):
            pos = positions.pop(sym)
            if slice_final is None:
                continue
            row = slice_final.loc[sym] if sym in slice_final.index else None
            if row is None:
                continue
            exit_px = float(row["close"])
            gross = pos.side * (exit_px - pos.entry_px) * pos.qty
            buy_val = pos.entry_px * pos.qty if pos.side > 0 else exit_px * pos.qty
            sell_val = exit_px * pos.qty if pos.side > 0 else pos.entry_px * pos.qty
            cost = round_trip_cost(buy_val, sell_val, cfg.cost)
            net = gross - cost
            capital += net
            trades.append(
                {
                    "symbol": sym,
                    "side": "LONG" if pos.side > 0 else "SHORT",
                    "entry_ts": pos.entry_ts,
                    "exit_ts": t_final,
                    "entry_px": pos.entry_px,
                    "exit_px": exit_px,
                    "qty": pos.qty,
                    "gross_inr": gross,
                    "cost_inr": cost,
                    "pnl_inr": net,
                    "return_pct": (exit_px / pos.entry_px - 1.0) * 100.0 * pos.side,
                    "exit_reason": "eod",
                    "sector": pos.sector,
                }
            )

    eq = pd.Series(
        [v for _, v in equity_rows],
        index=pd.DatetimeIndex([t for t, _ in equity_rows]),
        name="equity",
    )
    tr = pd.DataFrame(trades)
    return eq, tr
