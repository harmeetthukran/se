"""Paper-trading runner — NO REAL ORDERS.

Reads today's signals from cached data, simulates order placement at
the most recent close, tracks positions in a local parquet file, and
updates P&L each run.

Deliberately does *not* call any Upstox order-placement endpoint. When
you're confident enough to go live, the natural next step is to add a
broker adapter that mirrors this module's interface.

State files (under data/paper/):
    positions.parquet  — open positions (one row each)
    orders.parquet     — append-only log of every simulated order
    equity.parquet     — one row per run: {ts, cash, notional, equity}
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from nse_bot.backtest.risk import RiskConfig, atr, size_from_risk
from nse_bot.config import DATA_DIR

PAPER_DIR = DATA_DIR / "paper"
POSITIONS_PATH = PAPER_DIR / "positions.parquet"
ORDERS_PATH = PAPER_DIR / "orders.parquet"
EQUITY_PATH = PAPER_DIR / "equity.parquet"
IST = "Asia/Kolkata"

_POS_COLS = ["symbol", "side", "qty", "entry_ts", "entry_px", "stop_px", "target_px", "strategy"]
_ORD_COLS = ["ts", "symbol", "side", "action", "qty", "px", "reason", "strategy"]
_EQ_COLS = ["ts", "cash", "notional", "equity", "open_positions"]


@dataclass
class PaperConfig:
    capital: float = 100_000.0
    max_positions: int = 5


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_parquet(path)


def _write(path: Path, df: pd.DataFrame) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def current_positions() -> pd.DataFrame:
    return _read(POSITIONS_PATH, _POS_COLS)


def orders_log() -> pd.DataFrame:
    return _read(ORDERS_PATH, _ORD_COLS)


def equity_history() -> pd.DataFrame:
    return _read(EQUITY_PATH, _EQ_COLS)


def _append_order(row: dict) -> None:
    df = _read(ORDERS_PATH, _ORD_COLS)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _write(ORDERS_PATH, df)


def _current_cash(cfg: PaperConfig) -> float:
    """Cash = initial capital + realized PnL (sum of CLOSE order PnL)."""
    orders = orders_log()
    if orders.empty:
        return cfg.capital
    closes = orders.loc[orders["action"].astype(str).str.upper() == "CLOSE"]
    # Realised PnL is captured in the 'reason' payload — we store pnl_inr on close.
    realised = 0.0
    if "pnl_inr" in orders.columns:
        realised = float(orders.loc[orders["action"].astype(str).str.upper() == "CLOSE", "pnl_inr"].fillna(0).sum())
    return float(cfg.capital + realised)


def _close_position(
    pos: pd.Series,
    exit_px: float,
    ts: pd.Timestamp,
    reason: str,
) -> float:
    qty = float(pos["qty"])
    side = int(pos["side"])
    entry_px = float(pos["entry_px"])
    gross = side * (exit_px - entry_px) * qty
    _append_order(
        {
            "ts": ts,
            "symbol": str(pos["symbol"]),
            "side": "LONG" if side > 0 else "SHORT",
            "action": "CLOSE",
            "qty": qty,
            "px": exit_px,
            "reason": reason,
            "strategy": str(pos["strategy"]),
            "pnl_inr": float(gross),
        }
    )
    return gross


def _open_position(
    symbol: str,
    side: int,
    qty: float,
    entry_px: float,
    stop_px: float,
    target_px: float,
    strategy: str,
    ts: pd.Timestamp,
) -> dict:
    _append_order(
        {
            "ts": ts,
            "symbol": symbol,
            "side": "LONG" if side > 0 else "SHORT",
            "action": "OPEN",
            "qty": qty,
            "px": entry_px,
            "reason": "signal",
            "strategy": strategy,
            "pnl_inr": 0.0,
        }
    )
    return {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry_ts": ts,
        "entry_px": entry_px,
        "stop_px": stop_px,
        "target_px": target_px,
        "strategy": strategy,
    }


def run_once(
    signals: pd.DataFrame,
    latest_bars: pd.DataFrame,
    cfg: PaperConfig = PaperConfig(),
    risk: RiskConfig = RiskConfig(),
) -> dict:
    """Reconcile signals with state and (paper-)place orders.

    Args:
        signals: frame with {symbol, desired_pos, strategy}. desired_pos in
                 {-1, 0, +1}. One row per symbol currently evaluated.
        latest_bars: OHLCV bars used for (a) the latest price for fills,
                     (b) ATR-based stops. Required columns per symbol:
                     ts, open, high, low, close. Each symbol's last row
                     is treated as "now".

    Returns:
        dict summary: opens, closes, skipped, current positions count.
    """
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=IST)

    positions = current_positions()
    pos_by_sym = {r["symbol"]: r for _, r in positions.iterrows()}

    opens, closes, skipped = 0, 0, 0

    # 1) Close positions whose signal flipped or went to 0.
    for sym in list(pos_by_sym.keys()):
        row = signals.loc[signals["symbol"] == sym]
        if row.empty:
            continue
        desired = int(row.iloc[0]["desired_pos"])
        held_side = int(pos_by_sym[sym]["side"])
        if desired != held_side:
            bar = latest_bars.loc[latest_bars["symbol"] == sym]
            if bar.empty:
                continue
            exit_px = float(bar["close"].iloc[-1])
            _close_position(pos_by_sym[sym], exit_px, now, reason="signal_flip")
            del pos_by_sym[sym]
            closes += 1

    # 2) Open new positions for desired-nonzero symbols we don't hold, respecting cap.
    desired_open = signals.loc[signals["desired_pos"].astype(int) != 0].copy()
    for _, row in desired_open.iterrows():
        sym = row["symbol"]
        if sym in pos_by_sym:
            continue
        if len(pos_by_sym) >= cfg.max_positions:
            skipped += 1
            continue
        bar_sub = latest_bars.loc[latest_bars["symbol"] == sym]
        if bar_sub.empty:
            continue
        bar_df = bar_sub.sort_values("ts").reset_index(drop=True)
        entry_px = float(bar_df["close"].iloc[-1])
        atr_series = atr(bar_df, risk.atr_period)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        side = int(row["desired_pos"])
        if risk.enabled and atr_val > 0:
            stop_px = entry_px - side * risk.stop_atr_mult * atr_val
            target_px = entry_px + side * risk.target_atr_mult * atr_val
            cash = _current_cash(cfg)
            qty = size_from_risk(cash, entry_px, stop_px, risk.risk_per_trade_pct, risk.max_position_pct)
        else:
            stop_px = 0.0
            target_px = 0.0
            qty = _current_cash(cfg) / max(1, cfg.max_positions) / max(entry_px, 1e-6)

        if qty <= 0:
            skipped += 1
            continue
        pos_by_sym[sym] = _open_position(
            sym, side, qty, entry_px, stop_px, target_px, str(row.get("strategy", "")), now,
        )
        opens += 1

    # Persist positions.
    if pos_by_sym:
        new_positions = pd.DataFrame(list(pos_by_sym.values()))
    else:
        new_positions = pd.DataFrame(columns=_POS_COLS)
    _write(POSITIONS_PATH, new_positions)

    # 3) Mark to market and log equity row.
    cash = _current_cash(cfg)
    notional = 0.0
    if not new_positions.empty:
        for _, p in new_positions.iterrows():
            bar = latest_bars.loc[latest_bars["symbol"] == p["symbol"]]
            if bar.empty:
                continue
            px = float(bar["close"].iloc[-1])
            notional += int(p["side"]) * (px - float(p["entry_px"])) * float(p["qty"])

    equity = cash + notional
    eq_hist = _read(EQUITY_PATH, _EQ_COLS)
    eq_hist = pd.concat(
        [eq_hist, pd.DataFrame([{"ts": now, "cash": cash, "notional": notional,
                                 "equity": equity, "open_positions": len(new_positions)}])],
        ignore_index=True,
    )
    _write(EQUITY_PATH, eq_hist)

    return {
        "opens": opens,
        "closes": closes,
        "skipped": skipped,
        "open_positions": len(new_positions),
        "cash": cash,
        "notional": notional,
        "equity": equity,
    }
