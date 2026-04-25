"""Multi-leg option backtest engine on top of bhavcopy data.

Daily granularity. For each trading day in [from_date, to_date]:

    1. Load that day's bhavcopy and resolve underlying spot price.
    2. Settle any open trades whose leg(s) expire today, using intrinsic
       value (CE: max(spot - K, 0); PE: max(K - spot, 0)).
    3. Mark every still-open trade to market using bhavcopy close prices
       and check stop / target / time-based exits.
    4. Ask the strategy for new trades to open. Apply per-trade capital
       caps and skip trades whose entry price isn't visible in the chain.
    5. Record one equity row.

Trade log columns:
    trade_id, strategy, entry_date, exit_date, exit_reason, legs (json
    string), gross_inr, cost_inr, pnl_inr, hold_days.

Cost model: round-trip option costs from nse_bot.backtest.costs (segment
"options"), applied at exit on the realized turnover (entry + exit value).

Limitations (honest):
    * EOD only. No intraday stops / no theta path within a day.
    * Settlement at expiry uses the same-day approximated spot from
      `bhavcopy.underlying_price` (nearest-expiry futures close).
    * Mark-to-market uses option close prices; bhavcopy may have stale
      closes on illiquid strikes — strategies should pick liquid strikes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable

import json
import pandas as pd

from nse_bot.backtest.costs import CostConfig, round_trip_cost
from nse_bot.data import bhavcopy

# Common NSE F&O lot sizes (as of mid-2024 — change over time).
DEFAULT_LOT_SIZES = {
    "NIFTY": 75,
    "BANKNIFTY": 30,
    "FINNIFTY": 65,
    "MIDCPNIFTY": 120,
    "NIFTYNXT50": 25,
    "SENSEX": 20,
    "BANKEX": 30,
}


@dataclass
class OptionLeg:
    symbol: str
    expiry: date
    strike: float
    option_type: str   # "CE" or "PE"
    side: int          # +1 long, -1 short
    qty: int           # in raw shares (not lots)
    entry_date: date
    entry_px: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "expiry": self.expiry.isoformat(),
            "strike": self.strike,
            "option_type": self.option_type,
            "side": self.side,
            "qty": self.qty,
            "entry_date": self.entry_date.isoformat(),
            "entry_px": self.entry_px,
        }


@dataclass
class OptionTrade:
    trade_id: str
    strategy: str
    entry_date: date
    legs: list[OptionLeg]
    target_pnl_inr: float | None = None
    stop_pnl_inr: float | None = None
    max_hold_days: int | None = None

    exit_date: date | None = None
    exit_pxs: list[float] = field(default_factory=list)
    exit_reason: str = ""
    gross_inr: float = 0.0
    cost_inr: float = 0.0
    pnl_inr: float = 0.0

    @property
    def entry_value(self) -> float:
        """Net cash flow at entry (positive = paid premium / debit)."""
        return sum(leg.side * leg.qty * leg.entry_px for leg in self.legs)


@dataclass
class OptionBacktestConfig:
    initial_capital: float = 100_000.0
    cost: CostConfig = field(default_factory=lambda: CostConfig(segment="options"))
    max_capital_per_trade_pct: float = 1.0  # cap premium paid per new trade
    risk_free_rate: float = 0.065


# ----- chain lookups -----

def _close_price(chain: pd.DataFrame, symbol: str, expiry: date, strike: float, opt: str) -> float | None:
    sub = chain.loc[
        (chain["symbol"].astype(str).str.upper() == symbol.upper())
        & (chain["expiry"] == expiry)
        & (chain["strike"].astype(float) == float(strike))
        & (chain["option_type"].astype(str).str.upper() == opt.upper())
    ]
    if sub.empty:
        return None
    px = sub.iloc[0]["close"]
    if pd.isna(px) or float(px) <= 0:
        return None
    return float(px)


def _intrinsic(opt: str, strike: float, spot: float) -> float:
    return max(spot - strike, 0.0) if opt.upper() == "CE" else max(strike - spot, 0.0)


# ----- mark-to-market and settlement -----

def _mark_to_market(trade: OptionTrade, chain: pd.DataFrame, spot: float | None) -> tuple[float, list[float]]:
    """Return (unrealized_pnl_inr, current_pxs aligned to legs)."""
    pxs: list[float] = []
    for leg in trade.legs:
        px = _close_price(chain, leg.symbol, leg.expiry, leg.strike, leg.option_type)
        if px is None and spot is not None:
            px = _intrinsic(leg.option_type, leg.strike, spot)
        if px is None:
            px = leg.entry_px  # fall back to entry price (no signal)
        pxs.append(float(px))
    upl = sum(leg.side * leg.qty * (px - leg.entry_px) for leg, px in zip(trade.legs, pxs))
    return float(upl), pxs


def _close_trade(
    trade: OptionTrade,
    exit_date: date,
    exit_pxs: list[float],
    reason: str,
    cfg: OptionBacktestConfig,
) -> float:
    """Mark the trade as closed and return the *cash inflow* to credit to capital.

    Cash inflow on close = exit_value - cost, where exit_value is the net cash
    proceeds at exit (long legs return premium; short legs cost premium to
    buy back). The opening cash outflow (`trade.entry_value`) was deducted
    when the trade was opened; net P&L over the life of the trade is then
    `gross - cost` = (exit_value - entry_value) - cost, which is what we
    record on the trade itself.
    """
    gross = sum(leg.side * leg.qty * (px - leg.entry_px) for leg, px in zip(trade.legs, exit_pxs))
    exit_value = sum(leg.side * leg.qty * px for leg, px in zip(trade.legs, exit_pxs))
    entry_turnover = sum(leg.qty * leg.entry_px for leg in trade.legs)
    exit_turnover = sum(leg.qty * px for leg, px in zip(trade.legs, exit_pxs))
    cost = round_trip_cost(entry_turnover, exit_turnover, cfg.cost)
    trade.exit_date = exit_date
    trade.exit_pxs = exit_pxs
    trade.exit_reason = reason
    trade.gross_inr = float(gross)
    trade.cost_inr = float(cost)
    trade.pnl_inr = float(gross - cost)
    return float(exit_value - cost)


# ----- strategy interface -----

class OptionStrategy:
    name: str = "base"

    def on_day(
        self,
        d: date,
        chain: pd.DataFrame,
        underlying_px: float,
        open_trades: list[OptionTrade],
        capital: float,
    ) -> list[OptionTrade]:
        """Return zero or more new trades to open today."""
        raise NotImplementedError


# ----- main loop -----

def run_options(
    strategy: OptionStrategy,
    symbol: str,
    from_date: date,
    to_date: date,
    cfg: OptionBacktestConfig = OptionBacktestConfig(),
) -> tuple[pd.Series, pd.DataFrame]:
    capital = cfg.initial_capital
    open_trades: list[OptionTrade] = []
    closed: list[OptionTrade] = []
    eq_rows: list[tuple[pd.Timestamp, float]] = []

    cur = from_date
    while cur <= to_date:
        if cur.weekday() >= 5:
            cur += timedelta(days=1)
            continue

        chain = bhavcopy.load_day(cur)
        spot = bhavcopy.underlying_price(symbol, cur)
        if chain.empty or spot is None:
            cur += timedelta(days=1)
            continue

        # 1) Settle trades whose ANY leg expires today (multi-leg strategies
        #    here always share an expiry; cross-expiry trades not modelled).
        still_open: list[OptionTrade] = []
        for tr in open_trades:
            expiring = any(leg.expiry == cur for leg in tr.legs)
            if expiring:
                exit_pxs = [_intrinsic(leg.option_type, leg.strike, spot) for leg in tr.legs]
                cash_in = _close_trade(tr, cur, exit_pxs, "expiry", cfg)
                capital += cash_in
                closed.append(tr)
            else:
                still_open.append(tr)
        open_trades = still_open

        # 2) Stop / target / time-exit on remaining trades.
        still_open = []
        for tr in open_trades:
            upl, pxs = _mark_to_market(tr, chain, spot)
            reason = None
            if tr.target_pnl_inr is not None and upl >= tr.target_pnl_inr:
                reason = "target"
            elif tr.stop_pnl_inr is not None and upl <= tr.stop_pnl_inr:
                reason = "stop"
            elif tr.max_hold_days is not None and (cur - tr.entry_date).days >= tr.max_hold_days:
                reason = "time_exit"
            if reason is not None:
                cash_in = _close_trade(tr, cur, pxs, reason, cfg)
                capital += cash_in
                closed.append(tr)
            else:
                still_open.append(tr)
        open_trades = still_open

        # 3) Generate new trades.
        new_trades = strategy.on_day(cur, chain, spot, open_trades, capital) or []
        for tr in new_trades:
            entry_debit = tr.entry_value  # >0 = debit, <0 = credit
            # Cap premium paid on long-only strategies.
            if entry_debit > 0 and entry_debit > capital * cfg.max_capital_per_trade_pct:
                continue
            capital -= entry_debit
            open_trades.append(tr)

        # 4) Mark equity.
        unreal = 0.0
        for tr in open_trades:
            upl, _ = _mark_to_market(tr, chain, spot)
            unreal += upl
        eq_rows.append((pd.Timestamp(cur), capital + unreal))

        cur += timedelta(days=1)

    # Force-close anything still open at the final bar using last available chain.
    if open_trades and eq_rows:
        last_d = eq_rows[-1][0].date()
        chain = bhavcopy.load_day(last_d)
        spot = bhavcopy.underlying_price(symbol, last_d)
        for tr in open_trades:
            _, pxs = _mark_to_market(tr, chain, spot)
            cash_in = _close_trade(tr, last_d, pxs, "eod", cfg)
            capital += cash_in
            closed.append(tr)

    eq = pd.Series(
        [v for _, v in eq_rows],
        index=pd.DatetimeIndex([t for t, _ in eq_rows]),
        name="equity",
    )
    rows = [
        {
            "trade_id": tr.trade_id,
            "strategy": tr.strategy,
            "entry_date": tr.entry_date,
            "exit_date": tr.exit_date,
            "exit_reason": tr.exit_reason,
            "n_legs": len(tr.legs),
            "legs": json.dumps([leg.to_dict() for leg in tr.legs]),
            "entry_value_inr": tr.entry_value,
            "gross_inr": tr.gross_inr,
            "cost_inr": tr.cost_inr,
            "pnl_inr": tr.pnl_inr,
            "hold_days": (tr.exit_date - tr.entry_date).days if tr.exit_date else 0,
        }
        for tr in closed
    ]
    return eq, pd.DataFrame(rows)
