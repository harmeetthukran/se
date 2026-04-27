"""Option-strategy implementations on top of the multi-leg engine.

All strategies pick a strike using the live option chain (ATM = strike
nearest the current spot). Lot sizes default to NIFTY's 75; pass
`lot_size=` for other underlyings.

Strategies included:
    LongStraddle      — long ATM CE + long ATM PE (debit, profit on big moves)
    LongCall          — long ATM/OTM CE (directional bullish)
    LongPut           — long ATM/OTM PE (directional bearish)
    BullCallSpread    — long CE at K1, short CE at K2 (K2 > K1) — capped debit
    BearPutSpread     — long PE at K1, short PE at K2 (K2 < K1) — capped debit

Each strategy emits at most one new trade per `entry_freq_days` and only
when no trade for that strategy is currently open. Entries pick the
nearest weekly expiry that is at least `min_dte_days` away.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from nse_bot.options.engine import (
    DEFAULT_LOT_SIZES,
    OptionLeg,
    OptionStrategy,
    OptionTrade,
)


def _round_to_strike(spot: float, strikes: list[float]) -> float | None:
    if not strikes:
        return None
    return min(strikes, key=lambda k: abs(k - spot))


def _strikes_for(chain: pd.DataFrame, expiry: date, opt: str) -> list[float]:
    sub = chain.loc[
        (chain["expiry"] == expiry)
        & (chain["option_type"].astype(str).str.upper() == opt.upper())
    ]
    return sorted(float(s) for s in sub["strike"].dropna().unique())


def _entry_px(chain: pd.DataFrame, symbol: str, expiry: date, strike: float, opt: str) -> float | None:
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


def _select_expiry(chain: pd.DataFrame, symbol: str, as_of: date, min_dte: int, max_dte: int) -> date | None:
    sub = chain.loc[
        (chain["symbol"].astype(str).str.upper() == symbol.upper())
        & chain["instrument_type"].astype(str).str.upper().isin(["OPTIDX", "OPTSTK", "STO", "IDO"])
        & (chain["expiry"] > as_of)
    ]
    if sub.empty:
        return None
    expiries = sorted({d for d in sub["expiry"].dropna().unique()})
    for e in expiries:
        dte = (e - as_of).days
        if min_dte <= dte <= max_dte:
            return e
    return None


@dataclass
class _BaseConfig:
    symbol: str = "NIFTY"
    lot_size: int = 0
    lots: int = 1
    min_dte_days: int = 5
    max_dte_days: int = 14
    entry_freq_days: int = 7
    target_pnl_pct_of_premium: float | None = 0.5  # e.g. +50% return
    stop_pnl_pct_of_premium: float | None = -0.5   # e.g. -50% loss

    # IV-rank gate. If both are None, no gating.
    # For DEBIT strategies (long straddle/call/put/spreads): enter only when
    #   IV rank <= max_iv_rank (vol is cheap). Defaults None = no gate.
    # For CREDIT strategies (not yet implemented): use min_iv_rank.
    max_iv_rank: float | None = None
    min_iv_rank: float | None = None
    iv_lookback_days: int = 180

    def lot_qty(self) -> int:
        if self.lot_size > 0:
            return self.lot_size * self.lots
        return DEFAULT_LOT_SIZES.get(self.symbol.upper(), 1) * self.lots


class _SingleEntryPerWindow(OptionStrategy):
    """Mixin: throttle entries to at most one per `entry_freq_days`,
    and optionally gate by IV rank (only enter when ATM IV is cheap)."""

    def __init__(self) -> None:
        self._last_entry: date | None = None
        self._iv_history: pd.Series | None = None

    def _can_enter(self, d: date, freq_days: int, has_open_for_strategy: bool) -> bool:
        if has_open_for_strategy:
            return False
        if self._last_entry is None:
            return True
        return (d - self._last_entry).days >= freq_days

    def _passes_iv_gate(
        self,
        d: date,
        symbol: str,
        cfg: "_BaseConfig",
    ) -> bool:
        """Lazy-build IV history, then check current IV rank against cfg gates.

        Returns True when no gate is configured. Returns False if IV history
        can't be built (silent skip — better than entering on missing data).
        """
        if cfg.max_iv_rank is None and cfg.min_iv_rank is None:
            return True
        # Build history once, lazily.
        if self._iv_history is None:
            from datetime import timedelta as _td
            from nse_bot.options.iv import iv_history as _iv_history
            start = d - _td(days=cfg.iv_lookback_days * 2 + 30)
            self._iv_history = _iv_history(symbol, start, d)
        if self._iv_history is None or self._iv_history.empty:
            return False
        # Refresh today's IV if we have stale history.
        if d not in self._iv_history.index:
            from nse_bot.options.iv import atm_iv_for_day as _atm_iv
            v = _atm_iv(symbol, d)
            if v is None:
                return False
            self._iv_history.loc[d] = v
            self._iv_history = self._iv_history.sort_index()
        current = float(self._iv_history.loc[d])
        from nse_bot.options.iv import iv_rank as _rank
        rank = _rank(self._iv_history.loc[:d], current, lookback=cfg.iv_lookback_days)
        if pd.isna(rank):
            return False
        if cfg.max_iv_rank is not None and rank > cfg.max_iv_rank:
            return False
        if cfg.min_iv_rank is not None and rank < cfg.min_iv_rank:
            return False
        return True


def _existing_for(strategy_name: str, open_trades: list[OptionTrade]) -> bool:
    return any(t.strategy == strategy_name for t in open_trades)


def _trade_id(strategy: str, d: date) -> str:
    return f"{strategy}-{d.isoformat()}"


class LongStraddle(_SingleEntryPerWindow):
    """Long ATM CE + long ATM PE on the nearest weekly expiry."""

    name = "long_straddle"

    def __init__(self, cfg: _BaseConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or _BaseConfig()

    def on_day(self, d, chain, spot, open_trades, capital):
        if not self._can_enter(d, self.cfg.entry_freq_days, _existing_for(self.name, open_trades)):
            return []
        if not self._passes_iv_gate(d, self.cfg.symbol, self.cfg):
            return []
        expiry = _select_expiry(chain, self.cfg.symbol, d, self.cfg.min_dte_days, self.cfg.max_dte_days)
        if expiry is None:
            return []
        ce_strikes = _strikes_for(chain, expiry, "CE")
        pe_strikes = _strikes_for(chain, expiry, "PE")
        atm = _round_to_strike(spot, sorted(set(ce_strikes) & set(pe_strikes)))
        if atm is None:
            return []
        ce_px = _entry_px(chain, self.cfg.symbol, expiry, atm, "CE")
        pe_px = _entry_px(chain, self.cfg.symbol, expiry, atm, "PE")
        if ce_px is None or pe_px is None:
            return []
        qty = self.cfg.lot_qty()
        legs = [
            OptionLeg(self.cfg.symbol, expiry, atm, "CE", +1, qty, d, ce_px),
            OptionLeg(self.cfg.symbol, expiry, atm, "PE", +1, qty, d, pe_px),
        ]
        premium = sum(l.qty * l.entry_px for l in legs)
        target = self.cfg.target_pnl_pct_of_premium * premium if self.cfg.target_pnl_pct_of_premium else None
        stop = self.cfg.stop_pnl_pct_of_premium * premium if self.cfg.stop_pnl_pct_of_premium else None
        self._last_entry = d
        return [OptionTrade(_trade_id(self.name, d), self.name, d, legs, target, stop, max_hold_days=self.cfg.max_dte_days)]


class _SingleLeg(_SingleEntryPerWindow):
    """Helper for single-leg long strategies."""

    name: str = "single_leg"

    def __init__(self, cfg: _BaseConfig | None = None, opt: str = "CE", strike_offset_pct: float = 0.0) -> None:
        super().__init__()
        self.cfg = cfg or _BaseConfig()
        self.opt = opt.upper()
        self.strike_offset_pct = strike_offset_pct  # +0.01 => 1% OTM for CE, -1% for PE

    def on_day(self, d, chain, spot, open_trades, capital):
        if not self._can_enter(d, self.cfg.entry_freq_days, _existing_for(self.name, open_trades)):
            return []
        if not self._passes_iv_gate(d, self.cfg.symbol, self.cfg):
            return []
        expiry = _select_expiry(chain, self.cfg.symbol, d, self.cfg.min_dte_days, self.cfg.max_dte_days)
        if expiry is None:
            return []
        target_strike = spot * (1.0 + self.strike_offset_pct)
        strikes = _strikes_for(chain, expiry, self.opt)
        k = _round_to_strike(target_strike, strikes)
        if k is None:
            return []
        px = _entry_px(chain, self.cfg.symbol, expiry, k, self.opt)
        if px is None:
            return []
        qty = self.cfg.lot_qty()
        legs = [OptionLeg(self.cfg.symbol, expiry, k, self.opt, +1, qty, d, px)]
        premium = qty * px
        target = self.cfg.target_pnl_pct_of_premium * premium if self.cfg.target_pnl_pct_of_premium else None
        stop = self.cfg.stop_pnl_pct_of_premium * premium if self.cfg.stop_pnl_pct_of_premium else None
        self._last_entry = d
        return [OptionTrade(_trade_id(self.name, d), self.name, d, legs, target, stop, max_hold_days=self.cfg.max_dte_days)]


class LongCall(_SingleLeg):
    name = "long_call"

    def __init__(self, cfg: _BaseConfig | None = None, strike_offset_pct: float = 0.0) -> None:
        super().__init__(cfg=cfg, opt="CE", strike_offset_pct=strike_offset_pct)


class LongPut(_SingleLeg):
    name = "long_put"

    def __init__(self, cfg: _BaseConfig | None = None, strike_offset_pct: float = 0.0) -> None:
        super().__init__(cfg=cfg, opt="PE", strike_offset_pct=-abs(strike_offset_pct))


class _Spread(_SingleEntryPerWindow):
    name: str = "spread"

    def __init__(self, cfg: _BaseConfig | None = None, opt: str = "CE", width_pct: float = 0.01) -> None:
        super().__init__()
        self.cfg = cfg or _BaseConfig()
        self.opt = opt.upper()
        self.width_pct = abs(width_pct)

    def on_day(self, d, chain, spot, open_trades, capital):
        if not self._can_enter(d, self.cfg.entry_freq_days, _existing_for(self.name, open_trades)):
            return []
        if not self._passes_iv_gate(d, self.cfg.symbol, self.cfg):
            return []
        expiry = _select_expiry(chain, self.cfg.symbol, d, self.cfg.min_dte_days, self.cfg.max_dte_days)
        if expiry is None:
            return []
        strikes = _strikes_for(chain, expiry, self.opt)
        atm = _round_to_strike(spot, strikes)
        if atm is None:
            return []
        if self.opt == "CE":
            target_far = spot * (1.0 + self.width_pct)
            far = _round_to_strike(target_far, [s for s in strikes if s > atm])
        else:
            target_far = spot * (1.0 - self.width_pct)
            far = _round_to_strike(target_far, [s for s in strikes if s < atm])
        if far is None or far == atm:
            return []
        near_px = _entry_px(chain, self.cfg.symbol, expiry, atm, self.opt)
        far_px = _entry_px(chain, self.cfg.symbol, expiry, far, self.opt)
        if near_px is None or far_px is None:
            return []
        qty = self.cfg.lot_qty()
        legs = [
            OptionLeg(self.cfg.symbol, expiry, atm, self.opt, +1, qty, d, near_px),
            OptionLeg(self.cfg.symbol, expiry, far, self.opt, -1, qty, d, far_px),
        ]
        debit = qty * (near_px - far_px)
        if debit <= 0:
            return []
        target = self.cfg.target_pnl_pct_of_premium * debit if self.cfg.target_pnl_pct_of_premium else None
        stop = self.cfg.stop_pnl_pct_of_premium * debit if self.cfg.stop_pnl_pct_of_premium else None
        self._last_entry = d
        return [OptionTrade(_trade_id(self.name, d), self.name, d, legs, target, stop, max_hold_days=self.cfg.max_dte_days)]


class BullCallSpread(_Spread):
    name = "bull_call_spread"

    def __init__(self, cfg: _BaseConfig | None = None, width_pct: float = 0.01) -> None:
        super().__init__(cfg=cfg, opt="CE", width_pct=width_pct)


class BearPutSpread(_Spread):
    name = "bear_put_spread"

    def __init__(self, cfg: _BaseConfig | None = None, width_pct: float = 0.01) -> None:
        super().__init__(cfg=cfg, opt="PE", width_pct=width_pct)


REGISTRY = {
    "long_straddle": LongStraddle,
    "long_call": LongCall,
    "long_put": LongPut,
    "bull_call_spread": BullCallSpread,
    "bear_put_spread": BearPutSpread,
}
