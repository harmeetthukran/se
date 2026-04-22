"""NSE transaction cost model.

Figures below match a typical discount-broker cost stack (Zerodha/Upstox) and
are applied symmetrically to long and short trades. All values in INR unless
noted. These are conservative — slippage estimate is separate and added on top.

Sources (public brokerage calculators, as of 2024):
    * Brokerage:   min(0.03% of turnover, ₹20) per order — intraday equity
                   min(0.03% of turnover, ₹20) per order — futures
                   flat ₹20 per order                   — options
                   zero for equity delivery (Zerodha); ₹20 for Upstox.
    * STT:         intraday equity: 0.025% on sell side only
                   delivery equity: 0.1% on both sides
                   futures:         0.02% on sell side only
                   options:         0.1% on sell side (premium)
                   option exercise: 0.125% of settlement value (ignored here)
    * Exchange txn NSE: equity 0.00325% / futures 0.00188% / options 0.03503%
    * SEBI:        0.0001% (₹10 per crore) turnover
    * Stamp duty:  equity intraday 0.003% / delivery 0.015% on buy
                   futures 0.002% on buy / options 0.003% on buy
    * GST:         18% on (brokerage + exchange + SEBI)

Slippage is modelled as a percentage of entry/exit price, configurable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Segment = Literal["equity_intraday", "equity_delivery", "futures", "options"]


@dataclass(frozen=True)
class CostConfig:
    segment: Segment = "equity_intraday"
    brokerage_per_order: float = 20.0
    brokerage_pct_cap: float = 0.0003  # 0.03% cap if broker uses %
    slippage_pct: float = 0.0005       # 5 bps per side (entry + exit separately)


def _brokerage(turnover_side: float, cfg: CostConfig) -> float:
    if cfg.segment == "options":
        return cfg.brokerage_per_order
    if cfg.segment == "equity_delivery":
        return 0.0
    return min(cfg.brokerage_per_order, turnover_side * cfg.brokerage_pct_cap)


def round_trip_cost(
    buy_value: float,
    sell_value: float,
    cfg: CostConfig = CostConfig(),
) -> float:
    """Total INR cost for one complete round-trip trade (buy + sell)."""
    turnover = buy_value + sell_value
    brokerage = _brokerage(buy_value, cfg) + _brokerage(sell_value, cfg)

    if cfg.segment == "equity_intraday":
        stt = 0.00025 * sell_value
        exch = 0.0000325 * turnover
        stamp = 0.00003 * buy_value
    elif cfg.segment == "equity_delivery":
        stt = 0.001 * (buy_value + sell_value)
        exch = 0.0000325 * turnover
        stamp = 0.00015 * buy_value
    elif cfg.segment == "futures":
        stt = 0.0002 * sell_value
        exch = 0.0000188 * turnover
        stamp = 0.00002 * buy_value
    else:  # options — note turnover for options = premium, not notional
        stt = 0.001 * sell_value
        exch = 0.0003503 * turnover
        stamp = 0.00003 * buy_value

    sebi = 0.000001 * turnover
    gst = 0.18 * (brokerage + exch + sebi)

    slippage = cfg.slippage_pct * (buy_value + sell_value)
    return brokerage + stt + exch + sebi + stamp + gst + slippage
