"""Market-neutral pair trade backtester.

For a pair (A, B) of correlated stocks:
    1. Fit hedge ratio β by OLS regression of log(P_A) on log(P_B) over the
       lookback window.
    2. Compute spread S = log(P_A) − β · log(P_B) and its rolling z-score.
    3. Enter LONG-spread (long A, short B) when z ≤ -entry_z (cheap A, rich B).
       Enter SHORT-spread (short A, long B) when z ≥ +entry_z.
    4. Exit when |z| ≤ exit_z (mean-reverted) OR |z| ≥ stop_z (broken).

Why pair trade with ₹1L:
    Both legs have ~equal notional, so capital required ≈ max-leg-notional.
    Returns are driven by the spread converging, not the market direction —
    real diversification away from your other long-biased equity strategies.

Constraints:
    Short side requires intraday MIS or F&O. For cash-segment-only, we model
    pair trades as long A / SHORT B with B treated as MIS short. Slippage
    and borrow costs are folded into the existing intraday cost model.

Augmented Dickey-Fuller test is *not* implemented here — that requires
statsmodels. Instead, we provide a basic stationarity proxy: ratio of
spread std to spread range. Run pre-checks via `cointegration_score`
to filter pairs before backtesting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from nse_bot.backtest.costs import CostConfig, round_trip_cost


@dataclass
class PairTradeConfig:
    initial_capital: float = 100_000.0
    lookback: int = 60          # bars for hedge-ratio + z-score
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_capital_per_trade_pct: float = 1.0
    cost_per_leg: CostConfig = field(default_factory=lambda: CostConfig(segment="equity_intraday"))


def hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> float:
    """OLS slope of log_a on log_b (intercept implicit via centering)."""
    a = np.asarray(log_a, dtype=float)
    b = np.asarray(log_b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = (b * b).sum()
    if denom <= 0:
        return 1.0
    return float((a * b).sum() / denom)


def cointegration_score(prices_a: pd.Series, prices_b: pd.Series, lookback: int = 252) -> dict:
    """Diagnostic for whether (A, B) might cointegrate over the latest `lookback` bars.

    Returns:
        beta:               OLS hedge ratio.
        corr:               Pearson correlation of log returns (the loose proxy).
        spread_volatility:  std(spread) — should be modest if cointegrated.
        adf_proxy:          (max-spread − min-spread) / std(spread). Lower
                            means tighter mean-reversion. Below ~6 is usually
                            interesting; above ~10 is weak.
    """
    a = np.log(prices_a.astype(float).values[-lookback:])
    b = np.log(prices_b.astype(float).values[-lookback:])
    if len(a) < 30 or len(b) != len(a):
        return {"beta": float("nan"), "corr": float("nan"),
                "spread_volatility": float("nan"), "adf_proxy": float("nan")}
    beta = hedge_ratio(a, b)
    spread = a - beta * b
    rets_a = np.diff(a)
    rets_b = np.diff(b)
    if rets_a.std() > 0 and rets_b.std() > 0:
        corr = float(np.corrcoef(rets_a, rets_b)[0, 1])
    else:
        corr = float("nan")
    sd = float(spread.std(ddof=1))
    rng = float(spread.max() - spread.min())
    return {
        "beta": float(beta),
        "corr": corr,
        "spread_volatility": sd,
        "adf_proxy": rng / sd if sd > 0 else float("inf"),
    }


def _rolling_zscore(spread: np.ndarray, lookback: int) -> np.ndarray:
    n = spread.size
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        window = spread[i - lookback:i]
        sd = window.std(ddof=1)
        if sd > 0:
            out[i] = (spread[i] - window.mean()) / sd
    return out


def run_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    cfg: PairTradeConfig = PairTradeConfig(),
) -> tuple[pd.Series, pd.DataFrame]:
    """Backtest a single pair. prices_a/_b should be aligned by date index."""
    if len(prices_a) != len(prices_b):
        raise ValueError("price series must be same length")
    n = len(prices_a)
    if n < cfg.lookback + 5:
        return pd.Series(dtype=float), pd.DataFrame()

    pa = prices_a.astype(float).values
    pb = prices_b.astype(float).values
    log_a = np.log(pa)
    log_b = np.log(pb)

    # Rolling hedge ratio + spread + z-score.
    betas = np.full(n, np.nan)
    spreads = np.full(n, np.nan)
    for i in range(cfg.lookback, n):
        wa = log_a[i - cfg.lookback:i]
        wb = log_b[i - cfg.lookback:i]
        b = hedge_ratio(wa, wb)
        betas[i] = b
        spreads[i] = log_a[i] - b * log_b[i]
    z = _rolling_zscore(spreads, cfg.lookback)

    capital = cfg.initial_capital
    side = 0  # +1 long-spread (long A, short B), -1 short-spread
    qty_a = 0.0
    qty_b = 0.0
    entry_a = 0.0
    entry_b = 0.0
    entry_idx = -1
    trades: list[dict] = []
    equity = np.empty(n, dtype=float)
    equity[:cfg.lookback] = capital

    idx = prices_a.index

    for i in range(cfg.lookback, n):
        zi = z[i]
        # Mark-to-market open position.
        if side != 0:
            mtm_a = side * (pa[i] - entry_a) * qty_a
            mtm_b = -side * (pb[i] - entry_b) * qty_b
            unreal = mtm_a + mtm_b
        else:
            unreal = 0.0
        equity[i] = capital + unreal

        # Entry / exit logic.
        if not np.isfinite(zi):
            continue

        if side == 0:
            if zi <= -cfg.entry_z:
                # Long spread: long A, short B
                budget = capital * cfg.max_capital_per_trade_pct / 2
                qty_a = budget / pa[i] if pa[i] > 0 else 0
                qty_b = budget / pb[i] if pb[i] > 0 else 0
                if qty_a > 0 and qty_b > 0:
                    side = 1
                    entry_a, entry_b = pa[i], pb[i]
                    entry_idx = i
            elif zi >= cfg.entry_z:
                budget = capital * cfg.max_capital_per_trade_pct / 2
                qty_a = budget / pa[i] if pa[i] > 0 else 0
                qty_b = budget / pb[i] if pb[i] > 0 else 0
                if qty_a > 0 and qty_b > 0:
                    side = -1
                    entry_a, entry_b = pa[i], pb[i]
                    entry_idx = i
        else:
            reverted = (side == 1 and zi >= -cfg.exit_z) or (side == -1 and zi <= cfg.exit_z)
            broken = abs(zi) >= cfg.stop_z
            if reverted or broken:
                exit_a, exit_b = pa[i], pb[i]
                gross_a = side * (exit_a - entry_a) * qty_a
                gross_b = -side * (exit_b - entry_b) * qty_b
                # Round-trip cost on each leg.
                buy_a = entry_a * qty_a if side > 0 else exit_a * qty_a
                sell_a = exit_a * qty_a if side > 0 else entry_a * qty_a
                buy_b = exit_b * qty_b if side > 0 else entry_b * qty_b
                sell_b = entry_b * qty_b if side > 0 else exit_b * qty_b
                cost_a = round_trip_cost(buy_a, sell_a, cfg.cost_per_leg)
                cost_b = round_trip_cost(buy_b, sell_b, cfg.cost_per_leg)
                gross = gross_a + gross_b
                cost = cost_a + cost_b
                net = gross - cost
                capital += net
                trades.append({
                    "entry_ts": idx[entry_idx],
                    "exit_ts": idx[i],
                    "side": "LONG_SPREAD" if side > 0 else "SHORT_SPREAD",
                    "entry_a": entry_a, "exit_a": exit_a, "qty_a": qty_a,
                    "entry_b": entry_b, "exit_b": exit_b, "qty_b": qty_b,
                    "gross_inr": gross, "cost_inr": cost, "pnl_inr": net,
                    "exit_reason": "stop" if broken else "mean_revert",
                    "entry_z": z[entry_idx], "exit_z": zi,
                })
                side = 0
                qty_a = qty_b = 0.0

    eq = pd.Series(equity, index=idx, name="equity")
    return eq, pd.DataFrame(trades)


def screen_pairs(
    candidates: Iterable[tuple[str, str]],
    price_loader,
    *,
    lookback: int = 252,
    min_corr: float = 0.7,
    max_adf_proxy: float = 8.0,
) -> pd.DataFrame:
    """Filter candidate pairs by simple cointegration proxies.

    price_loader: callable(symbol) -> pd.Series (price by date) — typically
                  closes from cache.read(sym, 'day').
    """
    rows = []
    for a, b in candidates:
        pa = price_loader(a)
        pb = price_loader(b)
        if pa is None or pb is None or pa.empty or pb.empty:
            continue
        idx = pa.index.intersection(pb.index)
        if len(idx) < lookback + 5:
            continue
        pa = pa.reindex(idx).dropna()
        pb = pb.reindex(idx).dropna()
        idx = pa.index.intersection(pb.index)
        pa = pa.reindex(idx)
        pb = pb.reindex(idx)
        score = cointegration_score(pa, pb, lookback=lookback)
        rows.append({"a": a, "b": b, **score,
                     "passes": (score["corr"] >= min_corr and score["adf_proxy"] <= max_adf_proxy)})
    return pd.DataFrame(rows).sort_values(
        by=["passes", "adf_proxy"], ascending=[False, True]
    ).reset_index(drop=True) if rows else pd.DataFrame()
