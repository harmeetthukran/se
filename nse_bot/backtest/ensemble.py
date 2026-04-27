"""Combine multiple strategy equity curves into an ensemble portfolio.

Allocation methods:
    'equal'          — equal capital weight across strategies, rebalanced.
    'inverse_vol'    — weight inverse to trailing volatility (risk parity).
    'rolling_sharpe' — weight proportional to trailing-N-bar Sharpe (capped
                       at 0; underperformers get nothing).
    'regime_switch'  — pick the single best-performing strategy over the
                       trailing window each rebalance period (winner-takes-all).

Inputs:
    equity_curves: dict {strategy_name: pd.Series of equity by date}.
                   Series can have different lengths; we align on the
                   intersection of indices.
    initial_capital: starting capital for the ensemble.
    method: allocation rule (above).
    rebalance_days: rebalance frequency in trading bars.
    lookback_days: window for inverse_vol / rolling_sharpe / regime_switch.

Returns: combined equity curve, allocation history (one row per rebalance).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Method = Literal["equal", "inverse_vol", "rolling_sharpe", "regime_switch"]


@dataclass
class EnsembleConfig:
    initial_capital: float = 100_000.0
    method: Method = "equal"
    rebalance_days: int = 21
    lookback_days: int = 60
    min_weight: float = 0.0


def _align(equity_curves: dict[str, pd.Series]) -> pd.DataFrame:
    if not equity_curves:
        return pd.DataFrame()
    df = pd.concat(equity_curves, axis=1)
    df.columns = list(equity_curves.keys())
    df = df.dropna(how="any")
    return df


def _normalize_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy daily returns from equity curves."""
    return df.pct_change().fillna(0.0)


def _allocate(
    rets_window: pd.DataFrame,
    method: Method,
    min_weight: float,
) -> pd.Series:
    """Produce strategy weights from a window of returns."""
    cols = list(rets_window.columns)
    if rets_window.empty or len(rets_window) < 5:
        return pd.Series(np.full(len(cols), 1 / len(cols)), index=cols)

    if method == "equal":
        w = pd.Series(1.0, index=cols)
    elif method == "inverse_vol":
        vols = rets_window.std(ddof=1).replace(0, np.nan)
        if vols.isna().all():
            w = pd.Series(1.0, index=cols)
        else:
            w = (1.0 / vols).fillna(0.0)
    elif method == "rolling_sharpe":
        means = rets_window.mean()
        vols = rets_window.std(ddof=1).replace(0, np.nan)
        sharpes = (means / vols).fillna(0.0).clip(lower=0.0)
        if (sharpes <= 0).all():
            w = pd.Series(1.0, index=cols)  # fallback to equal
        else:
            w = sharpes
    elif method == "regime_switch":
        cum = (1.0 + rets_window).prod() - 1.0
        if (cum <= 0).all():
            w = pd.Series(1.0, index=cols)
        else:
            best = cum.idxmax()
            w = pd.Series(0.0, index=cols)
            w.loc[best] = 1.0
    else:
        raise ValueError(f"Unknown method: {method}")

    if w.sum() <= 0:
        w = pd.Series(1.0 / len(cols), index=cols)
    else:
        w = w / w.sum()

    if min_weight > 0:
        w = w.clip(lower=min_weight)
        w = w / w.sum()
    return w


def run_ensemble(
    equity_curves: dict[str, pd.Series],
    cfg: EnsembleConfig = EnsembleConfig(),
) -> tuple[pd.Series, pd.DataFrame]:
    aligned = _align(equity_curves)
    if aligned.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    rets = _normalize_returns(aligned)
    n = len(rets)
    cols = list(rets.columns)

    # Initial weights from first lookback window if available, else equal.
    if n >= cfg.lookback_days:
        weights = _allocate(rets.iloc[:cfg.lookback_days], cfg.method, cfg.min_weight)
    else:
        weights = pd.Series(1.0 / len(cols), index=cols)

    portfolio_eq = np.empty(n, dtype=float)
    portfolio_eq[0] = cfg.initial_capital

    alloc_rows = [{
        "ts": rets.index[0],
        **{f"w_{c}": float(weights[c]) for c in cols},
    }]

    for i in range(1, n):
        # Rebalance every rebalance_days bars, after the first lookback window has data.
        if i >= cfg.lookback_days and (i - cfg.lookback_days) % cfg.rebalance_days == 0:
            weights = _allocate(rets.iloc[i - cfg.lookback_days: i], cfg.method, cfg.min_weight)
            alloc_rows.append({
                "ts": rets.index[i],
                **{f"w_{c}": float(weights[c]) for c in cols},
            })
        # Apply weights to today's per-strategy returns.
        port_ret = float((rets.iloc[i] * weights).sum())
        portfolio_eq[i] = portfolio_eq[i - 1] * (1.0 + port_ret)

    eq = pd.Series(portfolio_eq, index=rets.index, name="equity")
    return eq, pd.DataFrame(alloc_rows).set_index("ts")
