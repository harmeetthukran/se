"""Monte Carlo bootstrap of trade PnL.

Given a trade log, resample (or reshuffle) trade outcomes to produce a
*distribution* of possible final equity curves. Surfaces what a single
backtest path hides: drawdown clustering, sequence risk, tail outcomes.

Two modes:
    'resample'  — sample trades with replacement. Assumes trades are iid;
                  tests "if the strategy's per-trade distribution held,
                  what's the range of outcomes?"
    'shuffle'   — permute trade order without replacement. Isolates
                  path-dependency / sequence risk; final equity is the
                  same across all sims but drawdowns differ wildly.

Output reports percentiles of final return and max drawdown plus
probability of loss and DD thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    n_sims: int
    initial_capital: float
    final_equity: np.ndarray
    max_drawdown: np.ndarray  # fractions, e.g. -0.18 for -18%

    def summary(self) -> pd.Series:
        if len(self.final_equity) == 0:
            return pd.Series(dtype=float)
        fe = self.final_equity
        md = self.max_drawdown
        ret_pct = (fe / self.initial_capital - 1.0) * 100.0
        return pd.Series(
            {
                "n_sims": self.n_sims,
                "return_p05_pct": float(np.percentile(ret_pct, 5)),
                "return_p25_pct": float(np.percentile(ret_pct, 25)),
                "return_p50_pct": float(np.percentile(ret_pct, 50)),
                "return_p75_pct": float(np.percentile(ret_pct, 75)),
                "return_p95_pct": float(np.percentile(ret_pct, 95)),
                "return_mean_pct": float(ret_pct.mean()),
                "max_dd_p05_pct": float(np.percentile(md * 100, 5)),
                "max_dd_p50_pct": float(np.percentile(md * 100, 50)),
                "max_dd_p95_pct": float(np.percentile(md * 100, 95)),
                "prob_loss_pct": float((fe < self.initial_capital).mean() * 100),
                "prob_dd_gt_10pct": float((md < -0.10).mean() * 100),
                "prob_dd_gt_20pct": float((md < -0.20).mean() * 100),
                "prob_dd_gt_30pct": float((md < -0.30).mean() * 100),
            }
        )


def bootstrap(
    trades: pd.DataFrame,
    n_sims: int = 10_000,
    initial_capital: float = 100_000.0,
    mode: Literal["resample", "shuffle"] = "resample",
    seed: int | None = None,
) -> MonteCarloResult:
    if trades is None or trades.empty or "pnl_inr" not in trades.columns:
        return MonteCarloResult(0, initial_capital, np.array([]), np.array([]))
    pnl = trades["pnl_inr"].astype(float).values
    n = len(pnl)
    if n == 0:
        return MonteCarloResult(0, initial_capital, np.array([]), np.array([]))

    rng = np.random.default_rng(seed)
    finals = np.empty(n_sims, dtype=float)
    max_dds = np.empty(n_sims, dtype=float)

    for i in range(n_sims):
        sample = rng.choice(pnl, n, replace=True) if mode == "resample" else rng.permutation(pnl)
        equity = initial_capital + np.cumsum(sample)
        finals[i] = equity[-1]
        peak = np.maximum.accumulate(np.concatenate([[initial_capital], equity]))
        eq_with_start = np.concatenate([[initial_capital], equity])
        dd = (eq_with_start - peak) / peak
        max_dds[i] = float(dd.min())

    return MonteCarloResult(n_sims, initial_capital, finals, max_dds)
