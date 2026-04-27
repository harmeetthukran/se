"""Run statistical edge tests on a saved equity / trade output.

Examples:
    # Edge of one strategy on one symbol:
    python scripts/test_edge.py --equity reports/momentum_daily_RELIANCE_day/equity.csv

    # Multiple-testing-aware: pass all the per-symbol Sharpes you tried
    # so DSR can penalize for the strategy hunt:
    python scripts/test_edge.py --equity reports/X/equity.csv \\
        --sr-trials reports/backtest_per_symbol.csv --sr-col sharpe
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.backtest.stats import edge_report


@click.command()
@click.option("--equity", "equity_path", type=click.Path(exists=True), required=True,
              help="CSV with a column named 'equity' (and any index).")
@click.option("--periods-per-year", type=int, default=252)
@click.option("--n-sims", type=int, default=5000)
@click.option("--sr-trials", "sr_trials_path", type=click.Path(), default="")
@click.option("--sr-col", type=str, default="sharpe", help="Column name in --sr-trials CSV.")
def main(equity_path: str, periods_per_year: int, n_sims: int,
         sr_trials_path: str, sr_col: str) -> None:
    eq = pd.read_csv(equity_path)
    if "equity" not in eq.columns:
        raise SystemExit(f"Expected an 'equity' column in {equity_path}")
    rets = eq["equity"].astype(float).pct_change().dropna().values

    sr_trials = None
    if sr_trials_path:
        df = pd.read_csv(sr_trials_path)
        if sr_col not in df.columns:
            raise SystemExit(f"Column '{sr_col}' not in {sr_trials_path}. Got: {list(df.columns)}")
        sr_trials = df[sr_col].astype(float).dropna().values

    report = edge_report(rets, periods_per_year=periods_per_year,
                        n_sims=n_sims, sr_trials=sr_trials)
    print(report.summary())
    print()
    if report.bootstrap_pvalue > 0.05:
        print("VERDICT: Sharpe is NOT statistically distinguishable from zero.")
        print("         Don't deploy this strategy on the strength of this backtest.")
    elif report.deflated_sr is not None and report.deflated_sr < 0.5:
        print("VERDICT: Likely a multiple-testing artifact.")
        print("         The strategy LOOKS good but is probably the lucky pick of N trials.")
    elif report.psr_zero < 0.95:
        print("VERDICT: Plausible but not robust. Want PSR(0) > 0.95 before deploying.")
    else:
        print("VERDICT: Statistically robust by these tests. Still paper-trade before going live.")


if __name__ == "__main__":
    main()
