"""Combine strategy equity curves into an ensemble portfolio.

Inputs are CSVs produced by the various run_*.py scripts (each with a
'ts' index column and an 'equity' column). Examples:

    python scripts/run_ensemble.py \\
        --curve momentum=reports/portfolio_equity_momentum_daily.csv \\
        --curve straddle=reports/options_long_straddle_NIFTY/equity.csv \\
        --method rolling_sharpe --rebalance 21 --lookback 60 --save
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.backtest import metrics
from nse_bot.backtest.ensemble import EnsembleConfig, run_ensemble
from nse_bot.config import REPORTS_DIR, ensure_dirs, load_config


def _load_equity(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if "equity" not in df.columns:
        raise SystemExit(f"Expected 'equity' column in {path}")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).set_index("ts")
    return df["equity"].astype(float)


@click.command()
@click.option("--curve", "-c", multiple=True,
              help="name=path/to/equity.csv (repeatable, at least 2)")
@click.option("--method",
              type=click.Choice(["equal", "inverse_vol", "rolling_sharpe", "regime_switch"]),
              default="equal")
@click.option("--rebalance", type=int, default=21)
@click.option("--lookback", type=int, default=60)
@click.option("--min-weight", type=float, default=0.0)
@click.option("--save/--no-save", default=False)
def main(curve, method, rebalance, lookback, min_weight, save):
    if len(curve) < 2:
        raise SystemExit("Provide at least two --curve name=path entries.")

    ensure_dirs()
    cfg = load_config()

    curves: dict[str, pd.Series] = {}
    for spec in curve:
        if "=" not in spec:
            raise SystemExit(f"Bad --curve {spec}: expected name=path")
        name, path = spec.split("=", 1)
        curves[name.strip()] = _load_equity(path.strip())

    print(f"Combining {list(curves)} via method={method} (rebalance={rebalance}, lookback={lookback})")

    ec = EnsembleConfig(
        initial_capital=cfg.capital_inr,
        method=method,
        rebalance_days=rebalance,
        lookback_days=lookback,
        min_weight=min_weight,
    )
    eq, alloc = run_ensemble(curves, ec)
    if eq.empty:
        print("No overlapping data among inputs.")
        return

    m = metrics.compute(eq, pd.DataFrame(), periods_per_year=252)
    print("\n=== Ensemble metrics ===")
    for k, v in m.to_dict().items():
        print(f"  {k:<22} {v:>12.3f}")

    print("\n=== Allocation history (first/last 5) ===")
    head = alloc.head(5).round(3)
    tail = alloc.tail(5).round(3)
    print(tabulate(head, headers="keys", tablefmt="github"))
    print("...")
    print(tabulate(tail, headers="keys", tablefmt="github"))

    if save:
        out_dir = REPORTS_DIR / f"ensemble_{method}"
        out_dir.mkdir(parents=True, exist_ok=True)
        eq.to_frame("equity").to_csv(out_dir / "equity.csv", index_label="ts")
        alloc.to_csv(out_dir / "allocations.csv")
        print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
