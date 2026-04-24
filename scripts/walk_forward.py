"""Walk-forward validation CLI.

Examples:
    # Daily momentum on RELIANCE: 1 year train, 3 months test, roll by 3 months.
    python scripts/walk_forward.py --strategy momentum_daily --symbol RELIANCE \\
        --train-bars 252 --test-bars 63

    # ORB on HDFCBANK, 15-min bars, 2 months train, 1 month test.
    python scripts/walk_forward.py --strategy orb --symbol HDFCBANK \\
        --interval 15minute --train-bars 2625 --test-bars 1313
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.backtest.walkforward import walk_forward
from nse_bot.config import REPORTS_DIR, ensure_dirs, load_config
from nse_bot.data import cache
from nse_bot.data.upstox_client import resample
from nse_bot.strategies import REGISTRY


def _load(sym: str, interval: str) -> pd.DataFrame:
    if interval in ("5minute", "15minute"):
        base = cache.read(sym, "1minute")
        if base.empty:
            return base
        return resample(base, "5min" if interval == "5minute" else "15min")
    return cache.read(sym, interval)


@click.command()
@click.option("--strategy", type=click.Choice(list(REGISTRY.keys())), required=True)
@click.option("--symbol", type=str, required=True)
@click.option("--interval", type=str, default="", help="Override the strategy's default interval.")
@click.option("--train-bars", type=int, default=252)
@click.option("--test-bars", type=int, default=63)
@click.option("--step-bars", type=int, default=0, help="0 = test_bars (non-overlapping folds).")
@click.option("--metric", type=click.Choice(["sharpe", "cagr", "total_return", "profit_factor"]), default="sharpe")
@click.option("--save/--no-save", default=False)
def main(strategy: str, symbol: str, interval: str, train_bars: int, test_bars: int,
         step_bars: int, metric: str, save: bool) -> None:
    ensure_dirs()
    cfg = load_config()
    strat_cls = REGISTRY[strategy]
    interval = interval or strat_cls().interval

    df = _load(symbol, interval)
    if df.empty:
        raise SystemExit(f"No cached {interval} data for {symbol}. Run fetch_history.py first.")

    print(f"{strategy} on {symbol} ({interval}): {len(df)} bars, "
          f"train={train_bars}, test={test_bars}, step={step_bars or test_bars}")
    print(f"Param grid: {strat_cls.param_grid() or '(none — single run per fold)'}")

    res = walk_forward(
        strat_cls, df,
        train_bars=train_bars, test_bars=test_bars,
        step_bars=step_bars or None,
        capital=cfg.capital_inr, metric=metric,
    )

    if res.folds.empty:
        print("\nNot enough data for a single fold.")
        return

    print("\n=== Per-fold results ===")
    display_cols = [
        "fold", "train_start", "test_start", "params",
        "is_sharpe", "oos_sharpe",
        "is_total_return_pct", "oos_total_return_pct",
        "is_max_dd_pct", "oos_max_dd_pct",
        "oos_trades",
    ]
    shown = res.folds[display_cols].copy()
    for c in ("train_start", "test_start"):
        shown[c] = shown[c].astype(str).str.slice(0, 10)
    print(tabulate(shown.round(2), headers="keys", tablefmt="github", showindex=False))

    summary = pd.Series({
        "folds": len(res.folds),
        "avg_is_sharpe": res.folds["is_sharpe"].mean(),
        "avg_oos_sharpe": res.folds["oos_sharpe"].mean(),
        "avg_is_total_return_pct": res.folds["is_total_return_pct"].mean(),
        "avg_oos_total_return_pct": res.folds["oos_total_return_pct"].mean(),
        "avg_oos_max_dd_pct": res.folds["oos_max_dd_pct"].mean(),
        "oos_sharpe_over_is": (res.folds["oos_sharpe"] / res.folds["is_sharpe"].replace(0, float("nan"))).mean(),
    }).round(3)
    print("\n=== Summary ===")
    print(summary.to_string())

    if save:
        out = REPORTS_DIR / f"walkforward_{strategy}_{symbol}_{interval}.csv"
        res.folds.to_csv(out, index=False)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
