"""Backtest an option strategy on cached F&O bhavcopies.

Examples:
    python scripts/run_option_backtest.py --strategy long_straddle \\
        --symbol NIFTY --years 2

    python scripts/run_option_backtest.py --strategy bull_call_spread \\
        --symbol BANKNIFTY --from 2023-01-01 --to 2024-12-31 --width-pct 0.015

    # 1 lot vs 2 lots:
    python scripts/run_option_backtest.py --strategy long_straddle \\
        --symbol NIFTY --years 1 --lots 2 --save

Requires bhavcopy data already in cache — run scripts/fetch_bhavcopy.py first.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.backtest import metrics
from nse_bot.config import REPORTS_DIR, ensure_dirs, load_config
from nse_bot.options.engine import OptionBacktestConfig, run_options
from nse_bot.options.strategies import REGISTRY, _BaseConfig


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@click.command()
@click.option("--strategy", type=click.Choice(list(REGISTRY.keys())), required=True)
@click.option("--symbol", type=str, default="NIFTY")
@click.option("--years", type=float, default=0.0)
@click.option("--from", "from_date", type=str, default="")
@click.option("--to", "to_date", type=str, default="")
@click.option("--lots", type=int, default=1)
@click.option("--lot-size", type=int, default=0, help="Override default lot size (0 = use NIFTY/BankNifty/etc lookup).")
@click.option("--min-dte", type=int, default=5)
@click.option("--max-dte", type=int, default=14)
@click.option("--entry-freq-days", type=int, default=7)
@click.option("--target-pct", type=float, default=0.5, help="Take profit at +N×premium. 0=disable.")
@click.option("--stop-pct", type=float, default=-0.5, help="Stop at -N×premium. 0=disable.")
@click.option("--width-pct", type=float, default=0.01, help="Spread strategies only.")
@click.option("--max-trade-pct", type=float, default=0.5, help="Max premium per trade as fraction of capital.")
@click.option("--save/--no-save", default=False)
def main(strategy, symbol, years, from_date, to_date, lots, lot_size, min_dte, max_dte,
         entry_freq_days, target_pct, stop_pct, width_pct, max_trade_pct, save):
    ensure_dirs()
    cfg = load_config()

    if from_date and to_date:
        start = _parse_date(from_date)
        end = _parse_date(to_date)
    elif years > 0:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=int(years * 365.25))
    else:
        raise SystemExit("Specify --years or both --from and --to.")

    base = _BaseConfig(
        symbol=symbol.upper(),
        lot_size=lot_size,
        lots=lots,
        min_dte_days=min_dte,
        max_dte_days=max_dte,
        entry_freq_days=entry_freq_days,
        target_pnl_pct_of_premium=target_pct if target_pct != 0 else None,
        stop_pnl_pct_of_premium=stop_pct if stop_pct != 0 else None,
    )
    strat_cls = REGISTRY[strategy]
    if strategy in ("bull_call_spread", "bear_put_spread"):
        strat = strat_cls(cfg=base, width_pct=width_pct)
    else:
        strat = strat_cls(cfg=base)

    bt_cfg = OptionBacktestConfig(
        initial_capital=cfg.capital_inr,
        max_capital_per_trade_pct=max_trade_pct,
    )

    print(f"{strategy} on {symbol} {start} → {end}  capital ₹{cfg.capital_inr:,.0f}")
    eq, trades = run_options(strat, symbol, start, end, bt_cfg)

    if eq.empty:
        print("\nNo bhavcopy data in range — run scripts/fetch_bhavcopy.py first.")
        return

    m = metrics.compute(eq, trades, periods_per_year=252)
    print("\n=== Result ===")
    for k, v in m.to_dict().items():
        print(f"  {k:<22} {v:>14.3f}")

    if not trades.empty:
        print(f"\n=== Last 10 trades ===")
        cols = [c for c in ["entry_date", "exit_date", "exit_reason", "entry_value_inr", "gross_inr", "cost_inr", "pnl_inr", "hold_days"] if c in trades.columns]
        print(tabulate(trades.tail(10)[cols].round(2), headers="keys", tablefmt="github", showindex=False))
        print(f"\nExit-reason mix: {trades['exit_reason'].value_counts().to_dict()}")

    if save:
        out_dir = REPORTS_DIR / f"options_{strategy}_{symbol}"
        out_dir.mkdir(parents=True, exist_ok=True)
        eq.to_frame("equity").to_csv(out_dir / "equity.csv", index_label="ts")
        trades.to_csv(out_dir / "trades.csv", index=False)
        print(f"\nSaved equity + trades to {out_dir}")


if __name__ == "__main__":
    main()
