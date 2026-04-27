"""Backtest pair trades on cached daily candles.

Examples:
    # Single pair:
    python scripts/run_pair_trade.py --pair HDFCBANK,ICICIBANK --years 3

    # Screen many candidates first:
    python scripts/run_pair_trade.py --screen \\
        --candidates HDFCBANK:ICICIBANK,RELIANCE:ONGC,TCS:INFY,SBIN:PNB

    # Custom z-score thresholds:
    python scripts/run_pair_trade.py --pair RELIANCE,ONGC --years 3 \\
        --entry-z 2.5 --exit-z 0.5 --stop-z 5
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
from nse_bot.backtest.pair_trade import (
    PairTradeConfig, run_pair, screen_pairs, cointegration_score,
)
from nse_bot.config import REPORTS_DIR, ensure_dirs, load_config
from nse_bot.data import cache


def _series(symbol: str, from_date: date | None, to_date: date | None) -> pd.Series:
    df = cache.read(symbol, "day")
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("ts")["close"].astype(float)
    if from_date is not None:
        s = s.loc[s.index.date >= from_date]
    if to_date is not None:
        s = s.loc[s.index.date <= to_date]
    return s


@click.command()
@click.option("--pair", type=str, default="", help="A,B (e.g. HDFCBANK,ICICIBANK)")
@click.option("--screen", is_flag=True, help="Just print cointegration scores for --candidates and exit.")
@click.option("--candidates", type=str, default="",
              help="Comma-separated A:B pairs for --screen (e.g. HDFC:ICICI,TCS:INFY).")
@click.option("--years", type=float, default=3.0)
@click.option("--from", "from_date", type=str, default="")
@click.option("--to", "to_date", type=str, default="")
@click.option("--lookback", type=int, default=60)
@click.option("--entry-z", type=float, default=2.0)
@click.option("--exit-z", type=float, default=0.5)
@click.option("--stop-z", type=float, default=4.0)
@click.option("--save/--no-save", default=False)
def main(pair, screen, candidates, years, from_date, to_date, lookback,
         entry_z, exit_z, stop_z, save):
    ensure_dirs()
    cfg = load_config()

    end = datetime.strptime(to_date, "%Y-%m-%d").date() if to_date else date.today() - timedelta(days=1)
    start = (datetime.strptime(from_date, "%Y-%m-%d").date() if from_date
             else end - timedelta(days=int(years * 365.25)))

    if screen:
        if not candidates:
            raise SystemExit("--screen requires --candidates A:B,A:B,...")
        pairs = []
        for spec in candidates.split(","):
            spec = spec.strip()
            if ":" in spec:
                a, b = spec.split(":")
                pairs.append((a.strip().upper(), b.strip().upper()))
        df = screen_pairs(pairs, lambda s: _series(s, start, end), lookback=lookback)
        if df.empty:
            print("No pairs returned scores — check that daily data is cached.")
            return
        print(tabulate(df.round(3), headers="keys", tablefmt="github", showindex=False))
        return

    if not pair or "," not in pair:
        raise SystemExit("Specify --pair A,B (or --screen --candidates A:B,...)")
    a, b = (s.strip().upper() for s in pair.split(",", 1))
    sa, sb = _series(a, start, end), _series(b, start, end)
    if sa.empty or sb.empty:
        raise SystemExit(f"Missing daily candles for {a if sa.empty else b}.")
    idx = sa.index.intersection(sb.index)
    sa, sb = sa.reindex(idx), sb.reindex(idx)

    score = cointegration_score(sa, sb, lookback=min(252, len(idx)))
    print(f"\n=== Pair {a} / {b} screen ({len(idx)} bars) ===")
    print(f"  beta: {score['beta']:.3f}   corr: {score['corr']:.3f}   "
          f"spread_vol: {score['spread_volatility']:.4f}   adf_proxy: {score['adf_proxy']:.2f}")

    pcfg = PairTradeConfig(
        initial_capital=cfg.capital_inr,
        lookback=lookback, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
    )
    eq, trades = run_pair(sa, sb, pcfg)
    if eq.empty:
        print("Series too short for the chosen lookback.")
        return

    m = metrics.compute(eq, trades, periods_per_year=252)
    print(f"\n=== Result ===")
    for k, v in m.to_dict().items():
        print(f"  {k:<22} {v:>12.3f}")

    if not trades.empty:
        print(f"\n=== Last 10 trades ===")
        cols = [c for c in ["entry_ts","exit_ts","side","entry_z","exit_z",
                            "gross_inr","cost_inr","pnl_inr","exit_reason"] if c in trades.columns]
        print(tabulate(trades.tail(10)[cols].round(3), headers="keys", tablefmt="github", showindex=False))

    if save:
        out_dir = REPORTS_DIR / f"pair_{a}_{b}"
        out_dir.mkdir(parents=True, exist_ok=True)
        eq.to_frame("equity").to_csv(out_dir / "equity.csv", index_label="ts")
        trades.to_csv(out_dir / "trades.csv", index=False)
        print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
