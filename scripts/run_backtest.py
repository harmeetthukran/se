"""Run one or more strategies against the cached universe and print a report.

Examples:
    # Run all strategies on the top 30 most-liquid names:
    python scripts/run_backtest.py --top 30

    # Run a single strategy on a custom list:
    python scripts/run_backtest.py --strategy orb --symbols RELIANCE,HDFCBANK,TCS

    # Save per-strategy summary to reports/:
    python scripts/run_backtest.py --top 30 --save
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.backtest import engine, metrics
from nse_bot.backtest.costs import CostConfig
from nse_bot.config import REPORTS_DIR, ensure_dirs, load_config
from nse_bot.data import cache, universe
from nse_bot.data.upstox_client import resample
from nse_bot.strategies import REGISTRY

INTRADAY_INTERVALS = {"1minute", "5minute", "15minute", "30minute"}


def _load(sym: str, interval: str) -> pd.DataFrame:
    if interval in ("5minute", "15minute"):
        base = cache.read(sym, "1minute")
        if base.empty:
            return base
        rule = "5min" if interval == "5minute" else "15min"
        return resample(base, rule)
    return cache.read(sym, interval)


def _run_one(strategy_name: str, sym: str, capital: float) -> dict | None:
    strat_cls = REGISTRY[strategy_name]
    strat = strat_cls()
    df = _load(sym, strat.interval)
    if df.empty or len(df) < 50:
        return None
    result = strat.generate(df)
    segment = "equity_intraday" if result.intraday else "equity_delivery"
    cfg = engine.BacktestConfig(
        initial_capital=capital,
        cost=CostConfig(segment=segment),
        intraday_squareoff=result.intraday,
    )
    eq, trades = engine.run(df, result.signal, cfg)
    periods_per_year = 252 * 375 if strat.interval in INTRADAY_INTERVALS else 252
    m = metrics.compute(eq, trades, periods_per_year=periods_per_year)
    row = m.to_dict()
    row.update({"strategy": strategy_name, "symbol": sym})
    return row


def _pick_symbol_column(df) -> str:
    for c in ("tradingsymbol", "trading_symbol", "symbol"):
        if c in df.columns:
            return c
    raise RuntimeError("No symbol column found in instruments frame")


@click.command()
@click.option("--strategy", type=str, default="all", help="Strategy name, or 'all'.")
@click.option("--symbols", type=str, default="", help="Comma-separated trading symbols.")
@click.option("--top", type=int, default=30, help="Take top-N from universe when no --symbols.")
@click.option("--save/--no-save", default=False, help="Save CSVs under reports/.")
def main(strategy: str, symbols: str, top: int, save: bool) -> None:
    ensure_dirs()
    cfg = load_config()

    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        inst = universe.nse_equity()
        col = _pick_symbol_column(inst)
        syms = inst[col].astype(str).head(top).tolist()

    strategies = list(REGISTRY.keys()) if strategy == "all" else [strategy]
    missing = [s for s in strategies if s not in REGISTRY]
    if missing:
        raise SystemExit(f"Unknown strategies: {missing}. Available: {list(REGISTRY)}")

    rows: list[dict] = []
    for strat_name in strategies:
        for sym in tqdm(syms, desc=strat_name, unit="sym"):
            row = _run_one(strat_name, sym, cfg.capital_inr)
            if row:
                rows.append(row)

    if not rows:
        print("No results — did you run fetch_history.py first?")
        return

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("strategy")
        .agg(
            symbols=("symbol", "nunique"),
            avg_return_pct=("total_return_pct", "mean"),
            avg_cagr_pct=("cagr_pct", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_max_dd_pct=("max_drawdown_pct", "mean"),
            total_trades=("trades", "sum"),
            avg_win_rate_pct=("win_rate_pct", "mean"),
            avg_profit_factor=("profit_factor", lambda s: s.replace([float("inf")], float("nan")).mean()),
        )
        .sort_values("avg_sharpe", ascending=False)
        .round(2)
    )

    print("\n=== Per-symbol results ===")
    print(tabulate(df.round(2).sort_values(["strategy", "total_return_pct"], ascending=[True, False]),
                   headers="keys", tablefmt="github", showindex=False))
    print("\n=== Per-strategy summary ===")
    print(tabulate(summary, headers="keys", tablefmt="github"))

    if save:
        out_dir = REPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "backtest_per_symbol.csv", index=False)
        summary.to_csv(out_dir / "backtest_summary.csv")
        print(f"\nSaved CSVs to {out_dir}")


if __name__ == "__main__":
    main()
