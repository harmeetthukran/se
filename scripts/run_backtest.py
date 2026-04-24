"""Run one or more strategies against the cached universe and print a report.

Examples:
    # Run all strategies on the top 30 most-liquid names:
    python scripts/run_backtest.py --top 30

    # Run a single strategy on a custom list:
    python scripts/run_backtest.py --strategy orb --symbols RELIANCE,HDFCBANK,TCS

    # Save the usual summary to reports/:
    python scripts/run_backtest.py --top 30 --save

    # Also attach news context to every trade and every day:
    python scripts/run_backtest.py --top 30 --save --narrate

The --narrate flag joins trades and daily returns against the local news
store (populated by scripts/snapshot_news.py). Dates before you started
snapshotting will honestly show "no news in store".
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
from nse_bot.backtest.narrative import annotate_trades, daily_narrative
from nse_bot.backtest.news_filter import apply_news_filter
from nse_bot.backtest.regime import apply_mask, nifty_regime, trend_above_sma, volatility_floor
from nse_bot.config import REPORTS_DIR, ensure_dirs, load_config
from nse_bot.data import cache, universe
from nse_bot.data.upstox_client import resample
from nse_bot.strategies import REGISTRY

INTRADAY_INTERVALS = {"1minute", "5minute", "15minute", "30minute"}
NIFTY_SYMBOLS = ("NIFTY 50", "NIFTY50", "NIFTY")


def _load(sym: str, interval: str) -> pd.DataFrame:
    if interval in ("5minute", "15minute"):
        base = cache.read(sym, "1minute")
        if base.empty:
            return base
        rule = "5min" if interval == "5minute" else "15min"
        return resample(base, rule)
    return cache.read(sym, interval)


def _nifty_daily() -> pd.DataFrame:
    for s in NIFTY_SYMBOLS:
        df = cache.read(s, "day")
        if not df.empty:
            return df
    return pd.DataFrame()


def _apply_regime(signal: pd.Series, df: pd.DataFrame, regime: str, nifty_df: pd.DataFrame) -> pd.Series:
    if regime == "none":
        return signal
    if regime == "self":
        return apply_mask(signal, trend_above_sma(df, period=200))
    if regime == "vol":
        return apply_mask(signal, volatility_floor(df, atr_period=14, min_atr_pct=0.005))
    if regime == "nifty":
        if nifty_df.empty:
            return signal
        return apply_mask(signal, nifty_regime(df["ts"], nifty_df, period=200))
    if regime == "self+nifty":
        sig = signal
        if not nifty_df.empty:
            sig = apply_mask(sig, nifty_regime(df["ts"], nifty_df, period=200))
        return apply_mask(sig, trend_above_sma(df, period=200))
    return signal


def _run_one(
    strategy_name: str,
    sym: str,
    capital: float,
    regime: str,
    nifty_df: pd.DataFrame,
    news_filter: str,
) -> tuple[dict, pd.Series, pd.DataFrame] | None:
    strat_cls = REGISTRY[strategy_name]
    strat = strat_cls()
    df = _load(sym, strat.interval)
    if df.empty or len(df) < 50:
        return None
    result = strat.generate(df)
    signal = _apply_regime(result.signal, df, regime, nifty_df)
    if news_filter != "off":
        signal = apply_news_filter(signal, df["ts"].reset_index(drop=True), sym, policy=news_filter)
    segment = "equity_intraday" if result.intraday else "equity_delivery"
    cfg = engine.BacktestConfig(
        initial_capital=capital,
        cost=CostConfig(segment=segment),
        intraday_squareoff=result.intraday,
    )
    eq, trades = engine.run(df, signal, cfg)
    periods_per_year = 252 * 375 if strat.interval in INTRADAY_INTERVALS else 252
    m = metrics.compute(eq, trades, periods_per_year=periods_per_year)
    row = m.to_dict()
    row.update({"strategy": strategy_name, "symbol": sym})
    if not trades.empty:
        trades = trades.assign(strategy=strategy_name, symbol=sym)
    return row, eq, trades


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
@click.option("--narrate/--no-narrate", default=False, help="Attach news context to trades and daily returns.")
@click.option(
    "--regime",
    type=click.Choice(["none", "self", "vol", "nifty", "self+nifty"]),
    default="none",
    help="Regime gate: self=symbol 200-SMA, vol=ATR floor, nifty=Nifty 200-SMA, self+nifty=both.",
)
@click.option(
    "--news-filter",
    type=click.Choice(["off", "require_confirmation", "block_contradiction"]),
    default="off",
    help="Mask signals by stored news sentiment.",
)
def main(strategy: str, symbols: str, top: int, save: bool, narrate: bool, regime: str, news_filter: str) -> None:
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

    nifty_df = _nifty_daily() if regime in ("nifty", "self+nifty") else pd.DataFrame()
    if regime in ("nifty", "self+nifty") and nifty_df.empty:
        print("Regime=nifty requested but no NIFTY daily candles in cache; falling back to per-symbol gate.")

    rows: list[dict] = []
    all_trades: list[pd.DataFrame] = []
    equity_curves: dict[tuple[str, str], pd.Series] = {}
    for strat_name in strategies:
        for sym in tqdm(syms, desc=strat_name, unit="sym"):
            result = _run_one(strat_name, sym, cfg.capital_inr, regime, nifty_df, news_filter)
            if result is None:
                continue
            row, eq, trades = result
            rows.append(row)
            equity_curves[(strat_name, sym)] = eq
            if not trades.empty:
                all_trades.append(trades)

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

    if save or narrate:
        out_dir = REPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

    if save:
        df.to_csv(out_dir / "backtest_per_symbol.csv", index=False)
        summary.to_csv(out_dir / "backtest_summary.csv")
        print(f"\nSaved summary CSVs to {out_dir}")

    if narrate:
        trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        annotated = annotate_trades(trades_all) if not trades_all.empty else trades_all
        if not annotated.empty:
            annotated.to_csv(out_dir / "trade_narratives.csv", index=False)
            print(f"Saved annotated trades to {out_dir / 'trade_narratives.csv'} ({len(annotated)} rows)")

        narratives = []
        for (strat_name, sym), eq in equity_curves.items():
            if eq.empty:
                continue
            n = daily_narrative(eq)
            if not n.empty:
                n.insert(0, "strategy", strat_name)
                n.insert(1, "symbol", sym)
                narratives.append(n)
        if narratives:
            daily_df = pd.concat(narratives, ignore_index=True)
            daily_df.to_csv(out_dir / "daily_narratives.csv", index=False)
            print(f"Saved per-day narratives to {out_dir / 'daily_narratives.csv'} ({len(daily_df)} rows)")

            print("\n=== Sample trade narratives (top 10 by |return|) ===")
            if not annotated.empty and "return_pct" in annotated.columns:
                sample_cols = ["strategy", "symbol", "side", "entry_ts", "exit_ts", "return_pct", "entry_news"]
                existing = [c for c in sample_cols if c in annotated.columns]
                top = annotated.assign(_abs=annotated["return_pct"].abs()).sort_values("_abs", ascending=False).head(10)
                print(tabulate(top[existing], headers="keys", tablefmt="github", showindex=False))


if __name__ == "__main__":
    main()
