"""Daily paper-trading runner.

Generates today's signals using cached data and the configured strategy,
then hands them to nse_bot.live.paper.run_once which simulates orders
and updates the local paper-trading state (no real orders placed).

Usage (run after markets close, or each morning on prior-day bars):
    python scripts/paper_trade.py --strategy momentum_daily --top 50 \
        --regime nifty --news-filter block_contradiction --skip-events 1

Inspect state:
    python scripts/paper_trade.py --status
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.backtest.news_filter import apply_news_filter
from nse_bot.backtest.regime import apply_mask, nifty_regime, trend_above_sma, volatility_floor
from nse_bot.backtest.risk import RiskConfig
from nse_bot.config import ensure_dirs, load_config
from nse_bot.data import cache, universe
from nse_bot.data.events import skip_around_events
from nse_bot.data.upstox_client import resample
from nse_bot.live.paper import PaperConfig, current_positions, equity_history, orders_log, run_once
from nse_bot.strategies import REGISTRY

NIFTY_SYMBOLS = ("NIFTY 50", "NIFTY50", "NIFTY")


def _load(sym: str, interval: str) -> pd.DataFrame:
    if interval in ("5minute", "15minute"):
        base = cache.read(sym, "1minute")
        if base.empty:
            return base
        return resample(base, "5min" if interval == "5minute" else "15min")
    return cache.read(sym, interval)


def _nifty() -> pd.DataFrame:
    for s in NIFTY_SYMBOLS:
        df = cache.read(s, "day")
        if not df.empty:
            return df
    return pd.DataFrame()


def _apply_regime(signal, df, regime, nifty_df):
    if regime == "none":
        return signal
    if regime == "self":
        return apply_mask(signal, trend_above_sma(df, period=200))
    if regime == "vol":
        return apply_mask(signal, volatility_floor(df))
    if regime == "nifty" and not nifty_df.empty:
        return apply_mask(signal, nifty_regime(df["ts"], nifty_df, period=200))
    if regime == "self+nifty":
        sig = signal
        if not nifty_df.empty:
            sig = apply_mask(sig, nifty_regime(df["ts"], nifty_df, period=200))
        return apply_mask(sig, trend_above_sma(df, period=200))
    return signal


def _pick_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None


def _print_status() -> None:
    pos = current_positions()
    eq = equity_history()
    ord_log = orders_log()
    print(f"\n=== Paper positions ({len(pos)}) ===")
    if not pos.empty:
        print(tabulate(pos.round(2), headers="keys", tablefmt="github", showindex=False))
    print(f"\n=== Recent orders ({len(ord_log)} total) ===")
    if not ord_log.empty:
        print(tabulate(ord_log.tail(15).round(2), headers="keys", tablefmt="github", showindex=False))
    print(f"\n=== Equity history ({len(eq)} rows) ===")
    if not eq.empty:
        print(tabulate(eq.tail(10).round(2), headers="keys", tablefmt="github", showindex=False))


@click.command()
@click.option("--strategy", type=click.Choice(list(REGISTRY.keys())), default="")
@click.option("--symbols", type=str, default="")
@click.option("--top", type=int, default=50)
@click.option("--max-positions", type=int, default=5)
@click.option("--regime",
              type=click.Choice(["none", "self", "vol", "nifty", "self+nifty"]), default="none")
@click.option("--news-filter",
              type=click.Choice(["off", "require_confirmation", "block_contradiction"]), default="off")
@click.option("--skip-events", type=int, default=0)
@click.option("--status", is_flag=True, help="Print current positions / equity / recent orders and exit.")
def main(strategy, symbols, top, max_positions, regime, news_filter, skip_events, status):
    ensure_dirs()
    if status:
        _print_status()
        return
    if not strategy:
        raise SystemExit("Must specify --strategy unless --status is set.")
    cfg = load_config()
    strat_cls = REGISTRY[strategy]
    strat = strat_cls()
    interval = strat.interval

    inst = universe.nse_equity()
    sym_col = _pick_col(inst, ("tradingsymbol", "trading_symbol", "symbol"))
    syms = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else inst[sym_col].astype(str).head(top).tolist()
    )

    nifty_df = _nifty() if regime in ("nifty", "self+nifty") else pd.DataFrame()

    sig_rows: list[dict] = []
    bars_rows: list[pd.DataFrame] = []
    for sym in syms:
        df = _load(sym, interval)
        if df.empty or len(df) < 50:
            continue
        result = strat.generate(df)
        signal = _apply_regime(result.signal, df, regime, nifty_df)
        if news_filter != "off":
            signal = apply_news_filter(signal, df["ts"].reset_index(drop=True), sym, policy=news_filter)
        if skip_events > 0:
            signal = skip_around_events(signal, df["ts"].reset_index(drop=True), sym, window_days=skip_events)
        # Today's desired position = last signal bar shifted forward (act on yesterday's close).
        desired = int(signal.iloc[-1]) if len(signal) else 0
        sig_rows.append({"symbol": sym, "desired_pos": desired, "strategy": strategy})
        tail = df.tail(60).copy()
        tail["symbol"] = sym
        bars_rows.append(tail)

    if not sig_rows:
        print("No symbols produced signals — did you run fetch_history.py?")
        return

    signals_df = pd.DataFrame(sig_rows)
    bars_df = pd.concat(bars_rows, ignore_index=True) if bars_rows else pd.DataFrame()

    summary = run_once(
        signals_df,
        bars_df,
        cfg=PaperConfig(capital=cfg.capital_inr, max_positions=max_positions),
        risk=RiskConfig(),
    )

    print("\n=== Paper run ===")
    print(f"Signals evaluated: {len(signals_df)}")
    print(f"  with non-zero desired_pos: {(signals_df['desired_pos'] != 0).sum()}")
    print(f"Opens: {summary['opens']}   Closes: {summary['closes']}   Skipped (over cap): {summary['skipped']}")
    print(f"Open positions now: {summary['open_positions']}")
    print(f"Cash: {summary['cash']:.0f}   Unrealised: {summary['notional']:.0f}   Equity: {summary['equity']:.0f}")

    _print_status()


if __name__ == "__main__":
    main()
