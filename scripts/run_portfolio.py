"""Portfolio-level backtest: one strategy across many symbols, shared capital.

Examples:
    python scripts/run_portfolio.py --strategy momentum_daily --top 50 \\
        --max-positions 5 --allocation equal --save

    python scripts/run_portfolio.py --strategy orb --top 30 \\
        --max-positions 3 --sector-cap 2 --regime nifty --save
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.backtest import metrics
from nse_bot.backtest.costs import CostConfig
from nse_bot.backtest.news_filter import apply_news_filter
from nse_bot.backtest.portfolio import PortfolioConfig, run_portfolio
from nse_bot.backtest.regime import apply_mask, nifty_regime, trend_above_sma, volatility_floor
from nse_bot.backtest.risk import RiskConfig
from nse_bot.config import REPORTS_DIR, ensure_dirs, load_config
from nse_bot.data import cache, universe
from nse_bot.data.upstox_client import resample
from nse_bot.strategies import REGISTRY

NIFTY_SYMBOLS = ("NIFTY 50", "NIFTY50", "NIFTY")


def _load(sym: str, interval: str) -> pd.DataFrame:
    if interval in ("5minute", "15minute"):
        base = cache.read(sym, "1minute")
        if base.empty:
            return base
        return resample(base, "5min" if interval == "5minute" else "15min")
    return cache.read(sym, interval)


def _nifty_daily() -> pd.DataFrame:
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


def _pick_col(inst: pd.DataFrame, names) -> str | None:
    for c in names:
        if c in inst.columns:
            return c
    return None


@click.command()
@click.option("--strategy", type=click.Choice(list(REGISTRY.keys())), required=True)
@click.option("--symbols", type=str, default="")
@click.option("--top", type=int, default=50)
@click.option("--max-positions", type=int, default=5)
@click.option("--allocation", type=click.Choice(["equal", "risk"]), default="equal")
@click.option("--sector-cap", type=int, default=0, help="0 = no cap.")
@click.option(
    "--regime",
    type=click.Choice(["none", "self", "vol", "nifty", "self+nifty"]),
    default="none",
)
@click.option(
    "--news-filter",
    type=click.Choice(["off", "require_confirmation", "block_contradiction"]),
    default="off",
)
@click.option("--save/--no-save", default=False)
def main(strategy, symbols, top, max_positions, allocation, sector_cap,
         regime, news_filter, save) -> None:
    ensure_dirs()
    cfg = load_config()
    strat_cls = REGISTRY[strategy]
    interval = strat_cls().interval

    inst = universe.nse_equity()
    sym_col = _pick_col(inst, ("tradingsymbol", "trading_symbol", "symbol"))
    sector_col = _pick_col(inst, ("sector", "industry"))
    sectors_map: dict[str, str] = {}
    if sector_col is not None and sym_col is not None:
        sectors_map = dict(
            zip(inst[sym_col].astype(str).str.upper(), inst[sector_col].astype(str).fillna(""))
        )

    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        syms = inst[sym_col].astype(str).head(top).tolist()

    nifty_df = _nifty_daily() if regime in ("nifty", "self+nifty") else pd.DataFrame()

    per_symbol: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for sym in tqdm(syms, desc="prep signals", unit="sym"):
        df = _load(sym, interval)
        if df.empty or len(df) < 50:
            continue
        strat = strat_cls()
        result = strat.generate(df)
        signal = _apply_regime(result.signal, df, regime, nifty_df)
        if news_filter != "off":
            signal = apply_news_filter(signal, df["ts"].reset_index(drop=True), sym, policy=news_filter)
        per_symbol[sym] = (df, signal)

    if not per_symbol:
        print("No cached data for any symbol.")
        return

    intraday = strat_cls().generate(pd.DataFrame(columns=["ts","open","high","low","close","volume","oi"])).intraday
    pcfg = PortfolioConfig(
        initial_capital=cfg.capital_inr,
        max_positions=max_positions,
        allocation=allocation,
        cost=CostConfig(segment="equity_intraday" if intraday else "equity_delivery"),
        risk=RiskConfig(),
        sector_cap=sector_cap if sector_cap > 0 else None,
    )

    eq, trades = run_portfolio(per_symbol, pcfg, sectors_map or None)

    if eq.empty:
        print("Portfolio ran but produced no equity curve — likely no triggered signals.")
        return

    periods_per_year = 252 * 375 if interval in {"1minute","5minute","15minute","30minute"} else 252
    m = metrics.compute(eq, trades, periods_per_year=periods_per_year)

    print(f"\n=== Portfolio ({strategy}) ===")
    print(f"Symbols:        {len(per_symbol)}")
    print(f"Max positions:  {max_positions}   Allocation: {allocation}   Sector cap: {sector_cap or '-'}")
    print(f"Regime: {regime}   News filter: {news_filter}")
    print(f"Trades:         {m.trades}")
    print(f"Total return:   {m.total_return_pct:.2f}%")
    print(f"CAGR:           {m.cagr_pct:.2f}%")
    print(f"Sharpe:         {m.sharpe:.2f}")
    print(f"Max DD:         {m.max_drawdown_pct:.2f}%")
    print(f"Win rate:       {m.win_rate_pct:.2f}%   Profit factor: {m.profit_factor:.2f}")

    if not trades.empty:
        print("\n=== Recent closed trades ===")
        tail = trades.tail(15).round(2)
        show_cols = [c for c in ["entry_ts","exit_ts","symbol","side","return_pct","pnl_inr","exit_reason","sector"] if c in tail.columns]
        print(tabulate(tail[show_cols], headers="keys", tablefmt="github", showindex=False))

    if save:
        out_trades = REPORTS_DIR / f"portfolio_trades_{strategy}.csv"
        out_eq = REPORTS_DIR / f"portfolio_equity_{strategy}.csv"
        trades.to_csv(out_trades, index=False)
        eq.to_frame("equity").to_csv(out_eq, index_label="ts")
        print(f"\nSaved trades -> {out_trades}")
        print(f"Saved equity -> {out_eq}")


if __name__ == "__main__":
    main()
