"""Download historical OHLCV for the NSE universe and populate the cache.

Examples:
    # 5 years of daily candles for the whole NSE equity universe:
    python scripts/fetch_history.py --interval day --years 5

    # 2 years of 30-minute bars for the top 200 by recent turnover:
    python scripts/fetch_history.py --interval 30minute --years 2 --top 200

    # 1 year of 1-minute bars for a custom list:
    python scripts/fetch_history.py --interval 1minute --years 1 \\
        --symbols RELIANCE,HDFCBANK,TCS

The cache is incremental — re-running only fetches missing date ranges.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.config import ensure_dirs, load_config
from nse_bot.data import cache, universe
from nse_bot.data.upstox_client import UpstoxClient

VALID_INTERVALS = ("1minute", "30minute", "day", "week", "month")


def _pick_symbol_column(df) -> str:
    for c in ("tradingsymbol", "trading_symbol", "symbol"):
        if c in df.columns:
            return c
    raise RuntimeError("No symbol column found in instruments frame")


def _pick_key_column(df) -> str:
    for c in ("instrument_key", "instrumentkey", "token"):
        if c in df.columns:
            return c
    raise RuntimeError("No instrument_key column found in instruments frame")


@click.command()
@click.option("--interval", type=click.Choice(VALID_INTERVALS), default="day")
@click.option("--years", type=float, default=5.0, help="How far back to fetch.")
@click.option("--symbols", type=str, default="", help="Comma-separated trading symbols (overrides universe).")
@click.option("--top", type=int, default=0, help="If >0, take the first N symbols of the equity universe.")
@click.option("--resume/--no-resume", default=True, help="Skip dates already cached.")
def main(interval: str, years: float, symbols: str, top: int, resume: bool) -> None:
    ensure_dirs()
    cfg = load_config()
    if not cfg.access_token:
        raise SystemExit("No UPSTOX_ACCESS_TOKEN. Run `python scripts/auth.py` first.")

    if symbols:
        # Explicit list — search the FULL instruments dump so indices,
        # F&O contracts, etc. resolve, not just NSE_EQ equities.
        full = universe.load_instruments()
        sym_col = _pick_symbol_column(full)
        key_col = _pick_key_column(full)
        raw_tokens = [s.strip() for s in symbols.split(",") if s.strip()]
        wanted: list[str] = []
        for tok in raw_tokens:
            expanded = universe.expand_universe_keyword(tok)
            if expanded is not None:
                wanted.extend(s.upper() for s in expanded)
            else:
                wanted.append(tok.upper())
        wanted = sorted(set(wanted))
        sub = full.loc[full[sym_col].astype(str).str.upper().isin(wanted)].copy()
        # Restrict to NSE-side instruments to avoid double-fetching from BSE.
        if "exchange" in sub.columns:
            sub = sub.loc[sub["exchange"].astype(str).str.upper().str.startswith("NSE_")]
        # If multiple matches per symbol (e.g. NSE_EQ + NSE_FO), prefer NSE_EQ then NSE_INDEX.
        if not sub.empty and "exchange" in sub.columns:
            order = {"NSE_EQ": 0, "NSE_INDEX": 1, "NSE_FO": 2}
            sub = sub.assign(_rank=sub["exchange"].astype(str).str.upper().map(order).fillna(9))
            sub = sub.sort_values(["_rank"]).drop_duplicates(subset=[sym_col]).drop(columns=["_rank"])
    else:
        inst = universe.nse_equity()
        sym_col = _pick_symbol_column(inst)
        key_col = _pick_key_column(inst)
        sub = inst.copy()
        if top > 0:
            sub = sub.head(top)

    to_date = date.today() - timedelta(days=1)
    from_date = to_date - timedelta(days=int(years * 365.25))

    print(f"Fetching {interval} candles for {len(sub)} symbols, {from_date} -> {to_date}")

    with UpstoxClient(cfg.access_token) as client:
        for _, row in tqdm(sub.iterrows(), total=len(sub), unit="sym"):
            sym = str(row[sym_col])
            key = str(row[key_col])
            start = from_date
            if resume:
                last = cache.last_ts(sym, interval)
                if last is not None:
                    candidate = (last.date() + timedelta(days=1))
                    if candidate > start:
                        start = candidate
            if start > to_date:
                continue
            try:
                df = client.fetch_candles(key, interval, start, to_date)
            except Exception as e:
                tqdm.write(f"  skip {sym}: {e}")
                continue
            cache.write(sym, interval, df)


if __name__ == "__main__":
    main()
