"""Stream live quotes (polling) for a list of instruments.

Examples:
    # Top NIFTY 50 names, poll every 2 seconds:
    python scripts/run_feed.py --symbols RELIANCE,HDFCBANK,TCS,INFY \\
        --interval 2

    # 5-minute test run:
    python scripts/run_feed.py --symbols NIFTY --interval 1 --max-runtime 300

Ticks are persisted to data/live/ticks_YYYY-MM-DD.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.config import ensure_dirs, load_config
from nse_bot.data import universe
from nse_bot.live.feed import FeedConfig, stream


def _resolve_instrument_keys(symbols: list[str]) -> list[str]:
    inst = universe.nse_equity()
    sym_col = next((c for c in ("tradingsymbol", "trading_symbol", "symbol") if c in inst.columns), None)
    key_col = next((c for c in ("instrument_key", "instrumentkey") if c in inst.columns), None)
    if sym_col is None or key_col is None:
        raise SystemExit("Could not resolve symbol/instrument_key columns from universe.")
    out = []
    upper = [s.upper() for s in symbols]
    sub = inst.loc[inst[sym_col].astype(str).str.upper().isin(upper)]
    for s in upper:
        match = sub.loc[sub[sym_col].astype(str).str.upper() == s]
        if not match.empty:
            out.append(str(match.iloc[0][key_col]))
        else:
            print(f"  warn: {s} not found in universe; skipping")
    return out


@click.command()
@click.option("--symbols", type=str, required=True, help="Comma-separated trading symbols.")
@click.option("--interval", type=float, default=2.0, help="Poll interval in seconds.")
@click.option("--max-runtime", type=float, default=0.0, help="Stop after N seconds (0 = run forever).")
@click.option("--no-persist", is_flag=True, help="Don't write ticks to parquet.")
def main(symbols, interval, max_runtime, no_persist):
    ensure_dirs()
    cfg = load_config()
    if not cfg.access_token:
        raise SystemExit("No UPSTOX_ACCESS_TOKEN. Run scripts/auth.py first.")

    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    keys = _resolve_instrument_keys(syms)
    if not keys:
        raise SystemExit("No instrument_keys resolved.")

    print(f"Streaming {len(keys)} instruments every {interval}s "
          f"({'no persist' if no_persist else 'persisting to data/live/ticks_*.parquet'})")
    if max_runtime > 0:
        print(f"Will stop after {max_runtime:.0f}s.")
    else:
        print("Press Ctrl+C to stop.")

    feed_cfg = FeedConfig(
        poll_interval_s=interval,
        persist_ticks=not no_persist,
        max_runtime_s=max_runtime if max_runtime > 0 else None,
    )

    last_print = [0.0]
    n = [0]

    def on_tick(tick: dict) -> None:
        n[0] += 1
        import time
        if time.monotonic() - last_print[0] > 5.0:
            print(f"  [{tick['ts'].strftime('%H:%M:%S')}] {tick['instrument_key']} "
                  f"ltp={tick['ltp']:.2f} bid={tick['bid']:.2f} ask={tick['ask']:.2f}  "
                  f"(total ticks: {n[0]})")
            last_print[0] = time.monotonic()

    try:
        total = stream(cfg.access_token, keys, on_tick=on_tick, cfg=feed_cfg)
    except KeyboardInterrupt:
        print("\nStopped.")
        return
    print(f"\nTotal ticks emitted: {total}")


if __name__ == "__main__":
    main()
