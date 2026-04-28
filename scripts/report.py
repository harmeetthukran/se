"""Generate a rich per-symbol / per-strategy HTML report.

Loads cached OHLCV, runs the strategy, runs the backtest, optionally runs
Monte Carlo, renders PNG charts, and writes an HTML index linking them.

Examples:
    python scripts/report.py --strategy momentum_daily --symbol RELIANCE
    python scripts/report.py --strategy orb --symbol HDFCBANK --monte-carlo 5000
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.backtest import engine, metrics
from nse_bot.backtest.costs import CostConfig
from nse_bot.backtest.montecarlo import bootstrap
from nse_bot.config import REPORTS_DIR, ensure_dirs, load_config
from nse_bot.data import cache
from nse_bot.data.upstox_client import resample
from nse_bot.reports.html import build_html_report
from nse_bot.reports.plots import (
    plot_equity,
    plot_mc_distributions,
    plot_monthly_heatmap,
    plot_trade_histogram,
    plot_underwater,
)
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
@click.option("--interval", type=str, default="", help="Override strategy default.")
@click.option("--monte-carlo", "mc_sims", type=int, default=5000, help="0 to skip MC.")
@click.option("--benchmark", type=str, default="", help="Symbol to overlay as benchmark (e.g. NIFTY 50).")
def main(strategy: str, symbol: str, interval: str, mc_sims: int, benchmark: str) -> None:
    ensure_dirs()
    cfg = load_config()
    strat_cls = REGISTRY[strategy]
    strat = strat_cls()
    interval = interval or strat.interval

    df = _load(symbol, interval)
    if df.empty:
        raise SystemExit(f"No cached {interval} data for {symbol}.")

    result = strat.generate(df)
    segment = "equity_intraday" if result.intraday else "equity_delivery"
    bt_cfg = engine.BacktestConfig(
        initial_capital=cfg.capital_inr,
        cost=CostConfig(segment=segment),
        intraday_squareoff=result.intraday,
    )
    eq, trades = engine.run(df, result.signal, bt_cfg)
    if eq.empty:
        raise SystemExit("Backtest produced no equity curve.")
    periods = 252 * 375 if interval in {"1minute", "5minute", "15minute", "30minute"} else 252
    m = metrics.compute(eq, trades, periods_per_year=periods)
    m_series = pd.Series(m.to_dict())

    out_dir = REPORTS_DIR / f"{strategy}_{symbol}_{interval}".replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist raw equity + trades so other tools (test_edge.py, ensemble) can read them.
    eq.to_frame("equity").to_csv(out_dir / "equity.csv", index_label="ts")
    if not trades.empty:
        trades.to_csv(out_dir / "trades.csv", index=False)

    bench_eq = None
    if benchmark:
        b_df = cache.read(benchmark, "day")
        if not b_df.empty:
            bench_eq = b_df.set_index("ts")["close"]

    eq_path = out_dir / "equity.png"
    uw_path = out_dir / "underwater.png"
    mh_path = out_dir / "monthly_heatmap.png"
    th_path = out_dir / "trade_histogram.png"

    plot_equity(eq, f"{strategy} on {symbol} — equity", eq_path, benchmark=bench_eq)
    plot_underwater(eq, f"{strategy} on {symbol} — drawdown", uw_path)
    plot_monthly_heatmap(eq, f"{strategy} on {symbol} — monthly returns", mh_path)
    plot_trade_histogram(trades, f"{strategy} on {symbol} — per-trade returns", th_path)

    mc_summary = pd.Series(dtype=float)
    mc_img = None
    if mc_sims > 0 and not trades.empty:
        mc_res = bootstrap(trades, n_sims=mc_sims, initial_capital=cfg.capital_inr, mode="resample", seed=42)
        mc_summary = mc_res.summary()
        mc_img = out_dir / "monte_carlo.png"
        plot_mc_distributions(
            mc_res.final_equity, mc_res.max_drawdown, cfg.capital_inr,
            f"{strategy} on {symbol} — Monte Carlo ({mc_sims} sims)", mc_img,
        )

    image_paths = [
        ("Equity curve", eq_path),
        ("Drawdown (underwater)", uw_path),
        ("Monthly returns heatmap", mh_path),
        ("Per-trade return distribution", th_path),
    ]
    if mc_img is not None:
        image_paths.append((f"Monte Carlo distribution ({mc_sims} sims)", mc_img))

    extra = []
    if not trades.empty and "exit_reason" in trades.columns:
        extra.append(("Exit reason breakdown", trades["exit_reason"].value_counts().to_frame("trades")))

    html_path = out_dir / "index.html"
    build_html_report(
        title=f"{strategy} on {symbol} ({interval})",
        out_path=html_path,
        metrics=m_series.round(3),
        mc_summary=mc_summary.round(3) if not mc_summary.empty else None,
        image_paths=image_paths,
        extra_tables=extra,
    )

    print(f"Report: {html_path}")
    print(f"Charts: {out_dir}")


if __name__ == "__main__":
    main()
