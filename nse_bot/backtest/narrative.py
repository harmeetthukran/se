"""Link backtest trades and per-day returns to stored news.

Two outputs:
    * `annotate_trades`  — per-trade: news on entry day + news on exit day
                           (symbol-matched when possible, else day-level).
    * `daily_narrative`  — per-day: return % + top headlines + avg sentiment.

Both honestly report "no news in store" for dates without coverage (i.e.
pre-snapshot history) rather than inventing context.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from nse_bot.data import news_store

IST = "Asia/Kolkata"
_MAX_HEADLINES_PER_ROW = 3


def _day(ts: pd.Timestamp) -> date:
    if ts.tzinfo is None:
        ts = ts.tz_localize(IST)
    return ts.tz_convert(IST).date()


def _format_headlines(df: pd.DataFrame, limit: int = _MAX_HEADLINES_PER_ROW) -> str:
    if df is None or df.empty:
        return ""
    ordered = df.assign(abs_sent=df["sentiment"].abs()).sort_values("abs_sent", ascending=False)
    lines = []
    for _, r in ordered.head(limit).iterrows():
        sign = "+" if r["sentiment"] >= 0 else "-"
        lines.append(f"[{sign}{abs(r['sentiment']):.2f}] {r['title']}")
    return " | ".join(lines)


def annotate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Add news-on-entry-day and news-on-exit-day columns to a trade log."""
    if trades is None or trades.empty:
        return trades.copy() if trades is not None else pd.DataFrame()

    out = trades.copy()
    news_all = news_store.read_all()

    if news_all.empty:
        out["entry_news"] = "no news in store"
        out["entry_sentiment"] = 0.0
        out["exit_news"] = "no news in store"
        out["exit_sentiment"] = 0.0
        return out

    news_all = news_all.copy()
    news_all["date"] = news_all["ts"].dt.tz_convert(IST).dt.date

    def _news_for(day_of: date, symbol: str) -> tuple[str, float]:
        day_news = news_all.loc[news_all["date"] == day_of]
        if day_news.empty:
            return "no news in store", 0.0
        if symbol:
            key = symbol.upper()
            mask = (
                day_news["title"].astype(str).str.upper().str.contains(rf"\b{key}\b", regex=True, na=False)
                | day_news["summary"].astype(str).str.upper().str.contains(rf"\b{key}\b", regex=True, na=False)
            )
            matched = day_news.loc[mask]
            if not matched.empty:
                return _format_headlines(matched), float(matched["sentiment"].mean())
        return _format_headlines(day_news), float(day_news["sentiment"].mean())

    entry_news, entry_sent, exit_news, exit_sent = [], [], [], []
    symbols = out.get("symbol", pd.Series([""] * len(out)))

    for (_, row), sym in zip(out.iterrows(), symbols):
        en_day = _day(pd.Timestamp(row["entry_ts"]))
        ex_day = _day(pd.Timestamp(row["exit_ts"]))
        en, es = _news_for(en_day, str(sym) if pd.notna(sym) else "")
        ex, xs = _news_for(ex_day, str(sym) if pd.notna(sym) else "")
        entry_news.append(en)
        entry_sent.append(es)
        exit_news.append(ex)
        exit_sent.append(xs)

    out["entry_news"] = entry_news
    out["entry_sentiment"] = entry_sent
    out["exit_news"] = exit_news
    out["exit_sentiment"] = exit_sent
    return out


def daily_narrative(equity: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Per-trading-day narrative: return + top headlines + sentiment.

    Args:
        equity:    Equity curve indexed by timestamp (any resolution).
        benchmark: Optional benchmark price series (e.g. Nifty close). If
                   provided, its daily % change is added for context.
    """
    if equity is None or equity.empty:
        return pd.DataFrame()

    eq = equity.copy()
    if eq.index.tz is None:
        eq.index = eq.index.tz_localize(IST)
    else:
        eq.index = eq.index.tz_convert(IST)
    daily_eq = eq.resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna() * 100

    start, end = daily_ret.index.min().date(), daily_ret.index.max().date()

    summary = news_store.daily_summary(start, end)
    summary_indexed = summary.set_index("date") if not summary.empty else pd.DataFrame()

    news_all = news_store.read_all()
    news_all = news_all.copy() if not news_all.empty else news_all
    if not news_all.empty:
        news_all["date"] = news_all["ts"].dt.tz_convert(IST).dt.date

    gaps = set(news_store.cover_gaps(start, end))

    rows = []
    bench = None
    if benchmark is not None and not benchmark.empty:
        b = benchmark.copy()
        if b.index.tz is None:
            b.index = b.index.tz_localize(IST)
        else:
            b.index = b.index.tz_convert(IST)
        bench = b.resample("1D").last().pct_change() * 100

    for ts, ret in daily_ret.items():
        d = ts.date()
        row = {
            "date": d,
            "strategy_return_pct": float(ret),
            "benchmark_return_pct": float(bench.loc[ts]) if bench is not None and ts in bench.index else None,
            "news_n": int(summary_indexed.loc[d, "n"]) if d in summary_indexed.index else 0,
            "avg_sentiment": float(summary_indexed.loc[d, "avg_sentiment"]) if d in summary_indexed.index else 0.0,
            "top_headlines": "",
        }
        if d in gaps:
            row["top_headlines"] = "no news in store"
        elif not news_all.empty:
            day_news = news_all.loc[news_all["date"] == d]
            row["top_headlines"] = _format_headlines(day_news)
        rows.append(row)
    return pd.DataFrame(rows)
