"""Snapshot today's news into the local store.

Run once per day. On Windows, schedule via Task Scheduler:

    Program:  C:\\path\\to\\.venv\\Scripts\\python.exe
    Argument: C:\\path\\to\\se\\scripts\\snapshot_news.py
    Trigger:  Daily 18:30 IST (after market close)

Historical note: this only captures news from the day you start running it.
For a true 5-year news history, you'd need a paid archive (NewsAPI.ai,
Benzinga, etc.) or the GDELT GKG dataset. The backtest narrative will mark
pre-snapshot dates as "no news in store" rather than faking them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.config import ensure_dirs
from nse_bot.data.news import snapshot


if __name__ == "__main__":
    ensure_dirs()
    result = snapshot()
    print(
        "Snapshot complete. "
        f"RSS: fetched {result['rss_fetched']}, new {result['rss_new']}. "
        f"NSE: fetched {result['nse_fetched']}, new {result['nse_new']}."
    )
