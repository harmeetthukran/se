"""One-shot diagnostic for the Upstox candle fetch.

Prints whether the token is set, looks up RELIANCE in the universe,
hits Upstox v2 historical-candle directly, and shows the raw response.

Usage:
    python scripts/_debug_fetch.py
"""
from __future__ import annotations

import sys
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from nse_bot.config import load_config
from nse_bot.data import universe


def main() -> None:
    cfg = load_config()
    print(f"Token set: {bool(cfg.access_token)}  (length: {len(cfg.access_token)})")

    inst = universe.nse_equity()
    print(f"NSE equity universe rows: {len(inst)}")

    rel = inst.loc[inst["tradingsymbol"].astype(str).str.upper() == "RELIANCE"]
    print(f"Rows matching RELIANCE: {len(rel)}")
    if rel.empty:
        print("ERROR: RELIANCE not present in universe — universe filter may still be wrong.")
        return
    key = str(rel.iloc[0]["instrument_key"])
    print(f"instrument_key: {key}")

    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=180)
    ik = urllib.parse.quote(key, safe="")

    print()
    print("--- v2 endpoint ---")
    url_v2 = f"https://api.upstox.com/v2/historical-candle/{ik}/day/{end}/{start}"
    print(f"GET {url_v2}")
    try:
        r = httpx.get(
            url_v2,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {cfg.access_token}",
            },
            timeout=30,
        )
        print(f"Status: {r.status_code}")
        print(f"Body (first 800 chars):\n{r.text[:800]}")
    except Exception as e:
        print(f"v2 request failed: {e}")

    print()
    print("--- v3 endpoint ---")
    # Upstox v3 historical: /v3/historical-candle/{key}/{unit}/{interval}/{to}/{from}
    url_v3 = f"https://api.upstox.com/v3/historical-candle/{ik}/days/1/{end}/{start}"
    print(f"GET {url_v3}")
    try:
        r = httpx.get(
            url_v3,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {cfg.access_token}",
            },
            timeout=30,
        )
        print(f"Status: {r.status_code}")
        print(f"Body (first 800 chars):\n{r.text[:800]}")
    except Exception as e:
        print(f"v3 request failed: {e}")


if __name__ == "__main__":
    main()
