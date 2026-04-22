"""Run the Upstox OAuth flow and save the access token to .env."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_bot.data.auth import run_oauth_flow


if __name__ == "__main__":
    run_oauth_flow()
