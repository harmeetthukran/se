"""Runtime configuration loaded from environment / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

# Load .env from project root if present.
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Config:
    api_key: str
    api_secret: str
    redirect_uri: str
    access_token: str
    capital_inr: float

    @property
    def has_access_token(self) -> bool:
        return bool(self.access_token)


def load_config() -> Config:
    return Config(
        api_key=os.getenv("UPSTOX_API_KEY", "").strip(),
        api_secret=os.getenv("UPSTOX_API_SECRET", "").strip(),
        redirect_uri=os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:5555/callback").strip(),
        access_token=os.getenv("UPSTOX_ACCESS_TOKEN", "").strip(),
        capital_inr=float(os.getenv("CAPITAL_INR", "50000")),
    )


def ensure_dirs() -> None:
    for d in (DATA_DIR, REPORTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
