"""Strategy base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyResult:
    signal: pd.Series  # +1 long / -1 short / 0 flat, aligned to df index
    intraday: bool = False  # engine should force-flat at session close


class Strategy(ABC):
    name: str = "base"
    interval: str = "day"  # preferred candle interval

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> StrategyResult:
        """Produce a signal series aligned to df (OHLCV with ts column)."""

    @classmethod
    def param_grid(cls) -> dict[str, list]:
        """Return the default parameter grid for walk-forward tuning.

        Override in subclasses. Empty grid = no tuning (fixed params).
        """
        return {}
