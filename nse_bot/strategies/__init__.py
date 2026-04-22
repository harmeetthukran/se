"""Strategy registry."""
from nse_bot.strategies.base import Strategy, StrategyResult
from nse_bot.strategies.orb import ORBStrategy
from nse_bot.strategies.vwap_revert import VWAPRevertStrategy
from nse_bot.strategies.momentum_daily import DailyMomentumStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "orb": ORBStrategy,
    "vwap_revert": VWAPRevertStrategy,
    "momentum_daily": DailyMomentumStrategy,
}

__all__ = ["Strategy", "StrategyResult", "REGISTRY"]
