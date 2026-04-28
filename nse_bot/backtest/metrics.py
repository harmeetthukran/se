"""Performance metrics for a backtest equity curve and trade log."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

# Indian market: ~252 trading days/year. For intraday strategies operating on
# bars, we annualize using trades-per-year from the actual data.
TRADING_DAYS = 252


@dataclass
class Metrics:
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    volatility_pct: float
    trades: int
    win_rate_pct: float
    avg_win_inr: float
    avg_loss_inr: float
    profit_factor: float
    expectancy_inr: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute(
    equity: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
) -> Metrics:
    if equity.empty:
        return _zero()

    equity = equity.astype(float).sort_index()
    rets = equity.pct_change().dropna()
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0

    years = max(1e-9, len(equity) / periods_per_year)
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) * 100.0

    vol = rets.std() * np.sqrt(periods_per_year) * 100.0
    sharpe = (
        float(rets.mean() / rets.std() * np.sqrt(periods_per_year))
        if rets.std() > 0
        else 0.0
    )
    downside = rets[rets < 0]
    sortino = (
        float(rets.mean() / downside.std() * np.sqrt(periods_per_year))
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )

    running_max = equity.cummax()
    dd = (equity / running_max - 1.0).min() * 100.0

    if trades is None or trades.empty:
        return Metrics(
            total_return_pct=float(total_return),
            cagr_pct=float(cagr),
            sharpe=float(sharpe),
            sortino=float(sortino),
            max_drawdown_pct=float(dd),
            volatility_pct=float(vol),
            trades=0,
            win_rate_pct=0.0,
            avg_win_inr=0.0,
            avg_loss_inr=0.0,
            profit_factor=0.0,
            expectancy_inr=0.0,
        )

    pnl = trades["pnl_inr"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    return Metrics(
        total_return_pct=float(total_return),
        cagr_pct=float(cagr),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown_pct=float(dd),
        volatility_pct=float(vol),
        trades=int(len(pnl)),
        win_rate_pct=float((pnl > 0).mean() * 100.0),
        avg_win_inr=float(wins.mean()) if len(wins) else 0.0,
        avg_loss_inr=float(losses.mean()) if len(losses) else 0.0,
        profit_factor=profit_factor,
        expectancy_inr=float(pnl.mean()),
    )


def _zero() -> Metrics:
    return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
