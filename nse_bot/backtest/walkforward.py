"""Walk-forward parameter tuning + out-of-sample validation.

For each rolling (train, test) window pair:
    1. Grid-search the strategy's param_grid on the train window.
    2. Pick the best parameter set by `metric` (default: Sharpe).
    3. Evaluate that parameter set on the immediately-following test window.
    4. Record both in-sample and out-of-sample metrics.

The output frame has one row per fold. If OOS metrics are systematically
much worse than IS (common!), the strategy is overfitting — take its
historical-summary Sharpe as illustrative, not predictive.

Usage:
    python scripts/walk_forward.py --strategy orb --symbol RELIANCE \\
        --train-bars 252 --test-bars 63
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import pandas as pd

from nse_bot.backtest import engine, metrics
from nse_bot.backtest.costs import CostConfig
from nse_bot.strategies.base import Strategy


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    combined_oos_equity: pd.Series


def _metric(m: metrics.Metrics, name: str) -> float:
    mapping = {
        "sharpe": m.sharpe,
        "cagr": m.cagr_pct,
        "total_return": m.total_return_pct,
        "profit_factor": m.profit_factor if m.profit_factor != float("inf") else 0.0,
    }
    return float(mapping.get(name, m.sharpe))


def _expand(grid: dict[str, list]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    return [dict(zip(keys, c)) for c in combos]


def _run_once(
    strat_cls: type[Strategy],
    params: dict,
    df: pd.DataFrame,
    capital: float,
    segment: str,
    *,
    warmup_df: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame, metrics.Metrics]:
    """Run the strategy + engine on `df`.

    When `warmup_df` is provided, it is concatenated in front of df so the
    strategy's rolling indicators have enough history, but the engine only
    runs on the df portion. Used for out-of-sample walk-forward folds.
    """
    strat = strat_cls(**params)
    if warmup_df is not None and not warmup_df.empty:
        full = pd.concat([warmup_df, df], ignore_index=True)
        result = strat.generate(full)
        test_slice = slice(len(warmup_df), len(full))
        signal = result.signal.iloc[test_slice].reset_index(drop=True)
        eval_df = df.reset_index(drop=True)
    else:
        result = strat.generate(df)
        signal = result.signal
        eval_df = df

    cfg = engine.BacktestConfig(
        initial_capital=capital,
        cost=CostConfig(segment=segment),
        intraday_squareoff=result.intraday,
    )
    eq, trades = engine.run(eval_df, signal, cfg)
    periods_per_year = 252 * 375 if strat.interval in {"1minute", "5minute", "15minute", "30minute"} else 252
    m = metrics.compute(eq, trades, periods_per_year=periods_per_year)
    return eq, trades, m


def walk_forward(
    strat_cls: type[Strategy],
    df: pd.DataFrame,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    capital: float = 100_000.0,
    metric: str = "sharpe",
    param_grid: dict[str, list] | None = None,
) -> WalkForwardResult:
    """Run the walk-forward loop over df.

    Args:
        strat_cls:  Strategy class (not instance).
        df:         OHLCV frame; must be long enough for at least one fold.
        train_bars: Bars in each training window.
        test_bars:  Bars in each out-of-sample test window.
        step_bars:  Step between folds (default = test_bars → non-overlapping).
        capital:    Initial capital per fold (reset each time).
        metric:     Metric used to rank params on train. sharpe|cagr|total_return|profit_factor.
        param_grid: Override the strategy's default grid.
    """
    if step_bars is None:
        step_bars = test_bars

    grid = _expand(param_grid or strat_cls.param_grid())
    intraday_guess = strat_cls().generate(df.iloc[:0]).intraday if len(df) > 0 else False
    segment = "equity_intraday" if intraday_guess else "equity_delivery"

    folds = []
    oos_segments: list[pd.Series] = []
    i = 0
    fold_id = 0
    while i + train_bars + test_bars <= len(df):
        train = df.iloc[i : i + train_bars].reset_index(drop=True)
        test = df.iloc[i + train_bars : i + train_bars + test_bars].reset_index(drop=True)

        best = None
        for params in grid:
            try:
                _eq, _tr, m_train = _run_once(strat_cls, params, train, capital, segment)
            except Exception:
                continue
            score = _metric(m_train, metric)
            if best is None or score > best[0]:
                best = (score, params, m_train)

        if best is None:
            i += step_bars
            continue

        chosen_params = best[1]
        m_train = best[2]

        try:
            eq_oos, _tr_oos, m_test = _run_once(
                strat_cls, chosen_params, test, capital, segment, warmup_df=train
            )
            oos_segments.append(eq_oos)
        except Exception:
            m_test = metrics.compute(pd.Series(dtype=float), pd.DataFrame())

        folds.append(
            {
                "fold": fold_id,
                "train_start": train["ts"].iloc[0],
                "train_end": train["ts"].iloc[-1],
                "test_start": test["ts"].iloc[0],
                "test_end": test["ts"].iloc[-1],
                "params": chosen_params,
                "is_sharpe": m_train.sharpe,
                "is_cagr_pct": m_train.cagr_pct,
                "is_total_return_pct": m_train.total_return_pct,
                "is_max_dd_pct": m_train.max_drawdown_pct,
                "is_trades": m_train.trades,
                "oos_sharpe": m_test.sharpe,
                "oos_cagr_pct": m_test.cagr_pct,
                "oos_total_return_pct": m_test.total_return_pct,
                "oos_max_dd_pct": m_test.max_drawdown_pct,
                "oos_trades": m_test.trades,
            }
        )
        fold_id += 1
        i += step_bars

    folds_df = pd.DataFrame(folds)
    if oos_segments:
        combined = pd.concat(oos_segments).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = pd.Series(dtype=float)
    return WalkForwardResult(folds=folds_df, combined_oos_equity=combined)
