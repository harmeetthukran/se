"""Matplotlib-based backtest plots. All plots save PNGs and return nothing.

Headless: matplotlib is set to the 'Agg' backend so these work on servers
and CI without a display.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity(
    eq: pd.Series,
    title: str,
    out_path: Path,
    benchmark: pd.Series | None = None,
) -> None:
    if eq is None or eq.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(eq.index, eq.values, label="Strategy", linewidth=1.6, color="#1f77b4")
    if benchmark is not None and not benchmark.empty:
        b = benchmark.reindex(eq.index).ffill().bfill()
        b = b / b.iloc[0] * eq.iloc[0]
        ax.plot(b.index, b.values, label="Benchmark", alpha=0.6, linewidth=1.2, color="#888")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (INR)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_underwater(eq: pd.Series, title: str, out_path: Path) -> None:
    if eq is None or eq.empty:
        return
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.25)
    ax.plot(dd.index, dd.values, color="darkred", linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_monthly_heatmap(eq: pd.Series, title: str, out_path: Path) -> None:
    if eq is None or eq.empty:
        return
    idx = eq.index
    if hasattr(idx, "tz") and idx.tz is not None:
        eq = eq.tz_convert(None) if idx.tz is not None else eq
    monthly = eq.resample("ME").last().pct_change() * 100
    frame = monthly.to_frame("ret").dropna()
    if frame.empty:
        return
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    pivot = frame.pivot(index="year", columns="month", values="ret")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.5 * pivot.shape[0] + 1.5)))
    vmax = max(5.0, np.nanmax(np.abs(pivot.values)))
    im = ax.imshow(pivot, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([month_labels[m - 1] for m in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(y) for y in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if not pd.isna(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > vmax * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Monthly return (%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trade_histogram(trades: pd.DataFrame, title: str, out_path: Path) -> None:
    if trades is None or trades.empty or "return_pct" not in trades.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vals = trades["return_pct"].astype(float).values
    ax.hist(vals, bins=50, color="#1f77b4", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(vals.mean(), color="orange", linestyle="--", label=f"Mean {vals.mean():.2f}%")
    ax.set_title(title)
    ax.set_xlabel("Return per trade (%)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_mc_distributions(
    final_equity: np.ndarray,
    max_drawdown: np.ndarray,
    initial_capital: float,
    title: str,
    out_path: Path,
) -> None:
    if final_equity is None or len(final_equity) == 0:
        return
    rets = (final_equity / initial_capital - 1.0) * 100.0
    dds = max_drawdown * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(rets, bins=60, color="#1f77b4", alpha=0.75, edgecolor="white")
    for pctl, color, label in [(5, "crimson", "5th"), (50, "orange", "Median"), (95, "green", "95th")]:
        axes[0].axvline(np.percentile(rets, pctl), color=color, linestyle="--",
                        label=f"{label} {np.percentile(rets, pctl):.1f}%")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title(f"{title}: Final return distribution")
    axes[0].set_xlabel("Total return (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(dds, bins=60, color="crimson", alpha=0.7, edgecolor="white")
    for pctl, color, label in [(5, "darkred", "5th (worst)"), (50, "orange", "Median"), (95, "green", "95th")]:
        axes[1].axvline(np.percentile(dds, pctl), color=color, linestyle="--",
                        label=f"{label} {np.percentile(dds, pctl):.1f}%")
    axes[1].set_title(f"{title}: Max drawdown distribution")
    axes[1].set_xlabel("Max drawdown (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
