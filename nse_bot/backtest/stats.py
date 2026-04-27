"""Statistical-edge tests on a backtest's per-period returns.

Three tools, in order of how much you should worry about them:

1. **Bootstrap Sharpe p-value** — H0: true Sharpe == 0. Resample mean-centered
   returns N times, compute null distribution of Sharpe, return the two-sided
   p-value of the observed Sharpe. p > 0.05: indistinguishable from random.

2. **Probabilistic Sharpe Ratio (PSR)** — Bailey & Lopez de Prado 2012.
   Probability that the *true* Sharpe exceeds a benchmark, given the observed
   Sharpe, sample size, skewness and kurtosis. Accounts for fat tails and
   short histories. PSR(0) > 0.95 ≈ "we're 95% confident this is real."

3. **Deflated Sharpe Ratio (DSR)** — PSR adjusted for multiple testing. If
   you tried N strategies, even random noise will produce some good-looking
   Sharpe; DSR(observed_SR | N) tells you whether the BEST one stands up.
   For a strategy hunt, this is the number that matters.

Inputs:
    rets : 1-D array-like of per-period returns (e.g. daily). Use raw
           returns, not %% — the math is unit-invariant for Sharpe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_inv(p: float, tol: float = 1e-9) -> float:
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return mid
    return 0.5 * (lo + hi)


def sharpe(rets: np.ndarray, periods_per_year: int = 252) -> float:
    a = np.asarray(rets, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    sd = a.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(a.mean() / sd * math.sqrt(periods_per_year))


@dataclass
class EdgeReport:
    n: int
    observed_sharpe: float
    bootstrap_pvalue: float
    psr_zero: float          # PSR(0): P(true SR > 0)
    psr_one: float           # PSR(1): P(true SR > 1)
    skew: float
    kurtosis: float
    deflated_sr: float | None  # DSR if `sr_trials` provided to the runner

    def summary(self) -> str:
        verdict_psr = "real" if self.psr_zero > 0.95 else ("plausible" if self.psr_zero > 0.80 else "noise")
        verdict_p = "significant" if self.bootstrap_pvalue < 0.05 else "not significant"
        line = (
            f"n={self.n}  Sharpe={self.observed_sharpe:.2f}  "
            f"bootstrap_p={self.bootstrap_pvalue:.3f} ({verdict_p})  "
            f"PSR(0)={self.psr_zero:.2f} ({verdict_psr})  "
            f"PSR(1)={self.psr_one:.2f}  "
            f"skew={self.skew:.2f}  kurt={self.kurtosis:.2f}"
        )
        if self.deflated_sr is not None:
            line += f"  DSR={self.deflated_sr:.2f}"
            if self.deflated_sr < 0.5:
                line += "  (likely a multiple-testing artifact)"
        return line


def bootstrap_sharpe_pvalue(
    rets: np.ndarray,
    n_sims: int = 10_000,
    periods_per_year: int = 252,
    seed: int | None = None,
) -> float:
    """Two-sided bootstrap p-value for H0: Sharpe == 0.

    Centers the returns on zero (imposing the null), resamples with
    replacement, computes the null Sharpe distribution, returns the
    fraction of null Sharpes whose absolute value meets or exceeds the
    observed.
    """
    a = np.asarray(rets, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 1.0
    observed = sharpe(a, periods_per_year)
    centered = a - a.mean()
    rng = np.random.default_rng(seed)
    null = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        sample = rng.choice(centered, a.size, replace=True)
        sd = sample.std(ddof=1)
        null[i] = sample.mean() / sd * math.sqrt(periods_per_year) if sd > 0 else 0.0
    return float((np.abs(null) >= abs(observed)).mean())


def probabilistic_sharpe_ratio(
    observed_sr: float,
    n: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    benchmark_sr: float = 0.0,
) -> float:
    """PSR(benchmark): probability that the true (annualized) Sharpe exceeds benchmark.

    Bailey & Lopez de Prado (2012). Inputs are PER-PERIOD Sharpe (not annualized)
    in their original paper; here we follow the common convention of using
    annualized SR throughout, which means n-1 acts as proxy for the effective
    sample size. For honest stats use raw period returns + period SR.
    """
    if n < 2:
        return 0.0
    denom = max(1e-12, 1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr ** 2)
    z = (observed_sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """Expected max of N i.i.d. Sharpe estimates under the null (Gumbel approx).

    Use to set the DSR benchmark: if you tried N strategies and their
    cross-strategy Sharpe variance is sr_var, you'd EXPECT the best one to
    achieve this even if all true Sharpes are zero.
    """
    if n_trials < 2 or sr_var <= 0:
        return 0.0
    euler_gamma = 0.5772156649015329
    z1 = _norm_inv(1.0 - 1.0 / n_trials)
    z2 = _norm_inv(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_var) * ((1 - euler_gamma) * z1 + euler_gamma * z2)


def deflated_sharpe_ratio(
    observed_sr: float,
    n: int,
    sr_trials: np.ndarray,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """DSR: PSR with benchmark = expected max from `len(sr_trials)` i.i.d. trials.

    sr_trials should be the FULL set of Sharpe ratios you computed during
    your strategy hunt (not just the best one). Variance is estimated from it.
    """
    arr = np.asarray(sr_trials, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return probabilistic_sharpe_ratio(observed_sr, n, skew, kurt, 0.0)
    sr_var = float(np.var(arr, ddof=1))
    benchmark = expected_max_sharpe(arr.size, sr_var)
    return probabilistic_sharpe_ratio(observed_sr, n, skew, kurt, benchmark)


def edge_report(
    rets: np.ndarray,
    *,
    periods_per_year: int = 252,
    n_sims: int = 5000,
    sr_trials: np.ndarray | None = None,
    seed: int | None = 7,
) -> EdgeReport:
    a = np.asarray(rets, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    sr = sharpe(a, periods_per_year)
    if n < 2:
        return EdgeReport(n=n, observed_sharpe=0.0, bootstrap_pvalue=1.0,
                          psr_zero=0.0, psr_one=0.0, skew=0.0, kurtosis=3.0,
                          deflated_sr=None)
    sd = a.std(ddof=1)
    if sd > 0:
        z = (a - a.mean()) / sd
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean())
    else:
        skew, kurt = 0.0, 3.0

    pvalue = bootstrap_sharpe_pvalue(a, n_sims=n_sims, periods_per_year=periods_per_year, seed=seed)
    psr_0 = probabilistic_sharpe_ratio(sr, n, skew, kurt, 0.0)
    psr_1 = probabilistic_sharpe_ratio(sr, n, skew, kurt, 1.0)
    dsr = (
        deflated_sharpe_ratio(sr, n, np.asarray(sr_trials, dtype=float), skew, kurt)
        if sr_trials is not None and len(sr_trials) >= 2
        else None
    )
    return EdgeReport(
        n=n,
        observed_sharpe=sr,
        bootstrap_pvalue=pvalue,
        psr_zero=psr_0,
        psr_one=psr_1,
        skew=skew,
        kurtosis=kurt,
        deflated_sr=dsr,
    )
