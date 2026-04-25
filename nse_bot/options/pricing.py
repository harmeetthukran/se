"""Black-Scholes option pricing + Greeks, stdlib-only.

All inputs in consistent units:
    S      underlying spot price
    K      strike
    T      time to expiry in YEARS (e.g. 7 days = 7/365)
    r      risk-free rate as decimal (e.g. 0.065 for 6.5%%)
    sigma  implied (or realized) volatility as decimal (e.g. 0.18 for 18%%)
    opt    "CE" (call) or "PE" (put)

Assumptions:
    * European-style exercise (all listed Indian index/stock options are
      European — aligned with the model).
    * No dividends (set a dividend yield q by replacing S with S*exp(-qT)
      if you need it; Nifty is treated as a total-return index, so q ≈ 0).

Greeks returned in their usual units: delta in [-1,1], gamma per unit spot,
vega per 1.00 (100%%) vol change, theta per YEAR (divide by 365 for per-day),
rho per 1.00 rate change.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def price(S: float, K: float, T: float, r: float, sigma: float, opt: str = "CE") -> float:
    if T <= 0.0:
        return max(S - K, 0.0) if opt.upper() == "CE" else max(K - S, 0.0)
    if sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if opt.upper() == "CE":
        return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def greeks(S: float, K: float, T: float, r: float, sigma: float, opt: str = "CE") -> Greeks:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        intr = max(S - K, 0.0) if opt.upper() == "CE" else max(K - S, 0.0)
        return Greeks(price=intr, delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    pdf = _norm_pdf(d1)
    if opt.upper() == "CE":
        p = S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta = -S * pdf * sigma / (2.0 * sqrt(T)) - r * K * exp(-r * T) * _norm_cdf(d2)
        rho = K * T * exp(-r * T) * _norm_cdf(d2)
    else:
        p = K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
        theta = -S * pdf * sigma / (2.0 * sqrt(T)) + r * K * exp(-r * T) * _norm_cdf(-d2)
        rho = -K * T * exp(-r * T) * _norm_cdf(-d2)
    gamma = pdf / (S * sigma * sqrt(T))
    vega = S * pdf * sqrt(T)
    return Greeks(price=p, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    opt: str = "CE",
    *,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> float:
    """Solve for sigma given market_price via Newton-Raphson; fallback to bisection."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0
    intrinsic = max(S - K, 0.0) if opt.upper() == "CE" else max(K - S, 0.0)
    if market_price < intrinsic:
        return 0.0

    sigma = 0.3
    for _ in range(max_iter):
        g = greeks(S, K, T, r, sigma, opt)
        diff = g.price - market_price
        if abs(diff) < tol:
            return sigma
        if g.vega <= 0:
            break
        sigma -= diff / g.vega
        if sigma <= 0.0 or sigma > 5.0:
            break

    lo, hi = 1e-4, 5.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if price(S, K, T, r, mid, opt) > market_price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return mid
    return 0.5 * (lo + hi)
