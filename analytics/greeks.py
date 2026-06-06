"""
Self-computed BSM Greeks via py_vollib.
Uses Peter Jäckel's "Let's Be Rational" — near-machine-precision IV inversion.

Why self-compute instead of trusting provider Greeks?
  - Single auditable assumption set (risk-free rate, dividend handling)
  - Enables "what-if" Greeks at hypothetical IVs (IV-crush sensitivity)
  - Unit-testable math
  - yfinance provides no Greeks at all

Graceful degradation: if py_vollib is not installed, all functions return
empty dicts / None — the app continues with provider Greeks or no Greeks.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Optional

try:
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
    from py_vollib.black_scholes.implied_volatility import implied_volatility

    PY_VOLLIB_AVAILABLE = True
except ImportError:
    PY_VOLLIB_AVAILABLE = False

# Approximate current Fed funds rate — update as needed
RISK_FREE_RATE = 0.0425  # 4.25%

# Minimum time to expiry to avoid division by zero (1 hour expressed in years)
MIN_T = 1 / (365 * 24)


def compute_greeks(
    flag: str,  # 'c' or 'p'
    spot: float,
    strike: float,
    dte: int,
    iv: float,
    r: float = RISK_FREE_RATE,
) -> dict:
    """
    Compute BSM Greeks for a single contract.
    Returns dict with bsm_delta, bsm_gamma, bsm_theta (per day), bsm_vega (per 1% IV).
    Returns {} if py_vollib unavailable or inputs invalid.
    """
    if not PY_VOLLIB_AVAILABLE:
        return {}
    if iv <= 0 or spot <= 0 or strike <= 0:
        return {}

    t = max(dte / 365.0, MIN_T)

    try:
        d = delta(flag, spot, strike, t, r, iv)
        g = gamma(flag, spot, strike, t, r, iv)
        th = theta(flag, spot, strike, t, r, iv) / 365  # → per calendar day
        v = vega(flag, spot, strike, t, r, iv) / 100    # → per 1% IV move

        # Sanity bounds
        if not (-1.01 <= d <= 1.01):
            return {}

        return {
            "bsm_delta": round(d, 4),
            "bsm_gamma": round(g, 6),
            "bsm_theta": round(th, 4),
            "bsm_vega": round(v, 4),
        }
    except Exception:
        return {}


def compute_iv_from_price(
    flag: str,
    price: float,
    spot: float,
    strike: float,
    dte: int,
    r: float = RISK_FREE_RATE,
) -> Optional[float]:
    """Back-solve IV from option market price. Returns None if impossible."""
    if not PY_VOLLIB_AVAILABLE or price <= 0:
        return None

    t = max(dte / 365.0, MIN_T)

    try:
        iv = implied_volatility(price, spot, strike, t, r, flag)
        return iv if 0 < iv < 5 else None
    except Exception:
        return None


def enrich_chain_with_greeks(contracts: list, spot: float, r: float = RISK_FREE_RATE) -> list:
    """
    Compute and attach BSM Greeks (bsm_delta, bsm_gamma, bsm_theta, bsm_vega)
    to each OptionContract in place. Uses the contract's implied_volatility.
    Silently skips contracts with missing/invalid IV.
    """
    if not PY_VOLLIB_AVAILABLE:
        return contracts

    today = date.today()

    for c in contracts:
        iv = c.implied_volatility
        if iv is None or iv <= 0:
            # Try to back-solve IV from mid price
            if c.mid and c.mid > 0:
                try:
                    exp_date = date.fromisoformat(c.expiry)
                    dte = max((exp_date - today).days, 0)
                    flag = "c" if c.option_type == "call" else "p"
                    iv = compute_iv_from_price(flag, c.mid, spot, c.strike, dte, r)
                    if iv:
                        c.implied_volatility = iv
                except Exception:
                    pass
            if iv is None or iv <= 0:
                continue

        try:
            exp_date = date.fromisoformat(c.expiry)
            dte = max((exp_date - today).days, 0)
        except Exception:
            continue

        flag = "c" if c.option_type == "call" else "p"
        greeks = compute_greeks(flag, spot, c.strike, dte, iv, r)
        for k, v in greeks.items():
            setattr(c, k, v)

    return contracts


def greek_discrepancy(provider_val: Optional[float], bsm_val: Optional[float]) -> Optional[float]:
    """Return absolute % difference between provider and BSM Greek. None if either missing."""
    if provider_val is None or bsm_val is None or provider_val == 0:
        return None
    return abs((bsm_val - provider_val) / provider_val) * 100
