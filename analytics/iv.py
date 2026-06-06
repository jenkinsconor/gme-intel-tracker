"""
IV Rank, IV Percentile, expected move, and constant-maturity 30-day IV.

Key formulas:
  IVR = (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
  IVP = (# days over past 252 where iv < current_iv) / 252 * 100

tastytrade thresholds (adjusted upward for GME's spike-distorted 52w range):
  Sell premium when IVR > 50 AND IVP > 70 (high conviction)
  Avoid selling when IVR < 30

GME-specific note: the 2021 spike anchors IVR's denominator for years.
IVP is more robust — always require both metrics before flagging high-conviction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class IVMetrics:
    current_iv: float
    ivr: Optional[float]       # 0–100; None if < 20 days of history
    ivp: Optional[float]       # 0–100; None if < 20 days of history
    iv_52w_high: Optional[float]
    iv_52w_low: Optional[float]
    history_days: int
    regime: str                # "high" | "elevated" | "normal" | "low" | "unknown"
    sell_signal: bool          # IVR > 50 AND IVP > 70 — high conviction sell
    cautious_sell: bool        # IVR > 50 OR IVP > 70
    do_not_sell: bool          # IVR < 30 — premium is too cheap; guardrail

    def summary(self) -> str:
        if self.ivr is None:
            return f"IV {self.current_iv*100:.0f}% (building history — {self.history_days}/20 days)"
        return (
            f"IV {self.current_iv*100:.0f}%  |  "
            f"IVR {self.ivr:.0f}  |  IVP {self.ivp:.0f}  |  {self.regime.upper()}"
        )


def compute_iv_metrics(current_iv: float, iv_history_df: pd.DataFrame) -> IVMetrics:
    """
    Compute IVR and IVP from current IV and a DataFrame with columns [date, iv_30d].
    Requires at least 20 data points for meaningful IVR; 252 for full IVP.
    """
    n = len(iv_history_df)

    # Fallback regime from raw IV when history is thin
    raw_regime = "high" if current_iv > 0.80 else ("low" if current_iv < 0.40 else "normal")

    if n < 20:
        return IVMetrics(
            current_iv=current_iv,
            ivr=None, ivp=None,
            iv_52w_high=None, iv_52w_low=None,
            history_days=n,
            regime="unknown",
            sell_signal=False,
            cautious_sell=current_iv > 0.60,
            do_not_sell=current_iv < 0.40,
        )

    history = iv_history_df.tail(252)
    iv_vals = history["iv_30d"].values

    high = float(np.max(iv_vals))
    low = float(np.min(iv_vals))

    ivr: Optional[float] = None
    if high != low:
        ivr = max(0.0, min(100.0, (current_iv - low) / (high - low) * 100))

    ivp = float(np.sum(iv_vals < current_iv) / len(iv_vals) * 100)

    # Regime
    if ivr is not None and ivr >= 70 and ivp >= 70:
        regime = "high"
    elif ivr is not None and ivr >= 50:
        regime = "elevated"
    elif ivr is not None and ivr < 30:
        regime = "low"
    else:
        regime = "normal"

    sell_signal = ivr is not None and ivr > 50 and ivp > 70
    cautious_sell = ivr is not None and (ivr > 50 or ivp > 70)
    do_not_sell = ivr is not None and ivr < 30

    return IVMetrics(
        current_iv=current_iv,
        ivr=ivr, ivp=ivp,
        iv_52w_high=high, iv_52w_low=low,
        history_days=n,
        regime=regime,
        sell_signal=sell_signal,
        cautious_sell=cautious_sell,
        do_not_sell=do_not_sell,
    )


def get_constant_maturity_30d_iv(chain_snapshot) -> Optional[float]:
    """
    Interpolate constant-maturity 30-day IV from the options chain.
    Interpolates in variance space between the two expirations bracketing 30 DTE.
    Uses ATM IV (average of 6 nearest strikes) per expiry.
    """
    today = date.today()
    target_dte = 30

    # Collect ATM IV per expiry
    expiry_data: dict[str, list[float]] = {}
    for c in chain_snapshot.contracts:
        iv = c.implied_volatility
        if iv is None or iv <= 0:
            continue
        try:
            exp_date = date.fromisoformat(c.expiry)
            dte = (exp_date - today).days
            if dte < 5:  # skip very near-term (distorted theta)
                continue
            key = c.expiry
            if key not in expiry_data:
                expiry_data[key] = []
            expiry_data[key].append((abs(c.strike - chain_snapshot.spot), iv))
        except Exception:
            continue

    if not expiry_data:
        return None

    # ATM IV = mean of 6 closest-to-spot strikes per expiry
    points: list[tuple[int, float]] = []  # (dte, atm_iv)
    for exp_str, data in expiry_data.items():
        data.sort(key=lambda x: x[0])
        atm_ivs = [iv for _, iv in data[:6]]
        if atm_ivs:
            try:
                exp_date = date.fromisoformat(exp_str)
                dte = (exp_date - today).days
                points.append((dte, sum(atm_ivs) / len(atm_ivs)))
            except Exception:
                continue

    if not points:
        return None

    points.sort()

    before = [(dte, iv) for dte, iv in points if dte <= target_dte]
    after = [(dte, iv) for dte, iv in points if dte > target_dte]

    if before and after:
        dte1, iv1 = before[-1]
        dte2, iv2 = after[0]
        # Interpolate in variance (IV² × t) — more stable than linear IV interpolation
        var1 = iv1 ** 2 * dte1 / 365
        var2 = iv2 ** 2 * dte2 / 365
        w = (target_dte - dte1) / (dte2 - dte1)
        var_interp = var1 + w * (var2 - var1)
        return math.sqrt(max(var_interp * 365 / target_dte, 1e-6))
    elif after:
        return after[0][1]
    elif before:
        return before[-1][1]

    return None


def compute_expected_move(spot: float, iv: float, dte: int) -> tuple[float, float]:
    """
    1-sigma expected move from IV (68% probability band).
    Returns (lower, upper) absolute price levels.
    """
    em = spot * iv * math.sqrt(dte / 365)
    return spot - em, spot + em


def compute_expected_move_straddle(spot: float, atm_straddle_price: float) -> tuple[float, float]:
    """
    Expected move from ATM straddle price.
    EM ≈ straddle × 0.85  (1-sigma approximation).
    Returns (lower, upper) absolute price levels.
    """
    em = atm_straddle_price * 0.85
    return spot - em, spot + em


def get_atm_straddle(chain_snapshot, expiry: Optional[str] = None) -> Optional[float]:
    """Get ATM straddle price (call + put closest to spot) for given expiry."""
    today = date.today()
    contracts = chain_snapshot.contracts

    if expiry:
        contracts = [c for c in contracts if c.expiry == expiry]
    else:
        future = [c for c in contracts if date.fromisoformat(c.expiry) > today]
        if not future:
            return None
        nearest = min(c.expiry for c in future)
        contracts = [c for c in contracts if c.expiry == nearest]

    if not contracts:
        return None

    spot = chain_snapshot.spot
    calls = [c for c in contracts if c.option_type == "call"]
    puts = [c for c in contracts if c.option_type == "put"]

    if not calls or not puts:
        return None

    atm_call = min(calls, key=lambda c: abs(c.strike - spot))
    atm_put = min(puts, key=lambda c: abs(c.strike - spot))

    call_price = atm_call.mid or atm_call.last
    put_price = atm_put.mid or atm_put.last

    if call_price is None or put_price is None:
        return None

    return call_price + put_price
