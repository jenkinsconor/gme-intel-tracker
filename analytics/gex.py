"""
Gamma Exposure (GEX) computation for GME.

For GME specifically, GEX is more predictive of intraday behavior than RSI or MAs.
- Positive GEX → dealers are net long gamma → they sell rallies / buy dips → price suppression
- Negative GEX → dealers are net short gamma → they buy rallies / sell dips → price amplification
                                                  (the structural mechanic behind both the 2021 squeeze
                                                   and the May 2024 Roaring Kitty rally)

Formula (per strike, assumes dealers net short all customer-bought options):
  GEX = gamma × OI × 100 × spot² × 0.01

  gamma × OI × 100  =  change in shares dealers hold per $1 spot move
  × spot² × 0.01    =  convert to dollar exposure per 1% spot move

  Net_GEX = Σ(call GEX) − Σ(put GEX)   [calls are positive — dealers long gamma]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class GEXProfile:
    """Gamma exposure metrics derived from the options chain."""
    net_gex: float                    # Total net GEX ($ per 1% spot move)
    call_wall: Optional[float]        # Strike with highest call GEX → resistance
    put_wall: Optional[float]         # Strike with highest put GEX → support
    gamma_flip: Optional[float]       # Price where cumulative GEX crosses zero
    regime: str                       # "positive" | "negative" | "unknown"
    gex_by_strike: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def is_negative(self) -> bool:
        return self.regime == "negative"

    @property
    def squeeze_risk(self) -> bool:
        """
        Squeeze setup: negative GEX regime.
        For premium sellers — negative GEX amplifies moves, so sold premium can
        blow through breakeven fast. Treat as a warning, not a hard stop.
        """
        return self.is_negative

    def regime_label(self) -> str:
        labels = {
            "positive": "Suppressing (dealers buy dips, sell rips)",
            "negative": "Amplifying (dealers sell dips, buy rips) ⚠️",
            "unknown": "Unknown (insufficient chain data)",
        }
        return labels.get(self.regime, self.regime)

    def net_gex_millions(self) -> float:
        return self.net_gex / 1_000_000


def compute_gex(chain_snapshot, spot: Optional[float] = None) -> GEXProfile:
    """
    Compute gamma exposure profile from an OptionsChainSnapshot.

    Uses BSM gamma (bsm_gamma) if available, falls back to provider gamma.
    Sums across all expirations — single-expiry GEX misses the full picture.
    """
    s = spot or chain_snapshot.spot
    if s <= 0:
        return _empty_profile()

    records = []
    for c in chain_snapshot.contracts:
        g = c.bsm_gamma if c.bsm_gamma is not None else c.gamma
        oi = c.open_interest

        if g is None or g <= 0 or oi is None or oi <= 0:
            continue

        raw_gex = g * oi * 100 * (s ** 2) * 0.01
        sign = 1.0 if c.option_type == "call" else -1.0

        records.append({
            "strike": c.strike,
            "expiry": c.expiry,
            "type": c.option_type,
            "gamma": g,
            "oi": oi,
            "signed_gex": raw_gex * sign,
            "call_gex": raw_gex if c.option_type == "call" else 0.0,
            "put_gex": raw_gex if c.option_type == "put" else 0.0,
        })

    if not records:
        return _empty_profile()

    df = pd.DataFrame(records)
    by_strike = (
        df.groupby("strike")
        .agg(
            net_gex=("signed_gex", "sum"),
            call_gex=("call_gex", "sum"),
            put_gex=("put_gex", "sum"),
        )
        .reset_index()
        .sort_values("strike")
    )

    net_gex_total = float(by_strike["net_gex"].sum())

    call_wall = float(by_strike.loc[by_strike["call_gex"].idxmax(), "strike"]) if not by_strike.empty else None
    put_wall = float(by_strike.loc[by_strike["put_gex"].idxmax(), "strike"]) if not by_strike.empty else None

    # Gamma flip: price where cumulative net GEX crosses zero (interpolated)
    by_strike["cumulative_gex"] = by_strike["net_gex"].cumsum()
    gamma_flip = _find_gamma_flip(by_strike)

    regime = "negative" if net_gex_total < 0 else "positive" if net_gex_total > 0 else "unknown"

    return GEXProfile(
        net_gex=net_gex_total,
        call_wall=call_wall,
        put_wall=put_wall,
        gamma_flip=gamma_flip,
        regime=regime,
        gex_by_strike=by_strike,
    )


def _find_gamma_flip(by_strike: pd.DataFrame) -> Optional[float]:
    """Linear interpolation between the two strikes where cumulative GEX crosses zero."""
    rows = by_strike.reset_index(drop=True)
    for i in range(len(rows) - 1):
        g1 = rows.loc[i, "cumulative_gex"]
        g2 = rows.loc[i + 1, "cumulative_gex"]
        if g1 * g2 < 0:  # sign flip
            s1 = rows.loc[i, "strike"]
            s2 = rows.loc[i + 1, "strike"]
            t = abs(g1) / (abs(g1) + abs(g2))
            return float(s1 + t * (s2 - s1))
    return None


def _empty_profile() -> GEXProfile:
    return GEXProfile(
        net_gex=0.0,
        call_wall=None,
        put_wall=None,
        gamma_flip=None,
        regime="unknown",
        gex_by_strike=pd.DataFrame(),
    )
