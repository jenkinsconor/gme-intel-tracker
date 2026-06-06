"""
Probability of Profit (POP), Probability of Touch, and Expected Value.

Key formulas:
  POP for short option ≈ 1 − |delta|   (standard retail approximation)
  Prob of touch        ≈ 2 × |delta|   (chance of touching strike at any point before expiry)
  EV = POP × max_profit − (1 − POP) × max_loss

  Note: EV beats POP alone. A 75% POP with 1:4 win/loss ratio is negative EV.
  Always display both.

ROC formulas:
  CSP:  ROC = premium / (strike × 100)          → annualized = ROC × (365/DTE)
  CC:   ROC = premium / (cost_basis × 100)       → use adjusted Wheel basis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeMetrics:
    """Full metrics for a single short-option trade."""
    pop: float                  # Probability of max profit (0–1)
    prob_of_touch: float        # Probability of touching strike (0–1)
    max_profit_per_contract: float    # Premium collected × 100 (dollars)
    max_loss_per_contract: float      # Max loss (dollars, positive value)
    expected_value: float             # EV per contract (dollars)
    annualized_roc_pct: float         # Annualized return on capital (%)
    capital_at_risk: float            # Cash required (CSP) or share value (CC)

    def pop_pct(self) -> str:
        return f"{self.pop * 100:.0f}%"

    def ev_str(self) -> str:
        return f"${self.expected_value:+.0f}"

    def roc_str(self) -> str:
        return f"{self.annualized_roc_pct:.0f}% ann."

    def is_positive_ev(self) -> bool:
        return self.expected_value > 0

    def summary(self) -> str:
        ev_icon = "✅" if self.is_positive_ev() else "❌"
        return (
            f"POP {self.pop_pct()}  |  "
            f"Touch {self.prob_of_touch*100:.0f}%  |  "
            f"EV {self.ev_str()} {ev_icon}  |  "
            f"Ann. ROC {self.roc_str()}"
        )


def compute_short_put_metrics(
    delta: float,        # option delta (negative for puts — pass absolute value or signed)
    premium: float,      # credit received per share (midpoint or bid)
    strike: float,       # put strike
    spot: float,
    dte: int,
) -> TradeMetrics:
    """
    Metrics for a cash-secured put.
    Max loss = assigned shares at $0 (theoretical) = strike − premium per share × 100.
    """
    abs_delta = abs(delta)
    pop = 1.0 - abs_delta
    prob_touch = min(2.0 * abs_delta, 1.0)

    max_profit = premium * 100
    max_loss = (strike - premium) * 100    # if assigned and stock goes to $0
    capital = strike * 100                  # cash required to secure the put

    ev = pop * max_profit - (1 - pop) * max_loss
    roc_cycle = max_profit / capital if capital > 0 else 0
    annualized_roc = roc_cycle * (365 / max(dte, 1)) * 100

    return TradeMetrics(
        pop=pop,
        prob_of_touch=prob_touch,
        max_profit_per_contract=max_profit,
        max_loss_per_contract=max_loss,
        expected_value=ev,
        annualized_roc_pct=annualized_roc,
        capital_at_risk=capital,
    )


def compute_covered_call_metrics(
    delta: float,
    premium: float,       # credit received per share
    strike: float,
    spot: float,
    dte: int,
    cost_basis: Optional[float] = None,   # adjusted Wheel basis per share
) -> TradeMetrics:
    """
    Metrics for a covered call.
    Max profit = premium collected per contract.
    "Max loss" here = if shares called away below cost basis (locking in basis loss).
    """
    abs_delta = abs(delta)
    pop = 1.0 - abs_delta
    prob_touch = min(2.0 * abs_delta, 1.0)

    max_profit = premium * 100
    basis = cost_basis or spot

    # If strike < basis, being called away locks in a realized loss on shares
    if strike < basis:
        max_loss = (basis - strike) * 100   # forced loss if assigned
    else:
        max_loss = 0.0  # called away above basis = still profitable overall

    capital = basis * 100   # value of shares held
    ev = pop * max_profit - (1 - pop) * max_loss
    roc_cycle = max_profit / capital if capital > 0 else 0
    annualized_roc = roc_cycle * (365 / max(dte, 1)) * 100

    return TradeMetrics(
        pop=pop,
        prob_of_touch=prob_touch,
        max_profit_per_contract=max_profit,
        max_loss_per_contract=max_loss,
        expected_value=ev,
        annualized_roc_pct=annualized_roc,
        capital_at_risk=capital,
    )
