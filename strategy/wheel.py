"""
Wheel Strategy — state machine, cost-basis ledger, roll suggestions.

The Wheel cycle:
  CSP_OPEN → assignment → SHARES_HELD → CC_OPEN → (CALLED_AWAY | back to SHARES_HELD)
  CSP_OPEN → expired worthless → CYCLE_COMPLETE (cash freed, start again)
  Either leg can be ROLLED at any point.

True adjusted cost basis per share:
  basis = strike_at_assignment
          − Σ(put_premia_received)
          − Σ(call_premia_received)
          + Σ(buyback_costs_paid)

This is the number that drives the covered-call basis guardrail:
  NEVER sell a CC at a strike below adjusted basis unless intentionally exiting at a loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class WheelState(str, Enum):
    CSP_OPEN      = "CSP_OPEN"
    SHARES_HELD   = "SHARES_HELD"
    CC_OPEN       = "CC_OPEN"
    CALLED_AWAY   = "CALLED_AWAY"
    CYCLE_COMPLETE = "CYCLE_COMPLETE"


class LegType(str, Enum):
    PUT_SELL     = "put_sell"      # Open CSP (credit received)
    PUT_BUYBACK  = "put_buyback"   # Close/roll CSP (debit paid)
    PUT_EXPIRED  = "put_expired"   # CSP expired worthless (cycle complete, cash freed)
    ASSIGNMENT   = "assignment"    # Assigned shares (no cash exchange — basis event)
    CALL_SELL    = "call_sell"     # Open CC (credit received)
    CALL_BUYBACK = "call_buyback"  # Close/roll CC (debit paid)
    CALL_EXPIRED = "call_expired"  # CC expired worthless (keep shares)
    CALLED_AWAY  = "called_away"   # Shares called away at strike


# Display-friendly labels and descriptions for the UI form
LEG_LABELS: dict[LegType, str] = {
    LegType.PUT_SELL:     "Sold a put (opened CSP)",
    LegType.PUT_BUYBACK:  "Bought back a put (closed/rolled CSP)",
    LegType.PUT_EXPIRED:  "Put expired worthless (kept premium, no shares)",
    LegType.ASSIGNMENT:   "Got assigned shares",
    LegType.CALL_SELL:    "Sold a call (opened covered call)",
    LegType.CALL_BUYBACK: "Bought back a call (closed/rolled CC)",
    LegType.CALL_EXPIRED: "Call expired worthless (kept premium, kept shares)",
    LegType.CALLED_AWAY:  "Shares called away (sold at strike)",
}

# Leg types that are credits (money in)
CREDIT_LEGS = {LegType.PUT_SELL, LegType.CALL_SELL}
# Leg types that are debits (money out)
DEBIT_LEGS  = {LegType.PUT_BUYBACK, LegType.CALL_BUYBACK}


@dataclass
class WheelLeg:
    id: Optional[int]
    cycle_id: int
    leg_date: date
    leg_type: LegType
    strike: float
    premium_per_share: float    # always positive; sign inferred from leg_type
    contracts: int              # number of option contracts (each = 100 shares)
    expiry: Optional[str]       # YYYY-MM-DD or None for assignment/expiry events
    notes: str = ""

    @property
    def total_premium(self) -> float:
        """Dollar amount of this leg (positive = received, negative = paid)."""
        gross = self.premium_per_share * self.contracts * 100
        if self.leg_type in DEBIT_LEGS:
            return -gross
        return gross

    @property
    def label(self) -> str:
        return LEG_LABELS.get(self.leg_type, self.leg_type.value)


@dataclass
class WheelCycle:
    """
    A full Wheel cycle with all its legs.
    Populated from DB records by WheelStore.
    """
    id: int
    ticker: str
    started_date: date
    closed_date: Optional[date]
    state: WheelState
    legs: list[WheelLeg] = field(default_factory=list)
    notes: str = ""

    # ----------------------------------------------------------------
    # Premium accounting
    # ----------------------------------------------------------------

    @property
    def put_credits(self) -> float:
        return sum(l.total_premium for l in self.legs if l.leg_type == LegType.PUT_SELL)

    @property
    def put_debits(self) -> float:
        return abs(sum(l.total_premium for l in self.legs if l.leg_type == LegType.PUT_BUYBACK))

    @property
    def call_credits(self) -> float:
        return sum(l.total_premium for l in self.legs if l.leg_type == LegType.CALL_SELL)

    @property
    def call_debits(self) -> float:
        return abs(sum(l.total_premium for l in self.legs if l.leg_type == LegType.CALL_BUYBACK))

    @property
    def total_net_premium(self) -> float:
        """Total net premium received across all legs."""
        return sum(l.total_premium for l in self.legs
                   if l.leg_type in (CREDIT_LEGS | DEBIT_LEGS))

    # ----------------------------------------------------------------
    # Assignment / cost basis
    # ----------------------------------------------------------------

    @property
    def assignment_leg(self) -> Optional[WheelLeg]:
        for leg in self.legs:
            if leg.leg_type == LegType.ASSIGNMENT:
                return leg
        return None

    @property
    def was_assigned(self) -> bool:
        return self.assignment_leg is not None

    @property
    def assignment_strike(self) -> Optional[float]:
        leg = self.assignment_leg
        return leg.strike if leg else None

    @property
    def adjusted_basis_per_share(self) -> Optional[float]:
        """
        True cost basis per share after all premium collected.

        Formula:
          basis = strike_at_assignment − put_credits_per_share − call_credits_per_share
                  + put_debits_per_share + call_debits_per_share

        Returns None if not yet assigned.
        """
        if not self.was_assigned:
            return None
        contracts = self.assignment_leg.contracts
        multiplier = contracts * 100

        # Premia per share (normalize by same contract count as assignment)
        put_net = (self.put_credits - self.put_debits) / multiplier
        call_net = (self.call_credits - self.call_debits) / multiplier

        return self.assignment_strike - put_net - call_net

    @property
    def shares_held(self) -> int:
        """Number of shares currently held (0 if not assigned or already called away)."""
        if not self.was_assigned:
            return 0
        if self.state in (WheelState.CALLED_AWAY, WheelState.CYCLE_COMPLETE):
            return 0
        return self.assignment_leg.contracts * 100

    # ----------------------------------------------------------------
    # Active open position
    # ----------------------------------------------------------------

    @property
    def open_put_leg(self) -> Optional[WheelLeg]:
        """The most recently opened (not yet closed) put, or None."""
        if self.state != WheelState.CSP_OPEN:
            return None
        put_sells = [l for l in self.legs if l.leg_type == LegType.PUT_SELL]
        return put_sells[-1] if put_sells else None

    @property
    def open_call_leg(self) -> Optional[WheelLeg]:
        """The most recently opened (not yet closed) call, or None."""
        if self.state != WheelState.CC_OPEN:
            return None
        call_sells = [l for l in self.legs if l.leg_type == LegType.CALL_SELL]
        return call_sells[-1] if call_sells else None

    # ----------------------------------------------------------------
    # P&L (realised at cycle end)
    # ----------------------------------------------------------------

    @property
    def realised_pnl(self) -> Optional[float]:
        """
        Total realised P&L when cycle is complete (CALLED_AWAY or CYCLE_COMPLETE).
        Returns None if cycle is still open.
        """
        if self.state not in (WheelState.CALLED_AWAY, WheelState.CYCLE_COMPLETE):
            return None

        pnl = self.total_net_premium

        # If called away: add the gain/loss on shares vs assignment price
        called = next((l for l in self.legs if l.leg_type == LegType.CALLED_AWAY), None)
        if called and self.assignment_strike:
            pnl += (called.strike - self.assignment_strike) * called.contracts * 100

        return pnl

    # ----------------------------------------------------------------
    # Unrealised P&L (mark-to-market)
    # ----------------------------------------------------------------

    def unrealised_pnl(self, current_price: float) -> float:
        """
        Approximate unrealised P&L at current_price.
        Premium collected so far + unrealised share gain/loss.
        """
        pnl = self.total_net_premium
        if self.was_assigned and self.shares_held > 0 and self.assignment_strike:
            pnl += (current_price - self.assignment_strike) * self.shares_held
        return pnl

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------

    def status_summary(self, current_price: Optional[float] = None) -> str:
        state_labels = {
            WheelState.CSP_OPEN:       "🔵 CSP open — collecting premium, watching for assignment",
            WheelState.SHARES_HELD:    "📦 Shares held — ready to sell covered call",
            WheelState.CC_OPEN:        "🟢 Covered call open — collecting premium",
            WheelState.CALLED_AWAY:    "✅ Shares called away — cycle complete",
            WheelState.CYCLE_COMPLETE: "✅ Put expired worthless — cycle complete, cash freed",
        }
        return state_labels.get(self.state, self.state.value)


# ----------------------------------------------------------------
# State transition logic
# ----------------------------------------------------------------

VALID_TRANSITIONS: dict[WheelState, set[LegType]] = {
    WheelState.CSP_OPEN: {
        LegType.PUT_BUYBACK,  # roll or close
        LegType.PUT_EXPIRED,  # expired worthless → cycle complete
        LegType.ASSIGNMENT,   # assigned → hold shares
        LegType.PUT_SELL,     # re-open after roll
    },
    WheelState.SHARES_HELD: {
        LegType.CALL_SELL,    # open covered call
    },
    WheelState.CC_OPEN: {
        LegType.CALL_BUYBACK,  # roll or close
        LegType.CALL_EXPIRED,  # expired worthless → hold shares again
        LegType.CALLED_AWAY,   # shares called away → cycle complete
        LegType.CALL_SELL,     # re-open after roll
    },
    WheelState.CALLED_AWAY:    set(),   # terminal
    WheelState.CYCLE_COMPLETE: set(),   # terminal
}


def next_state(current: WheelState, leg_type: LegType) -> WheelState:
    """Return the new WheelState after logging leg_type. Raises ValueError if invalid."""
    if leg_type not in VALID_TRANSITIONS.get(current, set()):
        raise ValueError(
            f"Cannot log '{leg_type.value}' when cycle is in state '{current.value}'. "
            f"Valid actions: {[t.value for t in VALID_TRANSITIONS.get(current, [])]}"
        )

    transitions = {
        LegType.PUT_SELL:     WheelState.CSP_OPEN,
        LegType.PUT_BUYBACK:  WheelState.CSP_OPEN,   # stays open (roll) or re-evaluated
        LegType.PUT_EXPIRED:  WheelState.CYCLE_COMPLETE,
        LegType.ASSIGNMENT:   WheelState.SHARES_HELD,
        LegType.CALL_SELL:    WheelState.CC_OPEN,
        LegType.CALL_BUYBACK: WheelState.CC_OPEN,    # stays open (roll) or re-evaluated
        LegType.CALL_EXPIRED: WheelState.SHARES_HELD,
        LegType.CALLED_AWAY:  WheelState.CALLED_AWAY,
    }
    return transitions[leg_type]


# ----------------------------------------------------------------
# Roll suggestions
# ----------------------------------------------------------------

@dataclass
class RollSuggestion:
    action: str
    current_strike: float
    current_expiry: str
    suggested_strike: float
    suggested_expiry: str
    rationale: str
    urgency: str   # "urgent" | "consider" | "watch"


def suggest_roll(cycle: WheelCycle, chain_snapshot, current_price: float) -> Optional[RollSuggestion]:
    """
    Suggest a roll if the open position meets the roll criteria.
    Returns None if no roll needed or no chain data available.
    """
    if chain_snapshot is None:
        return None

    today = date.today()

    if cycle.state == WheelState.CSP_OPEN and cycle.open_put_leg:
        return _suggest_put_roll(cycle.open_put_leg, chain_snapshot, current_price, today)

    if cycle.state == WheelState.CC_OPEN and cycle.open_call_leg:
        return _suggest_call_roll(cycle.open_call_leg, chain_snapshot, current_price, today,
                                   cycle.adjusted_basis_per_share)

    return None


def _suggest_put_roll(
    leg: WheelLeg, chain_snapshot, current_price: float, today: date
) -> Optional[RollSuggestion]:
    try:
        exp_date = date.fromisoformat(leg.expiry)
        dte = (exp_date - today).days
    except Exception:
        return None

    # Find current put in chain to check delta
    current_contracts = [
        c for c in chain_snapshot.contracts
        if c.option_type == "put"
        and abs(c.strike - leg.strike) < 0.5
        and c.expiry == leg.expiry
    ]
    current_delta = None
    if current_contracts:
        c = current_contracts[0]
        current_delta = abs(c.best_greek_delta) if c.best_greek_delta else None

    itm = current_price < leg.strike
    deep_itm = current_delta is not None and current_delta > 0.70
    near_expiry = dte <= 21

    if not (deep_itm or (itm and near_expiry)):
        return None

    urgency = "urgent" if deep_itm and dte <= 7 else "consider"

    # Find a lower-strike, later-expiry put for the roll
    further_expiries = sorted(set(
        c.expiry for c in chain_snapshot.contracts
        if c.expiry > leg.expiry and c.option_type == "put"
    ))
    if not further_expiries:
        return None

    target_expiry = further_expiries[0]
    # Target: 0.25–0.30 delta OTM put in the new expiry
    candidates = [
        c for c in chain_snapshot.contracts
        if c.option_type == "put"
        and c.expiry == target_expiry
        and c.strike < current_price  # OTM
        and c.best_greek_delta is not None
        and 0.20 <= abs(c.best_greek_delta) <= 0.35
    ]
    if not candidates:
        return None

    target = min(candidates, key=lambda c: abs(abs(c.best_greek_delta or 0) - 0.25))

    return RollSuggestion(
        action="Roll put down and out",
        current_strike=leg.strike,
        current_expiry=leg.expiry,
        suggested_strike=target.strike,
        suggested_expiry=target_expiry,
        rationale=(
            f"Current ${leg.strike:.0f} put is {'deep ITM' if deep_itm else 'ITM near expiry'}. "
            f"Roll to ${target.strike:.0f} / {target_expiry} for a lower strike and more time. "
            f"This lowers your potential cost basis if assigned. Always roll for a net credit."
        ),
        urgency=urgency,
    )


def _suggest_call_roll(
    leg: WheelLeg, chain_snapshot, current_price: float, today: date,
    adjusted_basis: Optional[float]
) -> Optional[RollSuggestion]:
    try:
        exp_date = date.fromisoformat(leg.expiry)
        dte = (exp_date - today).days
    except Exception:
        return None

    current_contracts = [
        c for c in chain_snapshot.contracts
        if c.option_type == "call"
        and abs(c.strike - leg.strike) < 0.5
        and c.expiry == leg.expiry
    ]
    current_delta = None
    if current_contracts:
        c = current_contracts[0]
        current_delta = abs(c.best_greek_delta) if c.best_greek_delta else None

    itm = current_price > leg.strike
    deep_itm = current_delta is not None and current_delta > 0.70
    near_expiry = dte <= 21

    if not (deep_itm or (itm and near_expiry)):
        return None

    urgency = "urgent" if deep_itm and dte <= 7 else "consider"

    further_expiries = sorted(set(
        c.expiry for c in chain_snapshot.contracts
        if c.expiry > leg.expiry and c.option_type == "call"
    ))
    if not further_expiries:
        return None

    target_expiry = further_expiries[0]
    # Target: higher strike that's still OTM and above basis
    min_strike = max(current_price, adjusted_basis or 0)
    candidates = [
        c for c in chain_snapshot.contracts
        if c.option_type == "call"
        and c.expiry == target_expiry
        and c.strike > min_strike
        and c.best_greek_delta is not None
        and 0.20 <= abs(c.best_greek_delta) <= 0.40
    ]
    if not candidates:
        return None

    target = min(candidates, key=lambda c: abs(abs(c.best_greek_delta or 0) - 0.30))

    basis_note = ""
    if adjusted_basis and target.strike > adjusted_basis:
        basis_note = f" Strike ${target.strike:.0f} is above your adjusted basis ${adjusted_basis:.2f} ✅"
    elif adjusted_basis and target.strike <= adjusted_basis:
        basis_note = f" ⚠️ Strike ${target.strike:.0f} is still below adjusted basis ${adjusted_basis:.2f} — consider waiting."

    return RollSuggestion(
        action="Roll call up and out",
        current_strike=leg.strike,
        current_expiry=leg.expiry,
        suggested_strike=target.strike,
        suggested_expiry=target_expiry,
        rationale=(
            f"Current ${leg.strike:.0f} call is {'deep ITM' if deep_itm else 'ITM near expiry'}. "
            f"Roll to ${target.strike:.0f} / {target_expiry} to recapture upside."
            + basis_note +
            " Always roll for a net credit."
        ),
        urgency=urgency,
    )
