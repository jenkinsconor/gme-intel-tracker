"""
Signal engine — composes analytics into actionable trade suggestions.

Guardrails (in priority order):
  1. Catalyst gate hard block (earnings < 7 DTE)  — suppress everything
  2. IVR < 30 guardrail                           — premium too cheap
  3. GEX negative warning                         — amplifying regime
  4. Wheel basis guardrail (Phase 2)              — don't sell CC below basis
"""
from __future__ import annotations

from datetime import date
from typing import Optional, TYPE_CHECKING

from analytics.iv import IVMetrics
from analytics.gex import GEXProfile
from analytics.pop import compute_short_put_metrics, compute_covered_call_metrics

if TYPE_CHECKING:
    from sentiment.catalyst import CatalystGate


def build_signals(
    df,
    sr_levels: list,
    iv_metrics: Optional[IVMetrics] = None,
    gex: Optional[GEXProfile] = None,
    chain_snapshot=None,
    cost_basis: Optional[float] = None,
    catalyst_gate: Optional["CatalystGate"] = None,
) -> tuple[list[dict], dict]:
    """
    Returns (signals, context_dict).
    """
    latest = df.iloc[-1]
    price = float(latest["Close"])
    rsi = float(latest["RSI"])
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    vol_spike = bool(latest["vol_spike"])

    p = lambda x: f"${x:.2f}"

    supports = sorted(
        [l for l in sr_levels if l.kind in ("support", "mixed")],
        key=lambda x: x.level, reverse=True
    )
    resistances = sorted(
        [l for l in sr_levels if l.kind in ("resistance", "mixed")],
        key=lambda x: x.level
    )
    nearest_sup = supports[0].level if supports else None
    nearest_res = resistances[0].level if resistances else None

    # ---- Trend ----
    if price > ma20 > ma50:
        trend, trend_note = "bullish", f"Price {p(price)} > MA20 {p(ma20)} > MA50 {p(ma50)}"
    elif price < ma20 < ma50:
        trend, trend_note = "bearish", f"Price {p(price)} < MA20 {p(ma20)} < MA50 {p(ma50)}"
    else:
        trend, trend_note = "neutral", f"Price {p(price)} between MAs — choppy/consolidating"

    rsi_zone = "oversold" if rsi < 35 else ("overbought" if rsi > 65 else "neutral")

    # ---- IV regime ----
    iv_regime = "unknown"
    if iv_metrics:
        iv_regime = iv_metrics.regime
    elif latest.get("iv_raw") is not None:
        iv_raw = float(latest["iv_raw"])
        iv_regime = "high" if iv_raw > 0.80 else ("low" if iv_raw < 0.40 else "normal")

    # ---- Guardrail evaluation ----
    guardrail_no_sell = iv_metrics is not None and iv_metrics.do_not_sell
    guardrail_gex_warning = gex is not None and gex.is_negative

    signals: list[dict] = []

    # ================================================================
    # 0. Situation assessment
    # ================================================================
    if trend == "bearish" and rsi_zone != "oversold":
        situation, action = "wait", "Not ideal to enter new positions right now"
        msg = (
            f"Price is in a **downtrend** (below MA20 and MA50) and RSI is not yet oversold. "
            f"This is not a great time to open new positions — you'd be fighting the trend. "
            f"**If you hold 100 shares**, a covered call is still valid to collect income while waiting."
        )
    elif trend == "bearish" and rsi_zone == "oversold":
        situation, action = "watch", "Oversold — watch for reversal signal"
        msg = (
            f"Price is in a downtrend BUT RSI is oversold (< 35) — selling may be exhausted. "
            f"Cautious watch zone. Wait for price to stop making lower lows before entering. "
            f"A LEAP could make sense here if you believe in the long-term thesis and IV is low."
        )
    elif trend == "neutral":
        situation, action = "ok", "Sideways market — good for premium selling"
        msg = (
            f"Price is choppy — not clearly trending. This is actually a good environment for "
            f"**premium selling** (covered calls, CSPs). You collect income while the stock moves sideways."
        )
    else:
        situation, action = "good", "Uptrend — good conditions overall"
        msg = (
            f"Price is in an **uptrend** (above both MA20 and MA50). "
            f"Good environment for covered calls above resistance or CSPs below support."
        )

    # Append GEX context to situation
    if gex and gex.regime != "unknown":
        if gex.is_negative:
            msg += (
                f"\n\n⚠️ **GEX regime: NEGATIVE** — dealers are net short gamma. "
                f"Moves are being *amplified*, not suppressed. Heightened assignment risk for short premium. "
                f"Gamma flip: **{p(gex.gamma_flip)}**" if gex.gamma_flip else ""
            )
        else:
            msg += (
                f"\n\n🟢 **GEX regime: POSITIVE** — dealers are net long gamma. "
                f"Price tends to be range-bound near high-OI strikes. "
                f"Good environment for premium selling."
            )

    signals.append({
        "type": "📍 Situation",
        "action": action,
        "why": msg,
        "when": "",
        "risk": "—",
        "beginner_note": "",
        "is_situation": True,
        "situation_key": situation,
    })

    # ================================================================
    # 1a. Catalyst Gate hard block (highest priority guardrail)
    # ================================================================
    if catalyst_gate is not None and catalyst_gate.hard_block:
        for reason in catalyst_gate.hard_block_reasons:
            signals.append({
                "type": "🚫 Catalyst Gate — HARD BLOCK",
                "action": reason,
                "why": (
                    "All premium-sell suggestions are suppressed until this resolves. "
                    "The premium you'd collect right now is the market pricing known event risk — "
                    "not free money. After the event, IV will crush and you can sell into "
                    "the elevated post-event IV from a position of information."
                ),
                "when": "Gate auto-lifts when the catalyst condition clears.",
                "risk": "—",
                "beginner_note": (
                    "Selling puts or calls into earnings is like selling fire insurance "
                    "while the building is already smoking. Premium looks great — because it is great — "
                    "for the buyer, not for you."
                ),
                "is_guardrail": True,
            })
        guardrail_no_sell = True  # suppress all downstream strategy signals

    # ================================================================
    # 1b. IV Guardrail (fires before any premium-sell suggestions)
    # ================================================================
    if guardrail_no_sell:
        signals.append({
            "type": "🚫 IV Guardrail Active",
            "action": f"IVR {iv_metrics.ivr:.0f} < 30 — premium is too cheap to sell",
            "why": (
                f"IV Rank is **{iv_metrics.ivr:.0f}** (below 30 threshold). "
                f"Current IV {iv_metrics.current_iv*100:.0f}% is near its 52-week low "
                f"({iv_metrics.iv_52w_low*100:.0f}%). "
                f"Selling premium when IV is this low means collecting little cash while still taking full assignment risk. "
                f"Wait for IVR > 50 before opening covered calls or CSPs."
            ),
            "when": "This guardrail auto-clears when IVR rises above 30.",
            "risk": "—",
            "beginner_note": "Think of IV like insurance pricing. When premiums are cheap, don't be the insurer. Wait until options are expensive again.",
            "is_guardrail": True,
        })

    # ================================================================
    # 2. Covered Call
    # ================================================================
    if nearest_res and not guardrail_no_sell:
        cc_strike = round(nearest_res * 1.02, 0)

        # Cost basis guardrail: never suggest CC below adjusted basis
        basis_warning = ""
        if cost_basis and cc_strike < cost_basis:
            basis_warning = (
                f"\n\n🚫 **Basis guardrail:** suggested strike {p(cc_strike)} is below your "
                f"adjusted Wheel basis {p(cost_basis)}. Accepting assignment here locks in a loss. "
                f"Consider waiting for a higher strike or accepting the basis loss intentionally."
            )

        cc_why = (
            f"Resistance at {p(nearest_res)} — price tends to stall there. "
            f"Sell a call just above it at {p(cc_strike)}. You collect premium upfront. "
            f"If GME stays below your strike at expiry, you keep the premium AND your 100 shares."
        )
        if trend == "bearish":
            cc_why = (
                f"Even in a downtrend, if you hold 100 GME shares you can sell a covered call to collect income. "
                f"Resistance at {p(nearest_res)} — sell just above at {p(cc_strike)}."
            )

        # POP / EV from best available chain data
        pop_note = _get_pop_note_cc(chain_snapshot, cc_strike, price, cost_basis)

        signals.append({
            "type": "🔵 Covered Call",
            "action": f"Sell the {p(cc_strike)} call — need 100 shares as collateral",
            "why": cc_why + basis_warning + (f"\n\n{pop_note}" if pop_note else ""),
            "when": "2–4 weeks to expiry (fastest theta decay). Strike above nearest resistance.",
            "risk": "Low",
            "beginner_note": (
                f"Max profit = premium collected. Your only 'loss' is if GME rockets past "
                f"your strike and shares get called away — you still made money, just missed extra upside."
            ),
        })

    # ================================================================
    # 3. Cash-Secured Put
    # ================================================================
    if nearest_sup and not guardrail_no_sell:
        if trend != "bearish":
            csp_strike = round(nearest_sup * 0.97, 0)

            gex_csp_warning = ""
            if guardrail_gex_warning:
                gex_csp_warning = (
                    f"\n\n⚠️ **GEX warning:** negative GEX regime active. If price drops, "
                    f"dealer hedging *amplifies* the move — your CSP could go deep ITM quickly. "
                    f"Size conservatively or wait for positive GEX."
                )

            pop_note = _get_pop_note_csp(chain_snapshot, csp_strike, nearest_sup)

            signals.append({
                "type": "🔵 Cash-Secured Put",
                "action": f"Sell the {p(csp_strike)} put — need ${csp_strike*100:,.0f} cash set aside",
                "why": (
                    f"Support at {p(nearest_sup)}. Sell a put just below at {p(csp_strike)}. "
                    f"If GME stays above that, you keep the cash. "
                    f"If it drops below, you own 100 shares at {p(csp_strike)} — a price below today's support."
                    + gex_csp_warning
                    + (f"\n\n{pop_note}" if pop_note else "")
                ),
                "when": "Only open if you genuinely want to own 100 GME shares at that price.",
                "risk": "Low-Med",
                "beginner_note": "This is how you start the Wheel. Collect premium → assigned shares → sell covered calls → repeat.",
            })
        else:
            signals.append({
                "type": "⏸️ CSP — Hold Off",
                "action": "Downtrend active — skip the cash-secured put",
                "why": (
                    f"Trend is bearish. Selling a CSP risks getting assigned shares that keep falling. "
                    f"Support at {p(nearest_sup)} is already under pressure — wait for stabilisation."
                ),
                "when": "Wait for price to reclaim MA20 for 2–3 days before opening a put position.",
                "risk": "—",
                "beginner_note": "Patience is part of the strategy.",
            })

    # ================================================================
    # 4. LEAP
    # ================================================================
    if iv_regime not in ("high", "unknown") and not guardrail_no_sell:
        leap_strike = round(price * 0.85, 0)
        leap_why = (
            f"A Jan 2027 call at {p(leap_strike)} (deep ITM, ~15% below spot) gives you high delta — "
            f"moves close to dollar-for-dollar with GME. Much cheaper than 100 shares. "
            f"IV is {iv_regime} so pricing is reasonable."
        )
        if trend == "bearish":
            leap_why += " Current downtrend — wait for reversal signal before buying."

        signals.append({
            "type": "🚀 LEAP",
            "action": f"Buy Jan 2027 {p(leap_strike)} call — deep ITM, high delta",
            "why": leap_why,
            "when": "Buy when IV is low (options cheap). Avoid buying LEAPs when IVR > 70 — IV crush will hurt you.",
            "risk": "Med",
            "beginner_note": f"Max loss = 100% of what you pay. Controls 100 shares for 1–2 years at a fraction of the capital.",
        })

    # ================================================================
    # 5. Volume Spike
    # ================================================================
    if vol_spike:
        signals.append({
            "type": "⚠️ Volume Spike",
            "action": "Unusual volume — check Reddit Intel for catalyst",
            "why": (
                f"Volume is {latest['vol_zscore']:.1f}σ above the 20-day average. "
                f"Big volume often precedes or confirms a directional move."
            ),
            "when": "Don't trade the spike itself. Confirm direction holds the next day.",
            "risk": "—",
            "beginner_note": "High volume + price up = conviction. High volume + price flat = indecision.",
        })

    # ================================================================
    # 6. GEX Key Levels (if computed)
    # ================================================================
    if gex and gex.regime != "unknown" and (gex.call_wall or gex.put_wall or gex.gamma_flip):
        level_parts = []
        if gex.gamma_flip:
            level_parts.append(f"Gamma flip: **{p(gex.gamma_flip)}** ({'above' if gex.gamma_flip > price else 'below'} current price)")
        if gex.call_wall:
            level_parts.append(f"Call wall (resistance): **{p(gex.call_wall)}**")
        if gex.put_wall:
            level_parts.append(f"Put wall (support): **{p(gex.put_wall)}**")

        signals.append({
            "type": "🎯 GEX Key Levels",
            "action": f"Dealer hedging map — {gex.regime_label()}",
            "why": (
                "\n".join(f"- {part}" for part in level_parts) +
                f"\n\nNet GEX: **${gex.net_gex_millions():.1f}M** per 1% move. "
                f"Call wall and put wall are better strike targets than static S/R for GME."
            ),
            "when": "Recalculates each time the chain refreshes. More reliable than pivot-based S/R on a meme stock.",
            "risk": "—",
            "beginner_note": (
                "The gamma flip is the price level where dealers switch from suppressing moves to amplifying them. "
                "Below the flip = dealers sell strength. Above the flip = dealers chase moves higher."
            ),
        })

    # ================================================================
    # 7. Confluence detector
    # ================================================================
    hits = []
    if rsi_zone == "oversold":
        hits.append(f"RSI oversold ({rsi:.0f})")
    if trend == "bullish":
        hits.append("Price above MA20 and MA50")
    if vol_spike:
        hits.append("Volume spike")
    if iv_metrics and (iv_metrics.sell_signal or iv_metrics.cautious_sell):
        hits.append(f"IV elevated (IVR {iv_metrics.ivr:.0f})" if iv_metrics.ivr else "IV elevated")
    if nearest_sup and price > nearest_sup * 0.99:
        hits.append(f"Price holding above support ({p(nearest_sup)})")
    if gex and not gex.is_negative:
        hits.append("GEX positive (suppressing regime)")

    if len(hits) >= 3:
        signals.append({
            "type": "🔥 Confluence",
            "action": f"{len(hits)} signals aligned — elevated conviction",
            "why": (
                "Multiple independent indicators point the same direction:\n\n" +
                "\n".join(f"- {h}" for h in hits)
            ),
            "when": "Still not a guarantee — but this is the setup to size up, not a single indicator firing alone.",
            "risk": "—",
            "beginner_note": "Confluence = multiple reasons to act. One indicator is noise. Three or more together is signal.",
        })

    ctx = {
        "price": price, "rsi": rsi, "trend": trend, "trend_note": trend_note,
        "rsi_zone": rsi_zone, "iv_regime": iv_regime, "vol_spike": vol_spike,
        "nearest_support": nearest_sup, "nearest_resistance": nearest_res,
        "guardrail_no_sell": guardrail_no_sell,
        "guardrail_gex_warning": guardrail_gex_warning,
        "catalyst_gate": catalyst_gate,
        "catalyst_gate_status": catalyst_gate.gate_status if catalyst_gate else "unknown",
    }

    return signals, ctx


# ============================================================
# Helpers: pull POP/EV from chain data
# ============================================================

def _get_pop_note_cc(chain_snapshot, strike: float, spot: float, cost_basis: Optional[float]) -> str:
    if chain_snapshot is None:
        return ""
    try:
        today = date.today()
        calls = [
            c for c in chain_snapshot.contracts
            if c.option_type == "call"
            and abs(c.strike - strike) < 2.0
            and (date.fromisoformat(c.expiry) - today).days > 5
        ]
        if not calls:
            return ""
        best = min(calls, key=lambda c: abs(c.strike - strike))
        delta = best.best_greek_delta
        premium = best.mid or best.last
        if delta is None or premium is None:
            return ""
        exp_date = date.fromisoformat(best.expiry)
        dte = max((exp_date - today).days, 1)
        metrics = compute_covered_call_metrics(
            delta=delta, premium=premium, strike=best.strike,
            spot=spot, dte=dte, cost_basis=cost_basis,
        )
        return f"**{metrics.summary()}**"
    except Exception:
        return ""


def _get_pop_note_csp(chain_snapshot, strike: float, support: float) -> str:
    if chain_snapshot is None:
        return ""
    try:
        today = date.today()
        puts = [
            c for c in chain_snapshot.contracts
            if c.option_type == "put"
            and abs(c.strike - strike) < 2.0
            and (date.fromisoformat(c.expiry) - today).days > 5
        ]
        if not puts:
            return ""
        best = min(puts, key=lambda c: abs(c.strike - strike))
        delta = best.best_greek_delta
        premium = best.mid or best.last
        if delta is None or premium is None:
            return ""
        exp_date = date.fromisoformat(best.expiry)
        dte = max((exp_date - today).days, 1)
        metrics = compute_short_put_metrics(
            delta=delta, premium=premium, strike=best.strike,
            spot=support, dte=dte,
        )
        return f"**{metrics.summary()}**"
    except Exception:
        return ""
