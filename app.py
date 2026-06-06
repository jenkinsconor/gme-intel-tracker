"""
GME Tactical Dashboard v5
Modular rebuild: real data providers · self-computed BSM Greeks · IVR/IVP · GEX · Wheel ledger
"""
import math
from datetime import date, datetime

import pandas as pd
import streamlit as st
import yfinance as yf

from data.providers import get_provider
from data.store import IVStore, WheelStore, PositionStore
from data.models import OptionsChainSnapshot
from analytics.greeks import enrich_chain_with_greeks, greek_discrepancy, PY_VOLLIB_AVAILABLE
from analytics.iv import (
    compute_iv_metrics, get_constant_maturity_30d_iv,
    compute_expected_move, compute_expected_move_straddle, get_atm_straddle,
)
from analytics.gex import compute_gex
from analytics.pop import compute_short_put_metrics, compute_covered_call_metrics
from modules.reddit import RedditScraper
from modules.technicals import get_support_resistance
from signals.engine import build_signals
from strategy.wheel import (
    WheelState, LegType, LEG_LABELS, VALID_TRANSITIONS,
    next_state, suggest_roll,
)
from sentiment.catalyst import build_catalyst_gate, CatalystGate
from sentiment.ftd import fetch_ftd_data, compute_ftd_metrics

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GME Tactical Dashboard",
    page_icon="🕹️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🕹️ GME Tactical Dashboard")
st.caption("Personal research tool. Not financial advice.")

# ---------------------------------------------------------------------------
# Provider + persistence (singletons)
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_provider():
    return get_provider()

@st.cache_resource
def _get_iv_store():
    return IVStore()

@st.cache_resource
def _get_wheel_store():
    return WheelStore()

@st.cache_resource
def _get_position_store():
    return PositionStore()

provider = _get_provider()
iv_store = _get_iv_store()
wheel_store = _get_wheel_store()
pos_store = _get_position_store()

# Provider badge
_rt_icon = "🟢 Real-time" if provider.is_real_time else "🟡 15-min delay"
st.sidebar.markdown(f"**Data provider:** {provider.provider_name}  \n{_rt_icon}")
if not provider.is_real_time:
    st.sidebar.info(
        "Upgrade to real-time data: open a **Tradier brokerage account** (free) and set "
        "`TRADIER_TOKEN` in your environment. App auto-switches on restart."
    )
if not PY_VOLLIB_AVAILABLE:
    st.sidebar.warning("`py_vollib` not installed — BSM Greeks and GEX unavailable. Run: `pip install py_vollib py_vollib_vectorized`")

tabs = st.tabs(["📈 Price & Signals", "🧠 Options Chain", "👾 Reddit Intel", "🎡 Wheel Tracker", "🔍 GME Context", "💼 My Positions"])

# Load wheel state at top level so Tab 2 CC Suggester can enforce the basis guardrail
_active_wheel_cycle = wheel_store.load_active_cycle("GME")
_tracked_basis: float | None = (
    _active_wheel_cycle.adjusted_basis_per_share if _active_wheel_cycle else None
)


# ---------------------------------------------------------------------------
# Shared data loaders  (must be defined before any top-level calls below)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_price_data():
    t = yf.Ticker("GME")
    df = t.history(period="1y")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()
    delta_ = df["Close"].diff()
    up    = delta_.clip(lower=0)
    down  = -delta_.clip(upper=0)
    df["RSI"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std  = df["Volume"].rolling(20).std()
    df["vol_zscore"] = (df["Volume"] - vol_mean) / vol_std.replace(0, float("nan"))
    df["vol_spike"]  = df["vol_zscore"] >= 2.0
    return df


@st.cache_data(ttl=120, show_spinner="Fetching options chain…")
def load_chain():
    """Load options chain, enrich with BSM Greeks, return as dict (cacheable)."""
    chain = provider.get_options_chain("GME", max_expiries=4)
    if PY_VOLLIB_AVAILABLE and chain.contracts:
        enrich_chain_with_greeks(chain.contracts, chain.spot)
    # Return as plain dict so Streamlit can serialize it
    return chain.model_dump()


@st.cache_data(ttl=3600)
def load_iv_history():
    return iv_store.get_history(days=365)


def _store_today_iv(iv_30d: float, spot: float) -> None:
    """Upsert today's IV snapshot — only writes once per day."""
    latest = iv_store.latest()
    if latest is None or latest[0] < date.today():
        iv_store.upsert(iv_30d, spot)


@st.cache_data(ttl=3600, show_spinner="Checking catalyst gate…")
def load_catalyst_gate(_reddit_df_hash, reddit_df):
    """Build catalyst gate. _reddit_df_hash is a hashable key to bust cache on refresh."""
    return build_catalyst_gate(reddit_df)


@st.cache_data(ttl=86400, show_spinner="Fetching SEC FTD data…")
def load_ftd():
    df = fetch_ftd_data(months_back=6)
    metrics = compute_ftd_metrics(df) if not df.empty else {}
    return df, metrics


@st.cache_data(ttl=300, show_spinner=False)
def _load_reddit_hot():
    scraper = RedditScraper(request_delay=1.5)
    return scraper.run(sort="hot")


# ---------------------------------------------------------------------------
# Top-level data that feeds multiple tabs (runs once per refresh)
# ---------------------------------------------------------------------------
_reddit_for_gate = _load_reddit_hot()
_reddit_hash = len(_reddit_for_gate) if _reddit_for_gate is not None and not _reddit_for_gate.empty else 0
_catalyst_gate = load_catalyst_gate(_reddit_hash, _reddit_for_gate)


# ===========================================================================
# TAB 1 — Price & Signals
# ===========================================================================
with tabs[0]:
    df = load_price_data()
    chain_dict = load_chain()
    iv_hist_df = load_iv_history()

    # Rebuild chain objects from dict
    chain = OptionsChainSnapshot.model_validate(chain_dict)

    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    price  = float(latest["Close"])
    price_delta = price - float(prev["Close"])
    price_pct   = price_delta / float(prev["Close"]) * 100

    sr_levels = get_support_resistance(df, price_col="Close", window=5, max_levels=8)

    # ---- Constant-maturity 30d IV + store snapshot ----
    iv_30d = get_constant_maturity_30d_iv(chain) if chain.contracts else None
    if iv_30d:
        _store_today_iv(iv_30d, price)

    # ---- IV metrics (IVR / IVP) ----
    iv_metrics = None
    if iv_30d:
        iv_metrics = compute_iv_metrics(iv_30d, iv_hist_df)

    # ---- GEX profile ----
    gex = compute_gex(chain) if chain.contracts else None

    # ---- Expected move ----
    em_iv = compute_expected_move(price, iv_30d, 30) if iv_30d else None
    straddle_price = get_atm_straddle(chain) if chain.contracts else None
    em_straddle = compute_expected_move_straddle(price, straddle_price) if straddle_price else None

    # ---- Build signals ----
    signals, ctx = build_signals(
        df, sr_levels,
        iv_metrics=iv_metrics,
        gex=gex,
        chain_snapshot=chain,
        catalyst_gate=_catalyst_gate,
    )

    # ================================================================
    # Metrics rows
    # ================================================================
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("GME Price", f"${price:.2f}", f"{price_delta:+.2f} ({price_pct:+.2f}%)")
    c2.metric("MA20", f"${latest['MA20']:.2f}", help="20-day moving average")
    c3.metric("MA50", f"${latest['MA50']:.2f}", help="50-day moving average")
    c4.metric("RSI", f"{latest['RSI']:.1f}",
              delta="oversold ✅" if ctx["rsi_zone"] == "oversold" else
                    "overbought ⚠️" if ctx["rsi_zone"] == "overbought" else None,
              delta_color="normal",
              help="Below 35 = oversold. Above 65 = overbought.")
    c5.metric("IV (30d CM)", f"{iv_30d*100:.0f}%" if iv_30d else "—",
              help="Constant-maturity 30-day implied volatility (interpolated). The cleanest single IV number.")
    c6.metric("Vol Spike", "YES ⚠️" if latest["vol_spike"] else "No",
              delta_color="off",
              help="Volume ≥ 2σ above 20-day average.")

    # IVR / IVP row
    if iv_metrics:
        ci1, ci2, ci3, ci4 = st.columns(4)
        ivr_str = f"{iv_metrics.ivr:.0f}" if iv_metrics.ivr is not None else f"({iv_metrics.history_days}/20d)"
        ivp_str = f"{iv_metrics.ivp:.0f}" if iv_metrics.ivp is not None else "—"
        hi_str = f"{iv_metrics.iv_52w_high*100:.0f}%" if iv_metrics.iv_52w_high else "—"
        lo_str = f"{iv_metrics.iv_52w_low*100:.0f}%" if iv_metrics.iv_52w_low else "—"
        ci1.metric("IV Rank (IVR)", ivr_str,
                   help="IVR = (current IV − 52w low) / (52w high − 52w low) × 100. Sell premium above 50.")
        ci2.metric("IV Percentile (IVP)", ivp_str,
                   help="IVP = % of past 252 days where IV was below current. More robust than IVR on spike-prone names like GME.")
        ci3.metric("52w IV High", hi_str)
        ci4.metric("52w IV Low", lo_str)

        # IV regime banner — guard against None IVR/IVP during history build-up
        _ivr = f"{iv_metrics.ivr:.0f}" if iv_metrics.ivr is not None else "—"
        _ivp = f"{iv_metrics.ivp:.0f}" if iv_metrics.ivp is not None else "—"
        regime_colors = {"high": "error", "elevated": "warning", "normal": "info", "low": "success", "unknown": "info"}
        regime_msgs = {
            "high":     f"IV REGIME: HIGH — IVR {_ivr} / IVP {_ivp}. **Strong sell-premium signal** (both thresholds exceeded).",
            "elevated": f"IV REGIME: ELEVATED — IVR {_ivr} / IVP {_ivp}. Cautious sell signal — one threshold met.",
            "normal":   f"IV REGIME: NORMAL — IVR {_ivr} / IVP {_ivp}. Neither threshold for high-conviction sell. Normal sizing.",
            "low":      f"IV REGIME: LOW — IVR {_ivr}. **Do not sell premium.** Wait for IVR > 30.",
            "unknown":  f"Building IV history ({iv_metrics.history_days}/20 days minimum). Raw IV {iv_30d*100:.0f}% in the meantime.",
        }
        msg = regime_msgs.get(iv_metrics.regime, "")
        if msg:
            getattr(st, regime_colors.get(iv_metrics.regime, "info"))(msg)

    # ---- Trend banner ----
    trend_colors = {"bullish": "success", "bearish": "error", "neutral": "info"}
    getattr(st, trend_colors[ctx["trend"]])(f"**Trend: {ctx['trend'].upper()}** — {ctx['trend_note']}")

    st.divider()

    # ================================================================
    # Expected Move display
    # ================================================================
    if em_iv or em_straddle:
        st.subheader("30-Day Expected Move (1σ ≈ 68% probability)")
        em_c1, em_c2, em_c3 = st.columns(3)
        if em_iv:
            em_c1.metric("IV-derived lower", f"${em_iv[0]:.2f}", help="Price − IV × √(30/365)")
            em_c2.metric("IV-derived upper", f"${em_iv[1]:.2f}")
            em_c3.metric("IV band width", f"${em_iv[1]-em_iv[0]:.2f} ({(em_iv[1]-em_iv[0])/price*100:.1f}%)")
        if straddle_price:
            st.caption(
                f"ATM straddle price: **${straddle_price:.2f}** → "
                f"Straddle-derived EM: **${em_straddle[0]:.2f} – ${em_straddle[1]:.2f}**  "
                f"(straddle × 0.85)"
            )
        with st.expander("How expected move works"):
            st.markdown("""
The expected move is the market's best guess at how far GME will move over 30 days, with ~68% confidence.

- **IV-derived:** `Spot × IV × √(DTE/365)` — uses implied volatility to estimate 1σ range
- **Straddle-derived:** `ATM straddle × 0.85` — derived from what buyers are paying for a straddle

**How to use for strike selection:**
- **CSP:** sell the put strike *outside* the lower expected-move bound for better odds
- **CC:** sell the call strike *above* the upper expected-move bound for cushion
- If both methods agree → higher confidence in the band. If they diverge significantly → be more conservative.
            """)

    # ================================================================
    # GEX display
    # ================================================================
    if gex and gex.regime != "unknown":
        st.subheader("Gamma Exposure (GEX) — Dealer Positioning")
        gex_c1, gex_c2, gex_c3, gex_c4 = st.columns(4)
        regime_icon = "🔴" if gex.is_negative else "🟢"
        gex_c1.metric("GEX Regime", f"{regime_icon} {'NEGATIVE' if gex.is_negative else 'POSITIVE'}",
                      help="Negative GEX = dealers amplify moves (squeeze risk). Positive = dealers suppress moves.")
        gex_c2.metric("Net GEX", f"${gex.net_gex_millions():.1f}M / 1% move")
        gex_c3.metric("Call Wall (resistance)", f"${gex.call_wall:.2f}" if gex.call_wall else "—",
                      help="Strike with highest call GEX. Dealers sell here aggressively. Useful CC strike reference.")
        gex_c4.metric("Put Wall (support)", f"${gex.put_wall:.2f}" if gex.put_wall else "—",
                      help="Strike with highest put GEX. Dealers buy here aggressively. Useful CSP strike reference.")

        if gex.gamma_flip:
            flip_side = "above" if gex.gamma_flip > price else "below"
            st.info(
                f"**Gamma Flip:** ${gex.gamma_flip:.2f} ({flip_side} current price ${price:.2f})  \n"
                f"Above the flip: positive GEX (suppressing). Below: negative GEX (amplifying).  \n"
                f"For premium sellers — price above flip = safer environment for short gamma."
            )

        if not gex_by_s.empty if (gex_by_s := gex.gex_by_strike) is not None and not gex.gex_by_strike.empty else False:
            with st.expander("📊 GEX by Strike"):
                chart_df = gex.gex_by_strike.set_index("strike")[["call_gex", "put_gex", "net_gex"]].copy()
                chart_df.columns = ["Call GEX", "Put GEX", "Net GEX"]
                st.bar_chart(chart_df[["Call GEX", "Put GEX"]], use_container_width=True)

        with st.expander("What is GEX and why it matters for GME"):
            st.markdown("""
**Gamma Exposure (GEX)** measures how much dollar hedging dealers must do per 1% move in GME's price.

**Positive GEX (green):**
- Dealers are net long gamma (they bought options)
- When price rises, dealers *sell* → suppresses rallies
- When price falls, dealers *buy* → cushions drops
- Result: range-bound, mean-reverting price action → ideal for covered calls and CSPs

**Negative GEX (red) ⚠️:**
- Dealers are net short gamma (they sold options to customers)
- When price rises, dealers *buy more* to hedge → amplifies the rally
- When price falls, dealers *sell more* → amplifies the drop
- This is the structural mechanic behind the 2021 squeeze **and** the May 2024 Roaring Kitty rally
- For premium sellers: a short put in a negative-GEX environment can blow through your breakeven before the open

**Gamma Flip:** the price level where GEX changes sign. Crossing above = suppressing regime, below = amplifying.
**Call Wall:** strike with max call GEX → dealers sell aggressively here (resistance)
**Put Wall:** strike with max put GEX → dealers buy aggressively here (support)
            """)

    st.divider()

    # ================================================================
    # Trade Signals
    # ================================================================
    st.subheader("Trade Signals")
    if not signals:
        st.info("No clear signals at this time.")
    else:
        for sig in signals:
            if sig.get("is_situation"):
                color = {
                    "wait": "error", "watch": "warning",
                    "ok": "info", "good": "success",
                }.get(sig.get("situation_key", "ok"), "info")
                getattr(st, color)(f"**{sig['type']} — {sig['action']}**\n\n{sig['why']}")
                continue

            if sig.get("is_guardrail"):
                st.error(f"**{sig['type']} — {sig['action']}**\n\n{sig['why']}")
                with st.expander("Details"):
                    if sig.get("when"):
                        st.markdown(f"**When clears:** {sig['when']}")
                    if sig.get("beginner_note"):
                        st.info(f"💡 {sig['beginner_note']}")
                continue

            risk_icon = {"Low": "🟢", "Low-Med": "🟡", "Med": "🟠", "—": "⚪"}.get(sig["risk"], "⚪")
            with st.expander(f"{sig['type']}  —  {sig['action']}", expanded=True):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"**Why:** {sig['why']}")
                    if sig.get("when"):
                        st.markdown(f"**When:** {sig['when']}")
                    if sig.get("beginner_note"):
                        st.info(f"💡 {sig['beginner_note']}")
                with col_b:
                    st.markdown("**Risk:**")
                    st.markdown(f"## {risk_icon} {sig['risk']}")

    with st.expander("Indicator reference"):
        st.markdown("""
**IV Rank (IVR):** (current IV − 52w low) / (52w high − 52w low) × 100
- Sell premium when IVR > 50. High conviction > 70. Do NOT sell when IVR < 30.
- GME caveat: the 2021 spike still anchors the denominator. IVP is more reliable.

**IV Percentile (IVP):** % of past 252 days where IV was below today's level.
- Less distorted by single spikes than IVR. Require IVP > 70 AND IVR > 50 for high-conviction sells.

**GEX:** Gamma Exposure — see the GEX section above.

**Expected Move (1σ):** Spot × IV × √(DTE/365). The 68% probability range.
        """)

    st.divider()

    # ================================================================
    # Charts
    # ================================================================
    chart_df = df[["Close", "MA20", "MA50", "MA100"]].dropna()
    date_range = f"{chart_df.index[0].strftime('%b %Y')} – {chart_df.index[-1].strftime('%b %Y')}"
    st.subheader(f"Price & Moving Averages ({date_range})")
    st.line_chart(chart_df, use_container_width=True)

    if em_iv:
        st.caption(
            f"30-day expected move band: **${em_iv[0]:.2f} – ${em_iv[1]:.2f}**  "
            f"(current: ${price:.2f})  ·  Use these as strike-selection boundaries."
        )

    col_rsi, col_vol = st.columns(2)
    with col_rsi:
        st.subheader("RSI (14)")
        st.line_chart(df[["RSI"]].dropna(), use_container_width=True)
        rsi_now = float(latest["RSI"])
        if rsi_now < 35:
            st.success(f"RSI {rsi_now:.1f} — oversold. Potential bounce. Consider LEAP or CSP.")
        elif rsi_now > 65:
            st.warning(f"RSI {rsi_now:.1f} — overbought. Consider covered call to collect premium on the run-up.")
        else:
            st.info(f"RSI {rsi_now:.1f} — neutral zone.")

    with col_vol:
        st.subheader("Volume")
        st.bar_chart(df[["Volume"]], use_container_width=True)
        if latest["vol_spike"]:
            st.warning(f"Volume spike ({latest['vol_zscore']:.1f}σ) — check Reddit Intel for catalysts.")
        else:
            st.info("Volume normal today.")

    if sr_levels:
        st.subheader("Support & Resistance Levels")
        sr_data = [{
            "Level": f"${l.level:.2f}",
            "Type": l.kind,
            "Strength": f"{l.strength:.2f}",
            "Touches": l.touches,
            "Last Touch": str(l.last_touch)[:10],
            "Use for": "CSP below" if l.kind == "support" else "CC above" if l.kind == "resistance" else "Either",
        } for l in sr_levels]
        st.dataframe(pd.DataFrame(sr_data), use_container_width=True)
        with st.expander("S/R vs GEX levels — which to use"):
            st.markdown("""
For a normal stock, pivot-based S/R is reliable. **For GME, GEX levels are more predictive.**

- S/R is based on *past price behaviour* — useful but GME can gap 50% past any pivot on a tweet
- GEX call wall / put wall are based on *current dealer hedging mechanics* — dynamic and forward-looking
- **Best practice:** use GEX levels as primary strike selection, S/R as secondary confirmation
            """)


# ===========================================================================
# TAB 2 — Options Chain
# ===========================================================================
with tabs[1]:
    st.subheader("Options Chain — IV, Greeks & Analytics")

    # Provider status
    rt_badge = "🟢 Real-time" if provider.is_real_time else "🟡 15-min delayed"
    st.caption(f"Data source: **{provider.provider_name}** · {rt_badge}")
    if not PY_VOLLIB_AVAILABLE:
        st.warning("BSM Greeks unavailable (`py_vollib` not installed). Install with `pip install py_vollib py_vollib_vectorized` then restart.")

    with st.expander("Options chain guide", expanded=False):
        st.markdown("""
| Column | What it means | Why you care |
|--------|--------------|--------------|
| **Strike** | Price you agree to buy/sell shares at | CC: above current price. LEAP: below (deep ITM). |
| **Bid / Ask / Mid** | Market prices for the contract | Your fill will be near mid. Mid = your actual premium. |
| **IV** | Implied volatility for this specific contract | High = expensive. Sell when high, buy when low. |
| **BSM δ (delta)** | Self-computed Black-Scholes delta | More reliable than yfinance deltas. Used for POP. |
| **BSM γ (gamma)** | Rate of delta change | High gamma near expiry = risky for sellers |
| **BSM θ (theta)** | Premium decay per day | Negative for buyers, works for you as a seller |
| **POP** | Probability of max profit ≈ 1 − |delta| | 80% POP short put = 20% chance of assignment |
| **Ann. ROC** | Annualized return on capital | Filter for > 30% to weed out low-IV environments |

**Covered Call checklist:** OTM, 0.20–0.35 delta, 2–4 weeks out, strike above GEX call wall or S/R resistance.
**LEAP checklist:** Deep ITM, 0.70+ delta, Jan 2027+, buy when IVR < 30 (cheap).
        """)

    chain_dict2 = load_chain()
    chain2 = OptionsChainSnapshot.model_validate(chain_dict2)

    if not chain2.contracts:
        st.warning("No options data available. Check your data provider.")
    else:
        price2 = chain2.spot
        expiries2 = chain2.expiries or sorted(set(c.expiry for c in chain2.contracts))

        st.info(f"Current GME price: **${price2:.2f}**")

        col1, col2 = st.columns(2)
        expiry_choice = col1.selectbox("Expiry date", expiries2,
            help="Covered calls: 2–4 weeks out. LEAPs: Jan 2027+.")
        n_strikes = col2.slider("Strikes around ATM", 2, 12, 6,
            help="Number of strikes above and below current price to show.")

        view_contracts = chain2.for_expiry(expiry_choice)
        calls = sorted(
            [c for c in view_contracts if c.option_type == "call"],
            key=lambda c: abs(c.strike - price2)
        )[:n_strikes]
        puts = sorted(
            [c for c in view_contracts if c.option_type == "put"],
            key=lambda c: abs(c.strike - price2)
        )[:n_strikes]

        today2 = date.today()

        def _dte(expiry_str):
            return max((date.fromisoformat(expiry_str) - today2).days, 1)

        dte2 = _dte(expiry_choice)

        def _contracts_to_df(contracts):
            rows = []
            for c in sorted(contracts, key=lambda x: x.strike):
                d = c.best_greek_delta
                g = c.bsm_gamma if c.bsm_gamma is not None else c.gamma
                th = c.bsm_theta if c.bsm_theta is not None else c.theta
                mid = c.mid

                pop_str = f"{(1-abs(d))*100:.0f}%" if d is not None else "—"
                roc_str = "—"
                if d is not None and mid is not None:
                    try:
                        if c.option_type == "put":
                            m = compute_short_put_metrics(d, mid, c.strike, price2, dte2)
                        else:
                            m = compute_covered_call_metrics(d, mid, c.strike, price2, dte2)
                        roc_str = f"{m.annualized_roc_pct:.0f}%"
                    except Exception:
                        pass

                rows.append({
                    "Strike": f"${c.strike:.1f}",
                    "Bid": f"${c.bid:.2f}" if c.bid else "—",
                    "Ask": f"${c.ask:.2f}" if c.ask else "—",
                    "Mid": f"${mid:.2f}" if mid else "—",
                    "IV": f"{c.implied_volatility*100:.0f}%" if c.implied_volatility else "—",
                    "BSM δ": f"{d:+.2f}" if d is not None else "—",
                    "BSM γ": f"{g:.4f}" if g is not None else "—",
                    "BSM θ": f"{th:.3f}" if th is not None else "—",
                    "OI": f"{c.open_interest:,}" if c.open_interest else "—",
                    "Volume": f"{c.volume:,}" if c.volume else "—",
                    "POP": pop_str,
                    "Ann. ROC": roc_str,
                })
            return pd.DataFrame(rows)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Calls** — *highlighted = good CC candidate (δ 0.20–0.35, OTM)*")
            calls_df = _contracts_to_df(calls)

            def highlight_cc(row):
                try:
                    strike_val = float(str(row["Strike"]).replace("$", ""))
                    delta_str = str(row["BSM δ"]).replace("+", "")
                    if delta_str == "—":
                        return [""] * len(row)
                    d = abs(float(delta_str))
                    if strike_val > price2 and 0.20 <= d <= 0.35:
                        return ["background-color: #1a3a1a"] * len(row)
                except Exception:
                    pass
                return [""] * len(row)

            try:
                st.dataframe(calls_df.style.apply(highlight_cc, axis=1), use_container_width=True)
            except Exception:
                st.dataframe(calls_df, use_container_width=True)

        with col_b:
            st.markdown("**Puts**")
            st.dataframe(_contracts_to_df(puts), use_container_width=True)

        # ---- Covered Call Suggester ----
        st.subheader("Covered Call Suggester")
        try:
            otm_calls = [c for c in calls if c.strike > price2]
            cc_candidates = [
                c for c in otm_calls
                if c.best_greek_delta is not None and 0.20 <= abs(c.best_greek_delta) <= 0.40
            ] or otm_calls

            if not cc_candidates:
                st.info("No covered call candidates — try adjusting expiry or strike range.")
            else:
                best_cc = min(cc_candidates, key=lambda c: abs(c.strike - price2))
                mid_cc = best_cc.mid or best_cc.last
                d_cc = best_cc.best_greek_delta
                iv_cc = best_cc.implied_volatility

                delta_str = f"{abs(d_cc)*100:.0f}% chance of assignment" if d_cc else ""
                iv_str = f"IV {iv_cc*100:.0f}%" if iv_cc else ""
                mid_str = f"${mid_cc:.2f}/share = ${mid_cc*100:.0f} per contract" if mid_cc else ""

                # POP / EV
                pop_block = ""
                if d_cc is not None and mid_cc:
                    m = compute_covered_call_metrics(
                        d_cc, mid_cc, best_cc.strike, price2, dte2,
                        cost_basis=_tracked_basis,
                    )
                    pop_block = f"\n\n{m.summary()}"

                # Basis guardrail from Wheel Tracker
                basis_guard = ""
                if _tracked_basis:
                    if best_cc.strike < _tracked_basis:
                        basis_guard = (
                            f"\n\n**BASIS GUARDRAIL:** This strike (${best_cc.strike:.2f}) is "
                            f"below your tracked adjusted basis (${_tracked_basis:.2f}). "
                            f"If called away here you lock in a loss. Select a higher strike or wait."
                        )
                    else:
                        basis_guard = (
                            f"\n\nStrike (${best_cc.strike:.2f}) is above adjusted basis "
                            f"(${_tracked_basis:.2f}) — safe to sell."
                        )

                st.success(
                    f"**Suggested CC:** Sell the **${best_cc.strike:.0f} call** expiring **{expiry_choice}** ({dte2} DTE)\n\n"
                    f"- Collect ~**{mid_str}**\n"
                    f"- {delta_str}  ·  {iv_str}"
                    + pop_block
                    + basis_guard
                )

                # Provider vs BSM Greek discrepancy warning
                if best_cc.delta is not None and best_cc.bsm_delta is not None:
                    disc = greek_discrepancy(best_cc.delta, best_cc.bsm_delta)
                    if disc and disc > 5:
                        st.warning(
                            f"Provider delta ({best_cc.delta:.2f}) differs from BSM delta ({best_cc.bsm_delta:.2f}) "
                            f"by {disc:.1f}%. Using BSM delta for POP."
                        )
        except Exception as e:
            st.info(f"CC suggester unavailable: {e}")

        # ---- Max Pain ----
        try:
            oi_by_strike: dict[float, int] = {}
            for c in view_contracts:
                if c.open_interest:
                    oi_by_strike[c.strike] = oi_by_strike.get(c.strike, 0) + c.open_interest
            if oi_by_strike:
                max_pain_strike = max(oi_by_strike, key=oi_by_strike.get)
                st.info(
                    f"**Max Pain: ${max_pain_strike:.2f}** — highest combined OI. "
                    f"Market makers incentivised to pin price here at expiry."
                )
        except Exception:
            pass

        # ---- IV Smile ----
        st.subheader("IV Smile")
        calls_iv = [(c.strike, c.implied_volatility * 100) for c in calls if c.implied_volatility]
        puts_iv  = [(c.strike, c.implied_volatility * 100) for c in puts if c.implied_volatility]
        if calls_iv or puts_iv:
            smile_df = pd.DataFrame({
                "Call IV%": {s: iv for s, iv in calls_iv},
                "Put IV%":  {s: iv for s, iv in puts_iv},
            }).sort_index()
            st.line_chart(smile_df, use_container_width=True)


# ===========================================================================
# TAB 3 — Reddit Intel
# ===========================================================================
with tabs[2]:
    st.subheader("👾 Reddit Intel — GME Community Pulse")

    col_r1, col_r2, col_r3, col_r4 = st.columns([1, 1, 1, 1])
    sort_feed   = col_r1.selectbox("Feed", ["hot", "new", "top", "rising"], index=0)
    show_filter = col_r2.selectbox("Filter", [
        "All posts", "Options-relevant", "Catalysts",
        "Strategy posts (Wheel/Spreads)", "Weekly Plays (u/Crybad style)",
        "DD only", "Key authors",
    ])
    sub_filter = col_r3.selectbox("Subreddit", ["All", "GMEOptions", "Superstonk", "GME", "wallstreetbets"])
    refresh    = col_r4.button("🔄 Refresh")

    @st.cache_data(ttl=300, show_spinner="Scraping Reddit feeds...")
    def load_reddit(sort: str):
        scraper = RedditScraper(request_delay=1.5)
        return scraper.run(sort=sort)

    if refresh:
        st.cache_data.clear()

    df_reddit = load_reddit(sort_feed)

    if df_reddit.empty:
        st.error("No Reddit data retrieved.")
    else:
        scraper_inst = RedditScraper()
        summary = scraper_inst.sentiment_summary(df_reddit)

        st.markdown("### Community Sentiment")
        s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
        icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(summary["overall"], "⚪")
        s1.metric("Signal Sentiment", f"{icon} {summary['overall'].upper()}",
                  help="Weighted — GMEOptions traders count more than general hype")
        s2.metric("Weighted Score", f"{summary['weighted_compound']:.3f}",
                  help=">0.05 = bullish, <-0.05 = bearish")
        gme_opts_val = summary.get("gmeoptions_compound")
        s3.metric("r/GMEOptions", f"{gme_opts_val:.3f}" if gme_opts_val is not None else "—",
                  help="Sentiment from options traders only — cleanest signal")
        s4.metric("🟢 Bullish",  summary["bullish"])
        s5.metric("⚪ Neutral",   summary["neutral"])
        s6.metric("🔴 Bearish",  summary["bearish"])
        s7.metric("⚡ Catalysts", summary["catalyst_posts"])

        if summary["catalyst_posts"] > 0:
            st.warning(f"**{summary['catalyst_posts']} catalyst posts** — potential event-driven move")
        if summary["key_author_posts"] > 0:
            st.success(f"**{summary['key_author_posts']} posts from key authors** (u/Crybad etc.)")

        view = df_reddit.copy()
        if sub_filter != "All":
            view = view[view["subreddit"] == sub_filter]
        if show_filter == "Options-relevant":
            view = view[view["is_options_relevant"]]
        elif show_filter == "Catalysts":
            view = view[view["is_catalyst"]]
        elif show_filter == "Strategy posts (Wheel/Spreads)":
            view = view[view["is_strategy"]]
        elif show_filter == "Weekly Plays (u/Crybad style)":
            view = view[view["is_weekly_plays"]] if "is_weekly_plays" in view.columns else view[view["is_strategy"]]
        elif show_filter == "DD only":
            view = view[view["is_dd"]]
        elif show_filter == "Key authors":
            view = view[view["is_key_author"]]

        st.markdown(f"### Posts ({len(view)} shown)")
        for _, row in view.head(60).iterrows():
            s_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(row["sentiment_label"], "⚪")
            tags = []
            if row.get("is_key_author"):       tags.append("📌 key author")
            if row.get("is_weekly_plays"):     tags.append("📅 weekly plays")
            if row.get("is_options_relevant"): tags.append("📊 options")
            if row.get("is_catalyst"):         tags.append("⚡ catalyst")
            if row.get("is_strategy"):         tags.append("🔵 strategy")
            if row.get("is_dd"):               tags.append("🔬 DD")

            with st.container():
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**[{row['title']}]({row['url']})**")
                    st.caption(
                        f"{s_icon} {row['sentiment_label']}  ·  "
                        f"r/{row['subreddit']}  ·  "
                        f"{row['age_hours']:.1f}h ago  ·  "
                        f"u/{row['author']}  " + "  ".join(tags)
                    )
                with c2:
                    st.markdown(f"`{row['sentiment_compound']:+.3f}`")
            st.divider()


# ===========================================================================
# TAB 4 — Wheel Tracker
# ===========================================================================
with tabs[3]:
    st.subheader("🎡 Wheel Strategy Tracker")
    st.caption(
        "Log every leg of your Wheel cycle. The ledger computes your true adjusted cost basis "
        "and enforces the rule: **never sell a covered call below adjusted basis**."
    )

    TICKER = "GME"

    with st.expander("How the Wheel works & why cost basis matters", expanded=False):
        st.markdown("""
**The Wheel cycle:**
1. **Sell a Cash-Secured Put (CSP)** at a strike below current price. Collect premium.
2. If the put *expires worthless* → keep cash, start again (full cycle complete).
3. If you get *assigned shares* → you now own 100 shares at the put strike.
4. **Sell a Covered Call (CC)** above your adjusted basis. Collect more premium.
5. If the call *expires worthless* → keep shares + premium, sell another call.
6. If shares get *called away* → sell at the call strike. Cycle complete.

**Why adjusted basis is critical:**
Your true cost per share is NOT just the price you were assigned at. It's:
```
Adjusted basis = assignment_strike
               − Σ(all put premiums collected)
               − Σ(all call premiums collected)
               + Σ(all buyback costs paid)
```
A common Wheel trap: during a drawdown, traders slide their CC strike down chasing premium —
and get shares called away *below their true basis*, locking in a loss they didn't realise they had.
The basis guardrail here prevents that.

**Rolling:** When a position moves against you, you can "roll" it — buy it back and sell a new
one at a different strike/expiry. Always roll for a *net credit*, not a debit.
        """)

    st.divider()

    # ================================================================
    # Load active cycle
    # ================================================================
    active_cycle = wheel_store.load_active_cycle(TICKER)
    all_cycles = wheel_store.get_all_cycles(TICKER)
    current_price_w = chain_dict.get("spot", 0.0) if chain_dict else 0.0

    # ================================================================
    # Active cycle summary panel
    # ================================================================
    col_status, col_actions = st.columns([3, 1])

    with col_status:
        if active_cycle is None:
            st.info("No active Wheel cycle. Start one below.")
        else:
            cycle = active_cycle
            st.markdown(f"### Cycle #{cycle.id} — {cycle.status_summary()}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("State", cycle.state.value.replace("_", " "))
            m1.caption(f"Started {cycle.started_date}")

            basis = cycle.adjusted_basis_per_share
            if basis is not None:
                basis_color = "normal" if current_price_w >= basis else "inverse"
                m2.metric(
                    "Adjusted Basis",
                    f"${basis:.2f}",
                    f"${current_price_w - basis:+.2f} vs current price",
                    delta_color=basis_color,
                    help="True cost per share after all premium. Never sell CC below this.",
                )
            else:
                m2.metric("Adjusted Basis", "Not yet assigned",
                          help="Will show after assignment leg is logged.")

            net_prem = cycle.total_net_premium
            m3.metric(
                "Net Premium Collected",
                f"${net_prem:,.2f}",
                help="Total cash collected minus buyback costs across all legs so far.",
            )

            if cycle.was_assigned and current_price_w > 0:
                unreal = cycle.unrealised_pnl(current_price_w)
                m4.metric(
                    "Unrealised P&L",
                    f"${unreal:+,.2f}",
                    delta_color="normal" if unreal >= 0 else "inverse",
                    help="Premium collected + unrealised share gain/loss at current price.",
                )
            elif cycle.realised_pnl is not None:
                m4.metric(
                    "Realised P&L",
                    f"${cycle.realised_pnl:+,.2f}",
                    delta_color="normal" if cycle.realised_pnl >= 0 else "inverse",
                )

            # Basis guardrail warning
            if basis is not None and current_price_w > 0:
                if basis > current_price_w * 1.05:
                    st.error(
                        f"**Basis guardrail:** current price (${current_price_w:.2f}) is more than 5% "
                        f"below your adjusted basis (${basis:.2f}). "
                        f"No strike above basis has meaningful premium right now. "
                        f"Wait for price recovery or consider rolling the original put."
                    )
                elif basis > current_price_w:
                    st.warning(
                        f"Current price (${current_price_w:.2f}) is below adjusted basis (${basis:.2f}). "
                        f"Any CC strike with decent premium will be below your basis — "
                        f"if called away you lock in a loss."
                    )
                else:
                    st.success(
                        f"Price (${current_price_w:.2f}) is above adjusted basis (${basis:.2f}). "
                        f"Safe to sell covered calls at strikes above ${basis:.2f}."
                    )

            # Open position summary
            if cycle.open_put_leg:
                leg = cycle.open_put_leg
                try:
                    exp_date = date.fromisoformat(leg.expiry)
                    dte_rem = (exp_date - date.today()).days
                except Exception:
                    dte_rem = None
                dte_str = f" · {dte_rem} DTE" if dte_rem is not None else ""
                st.info(
                    f"**Open CSP:** ${leg.strike:.0f} put expiring {leg.expiry}{dte_str}  "
                    f"· {leg.contracts} contract(s) · collected ${leg.premium_per_share:.2f}/share"
                )
            if cycle.open_call_leg:
                leg = cycle.open_call_leg
                try:
                    exp_date = date.fromisoformat(leg.expiry)
                    dte_rem = (exp_date - date.today()).days
                except Exception:
                    dte_rem = None
                dte_str = f" · {dte_rem} DTE" if dte_rem is not None else ""
                st.info(
                    f"**Open CC:** ${leg.strike:.0f} call expiring {leg.expiry}{dte_str}  "
                    f"· {leg.contracts} contract(s) · collected ${leg.premium_per_share:.2f}/share"
                )

    with col_actions:
        if active_cycle is None:
            if st.button("▶️ Start New Cycle", use_container_width=True, type="primary"):
                new_id = wheel_store.open_cycle(TICKER)
                st.success(f"Cycle #{new_id} started.")
                st.rerun()
        else:
            if st.button("🔁 Force New Cycle", use_container_width=True,
                         help="Close this cycle and start fresh. Use if you've manually closed all positions."):
                wheel_store.update_cycle_state(active_cycle.id, "CYCLE_COMPLETE", closed=True)
                new_id = wheel_store.open_cycle(TICKER)
                st.success(f"Cycle #{new_id} started.")
                st.rerun()

    st.divider()

    # ================================================================
    # Roll suggestion
    # ================================================================
    if active_cycle and chain_dict:
        chain_w = OptionsChainSnapshot.model_validate(chain_dict)
        roll = suggest_roll(active_cycle, chain_w, current_price_w)
        if roll:
            urgency_color = "error" if roll.urgency == "urgent" else "warning"
            getattr(st, urgency_color)(
                f"**{roll.action.upper()}**\n\n"
                f"Current: ${roll.current_strike:.0f} / {roll.current_expiry}  →  "
                f"Suggested: **${roll.suggested_strike:.0f} / {roll.suggested_expiry}**\n\n"
                f"{roll.rationale}"
            )

    # ================================================================
    # Log a new leg
    # ================================================================
    if active_cycle:
        st.subheader("Log a New Leg")

        current_state = active_cycle.state
        valid_leg_types = list(VALID_TRANSITIONS.get(current_state, set()))

        if not valid_leg_types:
            st.info(f"Cycle is in terminal state ({current_state.value}). Start a new cycle to continue.")
        else:
            with st.form("log_leg_form", clear_on_submit=True):
                leg_options = {LEG_LABELS[lt]: lt for lt in valid_leg_types}
                selected_label = st.selectbox(
                    "What happened?",
                    options=list(leg_options.keys()),
                    help=f"Current state: {current_state.value.replace('_', ' ')}. Only valid transitions shown."
                )
                selected_leg_type = leg_options[selected_label]

                fc1, fc2, fc3 = st.columns(3)
                strike = fc1.number_input("Strike price ($)", min_value=0.01, value=float(round(current_price_w, 0)) if current_price_w else 25.0, step=0.5)
                contracts = fc2.number_input("Contracts", min_value=1, value=1, step=1,
                                              help="Each contract = 100 shares")

                # Premium is 0 for non-money events
                no_premium_types = {LegType.ASSIGNMENT, LegType.PUT_EXPIRED, LegType.CALL_EXPIRED, LegType.CALLED_AWAY}
                premium = 0.0
                if selected_leg_type not in no_premium_types:
                    premium = fc3.number_input(
                        "Premium per share ($)",
                        min_value=0.0, value=1.00, step=0.01,
                        help="Enter as positive number. Sign (credit/debit) is inferred from the leg type."
                    )

                fd1, fd2 = st.columns(2)
                leg_date = fd1.date_input("Date", value=date.today())
                expiry_str = fd2.text_input(
                    "Option expiry (YYYY-MM-DD)",
                    value="",
                    placeholder="Leave blank for assignment/expiry events",
                )
                notes = st.text_input("Notes (optional)", placeholder="e.g. Rolled for $0.45 net credit")

                # Basis guardrail warning inside the form for CC sells
                if selected_leg_type == LegType.CALL_SELL and active_cycle.adjusted_basis_per_share:
                    basis = active_cycle.adjusted_basis_per_share
                    if strike <= basis:
                        st.warning(
                            f"Strike ${strike:.2f} is at or below your adjusted basis (${basis:.2f}). "
                            f"If called away here you take a loss. Consider a higher strike."
                        )

                submitted = st.form_submit_button("Log Leg", type="primary")

                if submitted:
                    try:
                        new_state = next_state(current_state, selected_leg_type)
                        expiry_val = expiry_str.strip() if expiry_str.strip() else None

                        wheel_store.add_leg(
                            cycle_id=active_cycle.id,
                            leg_type=selected_leg_type.value,
                            strike=float(strike),
                            premium_per_share=float(premium),
                            contracts=int(contracts),
                            expiry=expiry_val,
                            leg_date=leg_date,
                            notes=notes,
                        )

                        is_terminal = new_state in (WheelState.CALLED_AWAY, WheelState.CYCLE_COMPLETE)
                        wheel_store.update_cycle_state(active_cycle.id, new_state.value, closed=is_terminal)

                        if is_terminal:
                            st.success(f"Leg logged. Cycle complete. ({new_state.value})")
                        else:
                            st.success(f"Leg logged. State → {new_state.value.replace('_', ' ')}")
                        st.rerun()

                    except ValueError as e:
                        st.error(f"Invalid transition: {e}")
                    except Exception as e:
                        st.error(f"Error logging leg: {e}")

    st.divider()

    # ================================================================
    # Leg history
    # ================================================================
    if active_cycle and active_cycle.legs:
        st.subheader("Leg History — Current Cycle")

        rows = []
        running_basis = active_cycle.assignment_strike or 0.0
        running_credits = 0.0
        running_debits = 0.0

        for leg in active_cycle.legs:
            credit_debit = ""
            if leg.leg_type in {LegType.PUT_SELL, LegType.CALL_SELL}:
                credit_debit = f"+${leg.total_premium:,.2f}"
                running_credits += leg.total_premium
            elif leg.leg_type in {LegType.PUT_BUYBACK, LegType.CALL_BUYBACK}:
                credit_debit = f"-${abs(leg.total_premium):,.2f}"
                running_debits += abs(leg.total_premium)

            rows.append({
                "Date": str(leg.leg_date),
                "Action": leg.label,
                "Strike": f"${leg.strike:.2f}",
                "Premium/sh": f"${leg.premium_per_share:.2f}" if leg.premium_per_share else "—",
                "Contracts": leg.contracts,
                "Expiry": leg.expiry or "—",
                "Cash Flow": credit_debit or "—",
                "Notes": leg.notes,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Running basis table
        basis_now = active_cycle.adjusted_basis_per_share
        if basis_now is not None:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Assignment Strike", f"${active_cycle.assignment_strike:.2f}")
            b2.metric("− Total Credits", f"${running_credits:,.2f}")
            b3.metric("+ Total Debits", f"${running_debits:,.2f}")
            b4.metric("= Adjusted Basis", f"${basis_now:.2f}",
                      help="This is your true break-even. Never sell CC below this.")

        # Delete last leg (undo)
        if st.button("↩️ Delete last leg (undo)", use_container_width=False):
            last_leg = active_cycle.legs[-1]
            wheel_store.delete_leg(last_leg.id)
            # Re-derive state from remaining legs
            remaining = active_cycle.legs[:-1]
            if remaining:
                last_type = remaining[-1].leg_type
                try:
                    # Walk the state machine from scratch
                    state = WheelState.CSP_OPEN
                    for rl in remaining[1:]:
                        state = next_state(state, rl.leg_type)
                    wheel_store.update_cycle_state(active_cycle.id, state.value)
                except Exception:
                    pass
            else:
                wheel_store.update_cycle_state(active_cycle.id, WheelState.CSP_OPEN.value)
            st.rerun()

    # ================================================================
    # Cycle history
    # ================================================================
    if all_cycles:
        with st.expander(f"All cycles ({len(all_cycles)} total)"):
            cycle_rows = []
            for c in all_cycles:
                full = wheel_store.load_cycle(c["id"])
                row = {
                    "Cycle #": c["id"],
                    "State": c["state"].replace("_", " "),
                    "Started": c["started_date"],
                    "Closed": c["closed_date"] or "—",
                    "Net Premium": f"${full.total_net_premium:,.2f}" if full else "—",
                    "Adj. Basis": f"${full.adjusted_basis_per_share:.2f}" if full and full.adjusted_basis_per_share else "—",
                    "Realised P&L": f"${full.realised_pnl:+,.2f}" if full and full.realised_pnl is not None else "Open",
                    "Legs": len(full.legs) if full else 0,
                }
                cycle_rows.append(row)
            st.dataframe(pd.DataFrame(cycle_rows), use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 5 — GME Context Panel
# ===========================================================================
with tabs[4]:
    st.subheader("🔍 GME Context Panel")
    st.caption("SEC EDGAR filings · earnings · short interest · FTD data · Roaring Kitty detection")

    gate = _catalyst_gate

    # ================================================================
    # 1. Catalyst Gate Banner — always first
    # ================================================================
    getattr(st, gate.banner_color)(gate.banner_text())

    st.divider()

    # ================================================================
    # 2. Earnings Countdown
    # ================================================================
    st.subheader("Earnings")
    if gate.earnings_date:
        dte_color = "error" if (gate.earnings_dte or 99) <= 7 else \
                    "warning" if (gate.earnings_dte or 99) <= 21 else "success"
        e1, e2 = st.columns(2)
        e1.metric(
            "Next Earnings Date",
            str(gate.earnings_date),
            delta=f"{gate.earnings_dte} days away",
            delta_color="inverse" if (gate.earnings_dte or 99) <= 7 else "normal",
        )
        e2.metric("Earnings DTE", f"{gate.earnings_dte}")
        if (gate.earnings_dte or 99) <= 7:
            st.error(
                "**Earnings within 7 DTE — catalyst gate hard block active.** "
                "IV is artificially elevated into the event. After earnings, IV will crush — "
                "sell premium after, not before."
            )
        elif (gate.earnings_dte or 99) <= 21:
            st.warning(
                f"Earnings in {gate.earnings_dte} days. "
                "Avoid selling options that expire after the earnings date — "
                "IV crush will eliminate extrinsic value regardless of direction."
            )
    else:
        st.info("No confirmed earnings date found. GameStop typically reports quarterly — "
                "check sec.gov/cgi-bin/browse-edgar for 10-Q/10-K filing schedules.")

    st.divider()

    # ================================================================
    # 3. Short Interest
    # ================================================================
    st.subheader("Short Interest Snapshot")
    st.caption("Source: yfinance (FINRA bi-monthly data, ~2 week lag). For real-time: Ortex.")

    si1, si2, si3, si4 = st.columns(4)
    si1.metric(
        "Shares Short",
        f"{gate.shares_short/1e6:.1f}M" if gate.shares_short else "—",
        help="Total shares sold short (FINRA, bi-monthly)",
    )
    si2.metric(
        "Short % of Float",
        f"{gate.short_pct_float*100:.1f}%" if gate.short_pct_float else "—",
        delta=f"{gate.short_change_pct:+.1f}% MoM" if gate.short_change_pct else None,
        delta_color="inverse",
        help="Short interest as % of float. >20% = elevated squeeze potential.",
    )
    si3.metric(
        "Days to Cover",
        f"{gate.short_ratio:.1f}" if gate.short_ratio else "—",
        help="Shares short / avg daily volume. Higher = more days to unwind short positions.",
    )
    si4.metric(
        "Prior Month Shares Short",
        f"{gate.shares_short_prior_month/1e6:.1f}M" if gate.shares_short_prior_month else "—",
    )

    if gate.short_pct_float and gate.short_pct_float > 0.20:
        st.warning(
            f"Short interest at **{gate.short_pct_float*100:.1f}% of float** — "
            "elevated squeeze potential if combined with negative GEX and a catalyst. "
            "Not a timing signal — GME has been heavily shorted for years."
        )

    with st.expander("Short interest and options sellers"):
        st.markdown("""
**High short interest + negative GEX = the structural setup for both 2021 and May 2024.**

For options sellers on GME:
- **High short interest** means a large pool of shorts who must buy to cover if price spikes
- **Negative GEX** means dealer hedging *amplifies* that buying pressure
- **The ATM offering pattern:** GameStop management has now executed two massive dilutions
  into price spikes ($933M May 2024, $2.14B June 2024). This is the *single biggest risk*
  for call holders — the company prints shares at the spike, crushes price

**What to watch:**
- Short interest rising MoM + price stable = pressure building
- Short interest falling fast + price rising = short covering in progress (momentum trade)
- Days-to-cover > 5 = squeezable if catalyst hits
        """)

    st.divider()

    # ================================================================
    # 4. SEC EDGAR Filings
    # ================================================================
    st.subheader("SEC EDGAR — Recent Filings")

    col_8k, col_4 = st.columns(2)

    with col_8k:
        st.markdown("**8-K Filings (past 14 days)**")
        if gate.recent_8k_filings:
            atm_filings = [f for f in gate.recent_8k_filings if f.get("is_atm")]
            if atm_filings:
                st.error(f"🚨 {len(atm_filings)} POSSIBLE ATM OFFERING FILING(S) DETECTED")

            rows_8k = [{
                "Date": str(f["filed_date"]),
                "Title": f["title"][:80],
                "ATM?": "⚠️ POSSIBLE" if f.get("is_atm") else "—",
                "Link": f["url"],
            } for f in gate.recent_8k_filings]
            st.dataframe(pd.DataFrame(rows_8k), use_container_width=True, hide_index=True)
        else:
            st.success("No 8-K filings in the past 14 days.")

        with st.expander("Why 8-K ATM detection matters"):
            st.markdown("""
GameStop has executed two massive at-the-market (ATM) equity offerings into price spikes:
- **May 2024:** 45M shares → ~$933M gross proceeds (Form 8-K, May 24, 2024)
- **June 2024:** 75M shares → ~$2.14B completed June 11, 2024

**Pattern:** price spikes on social catalyst → company files 8-K announcing ATM → dilution crushes spike.
An 8-K in the past 14 days, especially one mentioning "prospectus supplement" or "sales agreement",
is a major warning for call holders. The company has demonstrated willingness to dilute into rallies.
            """)

    with col_4:
        st.markdown("**Form 4 — Insider Transactions (past 30 days)**")
        if gate.recent_form4_filings:
            rows_4 = [{
                "Date": str(f["filed_date"]),
                "Filing": f["title"][:80],
                "Link": f["url"],
            } for f in gate.recent_form4_filings]
            st.dataframe(pd.DataFrame(rows_4), use_container_width=True, hide_index=True)
            st.caption(
                "Ryan Cohen (RC Ventures) holds ~36.84M shares. Any Form 4 from him "
                "or board members is worth reading. Large buys = bullish signal. "
                "Sales into a spike = follow the playbook."
            )
        else:
            st.info("No Form 4 filings in the past 30 days.")

    st.divider()

    # ================================================================
    # 5. FTD Data
    # ================================================================
    st.subheader("Fail-to-Deliver (FTD) Data")
    st.caption("Source: SEC FOIA bi-monthly CSVs. Data lags ~2 weeks. Cached 24h.")

    ftd_df, ftd_metrics = load_ftd()

    if ftd_df.empty:
        st.warning("FTD data unavailable — SEC may be rate-limiting or data not yet published.")
    else:
        fm1, fm2, fm3, fm4 = st.columns(4)
        fm1.metric("Total FTDs (6 months)", f"{ftd_metrics.get('total', 0):,}")
        fm2.metric("Peak FTD Day", ftd_metrics.get("peak_date", "—"))
        fm2.caption(f"{ftd_metrics.get('peak_qty', 0):,} shares")
        fm3.metric("30-day Avg Daily FTD", f"{ftd_metrics.get('recent_30d_avg', 0):,.0f}")
        trend_icon = {"rising": "📈", "falling": "📉", "flat": "➡️", "unknown": "❓"}.get(
            ftd_metrics.get("trend", "unknown"), "❓"
        )
        fm4.metric("FTD Trend", f"{trend_icon} {ftd_metrics.get('trend', '—').upper()}")

        if ftd_metrics.get("trend") == "rising":
            st.warning(
                "FTD quantity is rising vs the prior 30-day period. "
                "Elevated FTDs suggest short sellers are having difficulty locating shares to borrow. "
                "The T+35 delivery cycle historically correlates with GME price pressure ~35 days later."
            )

        if "settle_date" in ftd_df.columns and "quantity" in ftd_df.columns:
            chart_df = ftd_df.set_index("settle_date")[["quantity"]].rename(
                columns={"quantity": "FTD Quantity (shares)"}
            )
            st.line_chart(chart_df, use_container_width=True)

        with st.expander("What are FTDs and why they matter"):
            st.markdown("""
A **Fail-to-Deliver** occurs when a seller doesn't deliver shares by the settlement date (T+2).

For GME, FTDs are significant because:
- **T+35 delivery cycle:** Under SEC Reg SHO Rule 204, fails in threshold securities trigger
  mandatory close-out obligations. The ~T+35 pattern around GME's high-volatility events has
  been documented in peer-reviewed research (Pastorek, *Finance a úvěr*, 2023)
- **Short interest proxy:** Persistent FTDs alongside high short interest can indicate
  synthetic short positions (selling without locating shares first)
- **Not a timing signal alone** — FTDs tell you *pressure exists*, not *when it releases*

Combine FTDs with: GEX regime, short interest, social volume, and catalyst presence
for a fuller picture of squeeze probability.
            """)

    st.divider()

    # ================================================================
    # 6. Roaring Kitty / Social Activity
    # ================================================================
    st.subheader("Roaring Kitty / Social Spike Detector")

    if gate.rk_spike:
        st.error(
            f"🚨 **RK SPIKE DETECTED** — {gate.rk_post_count} posts with Roaring Kitty / DFV "
            f"keywords on Reddit in the past 6 hours.\n\n"
            f"Historical pattern: RK post → multi-day momentum → ATM offering. "
            f"**Avoid selling covered calls.** If you hold shares, consider waiting "
            f"5–10 days before opening new CC positions."
        )
        if gate.rk_sample_titles:
            st.markdown("**Sample posts detected:**")
            for title in gate.rk_sample_titles:
                st.markdown(f"- {title}")
    else:
        st.success(
            f"No Roaring Kitty / DFV keyword spike detected in the past 6 hours "
            f"({gate.rk_post_count} matching posts — below threshold of 3)."
        )

    with st.expander("Keith Gill / Roaring Kitty — the GME catalyst"):
        st.markdown("""
**Keith Gill ("Roaring Kitty" / "DeepFuckingValue")** is the single most impactful individual
catalyst for GME options pricing.

Key events:
- **January 2021:** His r/WallStreetBets posts catalysed the initial squeeze from $17 → $483
- **May 13, 2024:** Returned to Twitter/X after 3 years → GME rallied, short sellers suffered
  **$838M mark-to-market loss** on that single day (S3 Partners / Ihor Dusaniwsky)
- **June 2, 2024:** Posted a Reddit position update showing 5M shares + 120,000 call contracts
  at the $20 strike expiring June 21, 2024

**What to do when RK spike fires:**
- Do NOT sell covered calls — the rally may continue for 3–7 days
- If you hold long shares, enjoy the ride but set mental stops
- Watch for a GameStop ATM filing (8-K) within 5–10 days of any major spike
- If you're considering a LEAP — wait for the post-spike IV crush before buying
        """)

    # Link to X / Superstonk for manual monitoring
    st.info(
        "**Manual monitoring:** "
        "[r/Superstonk](https://reddit.com/r/Superstonk) · "
        "[r/GMEOptions](https://reddit.com/r/GMEOptions) · "
        "[SEC EDGAR GME Filings](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001326380&type=&dateb=&owner=include&count=40)"
    )


# ===========================================================================
# TAB 6 — My Positions  (stored privately in gme_intel.db, gitignored)
# ===========================================================================
with tabs[5]:
    st.subheader("💼 My Positions")
    st.caption(
        "Stored in `gme_intel.db` (gitignored — never committed to the repo). "
        "Log shares, warrants, and long calls to track real P&L."
    )

    # ----------------------------------------------------------------
    # Current price (reuse what Tab 1 already fetched)
    # ----------------------------------------------------------------
    _pos_price = chain_dict.get("spot", 0.0) if chain_dict else 0.0
    _pos_chain = OptionsChainSnapshot.model_validate(chain_dict) if chain_dict else None

    # ----------------------------------------------------------------
    # Add / edit position form
    # ----------------------------------------------------------------
    st.subheader("➕ Log a Position")
    with st.form("add_position_form", clear_on_submit=True):
        pf1, pf2 = st.columns(2)
        ptype = pf1.selectbox(
            "Position type",
            ["shares", "warrant", "call"],
            format_func=lambda x: {"shares": "Shares", "warrant": "Warrant", "call": "Long Call"}[x],
        )
        qty = pf2.number_input(
            "Quantity",
            min_value=0.01, value=100.0, step=1.0,
            help="Shares: number of shares. Warrant / Call: number of units / contracts."
        )

        pf3, pf4 = st.columns(2)
        cost_basis = pf3.number_input(
            "Cost basis (per share / per unit / premium per share for calls)",
            min_value=0.0, value=0.0, step=0.01,
            help="Shares: avg cost per share. Warrant: price paid per warrant. Call: premium paid per share (contract = 100 shares)."
        )

        strike = None
        expiry = None
        if ptype in ("warrant", "call"):
            strike = pf4.number_input("Strike ($)", min_value=0.01, value=20.0, step=0.5)
            pf5, pf6 = st.columns(2)
            expiry_input = pf5.text_input(
                "Expiry (YYYY-MM-DD)",
                placeholder="e.g. 2027-01-15",
            )
            expiry = expiry_input.strip() if expiry_input.strip() else None
        else:
            pf4.empty()

        notes = st.text_input("Notes (optional)", placeholder="e.g. Bought on dip, avg down from $28")

        submitted_pos = st.form_submit_button("Add Position", type="primary")
        if submitted_pos:
            try:
                pos_store.upsert_position(
                    position_type=ptype,
                    quantity=float(qty),
                    cost_basis=float(cost_basis),
                    strike=float(strike) if strike else None,
                    expiry=expiry,
                    notes=notes,
                )
                st.success("Position saved.")
                st.rerun()
            except Exception as e:
                st.error(f"Error saving position: {e}")

    st.divider()

    # ----------------------------------------------------------------
    # Portfolio summary
    # ----------------------------------------------------------------
    positions = pos_store.get_positions()

    if not positions:
        st.info("No positions logged yet. Add one above.")
    else:
        summary = pos_store.summary(current_price=_pos_price, chain_snapshot=_pos_chain)

        # Top-line metrics
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("GME Price (live)", f"${_pos_price:.2f}" if _pos_price else "—")
        sm2.metric(
            "Total Cost",
            f"${summary['total_cost']:,.2f}",
            help="Sum of all cost bases across every position type."
        )
        sm3.metric(
            "Total Market Value",
            f"${summary['total_market_value']:,.2f}" if summary['total_market_value'] else "—",
            help="Shares: qty × live price. Calls: marked to chain mid. Warrants: intrinsic only."
        )
        pnl = summary["total_pnl"]
        pnl_pct = pnl / summary["total_cost"] * 100 if summary["total_cost"] else 0.0
        sm4.metric(
            "Total P&L",
            f"${pnl:+,.2f}" if summary['total_market_value'] else "—",
            delta=f"{pnl_pct:+.1f}%" if summary['total_market_value'] else None,
            delta_color="normal" if pnl >= 0 else "inverse",
        )

        # Wheel basis cross-reference (if active cycle exists)
        if _tracked_basis and summary["shares_qty"] > 0:
            tb_col1, tb_col2 = st.columns(2)
            tb_col1.metric(
                "Shares Avg Cost",
                f"${summary['shares_avg_cost']:.2f}",
                help="Average cost basis from your position ledger."
            )
            tb_col2.metric(
                "Wheel Adjusted Basis",
                f"${_tracked_basis:.2f}",
                delta=f"${summary['shares_avg_cost'] - _tracked_basis:+.2f} vs position cost",
                delta_color="off",
                help="Adjusted basis from Wheel Tracker (assignment − all premiums). "
                     "This is your true break-even for CC decisions."
            )

        st.divider()

        # ----------------------------------------------------------------
        # Position rows by type
        # ----------------------------------------------------------------
        shares_rows  = [p for p in summary["breakdown"] if p["Type"] == "Shares"]
        warrant_rows = [p for p in summary["breakdown"] if p["Type"] == "Warrant"]
        call_rows    = [p for p in summary["breakdown"] if p["Type"] == "Long Call"]

        def _delete_btn(pos_id: int, label: str) -> bool:
            return st.button(f"🗑️ Delete", key=f"del_{pos_id}", help=f"Remove {label}")

        if shares_rows:
            st.markdown("#### Shares")
            for row in shares_rows:
                with st.container():
                    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 2, 1])
                    c1.metric("Qty", f"{row['Qty']:,.0f}")
                    c2.metric("Avg Cost", row["Avg Cost"])
                    c3.metric("Mkt Value", row["Mkt Value"])
                    c4.metric("P&L", row["P&L"], delta=row["P&L %"], delta_color="normal" if "+" in row["P&L"] else "inverse")
                    c5.caption(row["Notes"] or "")
                    with c6:
                        if _delete_btn(row["id"], "shares"):
                            pos_store.delete_position(row["id"])
                            st.rerun()

        if warrant_rows:
            st.markdown("#### Warrants")
            for row in warrant_rows:
                with st.container():
                    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 2, 1])
                    c1.metric("Qty", f"{row['Qty']:,.0f}")
                    c2.metric("Strike", row.get("Strike", "—"))
                    c3.metric("Expiry", row.get("Expiry", "—"))
                    c4.metric("Intrinsic", row.get("Intrinsic", "—"))
                    c5.caption(row["Notes"] or "")
                    with c6:
                        if _delete_btn(row["id"], "warrant"):
                            pos_store.delete_position(row["id"])
                            st.rerun()

        if call_rows:
            st.markdown("#### Long Calls")
            for row in call_rows:
                with st.container():
                    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1, 2, 1])
                    c1.metric("Contracts", row["Qty"])
                    c2.metric("Strike", row.get("Strike", "—"))
                    c3.metric("Expiry", row.get("Expiry", "—"))
                    c4.metric("DTE", str(row.get("DTE", "—")) if row.get("DTE") is not None else "—")
                    c5.metric("Mkt Value", row.get("Mkt Value", "—"))
                    c6.caption(row["Notes"] or "")
                    with c7:
                        if _delete_btn(row["id"], "call"):
                            pos_store.delete_position(row["id"])
                            st.rerun()

        with st.expander("Position tracking notes"):
            st.markdown("""
- **Shares P&L** is marked to the live GME price fetched from your data provider.
- **Warrants** show intrinsic value only (max(spot − strike, 0)). Time value is not modelled — warrants may be worth more than shown.
- **Long Calls** are marked to the mid price in the current options chain. If the strike/expiry doesn't match any contract in the loaded chain, market value will show "—".
- All data lives in `gme_intel.db` (gitignored). Nothing here is ever committed to GitHub.
            """)
