"""
GME Tactical Dashboard v4
Tabs: Price & Signals | Options Chain | Reddit Intel
"""
import pandas as pd
import streamlit as st
import yfinance as yf

from modules.reddit import RedditScraper
from modules.technicals import get_support_resistance

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

tabs = st.tabs(["📈 Price & Signals", "🧠 Options Chain", "👾 Reddit Intel"])


# ---------------------------------------------------------------------------
# Shared data loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_price_data():
    ticker = yf.Ticker("GME")
    df = ticker.history(period="1y")   # fetch 1y so MA100 is valid throughout
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()
    delta = df["Close"].diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    df["RSI"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std  = df["Volume"].rolling(20).std()
    df["vol_zscore"] = (df["Volume"] - vol_mean) / vol_std.replace(0, float("nan"))
    df["vol_spike"]  = df["vol_zscore"] >= 2.0
    return df

@st.cache_data(ttl=300)
def get_iv_avg():
    try:
        t = yf.Ticker("GME")
        exp = t.options[0]
        chain = t.option_chain(exp)
        price = t.history(period="1d")["Close"].iloc[-1]
        all_opts = pd.concat([chain.calls, chain.puts])
        atm = all_opts.copy()
        atm["diff"] = abs(atm["strike"] - price)
        return atm.nsmallest(8, "diff")["impliedVolatility"].mean()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Signal engine
# ---------------------------------------------------------------------------
def build_signals(df, sr_levels, iv_avg=None):
    latest = df.iloc[-1]
    price  = latest["Close"]
    rsi    = latest["RSI"]
    ma20   = latest["MA20"]
    ma50   = latest["MA50"]
    vol_spike = latest["vol_spike"]

    supports    = sorted([l for l in sr_levels if l.kind in ("support", "mixed")],  key=lambda x: x.level, reverse=True)
    resistances = sorted([l for l in sr_levels if l.kind in ("resistance", "mixed")], key=lambda x: x.level)
    nearest_sup = supports[0].level    if supports    else None
    nearest_res = resistances[0].level if resistances else None

    if price > ma20 > ma50:
        trend, trend_note = "bullish", f"Price \\${price:.2f} > MA20 \\${ma20:.2f} > MA50 \\${ma50:.2f}"
    elif price < ma20 < ma50:
        trend, trend_note = "bearish", f"Price \\${price:.2f} < MA20 \\${ma20:.2f} < MA50 \\${ma50:.2f}"
    else:
        trend, trend_note = "neutral", f"Price \\${price:.2f} between MAs — choppy/consolidating"

    rsi_zone = "oversold" if rsi < 35 else ("overbought" if rsi > 65 else "neutral")

    if iv_avg is not None:
        iv_regime = "high" if iv_avg > 0.80 else ("low" if iv_avg < 0.40 else "normal")
    else:
        iv_regime = "unknown"

    p  = lambda x: f"USD {x:.2f}"   # safe price formatter — no $ in markdown
    pk = lambda x: f"USD {x:,.0f}"  # safe large number formatter

    signals = []

    # ---- Situation assessment (always shown first) ----
    if trend == "bearish" and rsi_zone != "oversold":
        situation = "wait"
        situation_msg = (
            f"Price is in a **downtrend** (below MA20 and MA50) and RSI is not yet oversold. "
            f"This is not a great time to buy options or open new positions — you'd be fighting the trend. "
            f"**If you hold 100 shares**, a covered call is still valid to collect some income while you wait."
        )
    elif trend == "bearish" and rsi_zone == "oversold":
        situation = "watch"
        situation_msg = (
            f"Price is in a downtrend BUT RSI is oversold (below 35) — meaning the selling may be exhausted. "
            f"This is a cautious watch zone. Wait for price to stop making lower lows before entering. "
            f"A LEAP could make sense here if you believe in the long-term thesis and IV is low."
        )
    elif trend == "neutral":
        situation = "ok"
        situation_msg = (
            f"Price is choppy — not clearly trending up or down. "
            f"This is actually a good environment for **premium selling** (covered calls, CSPs). "
            f"You collect income while the stock moves sideways."
        )
    else:  # bullish
        situation = "good"
        situation_msg = (
            f"Price is in an **uptrend** (above both MA20 and MA50). "
            f"Good environment for covered calls above resistance, CSPs below support, or a LEAP if IV is low."
        )

    signals.append({
        "type": "📍 Situation",
        "action": {"wait": "Not ideal to enter new positions right now",
                   "watch": "Oversold — watch for reversal signal",
                   "ok":   "Sideways market — good for premium selling",
                   "good": "Uptrend — good conditions overall"}.get(situation),
        "why": situation_msg,
        "when": "",
        "risk": "—",
        "beginner_note": "",
        "is_situation": True,
    })

    # COVERED CALL — works in any trend if you hold shares
    if nearest_res:
        cc_strike = round(nearest_res * 1.02, 0)
        if trend == "bearish":
            cc_why = (
                f"Even in a downtrend, if you hold 100 GME shares you can sell a covered call to collect income. "
                f"Resistance is at {p(nearest_res)} — sell a call above that at {p(cc_strike)}. "
                f"If the stock keeps drifting down your call expires worthless and you keep the premium."
            )
        else:
            cc_why = (
                f"Resistance at {p(nearest_res)} — price tends to stall there. "
                f"Sell a call just above it at {p(cc_strike)}. You collect premium upfront. "
                f"If GME stays below your strike at expiry, you keep the premium AND your 100 shares."
            )
        signals.append({
            "type": "🔵 Covered Call",
            "action": f"Sell the {p(cc_strike)} call — need 100 shares as collateral",
            "why": cc_why,
            "when": "2–4 weeks to expiry is the sweet spot (fastest time decay). Pick a strike above the nearest resistance level.",
            "risk": "Low",
            "beginner_note": f"Your max profit is the premium collected. Your only 'loss' is if GME rockets past your strike and your shares get called away — you still made money, just missed extra upside.",
        })

    # CSP — only suggest if trend isn't strongly bearish
    if nearest_sup and iv_regime in ("high", "normal") and trend != "bearish":
        csp_strike = round(nearest_sup * 0.97, 0)
        signals.append({
            "type": "🔵 Cash Secured Put",
            "action": f"Sell the {p(csp_strike)} put — need {pk(csp_strike * 100)} cash set aside",
            "why": (
                f"Support is at {p(nearest_sup)}. You agree to buy 100 shares at {p(csp_strike)} "
                f"(below support). If GME stays above that you keep the cash. "
                f"If it drops below, you own 100 shares at a price you were comfortable with anyway."
            ),
            "when": "Only do this if you genuinely want to own 100 GME shares. Don't do it just for the premium if you'd hate being assigned.",
            "risk": "Low-Med",
            "beginner_note": "This is how you start the Wheel strategy. Collect premium → get assigned shares → sell covered calls on those shares → repeat.",
        })
    elif trend == "bearish" and nearest_sup:
        signals.append({
            "type": "⏸️ CSP — Hold Off",
            "action": "Don't sell a cash secured put right now",
            "why": (
                f"Trend is bearish. Selling a CSP in a downtrend risks getting assigned shares "
                f"that keep falling. Support at {p(nearest_sup)} has already been tested — "
                f"wait for the trend to stabilise before opening a put position."
            ),
            "when": "Wait for price to reclaim MA20 and show at least 2–3 days of holding above a support level.",
            "risk": "—",
            "beginner_note": "Patience is part of the strategy. u/Crybad explicitly mentions watching for these conditions before opening new CSPs.",
        })

    # LEAPS
    if iv_regime not in ("high", "unknown"):
        leap_strike = round(price * 0.85, 0)
        if trend == "bearish":
            leap_why = (
                f"IF you have strong conviction GME goes higher over 1-2 years, a LEAP at {p(leap_strike)} "
                f"costs a fraction of buying 100 shares. Your max loss is exactly what you pay — nothing more. "
                f"But the current downtrend means wait for a reversal signal before buying."
            )
        else:
            leap_why = (
                f"A Jan 2027 call at {p(leap_strike)} (about 15% below current price) "
                f"gives you high delta — it moves close to dollar-for-dollar with GME. "
                f"Much cheaper than buying 100 shares outright. IV is {iv_regime} so pricing is reasonable."
            )
        signals.append({
            "type": "🚀 LEAP",
            "action": f"Buy Jan 2027 {p(leap_strike)} call — deep ITM, high delta",
            "why": leap_why,
            "when": "Buy when IV is low (options are cheap). Avoid buying LEAPs when IV is high — you overpay and IV crush will hurt you.",
            "risk": "Med",
            "beginner_note": f"Your maximum loss is 100% of what you pay for the LEAP — so only spend what you're comfortable losing entirely. In return you control 100 shares for 1-2 years.",
        })

    if vol_spike:
        signals.append({
            "type": "⚠️ Volume Spike",
            "action": "Unusual volume today — check Reddit Intel for a catalyst",
            "why": f"Volume is {latest['vol_zscore']:.1f} standard deviations above the 20-day average. Big volume often precedes or confirms a directional move.",
            "when": "Don't trade on the spike itself. See if price holds direction the next day before entering.",
            "risk": "—",
            "beginner_note": "High volume + price moving up = conviction. High volume + price flat = indecision. Check what's being discussed on Reddit.",
        })

    # CONFLUENCE DETECTOR — all signals pointing the same direction
    confluence_hits = []
    if rsi_zone == "oversold":          confluence_hits.append(f"RSI oversold ({rsi:.0f})")
    if trend == "bullish":              confluence_hits.append("price above MA20 and MA50")
    if vol_spike:                       confluence_hits.append("volume spike")
    if iv_avg and iv_avg > 0.50:        confluence_hits.append(f"IV elevated ({iv_avg*100:.0f}% — premium selling favoured)")
    if nearest_sup and price > nearest_sup * 0.99:
        confluence_hits.append(f"price holding above support ({p(nearest_sup)})")

    if len(confluence_hits) >= 3:
        signals.append({
            "type": "🔥 Confluence Signal",
            "action": f"{len(confluence_hits)}/5 signals aligned — stronger conviction setup",
            "why": "Multiple independent indicators are pointing in the same direction at the same time. This is when setups have the highest probability of working out.\n\n" +
                   "\n".join(f"- {h}" for h in confluence_hits),
            "when": "This doesn't guarantee anything — but it's the kind of setup u/Crybad looks for before sizing up a position.",
            "risk": "—",
            "beginner_note": "Confluence = multiple reasons to act. One indicator alone is noise. Three or more together is a signal worth paying attention to.",
        })

    return signals, {
        "price": price, "rsi": rsi, "trend": trend,
        "trend_note": trend_note, "rsi_zone": rsi_zone,
        "iv_regime": iv_regime, "vol_spike": vol_spike,
        "nearest_support": nearest_sup, "nearest_resistance": nearest_res,
    }


# ===========================================================================
# TAB 1 — Price & Signals
# ===========================================================================
with tabs[0]:
    df = load_price_data()
    iv_avg = get_iv_avg()
    sr_levels = get_support_resistance(df, price_col="Close", window=5, max_levels=8)
    signals, ctx = build_signals(df, sr_levels, iv_avg)

    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    price_delta = latest["Close"] - prev["Close"]
    price_pct   = price_delta / prev["Close"] * 100

    # ---- Metrics row ----
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("GME Price", f"${latest['Close']:.2f}", f"{price_delta:+.2f} ({price_pct:+.2f}%)")
    c2.metric("MA20",  f"${latest['MA20']:.2f}",  help="20-day moving average — short-term trend")
    c3.metric("MA50",  f"${latest['MA50']:.2f}",  help="50-day moving average — medium-term trend")
    c4.metric("RSI",   f"{latest['RSI']:.1f}",
              delta="oversold ✅" if ctx["rsi_zone"] == "oversold" else "overbought ⚠️" if ctx["rsi_zone"] == "overbought" else None,
              delta_color="normal",
              help="Relative Strength Index — measures momentum. Below 35 = oversold (potential bounce). Above 65 = overbought (potential pullback).")
    c5.metric("IV Regime", ctx["iv_regime"].upper() if iv_avg else "N/A",
              f"{iv_avg*100:.0f}%" if iv_avg else None, delta_color="off",
              help="Implied Volatility regime. HIGH = options are expensive, good time to SELL premium (covered calls). LOW = options are cheap, good time to BUY (LEAPS).")
    c6.metric("Vol Spike", "YES ⚠️" if latest["vol_spike"] else "No", delta_color="off",
              help="True if today's volume is 2+ standard deviations above the 20-day average. Big volume often precedes big moves.")

    # ---- Trend banner ----
    trend_colors = {"bullish": "success", "bearish": "error", "neutral": "info"}
    getattr(st, trend_colors[ctx["trend"]])(f"**Trend: {ctx['trend'].upper()}** — {ctx['trend_note']}")

    # ---- Indicator explainer ----
    with st.expander("📚 What do these indicators mean?"):
        st.markdown("""
**Moving Averages (MA20 / MA50 / MA100)**
A moving average smooths out daily price noise to show the underlying trend.
- Price **above** MA20 > MA50 = short and medium term both bullish — good environment for covered calls and LEAPS
- Price **below** all MAs = downtrend — wait for stabilisation before entering long plays
- MA20 crossing **above** MA50 = "golden cross" — strong bullish signal

**RSI (Relative Strength Index) — scale 0 to 100**
Measures whether a stock is being bought or sold too aggressively.
- **Below 35** = oversold — stock has been beaten down, potential bounce. Good time to sell a CSP or buy a LEAP.
- **Above 65** = overbought — stock has run up fast, may pull back. Good time to sell a covered call.
- **35–65** = neutral — no strong lean either way.

**IV (Implied Volatility)**
How expensive options are right now, expressed as a percentage.
- **High IV (>80%)** = options are pricey. SELL premium — covered calls, CSPs. You collect more cash.
- **Low IV (<40%)** = options are cheap. BUY options — LEAPS. You pay less for time.
- **Why does this matter?** If you buy a LEAP when IV is high you overpay. If you sell a covered call when IV is low you underearn.

**Support & Resistance**
- **Support** = a price floor where buyers historically step in. Good place for CSP strikes — below support.
- **Resistance** = a price ceiling where sellers historically push back. Good place for covered call strikes — above resistance.

**Volume Spike**
Unusual volume (2+ standard deviations above average) often signals institutional activity or a catalyst. Don't chase — watch for confirmation the next day.
        """)

    st.divider()

    # ---- Trade Signals ----
    st.subheader("🎯 Trade Signals")
    st.caption("Based on current price, trend, RSI, IV, and support/resistance levels.")

    if not signals:
        st.info("No clear signals at this time — conditions are mixed.")
    else:
        for sig in signals:
            if sig.get("is_situation"):
                color = {"wait": "error", "watch": "warning", "ok": "info", "good": "success"}.get(
                    "wait" if "Not ideal" in sig["action"] else
                    "watch" if "Oversold" in sig["action"] else
                    "good" if "Uptrend" in sig["action"] else "ok", "info")
                getattr(st, color)(f"**{sig['type']} — {sig['action']}**\n\n{sig['why']}")
                continue

            risk_icon = {"Low": "🟢", "Low-Med": "🟡", "Med": "🟠", "—": "⚪"}.get(sig["risk"], "⚪")
            with st.expander(f"{sig['type']}  —  {sig['action']}", expanded=True):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"**Why:** {sig['why']}")
                    if sig["when"]:
                        st.markdown(f"**When:** {sig['when']}")
                    if sig["beginner_note"]:
                        st.info(f"💡 {sig['beginner_note']}")
                with col_b:
                    st.markdown("**Risk:**")
                    st.markdown(f"## {risk_icon} {sig['risk']}")

    with st.expander("📚 Which play is right for me as a beginner?"):
        st.markdown("""
**Start here — ranked safest to most aggressive:**

1. **🔵 Covered Call** ← best starting point
   - You already own 100 shares of GME
   - You sell someone the right to buy them at a higher price (your strike)
   - You collect cash (premium) upfront regardless of outcome
   - Worst case: your shares get "called away" at your strike — you still profited
   - *u/Crybad runs this every single week and shows his exact trades in r/GMEOptions*

2. **🔵 Cash Secured Put (CSP)**
   - You don't own shares yet but want to buy them cheaper
   - You collect premium to agree to buy 100 shares at a lower price
   - If price stays above your strike — you keep the cash, no shares
   - If price drops below — you own 100 shares at your agreed price (which you wanted anyway)
   - Requires capital: e.g. $21 strike = $2,100 set aside

3. **🚀 LEAP (Long-dated Call)**
   - You buy the RIGHT to purchase 100 shares at a set price, over 1-2 years
   - Much cheaper than buying 100 shares outright
   - Buy deep ITM (in the money) for high delta — moves like owning shares
   - Best when IV is LOW (options are cheap)
   - This is the "dip your toe in" play — limited to what you pay, that's your max loss

4. **🟢 Put Credit Spread** — more advanced, but your loss is always capped

**The Golden Rules:**
- Never risk more than you can afford to lose completely
- Covered calls and CSPs = you're the casino (collecting premium)
- LEAPS = you're making a long-term directional bet with defined risk
- Check IV before every trade — it changes everything about whether to buy or sell
        """)

    st.divider()

    # ---- Charts ----
    chart_df = df[["Close", "MA20", "MA50", "MA100"]].dropna()
    date_range = f"{chart_df.index[0].strftime('%b %Y')} – {chart_df.index[-1].strftime('%b %Y')}"
    st.subheader(f"Price & Moving Averages ({date_range})")
    st.line_chart(chart_df, use_container_width=True)

    col_rsi, col_vol = st.columns(2)
    with col_rsi:
        st.subheader("RSI (14)")
        st.line_chart(df[["RSI"]].dropna(), use_container_width=True)
        rsi_now = latest["RSI"]
        if rsi_now < 35:
            st.success(f"RSI {rsi_now:.1f} — oversold. Potential bounce territory. Consider LEAP or CSP.")
        elif rsi_now > 65:
            st.warning(f"RSI {rsi_now:.1f} — overbought. Consider covered call to collect premium on the run-up.")
        else:
            st.info(f"RSI {rsi_now:.1f} — neutral zone. No strong momentum signal.")

    with col_vol:
        st.subheader("Volume")
        st.bar_chart(df[["Volume"]], use_container_width=True)
        if latest["vol_spike"]:
            st.warning(f"⚠️ Volume spike today ({latest['vol_zscore']:.1f}σ) — check Reddit Intel for catalysts.")
        else:
            st.info("Volume normal today.")

    # ---- S/R Table ----
    if sr_levels:
        st.subheader("Support & Resistance Levels")
        with st.expander("📚 How to use support & resistance for options"):
            st.markdown("""
- **For Covered Calls:** sell your call strike *above* the nearest resistance level. Price has to break through resistance to get your shares called away.
- **For CSPs:** sell your put strike *below* the nearest support level. Price has to break through support before you get assigned shares.
- **Strength score:** 0–1. Higher = more times price has bounced at that level = more reliable.
- **Touches:** how many times price has tested that level. More touches = stronger level.
            """)
        sr_data = [{
            "Level": f"${l.level:.2f}",
            "Type": l.kind,
            "Strength": f"{l.strength:.2f}",
            "Touches": l.touches,
            "Last Touch": str(l.last_touch)[:10],
            "Use for": "CSP strike below this" if l.kind == "support" else "CC strike above this" if l.kind == "resistance" else "Either",
        } for l in sr_levels]
        st.dataframe(pd.DataFrame(sr_data), use_container_width=True)


# ===========================================================================
# TAB 2 — Options Chain
# ===========================================================================
with tabs[1]:
    st.subheader("Options Chain — IV & Greeks")

    # ---- Beginner explainer at the top ----
    with st.expander("📚 Options chain explained — what am I looking at?", expanded=False):
        st.markdown("""
The options chain shows every available contract for GME, organised by **strike price** and **expiry date**.

---
**The columns that matter most:**

| Column | What it means | Why you care |
|--------|--------------|-------------|
| **Strike** | The price you agree to buy/sell shares at | For covered calls: pick above current price. For LEAPS: pick below (deep ITM). |
| **Bid / Ask** | What buyers will pay / what sellers want | Your actual fill will be between these. The "premium" you collect or pay. |
| **IV (impliedVolatility)** | How expensive this specific contract is | High = expensive. Sell when high, buy when low. |
| **Volume** | Contracts traded today | High volume = active contract, easier to fill |
| **Open Interest (OI)** | Total open contracts | High OI = liquid, easy to enter/exit |
| **Delta** | How much option moves per $1 stock move | 0.30 delta CC = good sweet spot for premium. 0.70+ delta LEAP = moves like owning stock. |
| **Theta** | Premium lost per day from time decay | Negative for buyers (hurts you). Positive effect for sellers (works for you). |
| **Gamma** | How fast delta changes | High gamma near expiry = risky for sellers near the strike |

---
**Covered Call checklist:**
- ✅ Pick strike **above** current price (OTM — out of the money)
- ✅ Aim for **0.25–0.35 delta** — sweet spot of decent premium with lower chance of assignment
- ✅ Pick expiry **2–4 weeks out** — theta decay accelerates, you collect premium faster
- ✅ Higher IV = more premium collected
- ✅ Strike above nearest **resistance level** = extra cushion

**LEAP checklist:**
- ✅ Pick expiry **Jan 2027 or later** — you want time on your side
- ✅ Pick strike **below** current price (ITM — in the money), aim for **0.70+ delta**
- ✅ Buy when IV is **low** — options are cheap, you get more value
- ✅ Deep ITM LEAP moves almost like owning shares but costs a fraction of the capital

---
**IV Smile chart:** shows how IV varies across strike prices. The U-shape is normal — OTM options (far from current price) are more expensive because they're bets on big moves.
        """)

    @st.cache_data(ttl=180)
    def load_options():
        ticker = yf.Ticker("GME")
        expiries = ticker.options
        if not expiries:
            return pd.DataFrame(), [], 0.0
        all_chains = []
        price = ticker.history(period="1d")["Close"].iloc[-1]
        for exp in expiries[:4]:
            chain = ticker.option_chain(exp)
            calls = chain.calls.assign(type="call", expiry=exp)
            puts  = chain.puts.assign(type="put",  expiry=exp)
            all_chains.append(pd.concat([calls, puts]))
        combined = pd.concat(all_chains, ignore_index=True)
        combined["abs_diff"] = abs(combined["strike"] - price)
        return combined, expiries[:4], price

    combined, expiries, current_price = load_options()

    if combined.empty:
        st.warning("No options data available.")
    else:
        st.info(f"Current GME price: **${current_price:.2f}**  —  Select expiry and how many strikes to show around ATM (at the money).")

        col1, col2 = st.columns(2)
        expiry_choice = col1.selectbox("Expiry date", expiries,
            help="Covered calls: 2–4 weeks out. LEAPS: Jan 2027 or later.")
        n_strikes = col2.slider("Strikes around current price", 2, 12, 6,
            help="Shows this many strikes above and below the current price.")

        view  = combined[combined["expiry"] == expiry_choice].copy()
        calls = view[view["type"] == "call"].nsmallest(n_strikes, "abs_diff")
        puts  = view[view["type"] == "put"].nsmallest(n_strikes, "abs_diff")

        display_cols = [c for c in ["type", "strike", "bid", "ask", "lastPrice",
                                     "impliedVolatility", "volume", "openInterest",
                                     "delta", "gamma", "theta"] if c in view.columns]

        # Highlight good CC candidates (delta 0.25–0.35, OTM)
        def highlight_cc(row):
            try:
                if row.get("type") == "call" and row.get("strike", 0) > current_price:
                    delta = abs(row.get("delta") or 0)
                    if 0.20 <= delta <= 0.40:
                        return ["background-color: #1a3a1a"] * len(row)
            except Exception:
                pass
            return [""] * len(row)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Calls** — *green = good covered call candidate (delta 0.20–0.40, OTM)*")
            try:
                st.dataframe(calls[display_cols].reset_index(drop=True).style.apply(highlight_cc, axis=1),
                             use_container_width=True)
            except Exception:
                st.dataframe(calls[display_cols].reset_index(drop=True), use_container_width=True)

        with col_b:
            st.markdown("**Puts**")
            st.dataframe(puts[display_cols].reset_index(drop=True), use_container_width=True)

        # ---- Covered call suggester ----
        st.subheader("🔵 Covered Call Suggester")
        try:
            otm_calls = calls[calls["strike"] > current_price].copy()
            has_delta = "delta" in otm_calls.columns and otm_calls["delta"].notna().any()

            if has_delta:
                cc_candidates = otm_calls[otm_calls["delta"].abs().between(0.20, 0.40)]
                if cc_candidates.empty:
                    cc_candidates = otm_calls  # relax filter if nothing in range
            else:
                cc_candidates = otm_calls

            if cc_candidates.empty:
                st.info("No covered call candidates in current view — try a longer expiry or more strikes.")
            else:
                best = cc_candidates.nsmallest(1, "abs_diff").iloc[0]
                bid = best.get("bid") if pd.notna(best.get("bid", float("nan"))) else None
                ask = best.get("ask") if pd.notna(best.get("ask", float("nan"))) else None
                last = best.get("lastPrice") if pd.notna(best.get("lastPrice", float("nan"))) else None
                premium = (bid + ask) / 2 if (bid and ask) else (last or 0)
                delta_val = best.get("delta") if has_delta and pd.notna(best.get("delta", float("nan"))) else None
                iv_val = best.get("impliedVolatility")

                delta_str = f"- Delta: {delta_val:.2f} — ~{abs(delta_val)*100:.0f}% chance of being assigned\n" if delta_val else ""
                iv_str    = f"- IV: {iv_val*100:.0f}%\n" if iv_val and pd.notna(iv_val) else ""

                st.success(
                    f"**Suggested CC:** Sell the **${best['strike']:.0f} call** expiring **{expiry_choice}**\n\n"
                    f"- Collect ~**${premium:.2f}/share** = **${premium*100:.0f} per contract**\n"
                    f"{delta_str}{iv_str}"
                    f"- If GME stays below **${best['strike']:.0f}** at expiry you keep the premium AND your shares."
                )
        except Exception as e:
            st.info(f"Covered call suggester unavailable: {e}")

        # ---- Max pain ----
        try:
            oi_by_strike = view.groupby("strike")["openInterest"].sum()
            max_pain_strike = oi_by_strike.idxmax()
            st.info(f"📌 **Max Pain: ${max_pain_strike:.2f}** — the strike with highest combined open interest. Market makers are incentivised to keep price near here at expiry. Good reference for CC strikes.")
        except Exception:
            pass

        # ---- IV Smile ----
        st.subheader("IV Smile")
        with st.expander("📚 What is the IV smile?"):
            st.markdown("""
The IV smile shows implied volatility across different strike prices.
- The lowest IV is typically near the current price (ATM)
- IV rises on both sides — OTM options are more expensive relative to their probability
- A steep smile = market pricing in big moves (good time to sell premium via covered calls)
- A flat smile = cheap options across the board (good time to buy LEAPS)
            """)
        calls_iv = calls[["strike", "impliedVolatility"]].rename(columns={"impliedVolatility": "Call IV%"})
        puts_iv  = puts[["strike",  "impliedVolatility"]].rename(columns={"impliedVolatility": "Put IV%"})
        calls_iv["Call IV%"] *= 100
        puts_iv["Put IV%"]   *= 100
        smile = calls_iv.merge(puts_iv, on="strike", how="outer").set_index("strike").sort_index()
        st.line_chart(smile, use_container_width=True)


# ===========================================================================
# TAB 3 — Reddit Intel
# ===========================================================================
with tabs[2]:
    st.subheader("👾 Reddit Intel — GME Community Pulse")

    col_r1, col_r2, col_r3, col_r4 = st.columns([1, 1, 1, 1])
    sort_feed   = col_r1.selectbox("Feed", ["hot", "new", "top", "rising"], index=0)
    show_filter = col_r2.selectbox("Filter", [
        "All posts",
        "Options-relevant",
        "Catalysts",
        "Strategy posts (Wheel/Spreads)",
        "Weekly Plays (u/Crybad style)",
        "DD only",
        "Key authors",
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
                  help="Weighted — GMEOptions traders count more than general hype posts")
        s2.metric("Weighted Score",   f"{summary['weighted_compound']:.3f}",
                  help="Compound sentiment score weighted by subreddit signal quality. >0.05 = bullish, <-0.05 = bearish")
        gme_opts_val = summary.get("gmeoptions_compound")
        s3.metric("r/GMEOptions",     f"{gme_opts_val:.3f}" if gme_opts_val is not None else "N/A",
                  help="Sentiment from actual options traders only — the cleanest signal")
        s4.metric("🟢 Bullish",       summary["bullish"])
        s5.metric("⚪ Neutral",        summary["neutral"])
        s6.metric("🔴 Bearish",        summary["bearish"])
        s7.metric("⚡ Catalysts",      summary["catalyst_posts"])

        if summary["catalyst_posts"] > 0:
            st.warning(f"⚡ **{summary['catalyst_posts']} catalyst posts detected** — potential event-driven move")
        if summary["key_author_posts"] > 0:
            st.success(f"📌 **{summary['key_author_posts']} posts from key authors** (u/Crybad, u/terroristcavin etc.) — check Strategy or Key Authors filter")

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
            if row.get("is_key_author"):    tags.append("📌 key author")
            if row.get("is_weekly_plays"):  tags.append("📅 weekly plays")
            if row.get("is_options_relevant"): tags.append("📊 options")
            if row.get("is_catalyst"):      tags.append("⚡ catalyst")
            if row.get("is_strategy"):      tags.append("🔵 strategy")
            if row.get("is_dd"):            tags.append("🔬 DD")

            with st.container():
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**[{row['title']}]({row['url']})**")
                    st.caption(
                        f"{s_icon} {row['sentiment_label']}  ·  "
                        f"r/{row['subreddit']}  ·  "
                        f"{row['age_hours']:.1f}h ago  ·  "
                        f"u/{row['author']}  "
                        + ("  ".join(tags))
                    )
                with c2:
                    st.markdown(f"`{row['sentiment_compound']:+.3f}`")
            st.divider()
