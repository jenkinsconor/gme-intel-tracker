# GME Intel

A personal options research dashboard for GME built with Streamlit. Focused on premium-selling strategies — covered calls, cash-secured puts, and the Wheel — with real analytics rather than guesswork.

---

## What it does

- Pulls a live options chain and computes Black-Scholes Greeks independently (not relying on broker-supplied values)
- Tracks implied volatility history to compute IVR and IVP — the two metrics that tell you whether conditions actually favour selling premium
- Computes Gamma Exposure (GEX) to understand dealer positioning and identify the structural levels that matter for strike selection
- Runs a Wheel strategy ledger that tracks your true adjusted cost basis across every leg, and enforces the rule that you never sell a covered call below that basis
- Monitors SEC EDGAR for 8-K filings (ATM offering detection) and earnings dates, and gates trade signals accordingly
- Tracks your actual positions (shares, warrants, long calls) in a local SQLite database that never touches the repo

---

## Data providers

The app auto-selects a provider at startup in this order:

| Priority | Provider | Data quality | Cost |
|----------|----------|-------------|------|
| 1 | **Tradier** | Real-time, accurate IV, ORATS Greeks | Free brokerage account |
| 2 | Alpaca | Real-time OPRA snapshot | Free with account |
| 3 | yfinance | 15-min delayed, no Greeks | Free, no account |

The current provider and delay status are shown in the sidebar on launch. **Tradier is strongly recommended** — yfinance returns no Greeks and delayed quotes, which degrades the GEX computation and CC suggester.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/jenkinsconor/gme-intel-tracker.git
cd gme-intel-tracker

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set up Tradier (recommended)

1. Open a brokerage account at [tradier.com](https://tradier.com) — free, no minimums
2. Once approved, go to [developer.tradier.com](https://developer.tradier.com) and generate a production API token
3. Request options trading access if not already enabled (Account Settings → Options Level)
4. Set the token in your environment:

```bash
# Add to ~/.zshrc or ~/.bashrc for persistence
export TRADIER_TOKEN="your_production_token_here"
```

5. Restart the app — it detects the token automatically and switches to real-time data

> **Note:** Tradier's sandbox token (the one shown by default on developer.tradier.com) returns dummy data. You need the **production** token from your live brokerage account.

### 3. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

Add an alias to your shell config for quick daily access:

```bash
alias gme='cd ~/gme-intel-tracker && source venv/bin/activate && streamlit run app.py'
```

---

## Analytics

### Implied Volatility

- **Constant-maturity 30-day IV** — interpolated in variance space between the two expirations bracketing 30 DTE, giving a single stable IV number regardless of which expiry you're looking at
- **IV Rank (IVR)** — `(current IV − 52w low) / (52w high − 52w low) × 100`. Sell premium above 50, hard block below 30.
- **IV Percentile (IVP)** — percentage of past 252 trading days where IV was below current. More robust than IVR on names with spike history like GME, where a single event can anchor the denominator for a year.
- Both require ~20 days of local history to compute. The app accumulates one snapshot per day automatically.

### Black-Scholes Greeks

Greeks are computed independently using `py_vollib` rather than relying on broker-supplied values, which can lag or be miscalculated on illiquid strikes.

| Greek | Meaning | Use |
|-------|---------|-----|
| Delta (δ) | $ change per $1 move | Strike selection, POP estimate |
| Gamma (γ) | Rate of delta change | Risk near expiry |
| Theta (θ) | Premium decay per day | Works for you as a seller |
| Vega (ν) | Sensitivity to IV per 1% | Sizing during IV spikes |

POP (probability of max profit) is estimated as `1 − |delta|` and shown on every strike.

### Gamma Exposure (GEX)

GEX measures how much dollar hedging market makers must do per 1% move in GME's price. Computed from the full options chain using BSM gamma, open interest, and spot price.

- **Positive GEX** — dealers are net long gamma. They sell into rallies and buy dips. Price action is suppressed and mean-reverting. Good environment for selling premium.
- **Negative GEX** — dealers are net short gamma. They amplify moves in both directions. The structural mechanic behind the 2021 squeeze and the May 2024 Roaring Kitty rally. Poor environment for short options.
- **Call wall / Put wall** — strikes with the highest call and put GEX respectively. These are the levels dealers defend most aggressively and the most reliable inputs for strike selection.
- **Gamma flip** — the price level where net GEX changes sign. Above it: suppressing regime. Below it: amplifying.

### Expected Move

Two methods, both displayed:

- **IV-derived:** `Spot × IV × √(DTE/365)` — 1σ range with ~68% probability
- **Straddle-derived:** `ATM straddle price × 0.85` — market-implied move from actual option prices

Use these as strike selection boundaries. CSP below the lower bound, CC above the upper bound.

---

## Wheel tracker

Tracks each leg of a Wheel cycle in a local SQLite database. The ledger computes your adjusted cost basis after every leg:

```
Adjusted basis = assignment strike
               − Σ(put premiums collected)
               − Σ(call premiums collected)
               + Σ(buyback costs paid)
```

The CC suggester enforces a guardrail: if you try to log a covered call at a strike below your adjusted basis, the app warns you before you lock in a loss.

Roll suggestions are generated when an open position is within 21 DTE or has moved significantly against you.

---

## Catalyst gate

Trade signals are gated by:

| Condition | Effect |
|-----------|--------|
| Earnings ≤ 7 DTE | Hard block — no premium selling signals |
| 8-K filing in past 14 days | Warning — check for ATM offering language |
| Roaring Kitty keyword spike on Reddit (≥3 posts / 6h) | Warning — avoid short calls |
| IVR < 30 | Hard block — IV too low to sell premium |

GameStop has executed two large at-the-market equity offerings into price spikes (May 2024: $933M, June 2024: $2.14B). The 8-K watcher checks SEC EDGAR's RSS feed for filings containing ATM offering keywords.

---

## Fail-to-deliver data

FTD data is pulled from SEC FOIA bi-monthly CSV files and cached locally for 24 hours. The dashboard shows a 6-month history, peak FTD day, and trend direction.

Elevated FTDs alongside high short interest suggest synthetic short pressure. The ~T+35 delivery cycle has historically correlated with price pressure events on GME, as documented in peer-reviewed research (Pastorek, *Finance a úvěr*, 2023).

---

## Private data

`gme_intel.db` is gitignored and never committed. It contains:

- IV history (daily snapshots for IVR/IVP)
- Wheel trade ledger
- Your positions (shares, warrants, long calls)

All position P&L is computed locally and stays on your machine.

---

## Project structure

```
gme-intel-tracker/
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── analytics/
│   ├── greeks.py                   # BSM Greeks via py_vollib
│   ├── gex.py                      # Gamma Exposure computation
│   ├── iv.py                       # IVR, IVP, constant-maturity IV, expected move
│   └── pop.py                      # Probability of profit, annualised ROC
├── data/
│   ├── models.py                   # Pydantic schemas (OptionContract, ChainSnapshot, etc.)
│   ├── store.py                    # SQLite persistence (IVStore, WheelStore, PositionStore)
│   └── providers/
│       ├── base.py                 # MarketDataProvider ABC
│       ├── tradier.py              # Tradier production API
│       ├── alpaca.py               # Alpaca OPRA snapshot
│       └── yfinance_provider.py    # yfinance fallback
├── signals/
│   └── engine.py                   # Signal generation with catalyst gating
├── strategy/
│   └── wheel.py                    # Wheel FSM, cost-basis ledger, roll suggestions
├── sentiment/
│   ├── catalyst.py                 # EDGAR RSS, earnings, short interest, RK detection
│   └── ftd.py                      # SEC FOIA FTD data
└── modules/
    ├── reddit.py                   # Reddit RSS scraper + VADER sentiment
    └── technicals.py               # Support/resistance, volume spike detection
```

---

## Disclaimer

Personal research tool. Not financial advice. Options trading involves significant risk of loss.
