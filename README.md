# GME Intel

A personal options trading research dashboard for GME built with Streamlit.

Pulls live price data, options chain data, and Reddit community sentiment to surface actionable signals — focused on low-risk plays like covered calls, cash-secured puts, and LEAPs.

---

## Features

### 📈 Price & Signals
- Live price, MA20/50/100, RSI(14), volume spike detection
- Support & resistance levels with strength scoring and touch counts
- **Trade signal engine** — generates covered call, CSP, LEAP, and spread suggestions based on current conditions
- **Confluence detector** — flags when 3+ indicators align
- Plain-English explanations of every indicator and signal for beginners

### 🧠 Options Chain
- Live options chain via yfinance (nearest 4 expiries)
- IV smile chart, max pain estimate
- **Covered call suggester** — highlights best delta 0.20–0.40 OTM candidates
- Full Greeks: delta, gamma, theta, vega

### 👾 Reddit Intel
- Scrapes **r/GMEOptions**, **r/Superstonk** (DD/Data/TA flair only), **r/GME** (DD/God Tier DD flair only), **r/wallstreetbets** (GME posts only) via public RSS — no API key required
- VADER sentiment scoring weighted by subreddit signal quality
- Flags: options-relevant, catalyst, strategy posts, DD, weekly plays
- Tracks high-signal authors (u/Crybad, u/terroristcavin, u/bobsmith808)
- Catalyst alert banner when event-driven posts detected

---

## Setup

```bash
git clone https://github.com/yourusername/gme-intel.git
cd gme-intel

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Project Structure

```
gme-intel/
├── app.py                  # Streamlit dashboard (main entry point)
├── requirements.txt
├── README.md
└── modules/
    ├── reddit.py           # Reddit RSS scraper + VADER sentiment
    └── technicals.py       # Support/resistance + volume spike detection
```

---

## Data Sources

| Data | Source | Cost |
|------|--------|------|
| Price & options chain | yfinance (Yahoo Finance) | Free |
| Reddit posts | Public RSS feeds | Free, no key needed |
| Sentiment scoring | VADER (offline NLP model) | Free |

No API keys required. All data is delayed ~15 minutes (yfinance limitation).

---

## Strategy Focus

This dashboard is built around **low-risk, premium-selling strategies** as described by the r/GMEOptions community:

- **Covered Call** — sell calls above resistance to collect weekly income on existing shares
- **Cash Secured Put (CSP)** — get paid to agree to buy shares below support
- **The Wheel** — CSP → get assigned shares → sell covered calls → repeat
- **LEAPS** — long-dated deep ITM calls as a cheaper alternative to buying 100 shares

---

## Disclaimer

This is a personal research tool. Nothing here is financial advice. Options trading involves significant risk. Do your own research before trading.
