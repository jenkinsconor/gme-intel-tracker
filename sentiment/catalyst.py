"""
Catalyst gate — combines all event-risk signals into a single gate object
that flows into the signal engine and suppresses premium-sell suggestions.

Sources (all free):
  • SEC EDGAR Atom RSS — 8-K filings (ATM offering detection), Form 4 (Cohen)
  • yfinance calendar — next earnings date
  • yfinance info — short interest snapshot
  • Existing Reddit DataFrame — Roaring Kitty keyword spike detection

Hard blocks (suppress all premium-sell signals):
  • Earnings within 7 DTE

Soft warnings (show in UI, don't block):
  • 8-K filed in past 14 days
  • Roaring Kitty spike on Reddit (3+ posts in past 6h)

SEC EDGAR rate-limit note:
  The User-Agent header is REQUIRED by SEC policy. Without it you get 403s.
  We make at most 2 requests per load (8-K + Form 4), sleep 1s between them.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests
import yfinance as yf

GME_CIK = "0001326380"
EDGAR_ATOM_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&CIK={cik}&type={form}&dateb=&owner=include"
    "&count=20&search_text=&output=atom"
)
EDGAR_HEADERS = {
    "User-Agent": "GME-Intel-Research/1.0 research@gme-intel.local",
    "Accept-Encoding": "gzip, deflate",
}

# Keywords that suggest an ATM equity offering in an 8-K filing
ATM_KEYWORDS = [
    "at-the-market", "at the market", "prospectus supplement",
    "sales agreement", "equity distribution", "common stock offering",
    "shares of common stock", "forward sale",
]

# Keywords for Roaring Kitty / DFV detection in Reddit posts
RK_PATTERNS = [
    r"\bkitty\b", r"\bdfv\b", r"\bdeepfuckingvalue\b",
    r"\broaring\b", r"\bkeith\s*gill\b", r"\brk\s+post\b",
    r"\brk\s+returns\b", r"\bdfw\b",
]
_RK_RE = re.compile("|".join(RK_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# CatalystGate dataclass
# ---------------------------------------------------------------------------

@dataclass
class CatalystGate:
    """
    Immutable snapshot of all active catalyst conditions for one app refresh.
    Constructed by build_catalyst_gate() and injected into signals/engine.py.
    """
    # Hard blocks — suppress all premium-sell suggestions
    hard_block: bool = False
    hard_block_reasons: list[str] = field(default_factory=list)

    # Soft warnings — displayed in UI, don't block signals
    warnings: list[str] = field(default_factory=list)

    # Earnings
    earnings_date: Optional[date] = None
    earnings_dte: Optional[int] = None

    # SEC 8-K filings (past 14 days)
    recent_8k_filings: list[dict] = field(default_factory=list)   # [{title, date, url, is_atm}]

    # SEC Form 4 — insider transactions (past 30 days)
    recent_form4_filings: list[dict] = field(default_factory=list)  # [{title, date, url}]

    # Roaring Kitty / social spike
    rk_spike: bool = False
    rk_post_count: int = 0
    rk_sample_titles: list[str] = field(default_factory=list)

    # Short interest (yfinance, bi-monthly FINRA sourced)
    shares_short: Optional[int] = None
    short_pct_float: Optional[float] = None   # decimal, e.g. 0.24 = 24%
    short_ratio: Optional[float] = None       # days to cover
    shares_short_prior_month: Optional[int] = None
    short_change_pct: Optional[float] = None  # month-over-month %

    @property
    def gate_status(self) -> str:
        if self.hard_block:
            return "blocked"
        if self.warnings:
            return "warning"
        return "clear"

    @property
    def banner_color(self) -> str:
        return {"blocked": "error", "warning": "warning", "clear": "success"}[self.gate_status]

    def banner_text(self) -> str:
        if self.hard_block:
            return "CATALYST GATE BLOCKED — " + "  |  ".join(self.hard_block_reasons)
        if self.warnings:
            return "CATALYST WARNING — " + "  |  ".join(self.warnings)
        return "Catalyst gate clear. No blocks or warnings active."

    @property
    def has_atm_filing(self) -> bool:
        return any(f.get("is_atm") for f in self.recent_8k_filings)


# ---------------------------------------------------------------------------
# SEC EDGAR helpers
# ---------------------------------------------------------------------------

def _fetch_edgar_atom(cik: str, form_type: str, session: requests.Session) -> list[dict]:
    """
    Fetch EDGAR Atom RSS for a given CIK and form type.
    Returns list of {title, filed_date, url} dicts.
    Returns [] on any error (don't crash the dashboard for a feed failure).
    """
    url = EDGAR_ATOM_URL.format(cik=cik, form=form_type)
    try:
        resp = session.get(url, headers=EDGAR_HEADERS, timeout=12)
        if not resp.ok:
            return []
        # SEC sometimes returns HTML errors with 200 status — check content type
        if "text/html" in resp.headers.get("Content-Type", ""):
            return []

        root = ET.fromstring(resp.content)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entries = []
        for entry in root.findall("a:entry", ns):
            title = entry.findtext("a:title", "", ns).strip()
            updated = entry.findtext("a:updated", "", ns)[:10]
            link_el = entry.find("a:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            try:
                filed = date.fromisoformat(updated)
            except Exception:
                continue
            entries.append({"title": title, "filed_date": filed, "url": link})
        return entries
    except Exception:
        return []


def _is_atm_filing(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in ATM_KEYWORDS)


def fetch_edgar_filings(
    cik: str = GME_CIK,
    form_type: str = "8-K",
    lookback_days: int = 14,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    """
    Return recent EDGAR filings within lookback_days.
    8-K entries are tagged with is_atm: True/False.
    """
    s = session or requests.Session()
    cutoff = date.today() - timedelta(days=lookback_days)
    all_entries = _fetch_edgar_atom(cik, form_type, s)

    results = []
    for e in all_entries:
        if e["filed_date"] < cutoff:
            continue
        entry = dict(e)
        if form_type == "8-K":
            entry["is_atm"] = _is_atm_filing(e["title"])
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------

def fetch_earnings_date(ticker: str = "GME") -> Optional[date]:
    """
    Return next earnings date from yfinance calendar.
    Returns None if unknown or already past.
    """
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            return None
        # yfinance ≥0.2.x returns a dict; older versions a DataFrame
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
        else:
            # DataFrame — try to extract from index
            try:
                dates = cal.loc["Earnings Date"].dropna().tolist()
            except Exception:
                return None

        if not dates:
            return None

        # dates is a list of datetime/Timestamp objects
        for d in dates:
            try:
                if hasattr(d, "date"):
                    ed = d.date()
                else:
                    ed = date.fromisoformat(str(d)[:10])
                if ed >= date.today():
                    return ed
            except Exception:
                continue
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Short interest
# ---------------------------------------------------------------------------

def fetch_short_interest(ticker: str = "GME") -> dict:
    """
    Pull short interest metrics from yfinance.info (bi-monthly FINRA data).
    Returns dict with shares_short, short_pct_float, short_ratio,
    shares_short_prior_month, short_change_pct.
    All values may be None if yfinance doesn't have them.
    """
    try:
        info = yf.Ticker(ticker).info
        shares_short = info.get("sharesShort")
        prior = info.get("sharesShortPriorMonth")
        short_pct = info.get("shortPercentOfFloat")
        short_ratio = info.get("shortRatio")

        change_pct = None
        if shares_short and prior and prior > 0:
            change_pct = (shares_short - prior) / prior * 100

        return {
            "shares_short": shares_short,
            "short_pct_float": short_pct,
            "short_ratio": short_ratio,
            "shares_short_prior_month": prior,
            "short_change_pct": change_pct,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Roaring Kitty / social spike
# ---------------------------------------------------------------------------

def detect_rk_activity(
    reddit_df,
    lookback_hours: float = 6.0,
    min_posts: int = 3,
) -> tuple[bool, int, list[str]]:
    """
    Scan the existing Reddit DataFrame for Roaring Kitty keyword spikes.
    No additional network calls — reuses what's already been fetched.

    Returns (is_spike, post_count, sample_titles).
    """
    try:
        import pandas as pd
        if reddit_df is None or reddit_df.empty:
            return False, 0, []

        recent = reddit_df[reddit_df.get("age_hours", float("inf")) <= lookback_hours]
        if recent.empty:
            return False, 0, []

        mask = recent["title"].str.contains(_RK_RE, na=False)
        # Also catch key-author posts (DFV aliases already in reddit.py key_authors)
        if "is_key_author" in recent.columns:
            mask = mask | recent["is_key_author"]

        hits = recent[mask]
        is_spike = len(hits) >= min_posts
        return is_spike, len(hits), hits["title"].tolist()[:5]
    except Exception:
        return False, 0, []


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_catalyst_gate(
    reddit_df=None,
    lookback_days_filings: int = 14,
    earnings_block_dte: int = 7,
    rk_spike_threshold: int = 3,
    rk_lookback_hours: float = 6.0,
) -> CatalystGate:
    """
    Build the full CatalystGate from all sources.
    Called once per Streamlit refresh, cached with ttl=3600.
    """
    session = requests.Session()
    gate = CatalystGate()

    # 1. Short interest (yfinance, fast)
    si = fetch_short_interest()
    gate.shares_short = si.get("shares_short")
    gate.short_pct_float = si.get("short_pct_float")
    gate.short_ratio = si.get("short_ratio")
    gate.shares_short_prior_month = si.get("shares_short_prior_month")
    gate.short_change_pct = si.get("short_change_pct")

    # 2. Earnings date (yfinance, fast)
    earnings = fetch_earnings_date()
    if earnings:
        gate.earnings_date = earnings
        gate.earnings_dte = (earnings - date.today()).days
        if gate.earnings_dte <= earnings_block_dte:
            gate.hard_block = True
            gate.hard_block_reasons.append(
                f"Earnings in {gate.earnings_dte} DTE ({earnings}) — "
                "IV will spike into the event then crush violently after. "
                "Premium collected now is just the market pricing in event risk."
            )

    # 3. SEC 8-K filings (1 request + sleep)
    try:
        gate.recent_8k_filings = fetch_edgar_filings(
            form_type="8-K", lookback_days=lookback_days_filings, session=session
        )
        if gate.recent_8k_filings:
            n = len(gate.recent_8k_filings)
            atm_count = sum(1 for f in gate.recent_8k_filings if f.get("is_atm"))
            if atm_count:
                gate.warnings.append(
                    f"{atm_count} likely ATM offering filing(s) in past {lookback_days_filings} days — "
                    "dilution risk active. GameStop has a documented pattern of filing ATM sales "
                    "into price spikes."
                )
            elif n:
                gate.warnings.append(
                    f"{n} 8-K filing(s) in past {lookback_days_filings} days — check for material news."
                )
        time.sleep(1.0)  # SEC rate limit courtesy
    except Exception:
        pass

    # 4. SEC Form 4 — insider transactions (1 request)
    try:
        gate.recent_form4_filings = fetch_edgar_filings(
            form_type="4", lookback_days=30, session=session
        )
        time.sleep(0.5)
    except Exception:
        pass

    # 5. Roaring Kitty spike (pure DataFrame op, no I/O)
    rk_spike, rk_count, rk_titles = detect_rk_activity(
        reddit_df, lookback_hours=rk_lookback_hours, min_posts=rk_spike_threshold
    )
    gate.rk_spike = rk_spike
    gate.rk_post_count = rk_count
    gate.rk_sample_titles = rk_titles
    if rk_spike:
        gate.warnings.append(
            f"Roaring Kitty / DFV keyword spike: {rk_count} posts in past "
            f"{rk_lookback_hours:.0f}h on Reddit. Historical pattern: "
            "multi-day momentum → ATM offering. Avoid selling calls."
        )

    return gate
