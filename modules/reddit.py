"""
GME Reddit Intelligence Module

Scrapes r/GMEOptions, r/Superstonk, r/GME, r/wallstreetbets
via Reddit's public RSS/Atom feeds — no credentials required.

Tracks:
  - Options flow mentions (calls, puts, IV, OI, sweeps, gamma, etc.)
  - FTD / short interest / DRS / Reg SHO signals
  - Sentiment scoring via VADER
  - DD post detection
  - Post velocity (trending right now)

Usage:
    from gme_reddit_module import RedditScraper
    scraper = RedditScraper()
    df = scraper.run()
    print(scraper.sentiment_summary(df))
"""

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Standard subreddits — scraped via hot/new/top feed
STANDARD_SUBREDDITS = ["GMEOptions", "wallstreetbets"]

# Flair-filtered subreddits — only pull specific high-signal flairs
SUPERSTONK_FLAIRS = ["Due Diligence", "Data", "Technical Analysis"]
GME_FLAIRS        = ["DD", "God Tier DD"]

# Signal weight per subreddit
SUB_SIGNAL_WEIGHT = {
    "GMEOptions":     1.0,  # actual options traders — highest trust
    "GME":            0.6,
    "Superstonk":     0.9,  # high weight now — flair-filtered = quality only
    "wallstreetbets": 0.5,
}

# High-quality authors — posts get flagged prominently
# u/Crybad: weekly wheel plays with actual strikes, premiums, P&L
KEY_AUTHORS = {"crybad", "crybad_", "u_crybad", "bobsmith808", "terroristcavin"}

RSS_URL        = "https://www.reddit.com/r/{sub}/{sort}.rss?limit=100"
RSS_SEARCH_URL = "https://www.reddit.com/r/{sub}/search.rss?q=flair%3A%22{flair}%22&restrict_sr=on&sort=new&t=month"
ATOM_NS = "http://www.w3.org/2005/Atom"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/125.0.0.0 Safari/537.36"
}

# ---------------------------------------------------------------------------
# Signal keywords
# ---------------------------------------------------------------------------

OPTIONS_KEYWORDS = [
    r"\bcalls?\b", r"\bputs?\b",
    r"\bIV\b", r"\bimplied\s+vol", r"\biv\s+crush\b", r"\biv\s+spike\b",
    r"\bopen\s+interest\b", r"\bOI\b",
    r"\bsqueeze\b", r"\bshort\s+squeeze\b",
    r"\bFTD\b", r"\bfail\s+to\s+deliver\b",
    r"\breg\s*sho\b", r"\bregSHO\b",
    r"\bunusual\s+(options?|activity|flow)\b",
    r"\bsweep\b", r"\bdark\s+pool\b",
    r"\bgamma\b", r"\bdelta\b", r"\btheta\b", r"\bvega\b",
    r"\bmax\s+pain\b", r"\bGEX\b", r"\bgamma\s+exposure\b",
    r"\bdeep\s+ITM\b", r"\bOTM\b", r"\bATM\b", r"\bITM\b",
    r"\bshort\s+interest\b", r"\bSI\b", r"\butilization\b",
    r"\bDRS\b", r"\bdirect\s+regist",
    r"\bprice\s+target\b", r"\bprice\s+suppress",
    r"\bhedge\b", r"\bhedging\b",
    r"\bexpir", r"\bstrike\b",
    r"\bRC\b", r"\bRyan\s+Cohen\b",
    r"\bshort\s+ladder\b", r"\bnaked\s+short",
    r"\bcitadel\b", r"\bmarket\s+maker\b",
    # Active catalyst: GameStop / eBay
    r"\bebay\b", r"\beBay\b",
    r"\b13[Dd]\b", r"\bHSR\b",
    r"\bpoison\s+pill\b",
    r"\bOTC\s+options?\b",
    r"\bETF\s+short\b",
    r"\btrojan\s+horse\b",
    r"\bacquisition\b", r"\bmerger\b",
    r"\bshareholder\s+meeting\b",
    r"\bMOASS\b",
]

_OPTIONS_RE = re.compile("|".join(OPTIONS_KEYWORDS), re.IGNORECASE)
DD_TITLE_RE = re.compile(r"\b(DD|due diligence|technical analysis|deep dive|TA)\b", re.IGNORECASE)

# Wheel / low-risk strategy posts
STRATEGY_KEYWORDS = [
    r"\bwheel\b",
    r"\bCSP\b", r"\bcash.secured.put\b",
    r"\bCC\b", r"\bcovered.call\b",
    r"\bcredit.spread\b", r"\bput.credit.spread\b", r"\bcall.credit.spread\b",
    r"\bdebit.spread\b", r"\bcall.debit.spread\b",
    r"\biron.condor\b",
    r"\bbutterfly\b",
    r"\btheta.gang\b", r"\bthetagang\b",
    r"\bpremium\b",
    r"\broll(ing)?\b",
    r"\bassigned\b", r"\bassignment\b",
    r"\bcollateral\b",
    r"\bcommission\b",
    r"\bwarrant\b",
    r"\bbreakeven\b", r"\bbreak.even\b",
    r"\boption plays? for week\b",  # u/Crybad weekly posts
    r"\bbrokerage\b",
]

# Detect u/Crybad-style weekly play posts specifically
WEEKLY_PLAYS_RE = re.compile(r"option plays? for week", re.IGNORECASE)
_STRATEGY_RE = re.compile("|".join(STRATEGY_KEYWORDS), re.IGNORECASE)

# Catalyst keywords — event-driven signals that could move price/IV
CATALYST_KEYWORDS = [
    r"\bearnings\b", r"\bearnings\s+call\b",
    r"\bsec\s+filing\b", r"\b13[dDgG]\b", r"\b8-?K\b", r"\b10-?[KQ]\b",
    r"\bproxy\b", r"\bshareholder\s+(meeting|vote)\b",
    r"\bdividend\b", r"\bsplit\b", r"\bbuyback\b", r"\bshare\s+repurchase\b",
    r"\bacquisition\b", r"\bmerger\b", r"\btakeover\b", r"\bdeal\b",
    r"\bpartnership\b", r"\bjoint\s+venture\b",
    r"\bRC\b", r"\bRyan\s+Cohen\b",
    r"\btweet\b", r"\bX\.com\b",
    r"\bpoison\s+pill\b", r"\bHSR\b",
    r"\bebay\b",
    r"\bshort\s+report\b", r"\bcitron\b", r"\bhindenburg\b",
    r"\bclass\s+action\b", r"\blawsuit\b", r"\bsec\s+invest",
    r"\bhalted?\b", r"\bcircuit\s+breaker\b",
    r"\bshort\s+squeeze\b", r"\bMOASS\b",
    r"\bregSHO\b", r"\breg\s*sho\b",
    r"\bFTD\b",
]
_CATALYST_RE = re.compile("|".join(CATALYST_KEYWORDS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _age_hours(dt: datetime) -> float:
    now = datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max((now - dt).total_seconds() / 3600.0, 0.01)


def _velocity(score: int, dt: datetime) -> float:
    return round(score / _age_hours(dt), 2)


def _is_options_relevant(title: str, body: str = "") -> bool:
    return bool(_OPTIONS_RE.search(f"{title} {body}"))


def _is_catalyst(title: str, body: str = "") -> bool:
    return bool(_CATALYST_RE.search(f"{title} {body}"))


def _is_strategy(title: str, body: str = "") -> bool:
    return bool(_STRATEGY_RE.search(f"{title} {body}"))


def _is_key_author(author: str) -> bool:
    return author.lower().replace("/u/", "").replace("u/", "") in KEY_AUTHORS


def _is_dd(title: str) -> bool:
    return bool(DD_TITLE_RE.search(title))


def _score_sentiment(analyzer: SentimentIntensityAnalyzer, text: str) -> dict:
    scores = analyzer.polarity_scores(text[:2000])
    compound = round(scores["compound"], 4)
    label = "neutral"
    if compound >= 0.05:
        label = "bullish"
    elif compound <= -0.05:
        label = "bearish"
    return {"compound": compound, "label": label}


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class RedditScraper:
    def __init__(self, request_delay: float = 1.5):
        self.delay = request_delay
        self.analyzer = SentimentIntensityAnalyzer()
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _fetch_rss_url(self, url: str) -> list[dict]:
        """Fetch and parse any Reddit Atom RSS URL."""
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"  fetch error: {e}")
            return []
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as e:
            print(f"  parse error: {e}")
            return []

        posts = []
        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            title_el   = entry.find(f"{{{ATOM_NS}}}title")
            link_el    = entry.find(f"{{{ATOM_NS}}}link")
            updated_el = entry.find(f"{{{ATOM_NS}}}updated")
            content_el = entry.find(f"{{{ATOM_NS}}}content")
            author_el  = entry.find(f"{{{ATOM_NS}}}author/{{{ATOM_NS}}}name")
            title  = title_el.text  if title_el  is not None else ""
            url_p  = link_el.get("href", "") if link_el is not None else ""
            body   = _strip_html(content_el.text if content_el is not None else "")
            author = author_el.text if author_el is not None else ""
            try:
                dt = datetime.fromisoformat(updated_el.text.replace("Z", "+00:00")) if updated_el is not None else datetime.now(tz=timezone.utc)
            except Exception:
                dt = datetime.now(tz=timezone.utc)
            posts.append({"title": title, "body": body, "url": url_p, "author": author, "dt": dt})
        return posts

    def _fetch_rss(self, subreddit: str, sort: str = "hot") -> list[dict]:
        return self._fetch_rss_url(RSS_URL.format(sub=subreddit, sort=sort))

    def _parse_post(self, raw: dict, subreddit: str) -> Optional[dict]:
        title  = raw["title"] or ""
        body   = raw["body"] or ""
        url    = raw["url"] or ""
        dt     = raw["dt"]

        # WSB: only include GME-relevant posts
        if subreddit.lower() == "wallstreetbets":
            if not re.search(r"\bgme\b", f"{title} {body}", re.IGNORECASE):
                return None

        # RSS doesn't expose upvote score — use comment-count proxy if available
        # Velocity is based on age alone as a recency signal
        age = _age_hours(dt)
        # For RSS we don't have score, so velocity = 1/age (newer = higher)
        velocity = round(100.0 / age, 2)

        sentiment = _score_sentiment(self.analyzer, f"{title} {body}")

        return {
            "subreddit": subreddit,
            "title": title,
            "author": raw["author"],
            "age_hours": round(age, 2),
            "velocity": velocity,
            "signal_weight": SUB_SIGNAL_WEIGHT.get(subreddit, 0.5),
            "is_key_author": _is_key_author(raw["author"]),
            "is_options_relevant": _is_options_relevant(title, body),
            "is_catalyst": _is_catalyst(title, body),
            "is_strategy": _is_strategy(title, body),
            "is_weekly_plays": bool(WEEKLY_PLAYS_RE.search(title)),
            "is_dd": _is_dd(title),
            "sentiment_compound": sentiment["compound"],
            "sentiment_label": sentiment["label"],
            "created_utc": dt.strftime("%Y-%m-%d %H:%M UTC"),
            "url": url,
        }

    def scrape_subreddit(self, subreddit: str, sort: str = "hot") -> list[dict]:
        raw_posts = self._fetch_rss(subreddit, sort=sort)
        rows = []
        for raw in raw_posts:
            parsed = self._parse_post(raw, subreddit)
            if parsed:
                rows.append(parsed)
        return rows

    def _scrape_flair_filtered(self, subreddit: str, flairs: list[str]) -> list[dict]:
        """Fetch posts from a subreddit filtered to specific flairs only."""
        rows = []
        seen = set()
        for flair in flairs:
            url = RSS_SEARCH_URL.format(sub=subreddit, flair=flair.replace(" ", "+"))
            raw_posts = self._fetch_rss_url(url)
            for raw in raw_posts:
                if raw["url"] in seen:
                    continue
                seen.add(raw["url"])
                parsed = self._parse_post(raw, subreddit)
                if parsed:
                    parsed["flair"] = flair
                    rows.append(parsed)
            time.sleep(self.delay)
        print(f"  r/{subreddit} (flair-filtered): {len(rows)} posts ({', '.join(flairs)})")
        return rows

    def run(
        self,
        sort: str = "hot",
        options_only: bool = False,
    ) -> pd.DataFrame:
        """
        Scrape all subreddits and return a combined DataFrame sorted by recency.

        Parameters
        ----------
        sort         : str   'hot', 'new', or 'top'
        options_only : bool  If True, return only options-relevant posts.
        """
        all_rows = []

        # Standard subreddits — full feed
        for sub in STANDARD_SUBREDDITS:
            rows = self.scrape_subreddit(sub, sort=sort)
            print(f"  r/{sub}: {len(rows)} posts")
            all_rows.extend(rows)
            time.sleep(self.delay)

        # Superstonk — DD / Data / TA flair only
        all_rows.extend(self._scrape_flair_filtered("Superstonk", SUPERSTONK_FLAIRS))

        # r/GME — DD / God Tier DD flair only
        all_rows.extend(self._scrape_flair_filtered("GME", GME_FLAIRS))

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df

        if options_only:
            df = df[df["is_options_relevant"]]

        return df.sort_values("velocity", ascending=False).reset_index(drop=True)

    def top_dd(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        return (
            df[df["is_dd"]]
            .sort_values("age_hours")
            .head(n)
            .reset_index(drop=True)
        )

    def sentiment_summary(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        counts = df["sentiment_label"].value_counts().to_dict()

        # Weighted sentiment — GMEOptions counts more than Superstonk hype
        weights = df["signal_weight"]
        weighted_avg = (df["sentiment_compound"] * weights).sum() / weights.sum()
        raw_avg = df["sentiment_compound"].mean()

        overall = "bullish" if weighted_avg >= 0.05 else ("bearish" if weighted_avg <= -0.05 else "neutral")

        # GMEOptions-only sentiment (cleanest signal)
        gme_opts = df[df["subreddit"] == "GMEOptions"]
        gme_opts_sentiment = round(gme_opts["sentiment_compound"].mean(), 4) if not gme_opts.empty else None

        return {
            "overall": overall,
            "weighted_compound": round(weighted_avg, 4),
            "raw_compound": round(raw_avg, 4),
            "gmeoptions_compound": gme_opts_sentiment,
            "bullish": counts.get("bullish", 0),
            "neutral": counts.get("neutral", 0),
            "bearish": counts.get("bearish", 0),
            "total_posts": len(df),
            "options_relevant": int(df["is_options_relevant"].sum()),
            "catalyst_posts": int(df["is_catalyst"].sum()),
            "strategy_posts": int(df["is_strategy"].sum()),
            "dd_posts": int(df["is_dd"].sum()),
            "key_author_posts": int(df["is_key_author"].sum()),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("GME Reddit Intel — scraping RSS feeds...\n")
    scraper = RedditScraper()

    df = scraper.run(sort="hot")

    if df.empty:
        print("No posts retrieved.")
    else:
        summary = scraper.sentiment_summary(df)
        print(f"\nSentiment: {summary['overall'].upper()}  "
              f"(weighted: {summary['weighted_compound']}  |  GMEOptions only: {summary['gmeoptions_compound']})")
        print(f"Posts: {summary['total_posts']} total | "
              f"{summary['options_relevant']} options-relevant | "
              f"{summary['dd_posts']} DD\n")
        print(f"Bullish: {summary['bullish']}  "
              f"Neutral: {summary['neutral']}  "
              f"Bearish: {summary['bearish']}")

        cols = ["subreddit", "title", "age_hours", "sentiment_label",
                "is_options_relevant", "is_dd", "url"]

        print("\n--- Most Recent Posts (options-relevant first) ---")
        view = df.sort_values(["is_options_relevant", "age_hours"], ascending=[False, True])
        print(view[cols].head(15).to_string(index=False))

        dd = scraper.top_dd(df, n=5)
        if not dd.empty:
            print("\n--- Top DD Posts ---")
            print(dd[cols].to_string(index=False))

        out = "reddit_data.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} posts to {out}")
