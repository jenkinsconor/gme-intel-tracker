"""
SEC Fail-to-Deliver (FTD) data fetcher.

Source: SEC FOIA data, published bi-monthly (first half = 'a', second half = 'b')
URL pattern: https://www.sec.gov/data/foiadocuments/docs/fails-to-deliver-data/cnsfails{YYYYMM}{a|b}.zip

The zip contains one pipe-delimited text file with columns:
  SETTLEMENT DATE | CUSIP | SYMBOL | QUANTITY (FAILS) | DESCRIPTION | PRICE

Why FTDs matter for GME:
  The T+35 FTD delivery cycle has been documented as statistically significant for GME
  (Pastorek, Finance a úvěr, 2023). A spike in FTDs → elevated short-seller delivery
  obligations ~35 days later → potential forced buying. Not a timing signal by itself,
  but an important context metric alongside short interest and GEX.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

FTD_BASE = "https://www.sec.gov/data/foiadocuments/docs/fails-to-deliver-data"
FTD_HEADERS = {"User-Agent": "GME-Intel-Research/1.0 research@gme-intel.local"}
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".ftd_cache")


def _ftd_url(year: int, month: int, period: str) -> str:
    return f"{FTD_BASE}/cnsfails{year:04d}{month:02d}{period}.zip"


def _cache_path(year: int, month: int, period: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"ftd_{year:04d}{month:02d}{period}.parquet")


def _download_and_parse(
    year: int, month: int, period: str, session: requests.Session
) -> Optional[pd.DataFrame]:
    """Download one FTD zip, parse it, cache to parquet. Returns None if not found."""
    cache = _cache_path(year, month, period)

    # Use cached file if present (FTD data never changes once published)
    if os.path.exists(cache):
        try:
            return pd.read_parquet(cache)
        except Exception:
            os.remove(cache)

    url = _ftd_url(year, month, period)
    try:
        resp = session.get(url, headers=FTD_HEADERS, timeout=30)
        if resp.status_code == 404:
            return None  # Period not yet published — normal for current month's 'b'
        resp.raise_for_status()
    except requests.HTTPError:
        return None
    except Exception:
        return None

    try:
        raw = resp.content
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            fname = zf.namelist()[0]
            with zf.open(fname) as f:
                # Use utf-8-sig to strip BOM; errors='replace' for SEC encoding quirks
                content = f.read().decode("utf-8-sig", errors="replace")

        df = pd.read_csv(
            io.StringIO(content),
            sep="|",
            dtype=str,
            on_bad_lines="skip",
        )

        # Normalize column names — SEC format has changed over time
        df.columns = [c.strip().upper() for c in df.columns]

        # Identify columns by content pattern
        sym_col = next((c for c in df.columns if "SYMBOL" in c), None)
        qty_col = next((c for c in df.columns if "QUANTITY" in c or "FAILS" in c), None)
        date_col = next((c for c in df.columns if "DATE" in c or "SETTLEMENT" in c), None)
        price_col = next((c for c in df.columns if "PRICE" in c), None)

        if not sym_col:
            return None

        # Filter to GME
        gme = df[df[sym_col].str.strip() == "GME"].copy()
        if gme.empty:
            return None

        # Build clean DataFrame
        result = pd.DataFrame()
        if date_col:
            result["settle_date"] = pd.to_datetime(
                gme[date_col].str.strip(), format="%Y%m%d", errors="coerce"
            )
        if qty_col:
            result["quantity"] = pd.to_numeric(
                gme[qty_col].str.strip(), errors="coerce"
            )
        if price_col:
            result["price"] = pd.to_numeric(
                gme[price_col].str.strip().str.replace(r"[^\d.]", "", regex=True),
                errors="coerce",
            )

        result = result.dropna(subset=["settle_date"])
        if result.empty:
            return None

        # Cache to parquet for fast future loads
        try:
            result.to_parquet(cache, index=False)
        except Exception:
            pass

        return result

    except Exception:
        return None


def fetch_ftd_data(
    symbol: str = "GME",
    months_back: int = 6,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Download and concatenate GME FTD data for the past N months.
    Returns DataFrame with columns: settle_date, quantity, price.
    Empty DataFrame if all downloads fail.
    """
    s = session or requests.Session()
    today = date.today()
    frames = []

    for i in range(months_back):
        # Walk back month by month
        target = date(today.year, today.month, 1) - timedelta(days=i * 28)
        year, month = target.year, target.month

        for period in ("b", "a"):  # 'b' = second half, 'a' = first half; try 'b' first
            df = _download_and_parse(year, month, period, s)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(0.4)  # SEC courtesy rate limit

    if not frames:
        return pd.DataFrame(columns=["settle_date", "quantity", "price"])

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined
        .drop_duplicates(subset=["settle_date"])
        .sort_values("settle_date")
        .reset_index(drop=True)
    )
    return combined


def compute_ftd_metrics(df: pd.DataFrame) -> dict:
    """
    Compute summary metrics from FTD DataFrame.
    Returns dict with keys: total, peak_date, peak_qty, recent_30d_avg,
    prior_30d_avg, trend.
    """
    if df.empty or "quantity" not in df.columns:
        return {}

    today = pd.Timestamp.today()
    df = df.copy()
    df["settle_date"] = pd.to_datetime(df["settle_date"])

    recent_mask = df["settle_date"] >= today - pd.Timedelta(days=30)
    prior_mask = (df["settle_date"] >= today - pd.Timedelta(days=60)) & \
                 (df["settle_date"] < today - pd.Timedelta(days=30))

    peak_idx = df["quantity"].idxmax()
    peak_row = df.loc[peak_idx] if not df.empty else None

    recent_avg = float(df[recent_mask]["quantity"].mean()) if df[recent_mask].shape[0] else 0.0
    prior_avg = float(df[prior_mask]["quantity"].mean()) if df[prior_mask].shape[0] else 0.0

    if prior_avg > 0:
        if recent_avg > prior_avg * 1.15:
            trend = "rising"
        elif recent_avg < prior_avg * 0.85:
            trend = "falling"
        else:
            trend = "flat"
    else:
        trend = "unknown"

    return {
        "total": int(df["quantity"].sum()),
        "peak_date": str(peak_row["settle_date"].date()) if peak_row is not None else "—",
        "peak_qty": int(peak_row["quantity"]) if peak_row is not None else 0,
        "recent_30d_avg": recent_avg,
        "prior_30d_avg": prior_avg,
        "trend": trend,
    }
