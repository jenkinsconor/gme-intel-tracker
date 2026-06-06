"""
Tradier Brokerage API provider.
Real-time options chain + Greeks courtesy of ORATS.

Setup:
  1. Open a Tradier Brokerage account at tradier.com (free, FINRA KYC)
  2. Get your production API token at developer.tradier.com
  3. Set TRADIER_TOKEN=<your token> in your environment or .env file
  4. The app auto-selects this provider when the token is present.

Note: sandbox token (no brokerage account) gives only 15-min delayed data.
      Production token gives real-time consolidated data.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import MarketDataProvider
from ..models import OptionContract, OptionsChainSnapshot, QuoteSnapshot

TRADIER_BASE = "https://api.tradier.com/v1"


class TradierProvider(MarketDataProvider):
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("TRADIER_TOKEN")
        if not self.token:
            raise ValueError(
                "TRADIER_TOKEN not set. "
                "Open a Tradier brokerage account → developer.tradier.com → copy your production token."
            )
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    @property
    def provider_name(self) -> str:
        return "Tradier (real-time, ORATS Greeks)"

    @property
    def is_real_time(self) -> bool:
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _get(self, path: str, params: dict = None) -> dict:
        resp = requests.get(
            f"{TRADIER_BASE}{path}",
            params=params or {},
            headers=self._headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_quote(self, ticker: str) -> QuoteSnapshot:
        data = self._get("/markets/quotes", {"symbols": ticker})
        q = data["quotes"]["quote"]
        return QuoteSnapshot(
            ticker=ticker,
            price=float(q["last"] or q.get("bid") or 0),
            bid=_safe_float(q.get("bid")),
            ask=_safe_float(q.get("ask")),
            volume=_safe_int(q.get("volume")),
            timestamp=datetime.now(timezone.utc),
            provider=self.provider_name,
        )

    def get_options_chain(self, ticker: str, max_expiries: int = 4) -> OptionsChainSnapshot:
        # Step 1: get expirations
        exp_data = self._get(
            "/markets/options/expirations",
            {"symbol": ticker, "includeAllRoots": "true"},
        )
        raw_exps = exp_data.get("expirations", {}).get("expiration", [])
        if isinstance(raw_exps, dict):
            raw_exps = [raw_exps]
        all_expiries = [e["date"] for e in raw_exps]
        selected = all_expiries[:max_expiries]

        quote = self.get_quote(ticker)
        contracts: list[OptionContract] = []

        for exp in selected:
            try:
                chain_data = self._get(
                    "/markets/options/chains",
                    {"symbol": ticker, "expiration": exp, "greeks": "true"},
                )
                options = chain_data.get("options", {}).get("option", [])
                if isinstance(options, dict):
                    options = [options]
                for opt in options:
                    c = _opt_to_contract(opt, exp)
                    if c:
                        contracts.append(c)
            except Exception:
                continue

        return OptionsChainSnapshot(
            ticker=ticker,
            spot=quote.price,
            timestamp=datetime.now(timezone.utc),
            contracts=contracts,
            provider=self.provider_name,
            expiries=selected,
        )


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f != 0 else None
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    f = _safe_float(v)
    return int(f) if f is not None else None


def _opt_to_contract(opt: dict, expiry: str) -> Optional[OptionContract]:
    try:
        greeks = opt.get("greeks") or {}
        option_type = "call" if opt.get("option_type") == "call" else "put"

        # Tradier surfaces bid_iv, ask_iv, mid_iv via ORATS
        iv = _safe_float(opt.get("mid_iv")) or _safe_float(opt.get("bid_iv"))

        return OptionContract(
            strike=float(opt["strike"]),
            expiry=expiry,
            option_type=option_type,
            bid=_safe_float(opt.get("bid")),
            ask=_safe_float(opt.get("ask")),
            last=_safe_float(opt.get("last")),
            volume=_safe_int(opt.get("volume")),
            open_interest=_safe_int(opt.get("open_interest")),
            implied_volatility=iv,
            delta=_safe_float(greeks.get("delta")),
            gamma=_safe_float(greeks.get("gamma")),
            theta=_safe_float(greeks.get("theta")),
            vega=_safe_float(greeks.get("vega")),
        )
    except Exception:
        return None
