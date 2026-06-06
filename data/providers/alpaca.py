"""
Alpaca Markets options data provider — free, OPRA-sourced, Greeks included.
History available from February 2024 (covers the Roaring Kitty resurgence).

Setup:
  1. Open a free Alpaca account at alpaca.markets
  2. Get API key + secret from the dashboard
  3. Set ALPACA_API_KEY and ALPACA_API_SECRET env vars.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import MarketDataProvider
from ..models import OptionContract, OptionsChainSnapshot, QuoteSnapshot

ALPACA_DATA_URL = "https://data.alpaca.markets/v1beta1"
ALPACA_TRADE_URL = "https://paper-api.alpaca.markets/v2"


class AlpacaProvider(MarketDataProvider):
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET not set.")
        self._headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    @property
    def provider_name(self) -> str:
        return "Alpaca (OPRA, free, Greeks included)"

    @property
    def is_real_time(self) -> bool:
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _get(self, url: str, params: dict = None) -> dict:
        resp = requests.get(url, params=params or {}, headers=self._headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_quote(self, ticker: str) -> QuoteSnapshot:
        data = self._get(f"{ALPACA_DATA_URL}/stocks/{ticker}/quotes/latest")
        q = data.get("quote", {})
        price = float(q.get("ap") or q.get("bp") or 0)
        return QuoteSnapshot(
            ticker=ticker,
            price=price,
            bid=float(q.get("bp", 0)) or None,
            ask=float(q.get("ap", 0)) or None,
            timestamp=datetime.now(timezone.utc),
            provider=self.provider_name,
        )

    def get_options_chain(self, ticker: str, max_expiries: int = 4) -> OptionsChainSnapshot:
        # Alpaca option snapshots endpoint returns latest quote + greeks per contract
        data = self._get(
            f"{ALPACA_DATA_URL}/options/snapshots/{ticker}",
            {"feed": "opra", "limit": 1000},
        )
        quote = self.get_quote(ticker)
        contracts: list[OptionContract] = []

        snapshots = data.get("snapshots", {})
        seen_expiries: set[str] = set()

        for symbol, snap in snapshots.items():
            c = _snap_to_contract(snap)
            if c:
                contracts.append(c)
                seen_expiries.add(c.expiry)

        # Only keep contracts from first max_expiries expirations
        sorted_expiries = sorted(seen_expiries)[:max_expiries]
        contracts = [c for c in contracts if c.expiry in sorted_expiries]

        return OptionsChainSnapshot(
            ticker=ticker,
            spot=quote.price,
            timestamp=datetime.now(timezone.utc),
            contracts=contracts,
            provider=self.provider_name,
            expiries=sorted_expiries,
        )


def _snap_to_contract(snap: dict) -> Optional[OptionContract]:
    try:
        details = snap.get("details", {})
        option_type = "call" if str(details.get("type", "")).lower() == "call" else "put"
        strike = float(details.get("strike_price", 0))
        expiry = details.get("expiration_date", "")  # YYYY-MM-DD
        if not expiry or strike <= 0:
            return None

        greeks = snap.get("greeks", {})
        quote_data = snap.get("latestQuote", {})
        trade_data = snap.get("latestTrade", {})

        iv = greeks.get("iv") or snap.get("impliedVolatility")

        return OptionContract(
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            bid=float(quote_data.get("bp", 0)) or None,
            ask=float(quote_data.get("ap", 0)) or None,
            last=float(trade_data.get("p", 0)) or None,
            volume=int(trade_data.get("s", 0)) or None,
            open_interest=None,  # not in snapshot endpoint
            implied_volatility=float(iv) if iv else None,
            delta=float(greeks.get("delta", 0)) or None,
            gamma=float(greeks.get("gamma", 0)) or None,
            theta=float(greeks.get("theta", 0)) or None,
            vega=float(greeks.get("vega", 0)) or None,
        )
    except Exception:
        return None
