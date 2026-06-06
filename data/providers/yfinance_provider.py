"""
yfinance provider — offline/dev fallback. 15-min delayed, no Greeks from source.
Greeks will be self-computed by analytics.greeks after fetching.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from .base import MarketDataProvider
from ..models import OptionContract, OptionsChainSnapshot, QuoteSnapshot


class YFinanceProvider(MarketDataProvider):
    @property
    def provider_name(self) -> str:
        return "Yahoo Finance (15-min delay, no source Greeks)"

    @property
    def is_real_time(self) -> bool:
        return False

    def get_quote(self, ticker: str) -> QuoteSnapshot:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        price = float(hist["Close"].iloc[-1])
        volume = int(hist["Volume"].iloc[-1])
        return QuoteSnapshot(
            ticker=ticker,
            price=price,
            volume=volume,
            timestamp=datetime.now(timezone.utc),
            provider=self.provider_name,
        )

    def get_options_chain(self, ticker: str, max_expiries: int = 4) -> OptionsChainSnapshot:
        t = yf.Ticker(ticker)
        all_expiries = list(t.options or [])
        if not all_expiries:
            return OptionsChainSnapshot(
                ticker=ticker,
                spot=0.0,
                timestamp=datetime.now(timezone.utc),
                contracts=[],
                provider=self.provider_name,
                expiries=[],
            )

        quote = self.get_quote(ticker)
        selected = all_expiries[:max_expiries]
        contracts: list[OptionContract] = []

        for exp in selected:
            try:
                chain = t.option_chain(exp)
                for _, row in chain.calls.iterrows():
                    c = self._row_to_contract(row, exp, "call")
                    if c:
                        contracts.append(c)
                for _, row in chain.puts.iterrows():
                    c = self._row_to_contract(row, exp, "put")
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

    def _row_to_contract(
        self, row: pd.Series, expiry: str, option_type: str
    ) -> Optional[OptionContract]:
        def safe_float(key: str) -> Optional[float]:
            v = row.get(key)
            if v is None:
                return None
            try:
                f = float(v)
                return None if math.isnan(f) else f
            except (TypeError, ValueError):
                return None

        def safe_int(key: str) -> Optional[int]:
            v = safe_float(key)
            return int(v) if v is not None else None

        try:
            iv = safe_float("impliedVolatility")
            return OptionContract(
                strike=float(row["strike"]),
                expiry=expiry,
                option_type=option_type,
                bid=safe_float("bid"),
                ask=safe_float("ask"),
                last=safe_float("lastPrice"),
                volume=safe_int("volume"),
                open_interest=safe_int("openInterest"),
                implied_volatility=iv,
                # yfinance closes Issue #1465 as "not planned" — no Greeks
                delta=safe_float("delta"),
                gamma=safe_float("gamma"),
                theta=safe_float("theta"),
                vega=safe_float("vega"),
            )
        except Exception:
            return None
