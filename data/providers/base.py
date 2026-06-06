from abc import ABC, abstractmethod
from typing import Optional


class MarketDataProvider(ABC):
    """Abstract base class for all market data providers."""

    @abstractmethod
    def get_quote(self, ticker: str) -> dict:
        """Return {ticker, price, bid, ask, volume, timestamp, provider}."""

    @abstractmethod
    def get_options_chain(self, ticker: str, max_expiries: int = 4) -> dict:
        """
        Return {ticker, spot, timestamp, provider, expiries: [...], contracts: [...]}.
        Each contract: {strike, expiry, option_type, bid, ask, last, volume,
                        open_interest, implied_volatility, delta, gamma, theta, vega}
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""

    @property
    def is_real_time(self) -> bool:
        return False
