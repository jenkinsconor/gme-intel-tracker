import os
from .base import MarketDataProvider
from .yfinance_provider import YFinanceProvider


def get_provider() -> MarketDataProvider:
    """
    Auto-selects the best available data provider.
    Priority: Tradier > Alpaca > yfinance (offline/dev fallback).
    Set TRADIER_TOKEN or ALPACA_API_KEY + ALPACA_API_SECRET to activate.
    """
    if os.environ.get("TRADIER_TOKEN"):
        try:
            from .tradier import TradierProvider
            return TradierProvider()
        except Exception:
            pass

    if os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_SECRET"):
        try:
            from .alpaca import AlpacaProvider
            return AlpacaProvider()
        except Exception:
            pass

    return YFinanceProvider()
