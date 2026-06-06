"""
Pydantic schemas for market data. Every provider must validate its output
against these models — silent NaN propagation into analytics ends here.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


class OptionContract(BaseModel):
    strike: float
    expiry: str  # YYYY-MM-DD
    option_type: Literal["call", "put"]
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None  # decimal, e.g. 0.80
    # Provider-supplied Greeks (may be None for yfinance)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    # Self-computed BSM Greeks (py_vollib) — populated post-fetch
    bsm_delta: Optional[float] = None
    bsm_gamma: Optional[float] = None
    bsm_theta: Optional[float] = None
    bsm_vega: Optional[float] = None

    @field_validator("implied_volatility")
    @classmethod
    def validate_iv(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v <= 0 or v > 5):
            return None
        return v

    @field_validator("strike")
    @classmethod
    def validate_strike(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Strike must be positive, got {v}")
        return v

    @field_validator("bid", "ask", "last")
    @classmethod
    def validate_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            return None
        return v

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None and self.bid >= 0 and self.ask >= 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def best_greek_delta(self) -> Optional[float]:
        """BSM delta preferred; fall back to provider delta."""
        return self.bsm_delta if self.bsm_delta is not None else self.delta

    @property
    def best_greek_gamma(self) -> Optional[float]:
        return self.bsm_gamma if self.bsm_gamma is not None else self.gamma


class OptionsChainSnapshot(BaseModel):
    ticker: str
    spot: float
    timestamp: datetime
    contracts: list[OptionContract]
    provider: str
    expiries: list[str] = []

    def calls(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.option_type == "call"]

    def puts(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.option_type == "put"]

    def for_expiry(self, expiry: str) -> list[OptionContract]:
        return [c for c in self.contracts if c.expiry == expiry]


class QuoteSnapshot(BaseModel):
    ticker: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    timestamp: datetime
    provider: str


class IVSnapshot(BaseModel):
    """Daily IV snapshot for SQLite persistence."""
    date: date
    iv_30d: float
    spot: float
