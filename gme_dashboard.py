# gme_dashboard.py
# Pulls technical indicators and now fetches live options chain data via IBKR.

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import sys
import asyncio
import time
from ib_insync import IB, Stock, Option, util
import urllib.request
from bs4 import BeautifulSoup

async def main():
    ib = IB()
    try:
        await ib.connectAsync("127.0.0.1", 4001, clientId=1)
    except Exception as e:
        print("❌ Failed to connect to IBKR API:", e)
        return

    print("📡 Fetching historical GME data from IBKR...")
    contract = Stock("GME", "SMART", "USD")
    qualified = await ib.qualifyContractsAsync(contract)

    if not qualified:
        print("❌ Failed to qualify stock contract.")
        return

    qualified_contract = qualified[0]

    bars = await ib.reqHistoricalDataAsync(
        qualified_contract,
        endDateTime="",
        durationStr="3 M",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1
    )

    if not bars:
        print("❌ No data received from IBKR.")
        return

    df = util.df(bars)
    df.rename(columns={"date": "Date", "close": "Close"}, inplace=True)
    df.set_index("Date", inplace=True)
    data = df.dropna()

    data["20_MA"] = data["Close"].rolling(window=20).mean()
    data["50_MA"] = data["Close"].rolling(window=50).mean()
    data["100_MA"] = data["Close"].rolling(window=100).mean()

    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    data["RSI"] = compute_rsi(data["Close"])

    latest = data.iloc[-1]
    print("\n📈 GME Price and Technical Indicators:")
    print(f"Current Price: ${latest['Close']:.2f}")
    print(f"20-day MA:     ${latest['20_MA']:.2f}")
    print(f"50-day MA:     ${latest['50_MA']:.2f}")
    print(f"100-day MA:    ${latest['100_MA']:.2f}")
    print(f"RSI:           {latest['RSI']:.2f}")

    print("\n📊 Pulling Option Chain from IBKR...")
    try:
        params = await ib.reqSecDefOptParamsAsync(
            qualified_contract.symbol,
            qualified_contract.exchange,
            qualified_contract.secType,
            qualified_contract.conId
        )
    except Exception as e:
        print("❌ Failed to load option parameters:", e)
        ib.disconnect()
        return

    if not params:
        print("❌ Failed to load option parameters.")
        ib.disconnect()
        return

    expirations = sorted(list(params[0].expirations))[:1]  # Get nearest expiration only
    strikes = sorted(params[0].strikes)
    underlying_price = latest['Close']
    atm_strikes = [s for s in strikes if abs(s - underlying_price) <= 5][:6]  # ATM ± few

    contracts = []
    for expiry in expirations:
        for strike in atm_strikes:
            contracts.append(Option("GME", expiry, strike, "C", "SMART"))
            contracts.append(Option("GME", expiry, strike, "P", "SMART"))

    contracts = await ib.qualifyContractsAsync(*contracts)
    tickers = ib.reqTickers(*contracts)
    time.sleep(2)

    print("\n🧠 Option Chain Snapshot (near-term, ATM ± few):")
    print(f"{'Type':<5} {'Strike':>6} {'IV':>8} {'OI':>6} {'Volume':>7} {'Delta':>8}")
    for t in tickers:
        opt = t.contract
        iv = t.modelGreeks.impliedVol if t.modelGreeks else None
        delta = t.modelGreeks.delta if t.modelGreeks else None
        print(f"{opt.right:<5} {opt.strike:>6.2f} {iv*100 if iv else 'N/A':>6} {t.openInterest:>6} {t.volume:>7} {delta if delta else 'N/A':>8}")

    print("\n🚨 Alerts:")
    if latest["Close"] > latest["20_MA"] > latest["50_MA"] > latest["100_MA"]:
        print("✅ Bullish alignment: Price > 20 > 50 > 100-day MA")
    if latest["RSI"] > 70:
        print("⚠️ RSI Overbought (>70)")
    elif latest["RSI"] < 30:
        print("📉 RSI Oversold (<30)")
    else:
        print("RSI in neutral range.")

    ib.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise

