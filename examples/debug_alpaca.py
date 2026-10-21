"""
Debug: Why is the Alpaca data stopping at Oct 7?
"""

from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from src.config import ALPACA_API_KEY_ID, ALPACA_API_SECRET


client = StockHistoricalDataClient(ALPACA_API_KEY_ID, ALPACA_API_SECRET)

# Request last 2 days specifically
end = datetime.now(timezone.utc)
start = end - timedelta(days=2)

print(f"Requesting AAPL bars from {start} to {end}")

request = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame(15, TimeFrameUnit.Minute),
    start=start,
    end=end,
    feed=DataFeed.IEX,
)

bars = client.get_stock_bars(request)
df = bars.df

if df.empty:
    print("NO DATA returned!")
else:
    df = df.reset_index(level="symbol", drop=True)
    print(f"\nGot {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nLast 5 bars:")
    print(df.tail(5))

# Also check without feed (default)
print("\n\n--- Without specifying feed ---")
request2 = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame(15, TimeFrameUnit.Minute),
    start=start,
    end=end,
)
try:
    bars2 = client.get_stock_bars(request2)
    df2 = bars2.df
    if df2.empty:
        print("NO DATA (default feed)")
    else:
        df2 = df2.reset_index(level="symbol", drop=True)
        print(f"Got {len(df2)} bars (default feed)")
        print(f"Date range: {df2.index[0]} to {df2.index[-1]}")
except Exception as e:
    print(f"Error with default feed: {e}")