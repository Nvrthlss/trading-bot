"""
Quick test: can we fetch Apple stock data from Alpaca?
"""

from src.exchanges.alpaca_client import AlpacaClient


client = AlpacaClient()
print(f"Client: {client.name}")

print("\nFetching AAPL 15m data (last 500 bars)...")
df = client.get_klines("AAPL", "15m", limit=500)

print(f"\nShape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Days covered: {(df.index[-1] - df.index[0]).days}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nLast 3 rows:")
print(df.tail(3))