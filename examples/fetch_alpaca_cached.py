from src.exchanges.alpaca_client import AlpacaClient
from src.data.cache import DataCache


client = AlpacaClient()
cache = DataCache(client)

print("First fetch (should MISS and download):")
df = cache.get_klines("AAPL", "15m", 500)
print(f"Got {len(df)} bars\n")

print("Second fetch (should HIT cache):")
df = cache.get_klines("AAPL", "15m", 500)
print(f"Got {len(df)} bars\n")

print("Last 3 bars:")
print(df.tail(3))