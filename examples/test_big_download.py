"""
Quick test: can we download a LOT of candles?
Tests pagination, API limits, and timing
"""

import time
from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache


client = BinanceClient()
cache = DataCache(client)

# Delete existing cache for this test to force fresh download
import os
cache_file = "data/BTCUSDT_15m.csv"
if os.path.exists(cache_file):
    os.remove(cache_file)
    print(f"Deleted {cache_file} to force fresh download")

LIMIT = 20000  # Start smaller, see how it goes

print(f"\nAttempting to download {LIMIT} 15m candles of BTCUSDT...")
start = time.time()
df = cache.get_klines("BTCUSDT", "15m", LIMIT)
elapsed = time.time() - start

print(f"\nResult:")
print(f"  Downloaded: {len(df)} candles")
print(f"  Time taken: {elapsed:.1f} seconds")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")
print(f"  Days covered: {(df.index[-1] - df.index[0]).days}")