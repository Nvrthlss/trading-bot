from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache

client = BinanceClient()
cache = DataCache(client)

# Első hívás: letölt és ment
print("--- First call ---")
df1 = cache.get_klines("BTCUSDT", "1h", 500)
print(f"Got {len(df1)} candles\n")

# Második hívás ugyanazzal: betölt a fájlból
print("--- Second call (same params) ---")
df2 = cache.get_klines("BTCUSDT", "1h", 500)
print(f"Got {len(df2)} candles\n")

# Harmadik hívás kevesebb candle-vel: a cache-ből kap
print("--- Third call (fewer candles) ---")
df3 = cache.get_klines("BTCUSDT", "1h", 100)
print(f"Got {len(df3)} candles\n")