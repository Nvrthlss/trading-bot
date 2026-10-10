from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache
from src.strategies.sma_crossover import SMACrossoverStrategy, BUY, SELL, HOLD

client = BinanceClient()
cache = DataCache(client)

df = cache.get_klines("BTCUSDT", "1h", 1000)

strategy = SMACrossoverStrategy(short_window=20, long_window=50)
df = strategy.prepare(df)

# Find all signals
signals = []
for i in range(len(df)):
    sig = strategy.signal(df, i)
    if sig != HOLD:
        signals.append((df.index[i], sig, df["close"].iloc[i]))

print(f"Total bars: {len(df)}")
print(f"Signals found: {len(signals)}")
print()
print("All signals:")
for ts, sig, price in signals:
    label = "BUY " if sig == BUY else "SELL"
    print(f"  {ts}  {label}  @ ${price:,.2f}")