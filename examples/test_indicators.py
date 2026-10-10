from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache
from src.indicators.trend import sma, ema
from src.indicators.momentum import rsi

client = BinanceClient()
cache = DataCache(client)

df = cache.get_klines("BTCUSDT", "1h", 100)

# Add indicators
df["sma_20"] = sma(df, window=20)
df["ema_20"] = ema(df, window=20)
df["rsi_14"] = rsi(df, window=14)

# Last 10 rows
print(df.tail(10)[["close", "sma_20", "ema_20", "rsi_14"]])

last = df.iloc[-1]
print(f"\nCurrent price: ${last['close']:,.2f}")
print(f"RSI 14:        {last['rsi_14']:.2f}")

if last["rsi_14"] > 70:
    print("→ OVERBOUGHT — possible reversal down")
elif last["rsi_14"] < 30:
    print("→ OVERSOLD — possible reversal up")
else:
    print("→ Neutral zone")