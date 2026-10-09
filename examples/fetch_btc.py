from src.exchanges.binance_client import BinanceClient

client = BinanceClient()
df = client.get_klines(symbol="BTCUSDT", interval="1h", limit=2500)

print(f"Shape: {df.shape}")
print(f"Period: {df.index[0]} → {df.index[-1]}")
print(f"Current BTC price: ${df['close'].iloc[-1]:,.2f}")
print()
print("First 3 candles:")
print(df.head(3))
print()
print("Last 3 candles:")
print(df.tail(3))