from src.exchanges.binance_client import BinanceClient

client = BinanceClient()
df = client.get_klines(symbol="BTCUSDT", interval="1h", limit=10)

print(df)
print()
print(f"Shape: {df.shape}")
print(f"Current BTC price: ${df['close'].iloc[-1]:,.2f}")