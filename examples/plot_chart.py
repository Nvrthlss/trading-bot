from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache
from src.indicators.trend import sma, ema
from src.indicators.momentum import rsi
from src.visualization.charts import plot_price_with_indicators

client = BinanceClient()
cache = DataCache(client)

df = cache.get_klines("BTCUSDT", "1h", 500)

# Add indicators
df["sma_20"] = sma(df, window=20)
df["sma_50"] = sma(df, window=50)
df["ema_20"] = ema(df, window=20)
df["rsi_14"] = rsi(df, window=14)

# Plot
plot_price_with_indicators(df, title="BTC/USDT 1h with SMA, EMA, RSI")