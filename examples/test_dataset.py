from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache
from src.ml.dataset import build_dataset, label_distribution

client = BinanceClient()
cache = DataCache(client)

df = cache.get_klines("BTCUSDT", "1h", 3000)
print(f"Downloaded: {len(df)} candles")

X, y = build_dataset(df, forward_periods=8, threshold_pct=1.0)

print(f"\nDataset size: {len(X)} samples, {len(X.columns)} features")
print(f"\nFeature columns:")
for col in X.columns:
    print(f"  - {col}")

print(f"\nLabel distribution:")
dist = label_distribution(y)
label_names = {1: "BUY", 0: "HOLD", -1: "SELL"}
for label, info in sorted(dist.items(), reverse=True):
    print(f"  {label_names[label]:5s} ({label:2d}): {info['count']:5d}  ({info['pct']:5.1f}%)")

print(f"\nFirst 3 feature rows:")
print(X.head(3).round(4).to_string())

print(f"\nFirst 10 labels:")
print(y.head(10).tolist())