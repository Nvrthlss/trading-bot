from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache
from src.ml.dataset import build_dataset, label_distribution
from src.ml.model import TradingModel, time_based_split

client = BinanceClient()
cache = DataCache(client)

print("Downloading data...")
df = cache.get_klines("BTCUSDT", "1h", 3000)

print("Building dataset...")
X, y = build_dataset(df, forward_periods=8, threshold_pct=1.0)

print(f"Dataset: {len(X)} samples, {len(X.columns)} features")

# Chronological split
X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=0.8)
print(f"Train: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")

# Train
print("\nTraining Random Forest...")
model = TradingModel(n_estimators=200, max_depth=10, min_samples_leaf=20)
model.train(X_train, y_train)

# Evaluate on train set (sanity check)
train_metrics = model.evaluate(X_train, y_train)
print(f"\nTrain accuracy: {train_metrics['accuracy']:.3f}")

# Evaluate on test set (the real metric)
test_metrics = model.evaluate(X_test, y_test)
print(f"Test accuracy:  {test_metrics['accuracy']:.3f}")

print("\nConfusion matrix (test):")
print("       pred:  SELL  HOLD  BUY")
labels = ["SELL", "HOLD", "BUY"]
for i, true_label in enumerate(labels):
    row = test_metrics["confusion_matrix"][i]
    print(f"true {true_label:5s}:  {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")

print("\nClassification report (test):")
print(test_metrics["report"])

print("\nFeature importance (top 10):")
print(model.feature_importance().head(10).to_string())