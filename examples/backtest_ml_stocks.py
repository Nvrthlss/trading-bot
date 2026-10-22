"""
Test the ML strategy on Us stock market data (AAPL)
Same pipeline as crypto, just with Alpaca as data source
"""

from src.exchanges.alpaca_client import AlpacaClient
from src.data.cache import DataCache
from src.strategies.ml_strategy import MLStrategy
from src.backtest.engine import BacktestEngine

client = AlpacaClient()
cache = DataCache(client)

SYMBOL = "AAPL"
TRADE_TF = "15m"
EXTRA_TFS = ["1h", "4h", "1d"]

print(f"Downloading {SYMBOL} data...")

df = cache.get_klines(SYMBOL, TRADE_TF, 3000)
print(f"Trade TF dataset: {len(df)} candles ({(df.index[-1] - df.index[0]).days} days)")

multi_tf_data = cache.get_multi_tf_klines(SYMBOL, EXTRA_TFS, limit=1000)
for tf, mdf in multi_tf_data.items():
    print(f"  {tf}: {len(mdf)} candles, latest: {mdf.index[-1]}")

strategy = MLStrategy(
    forward_periods=16,
    threshold_pct=0.5,
    train_ratio=0.7,
    min_confidence=0.5,
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    extra_tfs=EXTRA_TFS,
)
strategy.set_extra_tf_data(multi_tf_data)

engine = BacktestEngine(
    initial_balance=10_000.0,
    trade_size_pct=5.0,
    fee_rate=0.001,
    stop_loss_pct=1.5,
    take_profit_pct=3.0,
)

print(f"\nRunning backtest...")
result = engine.run(df, strategy)

print("\n" + "=" * 50)
print(f"ML STRATEGY BACKTEST — {SYMBOL}")
print("=" * 50)
print(f"Period:              {df.index[0]} to {df.index[-1]}")
print(f"Confidence filter:   {strategy.min_confidence}")
print()
print(f"Initial balance:     ${result.initial_balance:,.2f}")
print(f"Final balance:       ${result.final_balance:,.2f}")
print(f"Total return:        {result.total_return_pct:+.2f}%")
print(f"Buy & Hold return:   {result.buy_hold_return_pct:+.2f}%")
print()
print(f"Total trades:        {result.num_trades}")
print(f"Wins / Losses:       {result.num_wins} / {result.num_losses}")
print(f"Win rate:            {result.win_rate:.1f}%")
print(f"Total fees paid:     ${result.total_fees:,.2f}")
print()
print(f"Exit reasons:")
for reason, count in result.exit_reasons.items():
    print(f"  {reason}: {count}")