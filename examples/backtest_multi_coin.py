from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache
from src.strategies.ml_strategy import MLStrategy
from src.backtest.multi_coin_engine import MultiCoinBacktestEngine


client = BinanceClient()
cache = DataCache(client)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TRADE_TF = "15m"
EXTRA_TFS = ["1h", "4h", "1d"]

print("Downloading data for all symbols...")

trade_data = {}
multi_tf_data_per_symbol = {}
for sym in SYMBOLS:
    trade_data[sym] = cache.get_klines(sym, TRADE_TF, 20000)
    multi_tf_data_per_symbol[sym] = cache.get_multi_tf_klines(sym, EXTRA_TFS, limit=5000)

print("\nBuilding and preparing strategies...")
strategies = {}
for sym in SYMBOLS:
    strat = MLStrategy(
        forward_periods=16,
        threshold_pct=0.5,
        train_ratio=0.7,
        min_confidence=0.5,
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        extra_tfs=EXTRA_TFS,
    )
    strat.set_extra_tf_data(multi_tf_data_per_symbol[sym])
    strategies[sym] = strat
    print(f"  [READY] {sym}")

engine = MultiCoinBacktestEngine(
    initial_balance=10_000.0,
    trade_size_pct=5.0,
    fee_rate=0.001,
    stop_loss_pct=1.5,
    take_profit_pct=3.0,
    max_total_exposure_pct=60.0,
)

print("\nRunning multi-coin backtest...")
result = engine.run(trade_data, strategies)

print("\n" + "=" * 60)
print("MULTI-COIN BACKTEST RESULTS")
print("=" * 60)
print(f"Symbols:           {', '.join(SYMBOLS)}")
print(f"Trade TF:          {TRADE_TF}")
print(f"Extra feature TFs: {EXTRA_TFS}")
print()
print(f"Initial balance:   ${result.initial_balance:,.2f}")
print(f"Final balance:     ${result.final_balance:,.2f}")
print(f"Total return:      {result.total_return_pct:+.2f}%")
print()
print(f"Total trades:      {result.num_trades}")
print(f"Wins / Losses:     {result.num_wins} / {result.num_losses}")
print(f"Win rate:          {result.win_rate:.1f}%")
print(f"Total fees:        ${result.total_fees:,.2f}")
print()
print(f"Exit reasons:")
for reason, count in result.exit_reasons.items():
    print(f"  {reason}: {count}")
print()
print(f"Per-symbol breakdown:")
for sym, stats in result.per_symbol_stats.items():
    print(f"  {sym}: {stats['num_trades']} trades, "
          f"win rate {stats['win_rate']:.1f}%, "
          f"total PnL ${stats['total_pnl']:+,.2f}")