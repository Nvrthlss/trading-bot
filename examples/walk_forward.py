from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache
from src.strategies.ml_strategy import MLStrategy
from src.backtest.engine import BacktestEngine
from src.backtest.advanced import walk_forward, monte_carlo


client = BinanceClient()
cache = DataCache(client)

TRADE_TF = "15m"
EXTRA_TFS = ["1h", "4h", "1d"]      # 5m deleted temporarily

print(f"Trade TF: {TRADE_TF}, Extra feature TFs: {EXTRA_TFS}")
print("Downloading data for all timeframes...")

df = cache.get_klines("BTCUSDT", TRADE_TF, 20000)
print(f"Trade TF dataset: {len(df)} candles ({(df.index[-1] - df.index[0]).days} days)")

# Per-TF limits — matched to cover 208 days (20000 15m bars)
TF_LIMITS = {
    "5m": 60000,
    "15m": 20000,
    "1h": 5000,
    "4h": 1500,
    "1d": 300,
}

multi_tf_data = {}
for tf in EXTRA_TFS:
    limit = TF_LIMITS.get(tf, 3000)
    multi_tf_data[tf] = cache.get_klines("BTCUSDT", tf, limit)

def make_strategy():
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
    strat.set_extra_tf_data(multi_tf_data)
    return strat


engine = BacktestEngine(
    initial_balance=10_000.0,
    trade_size_pct=5.0,
    fee_rate=0.001,
    stop_loss_pct=1.5,
    take_profit_pct=3.0,
)

print("\nRunning walk-forward optimization...")
print("-" * 60)
result = walk_forward(
    df=df,
    strategy_factory=make_strategy,
    engine=engine,
    train_size=5000,
    test_size=2000,
    step=2000,
)

print("-" * 60)
print(f"\nWALK-FORWARD SUMMARY")
print(f"=" * 60)
print(f"Windows tested:    {len(result.windows)}")
print(f"Avg return:        {result.avg_return:+.2f}% per window")
print(f"Avg win rate:      {result.avg_win_rate:.1f}%")
print(f"Consistency:       {result.consistency:.0f}% of windows were profitable")
print(f"Total trades:      {result.total_trades}")

if result.consistency >= 70:
    print(f"\nVerdict: ROBUST — profitable across most market regimes")
elif result.consistency >= 50:
    print(f"\nVerdict: MIXED — works sometimes, needs more validation")
else:
    print(f"\nVerdict: WEAK — probably not generalizable")

print("\n" + "=" * 60)
print("MONTE CARLO SIMULATION")
print("=" * 60)
print("Running single backtest for MC baseline...")

strategy = make_strategy()
single_result = engine.run(df, strategy)

print(f"Base backtest: {single_result.num_trades} trades, "
      f"return={single_result.total_return_pct:+.2f}%")

if single_result.num_trades == 0:
    print("No trades in base backtest, skipping Monte Carlo.")
else:
    print(f"Running Monte Carlo (500 simulations)...")
    mc = monte_carlo(single_result, n_simulations=500)

    print(f"\nMonte Carlo Results:")
    print(f"  Final balance:              ${mc['final_balance']:,.2f}")
    print(f"  Total return:               {mc['total_return_pct']:+.2f}%")
    print()
    print(f"  Drawdown statistics across {mc['n_simulations']} random orderings:")
    print(f"    Mean max drawdown:        {mc['mean_max_drawdown']:.2f}%")
    print(f"    Worst case (highest DD):  {mc['worst_max_drawdown']:.2f}%")
    print(f"    95th percentile:          {mc['percentile_95_drawdown']:.2f}%")
    print()
    print(f"  Robustness:")
    print(f"    DD stayed under 5%:       {mc['pct_under_5pct_dd']:.1f}% of orderings")
    print(f"    DD stayed under 10%:      {mc['pct_under_10pct_dd']:.1f}% of orderings")