from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.backtest.engine import BacktestEngine

client = BinanceClient()
cache = DataCache(client)

df = cache.get_klines("BTCUSDT", "1h", 1000)

strategy = SMACrossoverStrategy(short_window=20, long_window=50)
engine = BacktestEngine(
    initial_balance=10_000.0,
    trade_size_pct=5.0,
    fee_rate=0.001,
    stop_loss_pct=2.0,
    take_profit_pct=4.0,
)

result = engine.run(df, strategy)

print("=" * 50)
print("BACKTEST RESULTS")
print("=" * 50)
print(f"Period:              {df.index[0]} to {df.index[-1]}")
print(f"Strategy:            SMA Crossover ({strategy.short_window}/{strategy.long_window})")
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
print()
print("Last 5 trades:")
for t in result.trades[-5:]:
    print(f"  {t.entry_time} -> {t.exit_time}  "
          f"${t.entry_price:,.2f} -> ${t.exit_price:,.2f}  "
          f"PnL: ${t.pnl:+,.2f} ({t.return_pct:+.2f}%)")