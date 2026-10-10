from src.exchanges.binance_client import BinanceClient
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.paper_trading.trader import PaperTrader


client = BinanceClient()
strategy = SMACrossoverStrategy(short_window=20, long_window=50)

trader = PaperTrader(
    client=client,
    strategy=strategy,
    symbol="BTCUSDT",
    interval="1h",
    candles_to_fetch=500,
    initial_balance=10_000.0,
    trade_size_pct=5.0,
    fee_rate=0.001,
    stop_loss_pct=2.0,
    take_profit_pct=4.0,
    scan_interval_seconds=60,
)

trader.run()