from src.exchanges.binance_client import BinanceClient
from src.strategies.ml_strategy import MLStrategy
from src.paper_trading.trader import PaperTrader


client = BinanceClient()

def make_strategy(symbol: str):
    return MLStrategy(
        forward_periods=8,
        threshold_pct=1.0,
        train_ratio=0.7,
        min_confidence=0.5,
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
    )

trader = PaperTrader(
    client=client,
    strategy_factory=make_strategy,
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    interval="5m",
    candles_to_fetch=2000,
    initial_balance=10_000.0,
    trade_size_pct=5.0,
    fee_rate=0.001,
    stop_loss_pct=2.0,
    take_profit_pct=4.0,
    max_total_exposure_pct=60.0,
    scan_interval_seconds=60,
)

trader.run()