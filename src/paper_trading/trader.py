"""
Paper trading loop
Runs a strategy on live data, simulating trades with a virtual balance
"""

import time
from datetime import datetime
from typing import Optional

from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache
from src.strategies.sma_crossover import BUY, SELL
from src.risk.manager import RiskManager, EXIT_SIGNAL
from src.paper_trading.state import (
    PaperState, OpenPosition, StateStore, TradeLog
)


class PaperTrader:
    def __init__(
        self,
        client: BinanceClient,
        strategy,
        symbol: str,
        interval: str,
        candles_to_fetch: int = 500,
        initial_balance: float = 10_000.0,
        trade_size_pct: float = 5.0,
        fee_rate: float = 0.001,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        scan_interval_seconds: int = 300,
        state_path: str = "data/paper_state.json",
        trade_log_path: str = "data/paper_trades.jsonl",
    ):
        self.client = client
        self.cache = DataCache(client)
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.candles_to_fetch = candles_to_fetch
        self.trade_size_pct = trade_size_pct
        self.fee_rate = fee_rate
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.scan_interval_seconds = scan_interval_seconds

        self.state_store = StateStore(state_path)
        self.trade_log = TradeLog(trade_log_path)

        if self.state_store.exists():
            self.state = self.state_store.load()
            print(f"[INIT] Loaded existing state: balance=${self.state.balance:.2f}, "
                  f"cycle={self.state.cycle_count}")
        else:
            self.state = PaperState(
                balance=initial_balance,
                initial_balance=initial_balance,
                started_at=datetime.utcnow().isoformat(),
            )
            self.state_store.save(self.state)
            print(f"[INIT] New paper trader, starting balance: ${initial_balance:.2f}")

        self.risk = RiskManager(
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
        )
        if self.state.open_position is not None:
            self.risk.open_position(self.state.open_position.entry_price)
            print(f"[INIT] Restored open position: {self.state.open_position.quantity:.8f} "
                  f"{self.state.open_position.symbol} @ ${self.state.open_position.entry_price:.2f}")

    def run_one_cycle(self):
        self.state.cycle_count += 1
        now = datetime.utcnow().isoformat(timespec="seconds")

        df = self.cache.get_klines(self.symbol, self.interval, self.candles_to_fetch)
        df = self.strategy.prepare(df)

        i = len(df) - 1
        price = df["close"].iloc[i]
        timestamp = df.index[i]
        sig = self.strategy.signal(df, i)

        sig_label = "BUY" if sig == BUY else "SELL" if sig == SELL else "HOLD"
        pos_label = "OPEN" if self.state.open_position else "FLAT"
        print(f"[{now}] cycle={self.state.cycle_count} "
              f"price=${price:.2f} signal={sig_label} position={pos_label}")

        if sig == BUY and self.state.open_position is None:
            self._open_position(price, timestamp)

        elif self.state.open_position is not None:
            exit_decision = self.risk.check_exit(price)
            exit_reason = None
            if exit_decision.should_exit:
                exit_reason = exit_decision.reason
            elif sig == SELL:
                exit_reason = EXIT_SIGNAL
            if exit_reason is not None:
                self._close_position(price, timestamp, exit_reason)

        self.state_store.save(self.state)

    def _open_position(self, price: float, timestamp):
        trade_amount = self.state.balance * (self.trade_size_pct / 100)
        fee = trade_amount * self.fee_rate
        qty = (trade_amount - fee) / price

        self.state.balance -= trade_amount
        self.state.total_fees_paid += fee

        sl = price * (1 - self.stop_loss_pct / 100)
        tp = price * (1 + self.take_profit_pct / 100)

        self.state.open_position = OpenPosition(
            symbol=self.symbol,
            entry_time=str(timestamp),
            entry_price=price,
            quantity=qty,
            stop_loss_price=sl,
            take_profit_price=tp,
        )
        self.risk.open_position(entry_price=price)

        print(f"  -> BUY {qty:.8f} {self.symbol} @ ${price:.2f} "
              f"(SL=${sl:.2f}, TP=${tp:.2f}, fee=${fee:.4f})")

    def _close_position(self, price: float, timestamp, reason: str):
        pos = self.state.open_position
        gross_proceeds = pos.quantity * price
        fee = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - fee

        self.state.balance += net_proceeds
        self.state.total_fees_paid += fee

        entry_cost = pos.quantity * pos.entry_price * (1 + self.fee_rate)
        pnl = net_proceeds - entry_cost
        return_pct = (pnl / entry_cost) * 100

        self.trade_log.append({
            "symbol": pos.symbol,
            "entry_time": pos.entry_time,
            "entry_price": pos.entry_price,
            "exit_time": str(timestamp),
            "exit_price": price,
            "quantity": pos.quantity,
            "pnl": pnl,
            "return_pct": return_pct,
            "exit_reason": reason,
        })

        self.state.open_position = None
        self.risk.close_position()

        print(f"  -> SELL {pos.quantity:.8f} {pos.symbol} @ ${price:.2f} "
              f"PnL=${pnl:+.2f} ({return_pct:+.2f}%) reason={reason}")

    def portfolio_value(self, current_price: Optional[float] = None) -> float:
        value = self.state.balance
        if self.state.open_position is not None and current_price is not None:
            value += self.state.open_position.quantity * current_price
        return value

    def run(self):
        print(f"=" * 60)
        print(f"PAPER TRADER STARTED")
        print(f"Symbol: {self.symbol}, Interval: {self.interval}")
        print(f"Scan every {self.scan_interval_seconds}s. Ctrl+C to stop.")
        print(f"=" * 60)

        try:
            while True:
                try:
                    self.run_one_cycle()
                except Exception as e:
                    print(f"[ERROR] Cycle failed: {e}")

                time.sleep(self.scan_interval_seconds)
        except KeyboardInterrupt:
            print("\n[STOPPED] Paper trader interrupted by user")
            self.print_summary()

    def print_summary(self):
        current_price = None
        try:
            df = self.cache.get_klines(self.symbol, self.interval, 1)
            current_price = df["close"].iloc[-1]
        except Exception:
            pass

        value = self.portfolio_value(current_price)
        total_return = (value - self.state.initial_balance) / self.state.initial_balance * 100
        trades = self.trade_log.read_all()

        print("=" * 60)
        print("PAPER TRADING SUMMARY")
        print("=" * 60)
        print(f"Initial balance:    ${self.state.initial_balance:,.2f}")
        print(f"Current balance:    ${self.state.balance:,.2f}")
        print(f"Portfolio value:    ${value:,.2f}")
        print(f"Total return:       {total_return:+.2f}%")
        print(f"Total fees paid:    ${self.state.total_fees_paid:,.4f}")
        print(f"Cycles completed:   {self.state.cycle_count}")
        print(f"Trades completed:   {len(trades)}")
        if self.state.open_position:
            p = self.state.open_position
            print(f"Open position:      {p.quantity:.8f} {p.symbol} @ ${p.entry_price:.2f}")
        else:
            print("Open position:      None")
        print("=" * 60)