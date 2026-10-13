"""
Paper trading loop.
Runs a strategy on live data across multiple coins.
"""

import time
from datetime import datetime
from typing import Optional, List

from src.exchanges.binance_client import BinanceClient
from src.data.cache import DataCache
from src.strategies.sma_crossover import BUY, SELL
from src.risk.manager import RiskManager, EXIT_SIGNAL
from src.paper_trading.state import (
    PaperState, OpenPosition, StateStore, TradeLog
)


class PaperTrader:
    """
    Multi-coin paper trader with risk-based exposure limits.
    """

    def __init__(
        self,
        client: BinanceClient,
        strategy_factory,
        symbols: List[str],
        interval: str,
        candles_to_fetch: int = 500,
        initial_balance: float = 10_000.0,
        trade_size_pct: float = 5.0,
        fee_rate: float = 0.001,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        max_total_exposure_pct: float = 60.0,
        scan_interval_seconds: int = 300,
        state_path: str = "data/paper_state.json",
        trade_log_path: str = "data/paper_trades.jsonl",
    ):
        self.client = client
        self.cache = DataCache(client)
        self.strategy_factory = strategy_factory
        self.strategies: dict = {}
        print(f"[INIT] Preparing strategies for {len(symbols)} symbols")
        for sym in symbols:
            strat = strategy_factory(sym)
            df_prep = self.cache.get_klines(sym, interval, candles_to_fetch)
            strat.prepare(df_prep)
            self.strategies[sym] = strat
            print(f"   [READY] {sym}")
        self.symbols = symbols
        self.interval = interval
        self.candles_to_fetch = candles_to_fetch
        self.trade_size_pct = trade_size_pct
        self.fee_rate = fee_rate
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.scan_interval_seconds = scan_interval_seconds

        self.state_store = StateStore(state_path)
        self.trade_log = TradeLog(trade_log_path)

        # Load existing state or create new
        if self.state_store.exists():
            self.state = self.state_store.load()
            print(f"[INIT] Loaded state: balance=${self.state.balance:.2f}, "
                  f"cycle={self.state.cycle_count}, "
                  f"open positions={len(self.state.open_positions)}")
        else:
            self.state = PaperState(
                balance=initial_balance,
                initial_balance=initial_balance,
                started_at=datetime.utcnow().isoformat(),
            )
            self.state_store.save(self.state)
            print(f"[INIT] New trader, balance: ${initial_balance:.2f}, "
                  f"watching {len(symbols)} coins")

        # Build per-symbol risk managers
        self.risk_managers: dict = {}
        for sym in self.symbols:
            rm = RiskManager(
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
            )
            # Restore state if there's an open position for this symbol
            if sym in self.state.open_positions:
                rm.open_position(self.state.open_positions[sym].entry_price)
            self.risk_managers[sym] = rm

    def _compute_exposure_pct(self, prices: dict) -> float:
        """
        Current total exposure as % of portfolio value.
        prices: dict of symbol -> current price (for mark-to-market)
        """
        position_value = 0.0
        for sym, pos in self.state.open_positions.items():
            price = prices.get(sym, pos.entry_price)  # fallback to entry if missing
            position_value += pos.quantity * price
        portfolio_value = self.state.balance + position_value
        if portfolio_value <= 0:
            return 0.0
        return (position_value / portfolio_value) * 100

    def _can_open_new_position(self, prices: dict, proposed_trade_amount: float) -> bool:
        """
        Check if opening a new position would push total exposure above the max.
        """
        current_position_value = sum(
            pos.quantity * prices.get(sym, pos.entry_price)
            for sym, pos in self.state.open_positions.items()
        )
        new_position_value = current_position_value + proposed_trade_amount
        portfolio_value = self.state.balance + current_position_value

        if portfolio_value <= 0:
            return False

        future_exposure_pct = (new_position_value / portfolio_value) * 100
        return future_exposure_pct <= self.max_total_exposure_pct

    def run_one_cycle(self):
        """Process every watched symbol once."""
        self.state.cycle_count += 1
        now = datetime.utcnow().isoformat(timespec="seconds")

        # Fetch latest price for every symbol (needed for exposure calc + exit checks)
        prices: dict = {}
        signals: dict = {}
        for sym in self.symbols:
            try:
                df = self.cache.get_klines(sym, self.interval, self.candles_to_fetch)
                strat = self.strategies[sym]
                df = strat.prepare_live(df) if hasattr(strat, "prepare_live") else strat.prepare(df)
                i = len(df) - 1
                prices[sym] = df["close"].iloc[i]
                signals[sym] = strat.signal(df, i)
            except Exception as e:
                print(f"  [ERROR] fetch failed for {sym}: {e}")

        exposure = self._compute_exposure_pct(prices)
        portfolio = self.state.balance + sum(
            pos.quantity * prices.get(sym, pos.entry_price)
            for sym, pos in self.state.open_positions.items()
        )

        print(f"[{now}] cycle={self.state.cycle_count} "
              f"portfolio=${portfolio:,.2f} balance=${self.state.balance:,.2f} "
              f"exposure={exposure:.1f}% positions={len(self.state.open_positions)}")

        # Phase 1: check exits for open positions
        for sym in list(self.state.open_positions.keys()):
            if sym not in prices:
                continue
            price = prices[sym]
            sig = signals.get(sym, 0)
            rm = self.risk_managers[sym]

            exit_decision = rm.check_exit(price)
            exit_reason = None
            if exit_decision.should_exit:
                exit_reason = exit_decision.reason
            elif sig == SELL:
                exit_reason = EXIT_SIGNAL

            if exit_reason is not None:
                self._close_position(sym, price, datetime.utcnow(), exit_reason)

        # Phase 2: look for new entries
        for sym in self.symbols:
            if sym in self.state.open_positions:
                continue  # already have a position in this coin
            if sym not in prices:
                continue
            if signals.get(sym) != BUY:
                continue

            price = prices[sym]
            proposed_trade_amount = self.state.balance * (self.trade_size_pct / 100)

            if proposed_trade_amount <= 0:
                continue
            if not self._can_open_new_position(prices, proposed_trade_amount):
                print(f"  [SKIP] {sym} BUY signal — exposure limit reached")
                continue

            self._open_position(sym, price, datetime.utcnow())

        # Save state
        self.state_store.save(self.state)

    def _open_position(self, symbol: str, price: float, timestamp):
        trade_amount = self.state.balance * (self.trade_size_pct / 100)
        fee = trade_amount * self.fee_rate
        qty = (trade_amount - fee) / price

        self.state.balance -= trade_amount
        self.state.total_fees_paid += fee

        sl = price * (1 - self.stop_loss_pct / 100)
        tp = price * (1 + self.take_profit_pct / 100)

        self.state.open_positions[symbol] = OpenPosition(
            symbol=symbol,
            entry_time=str(timestamp),
            entry_price=price,
            quantity=qty,
            stop_loss_price=sl,
            take_profit_price=tp,
        )
        self.risk_managers[symbol].open_position(entry_price=price)

        print(f"  -> BUY  {symbol} {qty:.8f} @ ${price:,.2f} "
              f"(SL=${sl:,.2f}, TP=${tp:,.2f}, fee=${fee:.4f})")

    def _close_position(self, symbol: str, price: float, timestamp, reason: str):
        pos = self.state.open_positions[symbol]
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

        del self.state.open_positions[symbol]
        self.risk_managers[symbol].close_position()

        print(f"  -> SELL {symbol} {pos.quantity:.8f} @ ${price:,.2f} "
              f"PnL=${pnl:+.2f} ({return_pct:+.2f}%) reason={reason}")

    def run(self):
        """Infinite loop. Ctrl+C to stop."""
        print(f"=" * 60)
        print(f"PAPER TRADER STARTED")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Interval: {self.interval}, scan every {self.scan_interval_seconds}s")
        print(f"Max exposure: {self.max_total_exposure_pct}%, trade size: {self.trade_size_pct}%")
        print(f"Ctrl+C to stop")
        print(f"=" * 60)

        try:
            while True:
                try:
                    self.run_one_cycle()
                except Exception as e:
                    print(f"[ERROR] Cycle failed: {e}")
                time.sleep(self.scan_interval_seconds)
        except KeyboardInterrupt:
            print("\n[STOPPED] Interrupted by user")
            self.print_summary()

    def print_summary(self):
        prices: dict = {}
        for sym in self.symbols:
            try:
                df = self.cache.get_klines(sym, self.interval, 1)
                prices[sym] = df["close"].iloc[-1]
            except Exception:
                pass

        position_value = sum(
            pos.quantity * prices.get(sym, pos.entry_price)
            for sym, pos in self.state.open_positions.items()
        )
        portfolio = self.state.balance + position_value
        total_return = (portfolio - self.state.initial_balance) / self.state.initial_balance * 100
        trades = self.trade_log.read_all()

        print("=" * 60)
        print("PAPER TRADING SUMMARY")
        print("=" * 60)
        print(f"Initial balance:   ${self.state.initial_balance:,.2f}")
        print(f"Free balance:      ${self.state.balance:,.2f}")
        print(f"Positions value:   ${position_value:,.2f}")
        print(f"Portfolio value:   ${portfolio:,.2f}")
        print(f"Total return:      {total_return:+.2f}%")
        print(f"Total fees paid:   ${self.state.total_fees_paid:,.4f}")
        print(f"Cycles:            {self.state.cycle_count}")
        print(f"Trades completed:  {len(trades)}")
        print(f"Open positions:    {len(self.state.open_positions)}")
        for sym, pos in self.state.open_positions.items():
            current = prices.get(sym, pos.entry_price)
            unrealized_pct = (current / pos.entry_price - 1) * 100
            print(f"  {sym}: {pos.quantity:.8f} @ ${pos.entry_price:,.2f}  "
                  f"now ${current:,.2f}  ({unrealized_pct:+.2f}%)")
        print("=" * 60)