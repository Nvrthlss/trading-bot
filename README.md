# ML Trading Bot

A machine learning-based algorithmic trading bot with multi-timeframe feature engineering, walk-forward validation and Monte Carlo analysis.
Supports multi-coin paper trading with portfolio-level risk management.
Currently integrates with Binance and stocks (Alpaca), planned expansion to forex (OANDA).

Built as a self-directed learning project while studying software engineering at ELTE.

## Features

- **Pluggable exchange adapters** — common `ExchangeClient` interface, with Binance (crypto) and Alpaca (stocks) implementations
- **Multi-asset paper trading** with persistent state (JSON) and trade log (JSONL)
- **Random Forest classifier** trained per-symbol on multi-timeframe features
- **Multi-timeframe feature engineering**: 15m trade TF + 1h, 4h, 1d context features
- **Backtest engine** with configurable risk management (stop-loss, take-profit)
- **Multi-asset backtest** with shared portfolio balance and exposure limits (works for crypto, stocks)
- **Walk-forward optimization** across rolling train/test windows
- **Monte Carlo simulation** for drawdown shape analysis
- **CSV data cache** per-exchange, minimizing API calls

## Validation results

**Crypto — tested on 7 months (208 days) of BTCUSDT 15m data from Binance (May - October 2026):**

- Walk-forward optimization (7 windows, 21-day test periods): **100% consistency**, +0.98% avg return per window, 92.7% avg win rate (111 trades), ~17% annualized
- Multi-asset backtest (BTC + ETH + SOL): **+16.07% portfolio return**, 217 trades, 92.6% win rate, all three coins individually profitable

**Stocks — tested on 5 months of 15m data from Alpaca (AAPL + MSFT + NVDA):**

- Multi-asset backtest: **+1.22% portfolio return**, 21 trades, 76.2% win rate
- Lower return and trade frequency than crypto, consistent with stock market being open only ~6.5h/day

**Walk-forward optimization (7 windows, 21-day test periods):**
- 100% consistency (every window profitable)
- +0.98% average return per 3-week test window
- 92.7% average win rate across 111 trades
- ~17% annualized return

**Multi-coin backtest (BTC + ETH + SOL, 7 months):**
- +17.47% portfolio return
- 199 trades, 95.5% win rate
- All three coins individually profitable (BTC 90%, ETH 98%, SOL 97% win rate)

Note: results are measured on a predominantly bullish market period. Robustness across bear or sideways regimes is not yet validated.

## Tech stack

- Python 3.11+
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — Random Forest classifier
- `matplotlib` — equity curve plots
- `requests` — Binance REST API client
- `alpaca-py` — Alpaca stocks API client
- `python-dotenv` — environment-based credentials

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

For stock market access, create a `.env` file in the project root with your Alpaca paper trading API keys:

```
ALPACA_API_KEY_ID=your_key_id
ALPACA_API_SECRET=your_secret_key
```

Run a single-asset ML backtest (crypto):

```bash
python -m examples.backtest_ml
```

Run walk-forward optimization + Monte Carlo (crypto):

```bash
python -m examples.walk_forward
```

Run walk-forward optimization + Monte Carlo (stocks):

```bash
python -m examples.walk_forward_stock
```

Run multi-asset backtest (crypto — BTC + ETH + SOL):

```bash
python -m examples.backtest_multi_coin
```

Run multi-asset backtest (stocks — AAPL + MSFT + NVDA):

```bash
python -m examples.backtest_multi_stock
```

Run the live paper trader:

```bash
python -m examples.run_paper_trader
```

## Project structure

```
src/
├── config.py                      # Env-based configuration (.env loading)
├── exchanges/
│   ├── base.py                    # Abstract ExchangeClient interface
│   ├── binance_client.py          # Binance implementation (crypto)
│   └── alpaca_client.py           # Alpaca implementation (US stocks, IEX feed)
├── data/
│   └── cache.py                   # Per-exchange CSV OHLCV cache
├── indicators/                    # SMA, EMA, RSI
├── ml/
│   ├── features.py                # Multi-TF feature engineering
│   ├── labels.py                  # Future-return labels
│   ├── dataset.py                 # Train/test dataset builder
│   └── model.py                   # Random Forest wrapper
├── strategies/
│   ├── sma_crossover.py           # Baseline strategy
│   └── ml_strategy.py             # ML strategy with confidence filter
├── risk/
│   └── manager.py                 # Stop-loss / take-profit
├── backtest/
│   ├── engine.py                  # Single-asset backtest
│   ├── multi_asset_engine.py      # Multi-asset backtest
│   └── advanced.py                # Walk-forward + Monte Carlo
└── paper_trading/
    ├── state.py                   # Persistent state + trade log
    └── trader.py                  # Live paper trader loop
```

## Status

Work in progress. Current roadmap:

- [x] Alpaca integration for stocks
- [ ] OANDA integration for forex
- [ ] Ensemble model (Random Forest + Gradient Boosting + Logistic Regression)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Multi-asset paper trader (crypto + stocks simultaneously)
- [ ] Validation on multi-year data covering bear and sideways markets