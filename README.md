# ML Trading Bot

A machine learning-based algorithmic trading bot with multi-timeframe feature engineering, walk-forward validation and Monte Carlo analysis.
Supports multi-coin paper trading with portfolio-level risk management.
Currently integrates with Binance, planned expansion to stocks (Alpaca) and forex (OANDA).

Built as a self-directed learning project while studying software engineering at ELTE.

## Features

- **Multi-coin paper trading** with persistent state (JSON) and trade log (JSONL)
- **Random Forest classifier** trained per-symbol on multi-timeframe features
- **Multi-timeframe feature engineering**: 15m trade TF + 1h, 4h, 1d context features
- **Backtest engine** with configurable risk management (stop-loss, take-profit)
- **Multi-coin backtest** with shared portfolio balance and exposure limits
- **Walk-forward optimization** across rolling train/test windows
- **Monte Carlo simulation** for drawdown shape analysis
- **CSV data cache** to minimize Binance API calls

## Validation results

Tested on 7 months (208 days) of BTCUSDT 15m data from Binance (May - October 2026):

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

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a single-coin ML backtest:

```bash
python -m examples.backtest_ml
```

Run walk-forward optimization + Monte Carlo:

```bash
python -m examples.walk_forward
```

Run multi-coin backtest (BTC + ETH + SOL):

```bash
python -m examples.backtest_multi_coin
```

Run the live paper trader:

```bash
python -m examples.run_paper_trader
```

## Project structure

```
src/
├── exchanges/binance_client.py # Paginated Binance REST client
├── data/cache.py # CSV-based OHLCV cache
├── indicators/ # SMA, EMA, RSI
├── ml/
│ ├── features.py # Multi-TF feature engineering
│ ├── labels.py # Future-return labels
│ ├── dataset.py # Train/test dataset builder
│ └── model.py # Random Forest wrapper
├── strategies/
│ ├── sma_crossover.py # Baseline strategy
│ └── ml_strategy.py # ML strategy with confidence filter
├── risk/manager.py # Stop-loss / take-profit
├── backtest/
│ ├── engine.py # Single-coin backtest
│ ├── multi_coin_engine.py # Multi-coin backtest
│ └── advanced.py # Walk-forward + Monte Carlo
└── paper_trading/
├── state.py # Persistent state + trade log
└── trader.py # Live paper trader loop
```

## Status

Work in progress. Current roadmap:

- [ ] Alpaca integration for stocks
- [ ] OANDA integration for forex
- [ ] Ensemble model (Random Forest + Gradient Boosting + Logistic Regression)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Validation on multi-year data covering bear and sideways markets