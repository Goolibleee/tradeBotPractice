# Trading Bot Practice

Backtesting-first crypto trading bot scaffold for `BTC/USDT` on the `1h` timeframe.

## Strategy

The initial strategy is long-only trend following:

- enter when the fast moving average is above the slow moving average
- require price to stay above the trend moving average
- require volume to stay above its recent average
- size each trade off an ATR-based stop and fixed account risk
- exit on stop loss, take profit, or trend breakdown

## Commands

### Backtest

```bash
PYTHONPATH=src python -m trading_bot.cli backtest --data-source csv --data data/sample_btcusdt_1h.csv
```

Use Binance spot candles directly:

```bash
PYTHONPATH=src python -m trading_bot.cli backtest --data-source binance --symbol BTCUSDT --interval 1h --limit 500
```

Use Binance Vision archives for larger historical pulls:

```bash
PYTHONPATH=src python -m trading_bot.cli backtest --data-source binance-vision --symbol BTCUSDT --interval 1h --archive-period daily --archive-start-date 2024-01-01 --archive-end-date 2024-01-03
```

If your environment uses a self-signed inspection proxy, you can opt in to bypass SSL verification:

```bash
PYTHONPATH=src python -m trading_bot.cli backtest --data-source binance --symbol BTCUSDT --interval 1h --limit 500 --allow-insecure-ssl
```

Override strategy parameters from the CLI:

```bash
PYTHONPATH=src python -m trading_bot.cli backtest --data-source binance --symbol BTCUSDT --interval 1h --limit 500 --fast-ma-period 8 --slow-ma-period 21
```

### Build Training Dataset

Export a CSV of engineered features and labels for model training:

```bash
PYTHONPATH=src python -m trading_bot.cli dataset \
  --data-source csv --data data/sample_btcusdt_1h.csv \
  --output data/datasets/sample_features.csv
```

With Binance Vision data:

```bash
PYTHONPATH=src python -m trading_bot.cli dataset \
  --data-source binance-vision --symbol BTCUSDT --interval 1h \
  --archive-period daily --archive-start-date 2024-01-01 --archive-end-date 2024-01-31 \
  --allow-insecure-ssl \
  --output data/datasets/btcusdt_1h_2024_01.csv
```

### Train a Model

Train a classifier on a dataset and save it:

```bash
PYTHONPATH=src python -m trading_bot.cli train \
  --dataset data/datasets/btcusdt_1h_2024_01.csv \
  --model-path models/btc_direction_v1.pkl \
  --model-type logistic_regression
```

Use a Random Forest instead:

```bash
PYTHONPATH=src python -m trading_bot.cli train \
  --dataset data/datasets/btcusdt_1h_2024_01.csv \
  --model-path models/btc_direction_rf_v1.pkl \
  --model-type random_forest
```

### Backtest with Model Filter

Run the base strategy but only enter when the model predicts an up move:

```bash
PYTHONPATH=src python -m trading_bot.cli backtest \
  --data-source binance-vision --symbol BTCUSDT --interval 1h \
  --archive-period daily --archive-start-date 2024-01-01 --archive-end-date 2024-01-31 \
  --allow-insecure-ssl \
  --model-path models/btc_direction_rf_v1.pkl
```

### Walk-Forward Out-of-Sample Test

The most honest way to test a model: train on one period, test on a completely different period:

```bash
PYTHONPATH=src python -m trading_bot.cli walkforward \
  --train-source binance-vision --symbol BTCUSDT --interval 1h \
  --archive-period daily --train-start-date 2024-01-01 --train-end-date 2024-01-31 \
  --test-source binance-vision --test-start-date 2024-02-01 --test-end-date 2024-02-15 \
  --allow-insecure-ssl \
  --model-type random_forest
```

## Testing

### Run Automated Unit Tests

```bash
python3 -m pip install pytest
PYTHONPATH=src python3 -m pytest tests/ -v
```

Tests cover:
- **Indicators** (`test_indicators.py`): SMA and ATR math correctness
- **Backtest** (`test_backtest.py`): equity tracking, cooldown, model filter wiring
- **Features** (`test_features.py`): feature computation, no lookahead, label correctness
- **Dataset** (`test_dataset.py`): CSV export, completeness filtering

### What to Test After Every Change

1. **Unit tests pass**: `pytest tests/ -v`
2. **No lookahead bias**: verify features only use data up to current candle
3. **Model generalizes**: use `walkforward` with train/test on different months
4. **Base strategy unchanged**: backtest without `--model-path` should match previous results

### Testing Philosophy

- **Unit tests** catch math bugs and regression
- **Walk-forward tests** catch overfitting (the model memorized training data)
- **Paper trading** catches execution bugs before real money is at risk
- **Never test a model on data it was trained on** — this is the #1 mistake in ML trading

## Workflow

1. **Generate dataset** from historical candles
2. **Train a model** on the dataset
3. **Run walk-forward test** to see if it generalizes to unseen data
4. **Backtest with model filter** to compare against base strategy
5. **Iterate** by adding more features or using more data

## Next Steps

- add more features (RSI, MACD, Bollinger Bands)
- add dedicated paper trading and live execution modules
- add live exchange execution and alerting
- experiment with different model architectures
