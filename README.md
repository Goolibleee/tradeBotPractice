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

## Next Steps

- add dedicated paper trading and live execution modules
- add live exchange execution and alerting
- add model training scripts that consume exported datasets
- add model-based signal filters on top of the base strategy
