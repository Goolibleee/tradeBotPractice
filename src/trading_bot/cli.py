from __future__ import annotations

import argparse

from trading_bot.backtest import run_backtest
from trading_bot.binance import BinanceCandleDataSource
from trading_bot.binance_vision import BinanceVisionCandleDataSource
from trading_bot.config import BacktestDataConfig, StrategyConfig, apply_overrides
from trading_bot.data import Candle, CandleRequest, CsvCandleDataSource
from trading_bot.dataset import build_dataset, export_dataset_to_csv
from trading_bot.reporting import build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading bot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest")
    _add_data_args(backtest_parser)
    _add_strategy_args(backtest_parser)

    # Dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Build and export a training dataset")
    _add_data_args(dataset_parser)
    _add_strategy_args(dataset_parser)
    dataset_parser.add_argument("--output", required=True, help="Output CSV path for the dataset")

    return parser


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-source", choices=["csv", "binance", "binance-vision"], default="csv")
    parser.add_argument("--data", help="Path to an OHLCV CSV file")
    parser.add_argument("--symbol", default="BTCUSDT", help="Market symbol for remote data sources")
    parser.add_argument("--interval", default="1h", help="Candle interval for remote data sources")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles to request")
    parser.add_argument("--cache-dir", default="data/binance", help="Cache directory for downloaded Binance candles")
    parser.add_argument("--allow-insecure-ssl", action="store_true", help="Disable SSL certificate verification for Binance requests")
    parser.add_argument("--archive-market", choices=["spot"], default="spot", help="Binance Vision market type")
    parser.add_argument("--archive-period", choices=["daily", "monthly"], default="daily", help="Binance Vision archive grouping")
    parser.add_argument("--archive-start-date", help="Binance Vision start date in YYYY-MM-DD format")
    parser.add_argument("--archive-end-date", help="Binance Vision end date in YYYY-MM-DD format")


def _add_strategy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fast-ma-period", type=int)
    parser.add_argument("--slow-ma-period", type=int)
    parser.add_argument("--trend-ma-period", type=int)
    parser.add_argument("--atr-period", type=int)
    parser.add_argument("--volume-period", type=int)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = _build_strategy_config(args)
    data_config = _build_data_config(args)
    candles = load_candles(
        data_config,
        cache_dir=args.cache_dir,
        allow_insecure_ssl=args.allow_insecure_ssl,
    )

    if args.command == "backtest":
        result = run_backtest(candles, config)
        print(build_report(result, config))
    elif args.command == "dataset":
        rows = build_dataset(candles, config)
        export_dataset_to_csv(rows, args.output)
        print(f"Dataset exported: {args.output}")
        print(f"Rows: {len(rows)}")
        print(f"Columns: {len(rows[0].__dict__) if rows else 0}")


def _build_strategy_config(args: argparse.Namespace) -> StrategyConfig:
    return apply_overrides(
        StrategyConfig(),
        symbol=_format_symbol_for_display(args.symbol),
        timeframe=args.interval,
        fast_ma_period=args.fast_ma_period,
        slow_ma_period=args.slow_ma_period,
        trend_ma_period=args.trend_ma_period,
        atr_period=args.atr_period,
        volume_period=args.volume_period,
    )


def _build_data_config(args: argparse.Namespace) -> BacktestDataConfig:
    return BacktestDataConfig(
        source=args.data_source,
        csv_path=args.data,
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        archive_market=args.archive_market,
        archive_period=args.archive_period,
        archive_start_date=args.archive_start_date,
        archive_end_date=args.archive_end_date,
    )


def load_candles(data_config: BacktestDataConfig, cache_dir: str, allow_insecure_ssl: bool) -> list[Candle]:
    request = CandleRequest(
        symbol=data_config.symbol,
        interval=data_config.interval,
        limit=data_config.limit,
    )
    if data_config.source == "csv":
        if not data_config.csv_path:
            raise ValueError("--data is required when --data-source=csv")
        source = CsvCandleDataSource(data_config.csv_path)
        return source.load(request)
    if data_config.source == "binance-vision":
        if not data_config.archive_start_date or not data_config.archive_end_date:
            raise ValueError(
                "--archive-start-date and --archive-end-date are required when --data-source=binance-vision"
            )
        source = BinanceVisionCandleDataSource(cache_dir=cache_dir, allow_insecure_ssl=allow_insecure_ssl)
        return source.load(
            request,
            market=data_config.archive_market,
            period=data_config.archive_period,
            start_date=data_config.archive_start_date,
            end_date=data_config.archive_end_date,
        )
    source = BinanceCandleDataSource(cache_dir=cache_dir, allow_insecure_ssl=allow_insecure_ssl)
    return source.load(request)


def _format_symbol_for_display(symbol: str) -> str:
    if symbol.endswith("USDT") and len(symbol) > 4:
        base = symbol[:-4]
        return f"{base}/USDT"
    return symbol


if __name__ == "__main__":
    main()
