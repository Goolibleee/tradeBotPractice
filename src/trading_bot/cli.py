from __future__ import annotations

import argparse

from trading_bot.backtest import run_backtest
from trading_bot.binance import BinanceCandleDataSource
from trading_bot.binance_vision import BinanceVisionCandleDataSource
from trading_bot.config import BacktestDataConfig, StrategyConfig, apply_overrides
from trading_bot.data import Candle, CandleRequest, CsvCandleDataSource
from trading_bot.reporting import build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trading bot backtest")
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
    parser.add_argument("--fast-ma-period", type=int)
    parser.add_argument("--slow-ma-period", type=int)
    parser.add_argument("--trend-ma-period", type=int)
    parser.add_argument("--atr-period", type=int)
    parser.add_argument("--volume-period", type=int)
    parser.add_argument("--atr-stop-multiple", type=float)
    parser.add_argument("--risk-reward-ratio", type=float)
    parser.add_argument("--risk-per-trade-pct", type=float)
    parser.add_argument("--max-daily-loss-pct", type=float)
    parser.add_argument("--cooldown-bars", type=int)
    parser.add_argument("--fee-rate", type=float)
    parser.add_argument("--slippage-rate", type=float)
    parser.add_argument("--starting-equity", type=float)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = apply_overrides(
        StrategyConfig(),
        symbol=_format_symbol_for_display(args.symbol),
        timeframe=args.interval,
        fast_ma_period=args.fast_ma_period,
        slow_ma_period=args.slow_ma_period,
        trend_ma_period=args.trend_ma_period,
        atr_period=args.atr_period,
        volume_period=args.volume_period,
        atr_stop_multiple=args.atr_stop_multiple,
        risk_reward_ratio=args.risk_reward_ratio,
        risk_per_trade_pct=args.risk_per_trade_pct,
        max_daily_loss_pct=args.max_daily_loss_pct,
        cooldown_bars=args.cooldown_bars,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        starting_equity=args.starting_equity,
    )
    data_config = BacktestDataConfig(
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
    candles = load_candles(
        data_config,
        cache_dir=args.cache_dir,
        allow_insecure_ssl=args.allow_insecure_ssl,
    )
    result = run_backtest(candles, config)
    print(build_report(result, config))


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
