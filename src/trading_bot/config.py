from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    fast_ma_period: int = 5
    slow_ma_period: int = 10
    trend_ma_period: int = 20
    atr_period: int = 14
    volume_period: int = 10
    atr_stop_multiple: float = 2.0
    risk_reward_ratio: float = 2.0
    risk_per_trade_pct: float = 0.005
    max_daily_loss_pct: float = 0.02
    cooldown_bars: int = 2
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    starting_equity: float = 10_000.0
    max_open_positions: int = 1


@dataclass(frozen=True)
class BacktestDataConfig:
    source: str = "csv"
    csv_path: str | None = None
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    limit: int = 500
    archive_market: str = "spot"
    archive_period: str = "daily"
    archive_start_date: str | None = None
    archive_end_date: str | None = None


def apply_overrides(config: StrategyConfig, **overrides: object) -> StrategyConfig:
    filtered = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **filtered)
