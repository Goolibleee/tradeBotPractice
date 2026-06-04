from __future__ import annotations

import math
from dataclasses import dataclass

from trading_bot.config import StrategyConfig
from trading_bot.data import Candle
from trading_bot.indicators import average_true_range, simple_moving_average


@dataclass(frozen=True)
class FeatureRow:
    """One row of features and labels for a single candle."""

    timestamp: str
    close: float

    # Price action
    log_return_1: float | None
    log_return_5: float | None
    log_return_10: float | None
    volatility_10: float | None

    # Technicals
    atr: float | None
    volume_ratio: float | None
    fast_ma_dist: float | None
    slow_ma_dist: float | None
    trend_ma_dist: float | None

    # Candle shape
    body_size: float | None
    upper_shadow: float | None
    lower_shadow: float | None

    # Time
    hour_of_day: int
    day_of_week: int

    # Labels
    future_return_1: float | None
    future_return_5: float | None
    future_direction_1: int | None

    def is_complete(self) -> bool:
        required = [
            self.log_return_1,
            self.log_return_5,
            self.log_return_10,
            self.volatility_10,
            self.atr,
            self.volume_ratio,
            self.fast_ma_dist,
            self.slow_ma_dist,
            self.trend_ma_dist,
            self.body_size,
            self.upper_shadow,
            self.lower_shadow,
            self.future_return_1,
            self.future_return_5,
            self.future_direction_1,
        ]
        return all(value is not None for value in required)


def build_feature_rows(candles: list[Candle], config: StrategyConfig) -> list[FeatureRow]:
    if len(candles) < max(config.trend_ma_period, config.volume_period, config.atr_period, 20):
        raise ValueError("Not enough candles to compute features")

    closes = [candle.close for candle in candles]
    volumes = [candle.volume for candle in candles]

    fast_ma = simple_moving_average(closes, config.fast_ma_period)
    slow_ma = simple_moving_average(closes, config.slow_ma_period)
    trend_ma = simple_moving_average(closes, config.trend_ma_period)
    atr_series = average_true_range(candles, config.atr_period)
    avg_volume = simple_moving_average(volumes, config.volume_period)

    log_returns = _compute_log_returns(closes)
    volatility_10 = _rolling_std(log_returns, 10)

    rows: list[FeatureRow] = []
    for i, candle in enumerate(candles):
        hour, dow = _parse_time(candle.timestamp)

        body = candle.close - candle.open
        upper = candle.high - max(candle.open, candle.close)
        lower = min(candle.open, candle.close) - candle.low

        future_ret_1 = _future_return(closes, i, 1)
        future_ret_5 = _future_return(closes, i, 5)
        future_dir_1 = _future_direction(future_ret_1)

        rows.append(
            FeatureRow(
                timestamp=candle.timestamp,
                close=candle.close,
                log_return_1=_safe_get(log_returns, i),
                log_return_5=_safe_get(_compute_log_returns_n(closes, 5), i),
                log_return_10=_safe_get(_compute_log_returns_n(closes, 10), i),
                volatility_10=_safe_get(volatility_10, i),
                atr=_safe_get(atr_series, i),
                volume_ratio=_volume_ratio(candle.volume, _safe_get(avg_volume, i)),
                fast_ma_dist=_ma_distance(candle.close, _safe_get(fast_ma, i)),
                slow_ma_dist=_ma_distance(candle.close, _safe_get(slow_ma, i)),
                trend_ma_dist=_ma_distance(candle.close, _safe_get(trend_ma, i)),
                body_size=body / candle.close if candle.close != 0 else None,
                upper_shadow=upper / candle.close if candle.close != 0 else None,
                lower_shadow=lower / candle.close if candle.close != 0 else None,
                hour_of_day=hour,
                day_of_week=dow,
                future_return_1=future_ret_1,
                future_return_5=future_ret_5,
                future_direction_1=future_dir_1,
            )
        )

    return rows


def _compute_log_returns(closes: list[float]) -> list[float | None]:
    result: list[float | None] = [None]
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            result.append(math.log(closes[i] / closes[i - 1]))
        else:
            result.append(None)
    return result


def _compute_log_returns_n(closes: list[float], n: int) -> list[float | None]:
    result: list[float | None] = [None] * len(closes)
    for i in range(n, len(closes)):
        if closes[i - n] > 0:
            result[i] = math.log(closes[i] / closes[i - n])
    return result


def _rolling_std(values: list[float | None], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = [v for v in values[i - period + 1 : i + 1] if v is not None]
        if len(window) < period:
            continue
        mean = sum(window) / len(window)
        variance = sum((v - mean) ** 2 for v in window) / (len(window) - 1)
        result[i] = math.sqrt(variance) if variance > 0 else 0.0
    return result


def _future_return(closes: list[float], current_index: int, bars_forward: int) -> float | None:
    target_index = current_index + bars_forward
    if target_index >= len(closes):
        return None
    current = closes[current_index]
    future = closes[target_index]
    if current == 0:
        return None
    return (future - current) / current


def _future_direction(future_return: float | None) -> int | None:
    if future_return is None:
        return None
    return 1 if future_return > 0 else 0


def _safe_get(values: list[float | None], index: int) -> float | None:
    if index < 0 or index >= len(values):
        return None
    return values[index]


def _volume_ratio(current_volume: float, avg_volume: float | None) -> float | None:
    if avg_volume is None or avg_volume == 0:
        return None
    return current_volume / avg_volume


def _ma_distance(close: float, ma: float | None) -> float | None:
    if ma is None or close == 0:
        return None
    return (close - ma) / close


def _parse_time(timestamp: str) -> tuple[int, int]:
    """Extract hour of day and day of week from ISO timestamp."""
    from datetime import UTC, datetime

    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return dt.hour, dt.weekday()
