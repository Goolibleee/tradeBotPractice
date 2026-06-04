from __future__ import annotations

from trading_bot.data import Candle


def simple_moving_average(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0:
        raise ValueError("period must be positive")

    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= period:
            running_sum -= values[index - period]
        if index >= period - 1:
            result[index] = running_sum / period
    return result


def average_true_range(candles: list[Candle], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")

    true_ranges: list[float] = []
    for index, candle in enumerate(candles):
        if index == 0:
            true_range = candle.high - candle.low
        else:
            previous_close = candles[index - 1].close
            true_range = max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        true_ranges.append(true_range)

    return simple_moving_average(true_ranges, period)
