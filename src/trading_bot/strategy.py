from __future__ import annotations

from dataclasses import dataclass

from trading_bot.config import StrategyConfig
from trading_bot.data import Candle
from trading_bot.indicators import average_true_range, simple_moving_average


@dataclass(frozen=True)
class SignalSnapshot:
    candle: Candle
    fast_ma: float | None
    slow_ma: float | None
    trend_ma: float | None
    atr: float | None
    average_volume: float | None

    @property
    def can_enter(self) -> bool:
        return all(
            value is not None
            for value in (self.fast_ma, self.slow_ma, self.trend_ma, self.atr, self.average_volume)
        )

    @property
    def is_long_entry(self) -> bool:
        if not self.can_enter:
            return False
        return (
            self.fast_ma > self.slow_ma
            and self.candle.close > self.trend_ma
            and self.candle.volume > self.average_volume
        )

    @property
    def is_trend_break(self) -> bool:
        if self.fast_ma is None or self.slow_ma is None or self.trend_ma is None:
            return False
        return self.fast_ma <= self.slow_ma or self.candle.close <= self.trend_ma


def build_signal_snapshots(candles: list[Candle], config: StrategyConfig) -> list[SignalSnapshot]:
    closes = [candle.close for candle in candles]
    volumes = [candle.volume for candle in candles]

    fast_ma = simple_moving_average(closes, config.fast_ma_period)
    slow_ma = simple_moving_average(closes, config.slow_ma_period)
    trend_ma = simple_moving_average(closes, config.trend_ma_period)
    atr = average_true_range(candles, config.atr_period)
    average_volume = simple_moving_average(volumes, config.volume_period)

    return [
        SignalSnapshot(
            candle=candle,
            fast_ma=fast_ma[index],
            slow_ma=slow_ma[index],
            trend_ma=trend_ma[index],
            atr=atr[index],
            average_volume=average_volume[index],
        )
        for index, candle in enumerate(candles)
    ]
