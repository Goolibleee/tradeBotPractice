"""Unit tests for technical indicators."""

import math

import pytest

from trading_bot.data import Candle
from trading_bot.indicators import average_true_range, simple_moving_average


def test_simple_moving_average_basic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = simple_moving_average(values, period=3)
    assert result[0] is None
    assert result[1] is None
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_simple_moving_average_invalid_period() -> None:
    with pytest.raises(ValueError):
        simple_moving_average([1.0, 2.0], period=0)


def test_average_true_range_basic() -> None:
    candles = [
        Candle(timestamp="T1", open=10.0, high=12.0, low=9.0, close=11.0, volume=100.0),
        Candle(timestamp="T2", open=11.0, high=13.0, low=10.0, close=12.0, volume=100.0),
        Candle(timestamp="T3", open=12.0, high=14.0, low=11.0, close=13.0, volume=100.0),
    ]
    result = average_true_range(candles, period=2)

    # Period 2 means first None, then average of first 2 true ranges
    assert result[0] is None
    assert result[1] is not None
    assert result[2] is not None


def test_average_true_range_with_gap() -> None:
    # Large gap down from previous close
    candles = [
        Candle(timestamp="T1", open=100.0, high=102.0, low=99.0, close=101.0, volume=100.0),
        Candle(timestamp="T2", open=90.0, high=91.0, low=85.0, close=86.0, volume=100.0),
    ]
    result = average_true_range(candles, period=1)

    # First bar: high-low = 3
    assert result[0] == pytest.approx(3.0)

    # Second bar: max(high-low=6, |high-prev_close|=15, |low-prev_close|=16) = 16
    assert result[1] == pytest.approx(16.0)
