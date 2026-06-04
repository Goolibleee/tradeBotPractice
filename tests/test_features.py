"""Unit tests for feature engineering."""

import pytest

from trading_bot.config import StrategyConfig
from trading_bot.data import Candle
from trading_bot.features import FeatureRow, build_feature_rows


def _make_candles(n: int) -> list[Candle]:
    candles = []
    for i in range(n):
        close = 100.0 + i
        day = 1 + i // 24
        hour = i % 24
        candles.append(
            Candle(
                timestamp=f"2024-01-{day:02d}T{hour:02d}:00:00Z",
                open=close,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1000.0 + i,
            )
        )
    return candles


def test_build_feature_rows_insufficient_data() -> None:
    config = StrategyConfig()
    candles = _make_candles(10)
    with pytest.raises(ValueError):
        build_feature_rows(candles, config)


def test_build_feature_rows_returns_correct_shape() -> None:
    config = StrategyConfig()
    candles = _make_candles(50)
    rows = build_feature_rows(candles, config)
    assert len(rows) == len(candles)


def test_feature_row_is_complete() -> None:
    config = StrategyConfig()
    candles = _make_candles(50)
    rows = build_feature_rows(candles, config)
    # Later rows should be complete since indicators have enough history
    complete_rows = [r for r in rows if r.is_complete()]
    assert len(complete_rows) > 0


def test_log_return_computation() -> None:
    config = StrategyConfig()
    # Create candles with known closes for predictable log returns
    candles = []
    for i in range(30):
        close = 100.0 * (1.01 ** i)
        day = 1 + i // 24
        hour = i % 24
        candles.append(
            Candle(
                timestamp=f"2024-01-{day:02d}T{hour:02d}:00:00Z",
                open=close,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1000.0,
            )
        )
    rows = build_feature_rows(candles, config)
    # Check that log_return_1 exists for rows after index 0
    assert rows[1].log_return_1 is not None
    assert rows[1].log_return_1 > 0  # Price is increasing


def test_future_direction_label() -> None:
    config = StrategyConfig()
    # Pad to get enough data for indicators
    candles = _make_candles(30)
    # Override closes with our pattern starting from index 20
    for i, close in enumerate([100.0, 102.0, 101.0, 103.0, 102.0, 104.0]):
        idx = 20 + i
        day = 1 + idx // 24
        hour = idx % 24
        candles[idx] = Candle(
            timestamp=f"2024-01-{day:02d}T{hour:02d}:00:00Z",
            open=close,
            high=close + 1.0,
            low=close - 1.0,
            close=close,
            volume=1000.0,
        )

    rows = build_feature_rows(candles, config)
    # Row 20: close=100, next=102 -> direction should be 1 (up)
    assert rows[20].future_direction_1 == 1
    # Row 21: close=102, next=101 -> direction should be 0 (down)
    assert rows[21].future_direction_1 == 0
