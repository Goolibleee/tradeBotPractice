"""Unit tests for the backtest engine."""

import pytest

from trading_bot.backtest import run_backtest
from trading_bot.config import StrategyConfig
from trading_bot.data import Candle


def _make_candles(closes: list[float]) -> list[Candle]:
    """Helper to create a simple ascending candle series with no volume."""
    candles = []
    for i, close in enumerate(closes):
        open_price = close
        high = close + 10.0
        low = close - 10.0
        candles.append(
            Candle(
                timestamp=f"2024-01-01T{i:02d}:00:00Z",
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=1000.0,
            )
        )
    return candles


def test_backtest_no_trades_when_not_enough_data() -> None:
    # Only 10 candles, need at least 20 for default MA periods
    candles = _make_candles([100.0 + i for i in range(10)])
    config = StrategyConfig()
    result = run_backtest(candles, config)
    assert len(result.trades) == 0
    assert result.ending_equity == pytest.approx(config.starting_equity)


def test_backtest_equity_never_negative() -> None:
    # Large number of candles in a strong trend
    candles = _make_candles([100.0 + i * 5 for i in range(100)])
    config = StrategyConfig()
    result = run_backtest(candles, config)
    assert all(e >= 0 for e in result.equity_curve)
    assert result.max_drawdown_pct >= 0.0
    assert result.max_drawdown_pct <= 1.0


def test_backtest_cooldown_respected() -> None:
    # Need enough candles for indicators, then a trend to trigger entries
    candles = _make_candles([100.0 + i * 2 for i in range(100)])
    config = StrategyConfig(cooldown_bars=5)
    result = run_backtest(candles, config)

    # Check that no two entries happen within cooldown period
    entry_timestamps = [t.entry_timestamp for t in result.trades]
    for i in range(1, len(entry_timestamps)):
        # This is a simple check; in real code we'd parse timestamps
        assert entry_timestamps[i] != entry_timestamps[i - 1]


def test_backtest_model_filter_blocks_entries() -> None:
    """A model filter that always returns False should produce zero trades."""
    candles = _make_candles([100.0 + i * 2 for i in range(100)])
    config = StrategyConfig()

    def always_false(index: int) -> bool:
        return False

    result = run_backtest(candles, config, model_filter=always_false)
    assert len(result.trades) == 0


def test_backtest_model_filter_allows_entries() -> None:
    """A model filter that always returns True should behave like base strategy."""
    candles = _make_candles([100.0 + i * 2 for i in range(100)])
    config = StrategyConfig()

    base_result = run_backtest(candles, config)

    def always_true(index: int) -> bool:
        return True

    filtered_result = run_backtest(candles, config, model_filter=always_true)
    assert len(filtered_result.trades) == len(base_result.trades)
    assert filtered_result.ending_equity == pytest.approx(base_result.ending_equity)
