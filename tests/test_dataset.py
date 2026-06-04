"""Unit tests for dataset builder."""

import csv
from pathlib import Path

import pytest

from trading_bot.config import StrategyConfig
from trading_bot.data import Candle
from trading_bot.dataset import DATASET_COLUMNS, build_dataset, export_dataset_to_csv


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


def test_build_dataset_filters_incomplete_rows() -> None:
    config = StrategyConfig()
    candles = _make_candles(50)
    rows = build_dataset(candles, config)
    # Should only return complete rows
    assert all(row.is_complete() for row in rows)


def test_build_dataset_no_rows_for_too_few_candles() -> None:
    config = StrategyConfig()
    candles = _make_candles(10)
    with pytest.raises(ValueError):
        build_dataset(candles, config)


def test_export_dataset_creates_csv(tmp_path: Path) -> None:
    config = StrategyConfig()
    candles = _make_candles(50)
    rows = build_dataset(candles, config)

    output_path = tmp_path / "test_dataset.csv"
    export_dataset_to_csv(rows, output_path)

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        assert headers == DATASET_COLUMNS
        csv_rows = list(reader)
        assert len(csv_rows) == len(rows)


def test_export_dataset_values_are_numeric(tmp_path: Path) -> None:
    config = StrategyConfig()
    candles = _make_candles(50)
    rows = build_dataset(candles, config)

    output_path = tmp_path / "test_dataset.csv"
    export_dataset_to_csv(rows, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Verify that numeric columns contain actual numbers
            float(row["log_return_1"])
            float(row["atr"])
            int(row["future_direction_1"])
            int(row["hour_of_day"])
