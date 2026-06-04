from __future__ import annotations

import csv
from pathlib import Path

from trading_bot.config import StrategyConfig
from trading_bot.data import Candle
from trading_bot.features import FeatureRow, build_feature_rows


DATASET_COLUMNS = [
    "timestamp",
    "close",
    "log_return_1",
    "log_return_5",
    "log_return_10",
    "volatility_10",
    "atr",
    "volume_ratio",
    "fast_ma_dist",
    "slow_ma_dist",
    "trend_ma_dist",
    "body_size",
    "upper_shadow",
    "lower_shadow",
    "hour_of_day",
    "day_of_week",
    "future_return_1",
    "future_return_5",
    "future_direction_1",
]


def build_dataset(candles: list[Candle], config: StrategyConfig) -> list[FeatureRow]:
    rows = build_feature_rows(candles, config)
    return [row for row in rows if row.is_complete()]


def export_dataset_to_csv(rows: list[FeatureRow], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_dict(row))


def _row_to_dict(row: FeatureRow) -> dict[str, str | float | int]:
    return {
        "timestamp": row.timestamp,
        "close": row.close,
        "log_return_1": _fmt(row.log_return_1),
        "log_return_5": _fmt(row.log_return_5),
        "log_return_10": _fmt(row.log_return_10),
        "volatility_10": _fmt(row.volatility_10),
        "atr": _fmt(row.atr),
        "volume_ratio": _fmt(row.volume_ratio),
        "fast_ma_dist": _fmt(row.fast_ma_dist),
        "slow_ma_dist": _fmt(row.slow_ma_dist),
        "trend_ma_dist": _fmt(row.trend_ma_dist),
        "body_size": _fmt(row.body_size),
        "upper_shadow": _fmt(row.upper_shadow),
        "lower_shadow": _fmt(row.lower_shadow),
        "hour_of_day": row.hour_of_day,
        "day_of_week": row.day_of_week,
        "future_return_1": _fmt(row.future_return_1),
        "future_return_5": _fmt(row.future_return_5),
        "future_direction_1": row.future_direction_1 if row.future_direction_1 is not None else "",
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10f}"
