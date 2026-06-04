from __future__ import annotations

from typing import Any

import numpy as np

from trading_bot.config import StrategyConfig
from trading_bot.data import Candle
from trading_bot.features import FeatureRow, build_feature_rows
from trading_bot.train import FEATURE_COLUMNS, load_model


class ModelSignalFilter:
    """Wraps a trained classifier to filter strategy entry signals."""

    def __init__(self, model_path: str, config: StrategyConfig) -> None:
        self.model = load_model(model_path)
        self.config = config
        self._feature_rows: list[FeatureRow] | None = None

    def prepare(self, candles: list[Candle]) -> None:
        """Precompute feature rows for the entire candle series."""
        self._feature_rows = build_feature_rows(candles, self.config)

    def should_enter(self, index: int) -> bool:
        """Returns True if the model predicts an upward move at the given index."""
        if self._feature_rows is None or index >= len(self._feature_rows):
            return False
        row = self._feature_rows[index]
        if not row.is_complete():
            return False

        features = np.array([
            [
                row.log_return_1,
                row.log_return_5,
                row.log_return_10,
                row.volatility_10,
                row.atr,
                row.volume_ratio,
                row.fast_ma_dist,
                row.slow_ma_dist,
                row.trend_ma_dist,
                row.body_size,
                row.upper_shadow,
                row.lower_shadow,
                row.hour_of_day,
                row.day_of_week,
            ]
        ])
        prediction = self.model.predict(features)
        return bool(prediction[0] == 1)
