from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class CandleRequest:
    symbol: str
    interval: str
    limit: int = 500


class CandleDataSource(Protocol):
    def load(self, request: CandleRequest) -> list[Candle]:
        """Load candles for a specific request."""


def load_candles_from_csv(path: str | Path) -> list[Candle]:
    candles: list[Candle] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required columns: {missing_columns}")

        for row in reader:
            candles.append(
                Candle(
                    timestamp=row["timestamp"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )

    if not candles:
        raise ValueError("CSV did not contain any candle rows")
    return candles


class CsvCandleDataSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, request: CandleRequest) -> list[Candle]:
        del request
        return load_candles_from_csv(self.path)
