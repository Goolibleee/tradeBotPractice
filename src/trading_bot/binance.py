from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from trading_bot.data import Candle, CandleRequest


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass(frozen=True)
class BinanceClient:
    base_url: str = BINANCE_KLINES_URL
    timeout_seconds: int = 30
    allow_insecure_ssl: bool = False

    def fetch_klines(self, request: CandleRequest) -> list[Candle]:
        params = urlencode(
            {
                "symbol": request.symbol,
                "interval": request.interval,
                "limit": request.limit,
            }
        )
        url = f"{self.base_url}?{params}"
        ssl_context = None
        if self.allow_insecure_ssl:
            ssl_context = ssl._create_unverified_context()

        try:
            with urlopen(url, timeout=self.timeout_seconds, context=ssl_context) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Binance request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Binance: {exc.reason}") from exc

        if not isinstance(payload, list):
            raise RuntimeError("Unexpected Binance response format")

        candles: list[Candle] = []
        for row in payload:
            candles.append(
                Candle(
                    timestamp=_format_timestamp(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )

        if not candles:
            raise RuntimeError("Binance returned no candles")
        return candles


class BinanceCandleDataSource:
    def __init__(
        self,
        client: BinanceClient | None = None,
        cache_dir: str | Path | None = None,
        allow_insecure_ssl: bool = False,
    ) -> None:
        self.client = client or BinanceClient(allow_insecure_ssl=allow_insecure_ssl)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def load(self, request: CandleRequest) -> list[Candle]:
        candles = self.client.fetch_klines(request)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / _cache_filename(request)
            cache_path.write_text(_serialize_candles(candles), encoding="utf-8")
        return candles


def _format_timestamp(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cache_filename(request: CandleRequest) -> str:
    return f"{request.symbol.lower()}_{request.interval}_{request.limit}.csv"


def _serialize_candles(candles: list[Candle]) -> str:
    lines = ["timestamp,open,high,low,close,volume"]
    for candle in candles:
        lines.append(
            ",".join(
                [
                    candle.timestamp,
                    str(candle.open),
                    str(candle.high),
                    str(candle.low),
                    str(candle.close),
                    str(candle.volume),
                ]
            )
        )
    return "\n".join(lines) + "\n"
