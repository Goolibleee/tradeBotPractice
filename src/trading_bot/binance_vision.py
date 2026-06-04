from __future__ import annotations

import csv
import io
import ssl
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from trading_bot.data import Candle, CandleRequest


BASE_URL = "https://data.binance.vision/data"


@dataclass(frozen=True)
class BinanceVisionRequest:
    symbol: str
    interval: str
    market: str
    period: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class BinanceVisionClient:
    base_url: str = BASE_URL
    timeout_seconds: int = 30
    allow_insecure_ssl: bool = False

    def fetch_klines(self, request: BinanceVisionRequest) -> list[Candle]:
        period_starts = _build_period_starts(request.start_date, request.end_date, request.period)
        candles: list[Candle] = []
        for period_start in period_starts:
            url = _build_archive_url(self.base_url, request, period_start)
            candles.extend(_download_archive(url, self.timeout_seconds, self.allow_insecure_ssl))

        if not candles:
            raise RuntimeError("Binance Vision returned no candles")
        return candles


class BinanceVisionCandleDataSource:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        client: BinanceVisionClient | None = None,
        allow_insecure_ssl: bool = False,
    ) -> None:
        self.client = client or BinanceVisionClient(allow_insecure_ssl=allow_insecure_ssl)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def load(self, request: CandleRequest, *, market: str, period: str, start_date: str, end_date: str) -> list[Candle]:
        vision_request = BinanceVisionRequest(
            symbol=request.symbol,
            interval=request.interval,
            market=market,
            period=period,
            start_date=start_date,
            end_date=end_date,
        )
        candles = self.client.fetch_klines(vision_request)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / _cache_filename(vision_request)
            cache_path.write_text(_serialize_candles(candles), encoding="utf-8")
        return candles


def _download_archive(url: str, timeout_seconds: int, allow_insecure_ssl: bool) -> list[Candle]:
    ssl_context = None
    if allow_insecure_ssl:
        ssl_context = ssl._create_unverified_context()

    try:
        with urlopen(url, timeout=timeout_seconds, context=ssl_context) as response:
            zipped_content = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Binance Vision request failed with HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Binance Vision: {exc.reason}") from exc

    with zipfile.ZipFile(io.BytesIO(zipped_content)) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise RuntimeError(f"Unexpected archive contents for {url}")
        with archive.open(names[0], "r") as handle:
            reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8"))
            return [_row_to_candle(row) for row in reader if row]


def _row_to_candle(row: list[str]) -> Candle:
    return Candle(
        timestamp=_format_timestamp(int(row[0])),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def _format_timestamp(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_period_starts(start_date_value: str, end_date_value: str, period: str) -> list[date]:
    start_value = date.fromisoformat(start_date_value)
    end_value = date.fromisoformat(end_date_value)
    if end_value < start_value:
        raise ValueError("archive end date must be on or after archive start date")
    if period == "daily":
        return _build_daily_starts(start_value, end_value)
    if period == "monthly":
        return _build_monthly_starts(start_value, end_value)
    raise ValueError(f"Unsupported archive period: {period}")


def _build_daily_starts(start_value: date, end_value: date) -> list[date]:
    values: list[date] = []
    current = start_value
    while current <= end_value:
        values.append(current)
        current += timedelta(days=1)
    return values


def _build_monthly_starts(start_value: date, end_value: date) -> list[date]:
    values: list[date] = []
    current = date(start_value.year, start_value.month, 1)
    end_month = date(end_value.year, end_value.month, 1)
    while current <= end_month:
        values.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return values


def _build_archive_url(base_url: str, request: BinanceVisionRequest, period_start: date) -> str:
    suffix = period_start.strftime("%Y-%m-%d") if request.period == "daily" else period_start.strftime("%Y-%m")
    filename = f"{request.symbol}-{request.interval}-{suffix}.zip"
    return (
        f"{base_url}/{request.market}/{request.period}/klines/"
        f"{request.symbol}/{request.interval}/{filename}"
    )


def _cache_filename(request: BinanceVisionRequest) -> str:
    return (
        f"{request.symbol.lower()}_{request.interval}_{request.period}_"
        f"{request.start_date}_{request.end_date}.csv"
    )


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
