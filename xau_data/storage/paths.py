"""Hive-style partition paths: year=YYYY/month=MM throughout."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..timeutils import require_utc

__all__ = ["partition_dir", "ticks_partition", "bars_partition", "context_partition"]


def partition_dir(base: Path, year: int, month: int) -> Path:
    if not 1 <= month <= 12:
        raise ValueError(f"month out of range: {month}")
    return base / f"year={year:04d}" / f"month={month:02d}"


def ticks_partition(base: Path, symbol: str, year: int, month: int) -> Path:
    return partition_dir(base / f"symbol={symbol}", year, month)


def bars_partition(base: Path, symbol: str, timeframe: str, year: int, month: int) -> Path:
    return partition_dir(base / f"symbol={symbol}" / f"timeframe={timeframe}", year, month)


def context_partition(base: Path, series_id: str, year: int, month: int) -> Path:
    return partition_dir(base / f"series={series_id}", year, month)


def month_of(dt: datetime) -> tuple[int, int]:
    require_utc(dt)
    return dt.year, dt.month
