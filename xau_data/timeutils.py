"""UTC-only time helpers.

Every timestamp in this package is a timezone-aware instant in UTC. Naive
datetimes are rejected at every boundary rather than coerced, because silently
assuming a timezone is how lookahead bias gets in.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Final, Iterator
from zoneinfo import ZoneInfo

from .errors import NaiveDatetimeError

__all__ = [
    "UTC", "EPOCH", "require_utc", "to_utc", "epoch_us", "from_epoch_us",
    "floor_to_period", "month_range", "hour_range",
]

UTC: Final[timezone] = timezone.utc
EPOCH: Final[datetime] = datetime(1970, 1, 1, tzinfo=UTC)


def require_utc(dt: datetime, *, name: str = "timestamp") -> datetime:
    """Return ``dt`` unchanged if it is tz-aware UTC; otherwise raise."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise NaiveDatetimeError(
            f"{name} is naive: {dt!r}. Every timestamp must be timezone-aware UTC."
        )
    if dt.utcoffset() != timedelta(0):
        raise NaiveDatetimeError(
            f"{name} is not UTC: {dt!r} (offset {dt.utcoffset()}). "
            "Convert with to_utc() before passing it in."
        )
    return dt


def to_utc(dt: datetime, *, assume: str | None = None) -> datetime:
    """Convert an aware datetime to UTC. Naive input raises unless ``assume`` names a tz."""
    if dt.tzinfo is None:
        if assume is None:
            raise NaiveDatetimeError(
                f"cannot convert naive datetime {dt!r} without an explicit "
                "`assume` timezone"
            )
        dt = dt.replace(tzinfo=ZoneInfo(assume))
    return dt.astimezone(UTC)


def epoch_us(dt: datetime) -> int:
    """Microseconds since the Unix epoch, UTC."""
    require_utc(dt)
    return int((dt - EPOCH) // timedelta(microseconds=1))


def from_epoch_us(us: int) -> datetime:
    """Inverse of :func:`epoch_us`."""
    return EPOCH + timedelta(microseconds=int(us))


def floor_to_period(us: int, period_us: int) -> int:
    """Floor an epoch-microsecond instant onto a UTC-anchored grid.

    Integer arithmetic on the UTC epoch. M1/M5/M15/H1 all divide an hour
    evenly, so the grid is unambiguous and completely DST-independent.
    """
    if period_us <= 0:
        raise ValueError("period_us must be positive")
    return us - (us % period_us)


def month_range(start: date, end: date) -> Iterator[tuple[int, int]]:
    """Yield (year, month) inclusive of both endpoints' months."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def hour_range(start: datetime, end: datetime) -> Iterator[datetime]:
    """Yield hour-aligned UTC instants in [start, end)."""
    require_utc(start, name="start")
    require_utc(end, name="end")
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        yield cur
        cur += timedelta(hours=1)
