"""Session calendar, anchored in exchange-local time.

Bar boundaries live on a UTC epoch grid and are DST-independent. Session
boundaries do not: the trading week and the daily rollover are defined at
17:00/18:00 New York, which is 21:00/22:00 UTC in summer and 22:00/23:00 UTC in
winter. Every session question is answered by converting to the exchange
timezone first, so the tz database handles DST rather than UTC arithmetic.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, time as dtime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Final, Iterator
from zoneinfo import ZoneInfo

from ..config import SessionConfig
from ..timeutils import UTC, require_utc

__all__ = ["SessionCalendar", "WEEKDAYS"]

WEEKDAYS: Final[dict[str, int]] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


class SessionCalendar:
    """Answers: is this instant tradeable, and when is the next rollover."""

    def __init__(self, cfg: SessionConfig) -> None:
        self._cfg = cfg
        self._tz = ZoneInfo(cfg.timezone)
        self._open_wd = WEEKDAYS[cfg.week_open_weekday.lower()]
        self._close_wd = WEEKDAYS[cfg.week_close_weekday.lower()]
        self._holidays: frozenset[date] = frozenset()
        if cfg.holidays_file and Path(cfg.holidays_file).exists():
            with Path(cfg.holidays_file).open() as fh:
                self._holidays = frozenset(
                    date.fromisoformat(r["date"]) for r in csv.DictReader(fh)
                )

    @property
    def timezone(self) -> ZoneInfo:
        return self._tz

    def local(self, ts: datetime) -> datetime:
        require_utc(ts)
        return ts.astimezone(self._tz)

    def is_holiday(self, ts: datetime) -> bool:
        return self.local(ts).date() in self._holidays

    def in_weekend(self, ts: datetime) -> bool:
        """True if the instant falls outside the Sunday-open/Friday-close week."""
        lt = self.local(ts)
        wd, t = lt.weekday(), lt.time()
        if wd == 5:
            return True
        if wd == self._open_wd and t < self._cfg.week_open_time:
            return True
        if wd == self._close_wd and t >= self._cfg.week_close_time:
            return True
        return False

    def in_daily_break(self, ts: datetime) -> bool:
        """True during the daily rollover break (17:00-18:00 exchange-local)."""
        lt = self.local(ts)
        if lt.weekday() in (5, 6):
            return False
        return self._cfg.break_start <= lt.time() < self._cfg.break_end

    def is_tradeable(self, ts: datetime) -> bool:
        return not (self.in_weekend(ts) or self.in_daily_break(ts) or self.is_holiday(ts))

    def rollover_utc(self, on: date) -> datetime:
        """The UTC instant of the rollover for a given exchange-local date.

        This is the DST-sensitive value: 21:00 UTC in summer, 22:00 in winter.
        """
        local = datetime.combine(on, self._cfg.break_start, tzinfo=self._tz)
        return local.astimezone(UTC)

    def expected_bar_opens(
        self, start: datetime, end: datetime, period_seconds: int
    ) -> Iterator[datetime]:
        """Every bar open in [start, end) that the session calendar says should exist."""
        require_utc(start, name="start")
        require_utc(end, name="end")
        step = timedelta(seconds=period_seconds)
        cur = start
        while cur < end:
            if self.is_tradeable(cur):
                yield cur
            cur += step
