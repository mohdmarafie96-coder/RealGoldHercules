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

from enum import Enum

from ..config import SessionConfig
from ..timeutils import UTC, require_utc

__all__ = ["SessionCalendar", "SessionLabel", "WEEKDAYS"]

WEEKDAYS: Final[dict[str, int]] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


class SessionLabel(str, Enum):
    """Which intraday session an instant belongs to."""

    ASIAN = "asian"
    LONDON = "london"
    OVERLAP = "overlap"
    NY = "ny"
    OFF = "off"


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

    # ------------------------------------------------------------------
    # Rollover. This class is the SINGLE SOURCE OF TRUTH for rollover
    # timing: forced exits, swap charges and session labels all come from
    # here. There is no UTC hour constant anywhere in the codebase.
    # ------------------------------------------------------------------

    def rollover_for_instant(self, ts: datetime) -> datetime:
        """The rollover instant governing the exchange-local day of ``ts``."""
        return self.rollover_utc(self.local(ts).date())

    def next_rollover(self, ts: datetime) -> datetime:
        """The first rollover strictly after ``ts``."""
        require_utc(ts)
        d = self.local(ts).date()
        for _ in range(8):
            r = self.rollover_utc(d)
            if r > ts:
                return r
            d += timedelta(days=1)
        raise RuntimeError("no rollover found within 8 days")

    def rollovers_crossed(self, start: datetime, end: datetime) -> list[datetime]:
        """Rollover instants in the half-open interval (start, end].

        This is what a swap charge iterates: exactly once per position per
        rollover crossed, with no double counting at either edge.
        """
        require_utc(start, name="start")
        require_utc(end, name="end")
        if end <= start:
            return []
        out: list[datetime] = []
        d = self.local(start).date() - timedelta(days=1)
        last = self.local(end).date() + timedelta(days=1)
        while d <= last:
            r = self.rollover_utc(d)
            if start < r <= end:
                out.append(r)
            d += timedelta(days=1)
        return sorted(out)

    def is_rollover_bar(self, ts_open: datetime, period_seconds: int) -> bool:
        """True if the bar [ts_open, ts_open+period) contains a rollover."""
        require_utc(ts_open, name="ts_open")
        end = ts_open + timedelta(seconds=period_seconds)
        return bool(self.rollovers_crossed(ts_open - timedelta(microseconds=1), end - timedelta(microseconds=1)))

    def last_bar_open_before_rollover(
        self, ts: datetime, period_seconds: int
    ) -> datetime:
        """Open time of the final bar that closes at or before the next rollover.

        This is the forced-exit deadline. A position must be flat by the close
        of this bar to avoid crossing the rollover.
        """
        require_utc(ts)
        roll = self.next_rollover(ts)
        step = timedelta(seconds=period_seconds)
        # walk back from the rollover to the last bar whose close is <= roll
        from ..timeutils import epoch_us, from_epoch_us, floor_to_period
        period_us = period_seconds * 1_000_000
        end_us = floor_to_period(epoch_us(roll), period_us)
        return from_epoch_us(end_us - period_us)

    # ------------------------------------------------------------------
    # Session labels
    # ------------------------------------------------------------------

    def _in_window(self, ts: datetime, name: str) -> bool:
        for w in self._cfg.windows:
            if w.name != name:
                continue
            lt = ts.astimezone(ZoneInfo(w.tz))
            if lt.weekday() >= 5:
                return False
            return w.start <= lt.time() < w.end
        return False

    def session_label(self, ts: datetime) -> "SessionLabel":
        """Which intraday session ``ts`` falls in. DST-aware by construction."""
        require_utc(ts)
        if not self.is_tradeable(ts):
            return SessionLabel.OFF
        in_london = self._in_window(ts, "london")
        in_ny = self._in_window(ts, "ny")
        if in_london and in_ny:
            return SessionLabel.OVERLAP
        if in_ny:
            return SessionLabel.NY
        if in_london:
            return SessionLabel.LONDON
        if self._in_window(ts, "asian"):
            return SessionLabel.ASIAN
        return SessionLabel.OFF
