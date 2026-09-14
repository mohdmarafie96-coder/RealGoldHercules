"""Resolving a bar where both the stop and the target lie inside the range.

Two modes, chosen by config:

``pessimistic`` (default)
    Assume the stop is hit first. Safe, and the only honest default without
    tick data, but it is a BIASED estimator, not a neutral one: it charges the
    worst ordering every time both levels are reachable. That bias is on top of
    costs, which is why the calibration test must not run in this mode.

``tick``
    Replay the bar's ticks in order and take whichever level is touched first.
    Exact, and the reason the tick validation slice exists.

The share of bars where this matters is measurable: across the history, M15
bars whose range exceeds three times their month's own median run from 1.52%
to 9.68% of bars depending on regime.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, Sequence

from ..types import ExitReason, Quotes, Side
from .position import Position

__all__ = ["IntrabarMode", "Resolution", "IntrabarResolver",
           "PessimisticResolver", "TickResolver", "make_resolver"]


class IntrabarMode(str, Enum):
    PESSIMISTIC = "pessimistic"
    TICK = "tick"


@dataclass(frozen=True, slots=True)
class Resolution:
    reason: ExitReason
    level: float | None
    ambiguous: bool = False
    """True when both levels were reachable in this bar and an ordering had to
    be assumed or resolved. Counted in the report so the reader knows how much
    of the result rests on the assumption."""


def _touched(pos: Position, q: Quotes) -> tuple[bool, bool]:
    """(stop_touched, target_touched) using the side the position exits on."""
    hi = q.exit_high(pos.side)
    lo = q.exit_low(pos.side)
    if pos.side is Side.LONG:
        stop = pos.stop_price is not None and lo <= pos.stop_price
        target = pos.target_price is not None and hi >= pos.target_price
    else:
        stop = pos.stop_price is not None and hi >= pos.stop_price
        target = pos.target_price is not None and lo <= pos.target_price
    return stop, target


class IntrabarResolver(Protocol):
    mode: IntrabarMode

    def resolve(self, pos: Position, q: Quotes) -> Resolution | None: ...


class PessimisticResolver:
    """Stop wins whenever both are reachable."""

    mode = IntrabarMode.PESSIMISTIC

    def resolve(self, pos: Position, q: Quotes) -> Resolution | None:
        stop, target = _touched(pos, q)
        if stop and target:
            return Resolution(ExitReason.STOP, pos.stop_price, ambiguous=True)
        if stop:
            return Resolution(ExitReason.STOP, pos.stop_price)
        if target:
            return Resolution(ExitReason.TARGET, pos.target_price)
        return None


class TickResolver:
    """Replay ticks within the bar and take whichever level is touched first.

    ``tick_source`` maps a bar open timestamp to that bar's ticks as
    (ts, bid, ask) in ascending order. Falls back to pessimistic when a bar has
    no ticks available, and says so via ``fallbacks``.
    """

    mode = IntrabarMode.TICK

    def __init__(self, tick_source) -> None:
        self._src = tick_source
        self._fallback = PessimisticResolver()
        self.fallbacks = 0

    def resolve(self, pos: Position, q: Quotes) -> Resolution | None:
        stop, target = _touched(pos, q)
        if not (stop or target):
            return None
        if not (stop and target):
            if stop:
                return Resolution(ExitReason.STOP, pos.stop_price)
            return Resolution(ExitReason.TARGET, pos.target_price)

        ticks = self._src(q.ts_open)
        if not ticks:
            self.fallbacks += 1
            return self._fallback.resolve(pos, q)

        for _ts, bid, ask in ticks:
            px = bid if pos.side is Side.LONG else ask
            if pos.side is Side.LONG:
                if pos.stop_price is not None and px <= pos.stop_price:
                    return Resolution(ExitReason.STOP, pos.stop_price, ambiguous=True)
                if pos.target_price is not None and px >= pos.target_price:
                    return Resolution(ExitReason.TARGET, pos.target_price, ambiguous=True)
            else:
                if pos.stop_price is not None and px >= pos.stop_price:
                    return Resolution(ExitReason.STOP, pos.stop_price, ambiguous=True)
                if pos.target_price is not None and px <= pos.target_price:
                    return Resolution(ExitReason.TARGET, pos.target_price, ambiguous=True)
        # touched per the bar extremes but not per the replay: trust the replay
        self._fallback.resolve(pos, q)
        return None


def make_resolver(mode: str | IntrabarMode, tick_source=None) -> IntrabarResolver:
    m = IntrabarMode(mode)
    if m is IntrabarMode.TICK:
        if tick_source is None:
            raise ValueError("tick mode requires a tick_source")
        return TickResolver(tick_source)
    return PessimisticResolver()
