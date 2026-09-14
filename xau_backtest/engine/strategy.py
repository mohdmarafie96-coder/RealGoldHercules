"""Strategy base and two reference implementations.

A strategy sees a ``BarWindow`` ending at the bar that just closed and a
read-only ``PortfolioView``. It returns orders. It cannot reach the future and
cannot mutate portfolio state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..types import Order, Side
from .position import PortfolioView
from .window import BarWindow

__all__ = ["Strategy", "RandomStrategy", "LondonBreakoutStrategy",
           "AsianRangeFadeStrategy"]


class Strategy(ABC):
    """Base class. Subclasses must not import or hold the full dataset."""

    name: str = "strategy"

    @abstractmethod
    def on_bar(self, window: BarWindow, portfolio: PortfolioView) -> Sequence[Order]:
        """Called once per bar, after that bar has closed."""

    def on_fill(self, fill: object) -> None:  # pragma: no cover - hook
        return None

    def on_exit(self, trade: object) -> None:  # pragma: no cover - hook
        return None


class RandomStrategy(Strategy):
    """Random entries, fixed stop and target in ATR units.

    Exists to calibrate the cost model. Random entries have zero edge by
    construction, so the measured expectancy should equal the cost and nothing
    else. If random entries look profitable, the engine is wrong.

    Determinism comes from an explicitly passed Generator. There is no module
    level RNG and no call to global numpy or random state anywhere.
    """

    name = "random"

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        entry_probability: float = 0.02,
        atr_period: int = 14,
        stop_atr: float = 1.5,
        target_atr: float = 1.5,
        max_bars: int = 32,
        size: float = 1.0,
        max_concurrent: int = 1,
        use_stops: bool = True,
    ) -> None:
        if not 0.0 < entry_probability <= 1.0:
            raise ValueError("entry_probability must be in (0, 1]")
        self._rng = rng
        self._p = entry_probability
        self._atr_period = atr_period
        self._stop_atr = stop_atr
        self._target_atr = target_atr
        self._max_bars = max_bars
        self._size = size
        self._max_concurrent = max_concurrent
        self._use_stops = use_stops

    def _atr(self, window: BarWindow) -> float | None:
        n = self._atr_period + 1
        if len(window) < n:
            return None
        hi = window.series("bid_high", n)
        lo = window.series("bid_low", n)
        cl = window.series("bid_close", n)
        tr = np.maximum.reduce([
            hi[1:] - lo[1:],
            np.abs(hi[1:] - cl[:-1]),
            np.abs(lo[1:] - cl[:-1]),
        ])
        v = float(np.mean(tr))
        return v if v > 0 else None

    def on_bar(self, window: BarWindow, portfolio: PortfolioView) -> Sequence[Order]:
        if len(portfolio) >= self._max_concurrent:
            return ()
        if self._rng.random() >= self._p:
            return ()
        atr = self._atr(window)
        if atr is None:
            return ()
        side = Side.LONG if self._rng.random() < 0.5 else Side.SHORT
        return (
            Order(
                side=side,
                size=self._size,
                stop_distance=self._stop_atr * atr if self._use_stops else None,
                target_distance=self._target_atr * atr if self._use_stops else None,
                max_bars=self._max_bars,
                tag="random",
            ),
        )


class LondonBreakoutStrategy(Strategy):
    """Break of the Asian range, taken during the London session.

    The Asian range is built only from bars already closed, so the signal at
    bar i uses nothing later than bar i.
    """

    name = "london_breakout"

    def __init__(
        self,
        *,
        atr_period: int = 14,
        stop_atr: float = 1.0,
        target_atr: float = 2.0,
        max_bars: int = 32,
        size: float = 1.0,
        max_concurrent: int = 1,
        min_range_atr: float = 0.25,
    ) -> None:
        self._atr_period = atr_period
        self._stop_atr = stop_atr
        self._target_atr = target_atr
        self._max_bars = max_bars
        self._size = size
        self._max_concurrent = max_concurrent
        self._min_range_atr = min_range_atr
        self._traded_days: set[object] = set()

    def _atr(self, window: BarWindow) -> float | None:
        n = self._atr_period + 1
        if len(window) < n:
            return None
        hi = window.series("bid_high", n)
        lo = window.series("bid_low", n)
        cl = window.series("bid_close", n)
        tr = np.maximum.reduce([
            hi[1:] - lo[1:], np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1]),
        ])
        v = float(np.mean(tr))
        return v if v > 0 else None

    def on_bar(self, window: BarWindow, portfolio: PortfolioView) -> Sequence[Order]:
        session = window.session()
        if session != "london" or len(portfolio) >= self._max_concurrent:
            return ()
        day = window.ts().date()
        if day in self._traded_days:
            return ()

        asian = window.session_slice("asian", day)
        if asian is None or len(asian["bid_high"]) < 4:
            return ()
        hi = float(np.max(asian["bid_high"]))
        lo = float(np.min(asian["bid_low"]))
        atr = self._atr(window)
        if atr is None or (hi - lo) < self._min_range_atr * atr:
            return ()

        close = window.value("bid_close")
        if close > hi:
            side = Side.LONG
        elif close < lo:
            side = Side.SHORT
        else:
            return ()

        self._traded_days.add(day)
        return (
            Order(
                side=side, size=self._size,
                stop_distance=self._stop_atr * atr,
                target_distance=self._target_atr * atr,
                max_bars=self._max_bars, tag="london_breakout",
            ),
        )


class AsianRangeFadeStrategy(Strategy):
    """Mean reversion: FADE breaks of the Asian range instead of following them.

    Deliberately the mirror of LondonBreakoutStrategy on the same signal, so the
    pair brackets the question rather than testing one direction. A break above
    the Asian high is sold, a break below is bought, the stop sits beyond the
    extreme and the target is the range midpoint.

    Being the mirror does NOT make its expectancy the negative of the breakout's:
    both pay the spread, both pay slippage, and their stop and target geometries
    differ. If both lose, the signal carries no directional information at this
    horizon, which is a more useful result than either alone.
    """

    name = "asian_fade"

    def __init__(
        self,
        *,
        atr_period: int = 14,
        stop_atr: float = 1.0,
        max_bars: int = 32,
        size: float = 1.0,
        max_concurrent: int = 1,
        min_range_atr: float = 0.25,
        target_fraction: float = 0.5,
    ) -> None:
        self._atr_period = atr_period
        self._stop_atr = stop_atr
        self._max_bars = max_bars
        self._size = size
        self._max_concurrent = max_concurrent
        self._min_range_atr = min_range_atr
        self._target_fraction = target_fraction
        self._traded_days: set[object] = set()

    def _atr(self, window: BarWindow) -> float | None:
        n = self._atr_period + 1
        if len(window) < n:
            return None
        hi = window.series("bid_high", n)
        lo = window.series("bid_low", n)
        cl = window.series("bid_close", n)
        tr = np.maximum.reduce([
            hi[1:] - lo[1:], np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1]),
        ])
        v = float(np.mean(tr))
        return v if v > 0 else None

    def on_bar(self, window: BarWindow, portfolio: PortfolioView) -> Sequence[Order]:
        if window.session() != "london" or len(portfolio) >= self._max_concurrent:
            return ()
        day = window.ts().date()
        if day in self._traded_days:
            return ()

        asian = window.session_slice("asian", day)
        if asian is None or len(asian["bid_high"]) < 4:
            return ()
        hi = float(np.max(asian["bid_high"]))
        lo = float(np.min(asian["bid_low"]))
        atr = self._atr(window)
        if atr is None or (hi - lo) < self._min_range_atr * atr:
            return ()

        close = window.value("bid_close")
        mid_range = (hi + lo) / 2.0
        if close > hi:
            side = Side.SHORT                     # fade the upside break
        elif close < lo:
            side = Side.LONG                      # fade the downside break
        else:
            return ()

        target_distance = abs(close - mid_range) * (self._target_fraction * 2.0)
        if target_distance <= 0:
            return ()

        self._traded_days.add(day)
        return (
            Order(
                side=side, size=self._size,
                stop_distance=self._stop_atr * atr,
                target_distance=target_distance,
                max_bars=self._max_bars, tag="asian_fade",
            ),
        )
