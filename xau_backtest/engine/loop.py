"""Event loop skeleton: shows how the window is constructed and handed out."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Protocol, Sequence

from xau_backtest.engine.window import BarHistory, BarWindow


class Order:
    """Placeholder for the real order type."""


class PortfolioView(Protocol):
    """Read-only portfolio surface handed to the strategy."""

    @property
    def equity(self) -> float: ...
    @property
    def free_margin(self) -> float: ...
    def open_positions(self) -> Sequence[object]: ...


class BarSource(Protocol):
    """Yields raw bar rows in ascending time order."""

    def iter_bars(self) -> Iterable[Mapping[str, float | int]]: ...


class Strategy(ABC):
    """Strategies see a window ending at the current bar, and nothing else."""

    @abstractmethod
    def on_bar(
        self,
        window: BarWindow,
        portfolio: PortfolioView,
    ) -> Sequence[Order]:
        """Called once per bar.

        ``window`` covers bars 0..window.index inclusive, where window.index is
        the bar that just closed. There is no argument carrying future data and
        no reachable reference to it.
        """

    def on_fill(self, fill: object) -> None:  # pragma: no cover - hook
        return None

    def on_exit(self, trade: object) -> None:  # pragma: no cover - hook
        return None


def run(
    source: BarSource,
    strategy: Strategy,
    portfolio: PortfolioView,
    *,
    broker: object | None = None,
) -> None:
    """The event loop.

    Ordering matters. A bar is appended to history *before* the strategy is
    called, so the strategy sees the bar that just closed, and the window can
    never contain a bar the clock has not reached.
    """
    history = BarHistory()

    for raw in source.iter_bars():
        # 1. reveal exactly one bar
        history.append(raw)

        # 2. build the window. O(1): truncated views, no copy of history.
        window = history.window()

        # 3. hand it to the strategy. `history` itself is never passed.
        orders = strategy.on_bar(window, portfolio)

        # 4. broker applies orders against the *next* bar's quotes, not this one
        #    (fill logic omitted in this skeleton)
        del orders
