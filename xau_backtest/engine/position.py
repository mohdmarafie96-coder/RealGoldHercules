"""Positions and the portfolio.

Multiple concurrent positions, each with its own stop, target and time limit.
Positions never net against each other: closing one leaves the others untouched.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterator, Sequence

from ..types import ExitReason, Fill, Side

__all__ = ["Position", "ClosedTrade", "Portfolio", "PortfolioView"]

_ids = itertools.count(1)


@dataclass(slots=True)
class Position:
    id: int
    side: Side
    size: float
    entry_ts: datetime
    entry_price: float
    entry_bar: int
    entry_spread: float
    entry_slippage: float
    stop_price: float | None = None
    target_price: float | None = None
    max_bars: int | None = None
    tag: str = ""
    swap_paid: float = 0.0
    #: Rollover dates already charged, so a re-entrant call cannot double-charge.
    swaps_charged: set[date] = field(default_factory=set)

    def bars_held(self, current_bar: int) -> int:
        return current_bar - self.entry_bar

    def unrealised(self, mark: float) -> float:
        return self.side.sign * (mark - self.entry_price) * self.size

    def time_expired(self, current_bar: int) -> bool:
        return self.max_bars is not None and self.bars_held(current_bar) >= self.max_bars


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    id: int
    side: Side
    size: float
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    entry_bar: int
    exit_bar: int
    reason: ExitReason
    gross_pnl: float
    spread_cost: float
    slippage_cost: float
    swap_cost: float
    net_pnl: float
    entry_spread: float
    exit_spread: float
    session: str
    tag: str = ""

    @property
    def bars_held(self) -> int:
        return self.exit_bar - self.entry_bar

    @property
    def cost_total(self) -> float:
        return self.spread_cost + self.slippage_cost + self.swap_cost


class PortfolioView:
    """Read-only surface handed to a strategy. Cannot mutate anything."""

    __slots__ = ("_p",)

    def __init__(self, portfolio: "Portfolio") -> None:
        self._p = portfolio

    @property
    def equity(self) -> float:
        return self._p.equity

    @property
    def cash(self) -> float:
        return self._p.cash

    @property
    def free_margin(self) -> float:
        return self._p.free_margin

    @property
    def used_margin(self) -> float:
        return self._p.used_margin

    def open_positions(self) -> tuple[Position, ...]:
        # copies, so a strategy cannot mutate live state
        return tuple(
            Position(
                p.id, p.side, p.size, p.entry_ts, p.entry_price, p.entry_bar,
                p.entry_spread, p.entry_slippage, p.stop_price, p.target_price,
                p.max_bars, p.tag, p.swap_paid, set(p.swaps_charged),
            )
            for p in self._p.positions
        )

    def __len__(self) -> int:
        return len(self._p.positions)


class Portfolio:
    """Open positions, realised cash and margin accounting."""

    def __init__(self, starting_cash: float, margin_per_unit: float) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if margin_per_unit < 0:
            raise ValueError("margin_per_unit must be non-negative")
        self.cash = float(starting_cash)
        self.starting_cash = float(starting_cash)
        self._margin_per_unit = float(margin_per_unit)
        self.positions: list[Position] = []
        self.closed: list[ClosedTrade] = []
        self._mark: float = 0.0

    # ---- accounting --------------------------------------------------

    @property
    def used_margin(self) -> float:
        return sum(p.size for p in self.positions) * self._margin_per_unit

    @property
    def unrealised(self) -> float:
        if not self.positions:
            return 0.0
        return sum(p.unrealised(self._mark) for p in self.positions)

    @property
    def equity(self) -> float:
        return self.cash + self.unrealised

    @property
    def free_margin(self) -> float:
        return self.equity - self.used_margin

    def mark_to_market(self, mid: float) -> None:
        self._mark = mid

    def can_open(self, size: float) -> bool:
        return self.free_margin >= size * self._margin_per_unit

    # ---- lifecycle ---------------------------------------------------

    def open(self, fill: Fill, *, stop_price: float | None,
             target_price: float | None, max_bars: int | None,
             tag: str = "") -> Position:
        pos = Position(
            id=next(_ids), side=fill.side, size=fill.size, entry_ts=fill.ts,
            entry_price=fill.price, entry_bar=fill.bar_index,
            entry_spread=fill.spread, entry_slippage=fill.slippage,
            stop_price=stop_price, target_price=target_price,
            max_bars=max_bars, tag=tag,
        )
        self.positions.append(pos)
        return pos

    def close(self, pos: Position, *, exit_ts: datetime, exit_price: float,
              exit_bar: int, exit_spread: float, exit_slippage: float,
              reason: ExitReason, session: str) -> ClosedTrade:
        gross = pos.side.sign * (exit_price - pos.entry_price) * pos.size
        # Spread cost is the half-spread paid on each side, per unit, times size.
        spread_cost = (pos.entry_spread / 2.0 + exit_spread / 2.0) * pos.size
        slip_cost = (pos.entry_slippage + exit_slippage) * pos.size
        net = gross - pos.swap_paid
        trade = ClosedTrade(
            id=pos.id, side=pos.side, size=pos.size,
            entry_ts=pos.entry_ts, exit_ts=exit_ts,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_bar=pos.entry_bar, exit_bar=exit_bar, reason=reason,
            gross_pnl=gross, spread_cost=spread_cost, slippage_cost=slip_cost,
            swap_cost=pos.swap_paid, net_pnl=net,
            entry_spread=pos.entry_spread, exit_spread=exit_spread,
            session=session, tag=pos.tag,
        )
        self.cash += net
        self.positions.remove(pos)
        self.closed.append(trade)
        return trade

    def view(self) -> PortfolioView:
        return PortfolioView(self)
