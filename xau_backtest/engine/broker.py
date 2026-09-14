"""Fills, slippage, swap and margin.

Fill rules, without exception:

- A buy fills at the ASK. A sell fills at the BID.
- Slippage is always adverse: it raises a buy price and lowers a sell price.
- A stop that the bar GAPPED THROUGH fills at the gapped price, never at the
  stop price. Filling a gap at the stop is the single most common way a
  backtest flatters itself.

Spread is never a constant here. It is read from the bar being filled on, which
is what makes the cost model hold across a regime where spread ran from 0.32 to
1.03 USD/oz while relative cost stayed near 2 bps.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from ..types import (
    ExitReason, Fill, Order, Quotes, RejectReason, Rejection, Side, SwapCharge,
)
from .position import Portfolio, Position

__all__ = ["Broker", "SlippageModel", "SwapModel"]


@dataclass(frozen=True, slots=True)
class SlippageModel:
    """Fixed component plus a volatility-scaled component.

    ``per_side`` is USD/oz. ``vol_coefficient`` multiplies a volatility measure
    supplied per bar, normally ATR, so slippage widens when the market is fast.
    ``bps`` is a relative floor so the model tracks the price regime.
    """

    per_side: float
    vol_coefficient: float = 0.0
    bps: float = 0.0

    def __post_init__(self) -> None:
        if self.per_side < 0 or self.vol_coefficient < 0 or self.bps < 0:
            raise ValueError("slippage parameters must be non-negative")

    def amount(self, price: float, volatility: float | None = None) -> float:
        base = max(self.per_side, price * self.bps / 10_000.0)
        if volatility is not None and self.vol_coefficient > 0:
            base += self.vol_coefficient * volatility
        return base


@dataclass(frozen=True, slots=True)
class SwapModel:
    """Per-unit financing charged once per rollover crossed, per position.

    Rates are signed: a negative rate is a cost. Long and short differ, which is
    why they are separate config values rather than one number with a sign flip.
    """

    long_per_unit: float = 0.0
    short_per_unit: float = 0.0

    def rate(self, side: Side) -> float:
        return self.long_per_unit if side is Side.LONG else self.short_per_unit


class Broker:
    """Applies orders and exits against bar quotes."""

    def __init__(
        self,
        portfolio: Portfolio,
        slippage: SlippageModel,
        swap: SwapModel,
    ) -> None:
        self._pf = portfolio
        self._slip = slippage
        self._swap = swap
        self.rejections: list[Rejection] = []
        self.swaps: list[SwapCharge] = []

    # ---- entries -----------------------------------------------------

    def submit(
        self, order: Order, quotes: Quotes, bar_index: int,
        *, volatility: float | None = None,
    ) -> Fill | Rejection:
        if order.size <= 0:
            rej = Rejection(quotes.ts_open, order, RejectReason.BAD_SIZE,
                            f"size {order.size}")
            self.rejections.append(rej)
            return rej
        if not self._pf.can_open(order.size):
            rej = Rejection(
                quotes.ts_open, order, RejectReason.INSUFFICIENT_MARGIN,
                f"free margin {self._pf.free_margin:.2f} < required "
                f"{order.size * self._pf._margin_per_unit:.2f}",
            )
            self.rejections.append(rej)
            return rej

        raw = quotes.entry_price(order.side)
        slip = self._slip.amount(raw, volatility)
        price = raw + slip if order.side is Side.LONG else raw - slip
        return Fill(
            ts=quotes.ts_open, side=order.side, size=order.size, price=price,
            spread=quotes.spread_at_entry(), slippage=slip, bar_index=bar_index,
        )

    # ---- exits -------------------------------------------------------

    def exit_fill_price(
        self,
        pos: Position,
        quotes: Quotes,
        level: float | None,
        reason: ExitReason,
        *,
        volatility: float | None = None,
    ) -> tuple[float, float]:
        """Return (fill_price, slippage) for an exit.

        For STOP and TARGET, ``level`` is the trigger price. Gap handling:
        if the bar OPENS beyond the level in the adverse direction, the fill is
        at the open, not the level. That is the whole point of modelling gaps.
        """
        exit_side = pos.side           # a long exits by selling on the bid
        open_px = quotes.exit_open(exit_side)

        if reason is ExitReason.STOP and level is not None:
            gapped = open_px <= level if pos.side is Side.LONG else open_px >= level
            raw = open_px if gapped else level
        elif reason is ExitReason.TARGET and level is not None:
            # a target gap is favourable; fill at the level, never better than
            # the market actually offered at the open
            gapped = open_px >= level if pos.side is Side.LONG else open_px <= level
            raw = open_px if gapped else level
        else:
            raw = quotes.exit_price(exit_side)

        slip = self._slip.amount(raw, volatility)
        # adverse on exit: a long sells lower, a short buys higher
        price = raw - slip if pos.side is Side.LONG else raw + slip
        return price, slip

    # ---- financing ---------------------------------------------------

    def charge_swaps(
        self, rollovers: Sequence[datetime], positions: Sequence[Position]
    ) -> list[SwapCharge]:
        """Charge each open position once per rollover crossed.

        Idempotent by construction: the rollover's date is recorded on the
        position, so a repeated call for the same rollover charges nothing.
        """
        out: list[SwapCharge] = []
        for roll in rollovers:
            key = roll.date()
            for pos in positions:
                if key in pos.swaps_charged:
                    continue
                amount = self._swap.rate(pos.side) * pos.size
                pos.swaps_charged.add(key)
                pos.swap_paid -= amount          # negative rate is a cost
                charge = SwapCharge(roll, pos.id, pos.side, pos.size, amount)
                out.append(charge)
                self.swaps.append(charge)
        return out
