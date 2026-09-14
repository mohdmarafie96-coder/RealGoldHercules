"""Core value types for the backtest engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Final

__all__ = [
    "Side", "OrderKind", "ExitReason", "RejectReason",
    "Order", "Fill", "Rejection", "Quotes", "SwapCharge",
]


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1

    @property
    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


class OrderKind(str, Enum):
    MARKET = "market"


class ExitReason(str, Enum):
    STOP = "stop"
    TARGET = "target"
    TIME = "time"
    ROLLOVER = "rollover"
    MARGIN = "margin"
    END_OF_DATA = "end_of_data"


class RejectReason(str, Enum):
    INSUFFICIENT_MARGIN = "insufficient_margin"
    BAD_SIZE = "bad_size"
    NO_QUOTES = "no_quotes"


@dataclass(frozen=True, slots=True)
class Quotes:
    """The prices available for filling on one bar.

    A long exits on the BID and a short exits on the ASK, so stop and target
    triggers are evaluated against the side the position would exit on. Using
    mid for the trigger and bid/ask for the fill mis-counts half a spread.
    """

    ts_open: datetime
    ts_end_exclusive: datetime
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float
    ask_high: float
    ask_low: float
    ask_close: float

    @property
    def mid_open(self) -> float:
        return (self.bid_open + self.ask_open) / 2.0

    @property
    def mid_close(self) -> float:
        return (self.bid_close + self.ask_close) / 2.0

    @property
    def spread_open(self) -> float:
        return self.ask_open - self.bid_open

    @property
    def spread_close(self) -> float:
        return self.ask_close - self.bid_close

    def entry_price(self, side: Side) -> float:
        """Buy at ask, sell at bid. No exception, anywhere."""
        return self.ask_open if side is Side.LONG else self.bid_open

    def exit_price(self, side: Side) -> float:
        return self.bid_close if side is Side.LONG else self.ask_close

    def exit_high(self, side: Side) -> float:
        return self.bid_high if side is Side.LONG else self.ask_high

    def exit_low(self, side: Side) -> float:
        return self.bid_low if side is Side.LONG else self.ask_low

    def exit_open(self, side: Side) -> float:
        return self.bid_open if side is Side.LONG else self.ask_open

    def spread_at_entry(self) -> float:
        return self.ask_open - self.bid_open


@dataclass(frozen=True, slots=True)
class Order:
    side: Side
    size: float
    stop_distance: float | None = None
    target_distance: float | None = None
    max_bars: int | None = None
    tag: str = ""
    kind: OrderKind = OrderKind.MARKET


@dataclass(frozen=True, slots=True)
class Fill:
    ts: datetime
    side: Side
    size: float
    price: float
    spread: float
    slippage: float
    bar_index: int


@dataclass(frozen=True, slots=True)
class Rejection:
    ts: datetime
    order: Order
    reason: RejectReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SwapCharge:
    ts: datetime
    position_id: int
    side: Side
    size: float
    amount: float
