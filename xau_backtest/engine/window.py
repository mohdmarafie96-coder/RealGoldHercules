"""Lookahead-proof bar window.

The strategy never holds a reference to future data. Bars are revealed one at a
time into an append-only buffer; the window exposes views truncated to exactly
the number of bars revealed so far. Capacity beyond the revealed length is
zero-filled and never written with future values, so even reaching through
``ndarray.base`` yields nothing but zeros.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["LookaheadError", "BarWindow", "BarHistory", "BarRow", "BAR_COLUMNS"]


class LookaheadError(IndexError):
    """Raised when a strategy attempts to read at or beyond the next bar."""


BAR_COLUMNS: Final[tuple[str, ...]] = (
    "ts_open",
    "ts_end_exclusive",
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
    "mid_open", "mid_high", "mid_low", "mid_close",
    "spread_mean", "spread_max", "spread_close",
    "tick_count",
)

_INT_COLUMNS: Final[frozenset[str]] = frozenset(
    {"ts_open", "ts_end_exclusive", "tick_count"}
)

_EPOCH: Final[datetime] = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _dtype_for(col: str) -> np.dtype:
    return np.dtype(np.int64) if col in _INT_COLUMNS else np.dtype(np.float64)


@dataclass(frozen=True, slots=True)
class BarRow:
    """A single revealed bar, materialised on demand."""

    index: int
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
    mid_open: float
    mid_high: float
    mid_low: float
    mid_close: float
    spread_mean: float
    spread_max: float
    spread_close: float
    tick_count: int


class BarWindow:
    """Read-only view over bars 0..n-1, where n-1 is the current bar.

    Construction is O(1): the column arrays handed in are numpy slice views,
    not copies. Every accessor is bounded by ``n``; nothing in this object
    references a row past the current bar.
    """

    __slots__ = ("_cols", "_n")

    def __init__(self, cols: Mapping[str, npt.NDArray], n: int) -> None:
        if n < 1:
            raise ValueError("a window must contain at least the current bar")
        for name, arr in cols.items():
            if len(arr) != n:
                raise ValueError(
                    f"column {name!r} has length {len(arr)}, expected {n}; "
                    "the window must be handed truncated views"
                )
            if arr.flags.writeable:
                raise ValueError(f"column {name!r} must be read-only")
        self._cols = dict(cols)
        self._n = n

    # ---- identity -------------------------------------------------------

    @property
    def index(self) -> int:
        """Absolute index of the current bar."""
        return self._n - 1

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return f"<BarWindow bars=0..{self.index} current={self.ts().isoformat()}>"

    # ---- scalar access --------------------------------------------------

    def value(self, col: str, lag: int = 0) -> float:
        """Value of ``col``; lag=0 is the current bar, lag=1 the previous."""
        if lag < 0:
            raise LookaheadError(
                f"negative lag {lag} would read {-lag} bar(s) into the future"
            )
        i = self._n - 1 - lag
        if i < 0:
            raise IndexError(f"lag {lag} reaches before the first bar")
        return self._cols[col][i].item()

    def ts(self, lag: int = 0) -> datetime:
        """Bar open time, tz-aware UTC."""
        return _EPOCH.fromtimestamp(
            self.value("ts_open", lag) / 1_000_000, tz=timezone.utc
        )

    # ---- series access --------------------------------------------------

    def series(self, col: str, n: int | None = None) -> npt.NDArray:
        """Read-only view of the last ``n`` values, ending at the current bar."""
        arr = self._cols[col]
        if n is None:
            return arr
        if n < 0:
            raise ValueError("n must be non-negative")
        if n > self._n:
            raise IndexError(f"requested {n} bars, only {self._n} revealed")
        return arr[self._n - n:]

    # ---- indexing -------------------------------------------------------

    def __getitem__(self, key: int | slice) -> BarRow | dict[str, npt.NDArray]:
        if isinstance(key, slice):
            return self._slice(key)
        i = key
        if i < 0:
            i += self._n
            if i < 0:
                raise IndexError(f"index {key} is before the first bar")
        elif i >= self._n:
            raise LookaheadError(
                f"index {i} is at or beyond the next bar; current bar is "
                f"{self.index}. The future is not reachable from this window."
            )
        return self._row(i)

    def _slice(self, sl: slice) -> dict[str, npt.NDArray]:
        # Python clamps out-of-range slice bounds silently. That is exactly the
        # failure mode we are preventing, so bounds are checked before clamping.
        for attr in ("start", "stop"):
            raw = getattr(sl, attr)
            if raw is None:
                continue
            bound = raw + self._n if raw < 0 else raw
            limit = self._n if attr == "stop" else self._n - 1
            if bound > limit:
                raise LookaheadError(
                    f"slice {attr}={raw} reaches past the current bar "
                    f"({self.index}); slices are not clamped"
                )
        return {name: arr[sl] for name, arr in self._cols.items()}

    def _row(self, i: int) -> BarRow:
        g = lambda c: self._cols[c][i].item()  # noqa: E731
        return BarRow(
            index=i,
            ts_open=_EPOCH.fromtimestamp(g("ts_open") / 1_000_000, tz=timezone.utc),
            ts_end_exclusive=_EPOCH.fromtimestamp(
                g("ts_end_exclusive") / 1_000_000, tz=timezone.utc
            ),
            bid_open=g("bid_open"), bid_high=g("bid_high"),
            bid_low=g("bid_low"), bid_close=g("bid_close"),
            ask_open=g("ask_open"), ask_high=g("ask_high"),
            ask_low=g("ask_low"), ask_close=g("ask_close"),
            mid_open=g("mid_open"), mid_high=g("mid_high"),
            mid_low=g("mid_low"), mid_close=g("mid_close"),
            spread_mean=g("spread_mean"), spread_max=g("spread_max"),
            spread_close=g("spread_close"), tick_count=int(g("tick_count")),
        )

    def current(self) -> BarRow:
        return self._row(self._n - 1)


class BarHistory:
    """Append-only store of revealed bars. Owned by the engine, never by a strategy.

    Future bars are not present in this object at any point: they are appended
    only when the clock reaches them.
    """

    __slots__ = ("_buf", "_n", "_cap", "_columns")

    def __init__(
        self,
        columns: Sequence[str] = BAR_COLUMNS,
        initial_capacity: int = 4096,
    ) -> None:
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be positive")
        self._columns = tuple(columns)
        self._cap = initial_capacity
        self._n = 0
        self._buf: dict[str, npt.NDArray] = {
            c: np.zeros(self._cap, dtype=_dtype_for(c)) for c in self._columns
        }

    def __len__(self) -> int:
        return self._n

    def append(self, row: Mapping[str, float | int]) -> None:
        if self._n == self._cap:
            self._grow()
        i = self._n
        for c in self._columns:
            self._buf[c][i] = row[c]
        self._n = i + 1

    def _grow(self) -> None:
        self._cap *= 2
        for c in self._columns:
            bigger = np.zeros(self._cap, dtype=_dtype_for(c))
            bigger[: self._n] = self._buf[c][: self._n]
            self._buf[c] = bigger

    def window(self) -> BarWindow:
        """O(1). Returns truncated, read-only views ending at the current bar."""
        if self._n == 0:
            raise ValueError("no bars revealed yet")
        cols: dict[str, npt.NDArray] = {}
        for c in self._columns:
            v = self._buf[c][: self._n]
            v.flags.writeable = False
            cols[c] = v
        return BarWindow(cols, self._n)
