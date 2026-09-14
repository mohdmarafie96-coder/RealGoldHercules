"""Feature-ready dataset: bars plus derived columns.

Every column here is CAUSAL. A value at bar ``i`` is computed only from bars
``0..i`` inclusive. Nothing is centred, nothing is smoothed with a future
window, and nothing is back-filled. Warm-up rows where a feature is not yet
defined are NULL, not forward-filled and not dropped.

The headline column is ``cost_to_atr_ratio``: the round-trip cost of a trade
divided by current volatility, so a model can see how expensive trading was at
any point in time. Cost that is constant in dollars per ounce is not constant
in economic terms; this column is what makes that visible.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from ..errors import SchemaError
from ..storage.schemas import BARS, TS

__all__ = [
    "true_range", "atr_wilder", "add_cost_features", "FEATURES",
    "DEFAULT_ATR_PERIOD",
]

DEFAULT_ATR_PERIOD: Final[int] = 14
#: Deliberately absent: there is no default slippage. Callers pass it from
#: config so a cost assumption can never hide as a module constant.

#: Columns added on top of BARS.
_ADDED: Final[tuple[tuple[str, pa.DataType], ...]] = (
    ("atr", pa.float64()),
    ("round_trip_cost", pa.float64()),
    ("cost_to_atr_ratio", pa.float64()),
    ("cost_bps", pa.float64()),
)

FEATURES: Final[pa.Schema] = pa.schema(
    list(BARS) + [pa.field(n, t, nullable=True) for n, t in _ADDED],
    metadata={
        b"causality": (
            b"Every derived column at bar i uses only bars 0..i. Warm-up rows "
            b"are NULL. Nothing is forward-filled or back-filled."
        ),
        b"round_trip_cost": (
            b"spread_mean + 2 * slippage_per_side, in USD per ounce. One spread "
            b"per round trip: buy at ask, sell at bid."
        ),
        b"cost_to_atr_ratio": (
            b"round_trip_cost / atr. Dimensionless. How much of a typical bar's "
            b"range is consumed by transacting once."
        ),
        b"atr": b"Wilder ATR on MID prices, causal, NULL until warmed up.",
    },
)


def true_range(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """True range. TR[0] is high-low; thereafter it uses the PREVIOUS close."""
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low and close must be the same length")
    n = len(high)
    tr = np.empty(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    if n > 1:
        prev = close[:-1]
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - prev),
            np.abs(low[1:] - prev),
        ])
    return tr


def atr_wilder(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period: int = DEFAULT_ATR_PERIOD,
) -> npt.NDArray[np.float64]:
    """Wilder's ATR. Causal by construction; warm-up rows are NaN.

    The seed is the simple mean of the first ``period`` true ranges, placed at
    index ``period-1``. Every later value depends only on earlier ones.
    """
    if period < 1:
        raise ValueError("period must be positive")
    tr = true_range(high, low, close)
    n = len(tr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    seed = float(np.mean(tr[:period]))
    out[period - 1] = seed
    alpha = 1.0 / period
    prev = seed
    for i in range(period, n):
        prev = prev + alpha * (tr[i] - prev)
        out[i] = prev
    return out


def add_cost_features(
    bars: pa.Table,
    *,
    slippage_per_side: float,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> pa.Table:
    """Return ``bars`` with atr, round_trip_cost, cost_to_atr_ratio and cost_bps.

    Input must be sorted by ``ts_open``. The caller is responsible for not
    concatenating across a gap it does not want the ATR to smooth over; ATR is
    computed on the table as given.
    """
    for col in ("mid_high", "mid_low", "mid_close", "spread_mean", "ts_open"):
        if col not in bars.column_names:
            raise SchemaError(f"bars table missing column {col!r}")
    if slippage_per_side < 0:
        raise ValueError("slippage_per_side must be non-negative")

    bars = bars.sort_by([("ts_open", "ascending")])
    n = bars.num_rows
    if n == 0:
        return FEATURES.empty_table()

    hi = bars["mid_high"].to_numpy(zero_copy_only=False)
    lo = bars["mid_low"].to_numpy(zero_copy_only=False)
    cl = bars["mid_close"].to_numpy(zero_copy_only=False)
    sp = bars["spread_mean"].to_numpy(zero_copy_only=False)

    atr = atr_wilder(hi, lo, cl, atr_period)
    cost = sp + 2.0 * slippage_per_side

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(atr > 0, cost / atr, np.nan)
        bps = np.where(cl > 0, cost / cl * 10_000.0, np.nan)

    def nullable(a: npt.NDArray[np.float64]) -> pa.Array:
        return pa.array(a, type=pa.float64(), mask=np.isnan(a))

    out = bars
    for name, arr in (("atr", atr), ("round_trip_cost", cost),
                      ("cost_to_atr_ratio", ratio), ("cost_bps", bps)):
        out = out.append_column(pa.field(name, pa.float64(), nullable=True),
                                nullable(arr))
    return out.select(FEATURES.names).cast(FEATURES)
