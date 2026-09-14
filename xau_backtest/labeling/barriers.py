"""Triple-barrier labelling.

A label answers: if a position were opened at the next bar's open, which
happened first, the target, the stop, or the time limit?

Three rules keep labels honest and make them mean the same thing the engine
means:

1. **Same execution model as the engine.** Entry fills at the NEXT bar's open,
   buy at ask and sell at bid, and the exit fills on the side the position
   exits on. A label computed on mid prices would describe a trade the engine
   cannot execute.
2. **Same intrabar assumption.** When stop and target both sit inside one bar,
   the pessimistic rule applies: the stop wins. That is a biased estimator, not
   a neutral one, and the bias is recorded per label so it can be measured
   against the tick slice.
3. **Same forced exit.** The horizon is capped by the rollover, exactly as the
   engine caps it. A label may not describe holding through a rollover the
   engine would have closed.

Barriers are sized in ATR, never dollars: ATR ran 2.17 to 11.69 over the sample,
so a fixed dollar barrier would be a different trade in every regime.

Labels are forward-looking by construction; that is their job. The guard is that
nothing in the FEATURE set may see them, which is what the causality tests on
`xau_backtest.features` enforce from the other side.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pyarrow as pa

__all__ = ["BarrierConfig", "LabelReason", "label_triple_barrier", "LABEL_SCHEMA"]

F = npt.NDArray[np.float64]


class LabelReason(str, Enum):
    TARGET = "target"
    STOP = "stop"
    TIME = "time"
    ROLLOVER = "rollover"
    END_OF_DATA = "end_of_data"
    NO_ENTRY = "no_entry"


_REASON_CODE: Final[dict[LabelReason, int]] = {
    LabelReason.TARGET: 1, LabelReason.STOP: -1, LabelReason.TIME: 0,
    LabelReason.ROLLOVER: 2, LabelReason.END_OF_DATA: 3, LabelReason.NO_ENTRY: 9,
}


@dataclass(frozen=True, slots=True)
class BarrierConfig:
    """All barriers in ATR units. Never dollars."""

    target_atr: float
    stop_atr: float
    slippage_per_side: float        # REQUIRED: no default. A cost assumption
                                    # must not be able to hide as a dataclass
                                    # default, same rule as add_cost_features.
    max_bars: int = 48              # 12h at M15, the top of the holding window
    min_bars: int = 16              # 4h, the bottom
    atr_period: int = 14

    def __post_init__(self) -> None:
        if self.target_atr <= 0 or self.stop_atr <= 0:
            raise ValueError("barriers must be positive")
        if self.max_bars < self.min_bars:
            raise ValueError("max_bars must be >= min_bars")


LABEL_SCHEMA: Final[tuple[str, ...]] = (
    "label_long_ret_atr_net", "label_long_ret_atr_gross", "label_long_reason",
    "label_long_bars", "label_short_ret_atr_net", "label_short_ret_atr_gross",
    "label_short_reason", "label_short_bars", "label_ambiguous",
    "label_exit_index",
)


def _side_labels(
    n: int, is_long: bool, cfg: BarrierConfig,
    bid_open: F, bid_high: F, bid_low: F, bid_close: F,
    ask_open: F, ask_high: F, ask_low: F, ask_close: F,
    atr: F, spread: F, forced_exit_idx: npt.NDArray[np.int64],
) -> dict[str, npt.NDArray]:
    ret_net = np.full(n, np.nan)
    ret_gross = np.full(n, np.nan)
    reason = np.full(n, _REASON_CODE[LabelReason.NO_ENTRY], dtype=np.int8)
    bars = np.full(n, -1, dtype=np.int32)
    ambiguous = np.zeros(n, dtype=bool)
    exit_idx = np.full(n, -1, dtype=np.int64)
    slip = cfg.slippage_per_side

    for i in range(n - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        e = i + 1                                   # fill at the NEXT bar's open
        if e >= n:
            continue
        entry = ask_open[e] + slip if is_long else bid_open[e] - slip
        if not np.isfinite(entry):
            continue
        sign = 1.0 if is_long else -1.0
        target = entry + sign * cfg.target_atr * a
        stop = entry - sign * cfg.stop_atr * a

        limit = min(e + cfg.max_bars - 1, n - 1)
        forced = forced_exit_idx[e]
        capped_by_rollover = 0 <= forced < limit
        if capped_by_rollover:
            limit = forced

        hit = None
        for j in range(e, limit + 1):
            # a long exits on the BID, a short on the ASK
            hi = bid_high[j] if is_long else ask_high[j]
            lo = bid_low[j] if is_long else ask_low[j]
            if is_long:
                t_hit, s_hit = hi >= target, lo <= stop
            else:
                t_hit, s_hit = lo <= target, hi >= stop
            if t_hit and s_hit:
                ambiguous[i] = True
                hit = (j, LabelReason.STOP, stop)    # pessimistic
                break
            if s_hit:
                hit = (j, LabelReason.STOP, stop)
                break
            if t_hit:
                hit = (j, LabelReason.TARGET, target)
                break
        if hit is None:
            j = limit
            px = bid_close[j] if is_long else ask_close[j]
            why = (LabelReason.ROLLOVER if capped_by_rollover
                   else LabelReason.END_OF_DATA if j == n - 1 and
                   j < e + cfg.max_bars - 1 else LabelReason.TIME)
            hit = (j, why, px)

        j, why, level = hit
        raw = level
        exit_px = raw - slip if is_long else raw + slip
        # GROSS is mid-to-mid: before spread AND before slippage. Measuring it
        # from the entry-side quote would leave the spread inside gross, so
        # gross-minus-net would equal slippage alone (0.0237 ATR) instead of
        # the true round trip. Phase 7 needs the real split to tell a constant
        # edge with falling costs apart from a genuinely improving edge.
        entry_mid = (bid_open[e] + ask_open[e]) / 2.0
        half = spread[j] / 2.0
        exit_mid = raw + half if is_long else raw - half
        gross_mid = sign * (exit_mid - entry_mid)
        net = sign * (exit_px - entry)
        ret_net[i] = net / a
        ret_gross[i] = gross_mid / a
        reason[i] = _REASON_CODE[why]
        bars[i] = j - e + 1
        exit_idx[i] = j

    p = "long" if is_long else "short"
    return {
        f"label_{p}_ret_atr_net": ret_net,
        f"label_{p}_ret_atr_gross": ret_gross,
        f"label_{p}_reason": reason,
        f"label_{p}_bars": bars,
        "_ambiguous": ambiguous,
        "_exit_idx": exit_idx,
    }


def label_triple_barrier(
    bars: pa.Table, *, calendar, cfg: BarrierConfig, period_seconds: int = 900
) -> pa.Table:
    """Return ts_open plus triple-barrier labels for both sides."""
    bars = bars.sort_by([("ts_open", "ascending")])
    n = bars.num_rows
    g = lambda c: bars[c].to_numpy(zero_copy_only=False).astype(np.float64)

    from ..features.groups import _atr
    atr = _atr(g("bid_high"), g("bid_low"), g("bid_close"), cfg.atr_period)
    spread = (g("spread_open") + g("spread_close")) / 2.0
    ts = bars["ts_open"].to_pylist()

    # index of the last bar before each bar's next rollover, precomputed once
    forced = np.full(n, -1, dtype=np.int64)
    deadlines = [calendar.last_bar_open_before_rollover(t, period_seconds) for t in ts]
    j = 0
    for i in range(n):
        d = deadlines[i]
        j = max(j, i)
        while j + 1 < n and ts[j + 1] <= d:
            j += 1
        forced[i] = j

    common = dict(bid_open=g("bid_open"), bid_high=g("bid_high"),
                  bid_low=g("bid_low"), bid_close=g("bid_close"),
                  ask_open=g("ask_open"), ask_high=g("ask_high"),
                  ask_low=g("ask_low"), ask_close=g("ask_close"),
                  atr=atr, spread=spread, forced_exit_idx=forced)
    lo = _side_labels(n, True, cfg, **common)
    sh = _side_labels(n, False, cfg, **common)

    amb = lo.pop("_ambiguous") | sh.pop("_ambiguous")
    ex = np.maximum(lo.pop("_exit_idx"), sh.pop("_exit_idx"))

    cols: dict[str, pa.Array] = {"ts_open": bars["ts_open"]}
    for d in (lo, sh):
        for k, v in d.items():
            if v.dtype.kind == "f":
                cols[k] = pa.array(v, type=pa.float64(), mask=~np.isfinite(v))
            else:
                cols[k] = pa.array(v.astype(np.int32), type=pa.int32())
    cols["label_ambiguous"] = pa.array(amb, type=pa.bool_())
    cols["label_exit_index"] = pa.array(ex, type=pa.int64())
    return pa.table(cols)
