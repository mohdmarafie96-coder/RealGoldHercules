"""Rank every month on three volatility measures, to pick the tick slice.

The three measures answer different questions, which is why one is not enough:

1. ``realised_vol`` — annualised standard deviation of M15 log returns. The
   conventional measure. Says how noisy the month was on average.
2. ``max_bar_range`` — the largest single M15 bar range in the month. Says how
   violent the worst moment was. A calm month with one shock scores low on (1)
   and high on this.
3. ``wide_bar_count`` — how many M15 bars had a range above three times that
   month's own median range. This is the closest proxy for the actual intrabar
   failure mode: a bar wide enough that a stop and a target can both sit inside
   it, so their ordering within the bar decides the trade.

Measure 3 is deliberately normalised against the month's *own* median, not a
global threshold, so it measures how often a month produced bars that were wide
*for that regime* rather than simply reflecting the price level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyarrow as pa

from ..transform.resample import bar_range, resample_m1

__all__ = ["MonthStats", "rank_months", "render_table"]

_BARS_PER_YEAR = 96 * 252  # M15 bars in a trading year, for annualising


@dataclass(frozen=True, slots=True)
class MonthStats:
    month: str
    bars: int
    days: int
    mid_price: float
    realised_vol: float
    max_bar_range: float
    median_bar_range: float
    wide_bar_count: int
    wide_bar_rate: float
    mean_spread: float

    @property
    def incomplete(self) -> bool:
        return self.days < 15


def _month_key(ts) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def rank_months(
    bars_m1: pa.Table, *, timeframe: str = "M15", wide_multiple: float = 3.0
) -> list[MonthStats]:
    """Compute the three measures per calendar month."""
    if bars_m1.num_rows == 0:
        return []
    m15 = resample_m1(bars_m1, timeframe=timeframe)
    ts = m15["ts_open"].to_pylist()
    rng = bar_range(m15, "bid")
    close = m15["bid_close"].to_numpy(zero_copy_only=False)
    spread = (m15["spread_close"].to_numpy(zero_copy_only=False)
              + m15["spread_open"].to_numpy(zero_copy_only=False)) / 2.0

    buckets: dict[str, list[int]] = {}
    for i, t in enumerate(ts):
        buckets.setdefault(_month_key(t), []).append(i)

    out: list[MonthStats] = []
    for key, idx in sorted(buckets.items()):
        ix = np.asarray(idx)
        c = close[ix]
        r = rng[ix]
        if len(c) < 2:
            continue
        # returns computed within the month only; the cross-month gap is skipped
        logret = np.diff(np.log(c))
        vol = float(np.std(logret, ddof=1) * np.sqrt(_BARS_PER_YEAR)) if len(logret) > 1 else float("nan")
        med = float(np.median(r))
        wide = int(np.sum(r > wide_multiple * med)) if med > 0 else 0
        out.append(MonthStats(
            month=key,
            bars=len(ix),
            days=len({ts[i].date() for i in idx}),
            mid_price=float(np.mean(c)),
            realised_vol=vol,
            max_bar_range=float(np.max(r)),
            median_bar_range=med,
            wide_bar_count=wide,
            wide_bar_rate=wide / len(ix),
            mean_spread=float(np.mean(spread[ix])),
        ))
    return out


def _rank(values: Sequence[float], descending: bool = True) -> list[int]:
    order = np.argsort(values)
    if descending:
        order = order[::-1]
    ranks = [0] * len(values)
    for pos, i in enumerate(order, 1):
        ranks[int(i)] = pos
    return ranks


def render_table(stats: Sequence[MonthStats], *, only_complete: bool = True) -> str:
    rows = [s for s in stats if not (only_complete and s.incomplete)]
    if not rows:
        return "(no complete months)"
    rv = _rank([s.realised_vol for s in rows])
    mr = _rank([s.max_bar_range for s in rows])
    wb = _rank([s.wide_bar_rate for s in rows])
    combined = [(rv[i] + mr[i] + wb[i]) / 3.0 for i in range(len(rows))]
    order = np.argsort(combined)

    L = [
        f"{'month':>8} {'days':>4} {'bars':>5} {'price':>8} {'r_vol':>7} {'rk':>3} "
        f"{'max_rng':>8} {'rk':>3} {'wide%':>6} {'rk':>3} {'spread':>7} {'mean_rk':>7}",
        "-" * 92,
    ]
    for i in order:
        s = rows[int(i)]
        L.append(
            f"{s.month:>8} {s.days:>4} {s.bars:>5} {s.mid_price:>8.1f} "
            f"{s.realised_vol:>7.3f} {rv[int(i)]:>3} "
            f"{s.max_bar_range:>8.2f} {mr[int(i)]:>3} "
            f"{s.wide_bar_rate*100:>6.2f} {wb[int(i)]:>3} "
            f"{s.mean_spread:>7.4f} {combined[int(i)]:>7.1f}"
        )
    skipped = [s for s in stats if s.incomplete]
    if skipped and only_complete:
        L.append("")
        L.append(f"excluded as incomplete (<15 trading days): "
                 + ", ".join(f"{s.month}({s.days}d)" for s in skipped))
    return "\n".join(L)
