"""The event loop.

Ordering is the whole design:

1. Reveal bar i into history. Nothing later exists in any reachable object.
2. Build the window, O(1), truncated read-only views ending at bar i.
3. Manage OPEN positions against bar i: swap on rollover crossings, forced exit
   before rollover, stop/target via the intrabar resolver, time exit.
4. Ask the strategy for orders, given only bars 0..i.
5. Fill those orders on bar i+1's OPEN. A signal computed from a bar that has
   closed cannot be executed inside that same bar.

Step 5 is what stops the classic same-bar fill leak. The strategy decides on
the close of bar i; the market it actually trades is bar i+1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from ..types import ExitReason, Fill, Order, Quotes, Rejection, Side
from .broker import Broker, SlippageModel, SwapModel
from .intrabar import IntrabarResolver, PessimisticResolver, Resolution
from .position import ClosedTrade, Portfolio, Position
from .strategy import Strategy
from .window import BarHistory, BarWindow

__all__ = ["BacktestResult", "run_backtest", "BarSource"]


class BarSource(Protocol):
    def iter_bars(self) -> Iterable[Mapping[str, float | int]]: ...


@dataclass(slots=True)
class BacktestResult:
    trades: list[ClosedTrade] = field(default_factory=list)
    equity_ts: list[datetime] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    swaps_charged: int = 0
    ambiguous_bars: int = 0
    bars: int = 0
    starting_cash: float = 0.0

    @property
    def final_equity(self) -> float:
        return self.equity[-1] if self.equity else self.starting_cash


def _quotes_from(window: BarWindow, lag: int = 0, *, zero_spread: bool = False) -> Quotes:
    g = lambda c: window.value(c, lag)  # noqa: E731
    if zero_spread:
        # Collapse both sides onto mid. Used ONLY by the calibration test, so
        # the paired run has identical price paths with the spread removed and
        # the per-trade difference is exactly the round-trip cost.
        mo = (g("bid_open") + g("ask_open")) / 2.0
        mc = (g("bid_close") + g("ask_close")) / 2.0
        mh = (g("bid_high") + g("ask_high")) / 2.0
        ml = (g("bid_low") + g("ask_low")) / 2.0
        return Quotes(
            ts_open=window.ts(lag), ts_end_exclusive=window.ts(lag),
            bid_open=mo, bid_high=mh, bid_low=ml, bid_close=mc,
            ask_open=mo, ask_high=mh, ask_low=ml, ask_close=mc,
        )
    return Quotes(
        ts_open=window.ts(lag),
        ts_end_exclusive=window.ts(lag),
        bid_open=g("bid_open"), bid_high=g("bid_high"),
        bid_low=g("bid_low"), bid_close=g("bid_close"),
        ask_open=g("ask_open"), ask_high=g("ask_high"),
        ask_low=g("ask_low"), ask_close=g("ask_close"),
    )


def _causal_atr(window: BarWindow, period: int = 14) -> float | None:
    """Wilder-style ATR over the last `period` closed bars. Causal by construction."""
    n = period + 1
    if len(window) < n:
        return None
    hi = window.series("bid_high", n)
    lo = window.series("bid_low", n)
    cl = window.series("bid_close", n)
    tr = np.maximum.reduce([hi[1:] - lo[1:],
                            np.abs(hi[1:] - cl[:-1]),
                            np.abs(lo[1:] - cl[:-1])])
    v = float(np.mean(tr))
    return v if v > 0 else None


def run_backtest(
    source: BarSource,
    strategy: Strategy,
    *,
    calendar,
    period_seconds: int,
    starting_cash: float = 100_000.0,
    margin_per_unit: float = 0.0,
    slippage: SlippageModel,
    swap: SwapModel | None = None,
    resolver: IntrabarResolver | None = None,
    atr_period: int = 14,
    columns: Sequence[str] | None = None,
    zero_cost: bool = False,
) -> BacktestResult:
    """Run one backtest. Deterministic given the strategy's seed and this input."""
    swap = swap or SwapModel()
    resolver = resolver or PessimisticResolver()
    pf = Portfolio(starting_cash, margin_per_unit)
    broker = Broker(pf, slippage, swap)
    history = BarHistory(columns=tuple(columns)) if columns else BarHistory()
    result = BacktestResult(starting_cash=starting_cash)

    pending: list[Order] = []
    prev_ts: datetime | None = None

    for raw in source.iter_bars():
        history.append(raw)                                  # 1. reveal bar i
        window = history.window(calendar)                    # 2. O(1) view
        i = window.index
        result.bars += 1
        q = _quotes_from(window, zero_spread=zero_cost)
        pf.mark_to_market(q.mid_close)

        # --- 5 (deferred from last bar): fill orders on THIS bar's open ---
        for order in pending:
            out = broker.submit(order, q, i)
            if isinstance(out, Rejection):
                result.rejections.append(out)
                continue
            sign = out.side.sign
            stop = (out.price - sign * order.stop_distance
                    if order.stop_distance else None)
            target = (out.price + sign * order.target_distance
                      if order.target_distance else None)
            pf.open(out, stop_price=stop, target_price=target,
                    max_bars=order.max_bars, tag=order.tag,
                    entry_atr=_causal_atr(window, atr_period))
            strategy.on_fill(out)
        pending = []

        # --- 3. manage open positions against bar i ---
        if pf.positions:
            if prev_ts is not None:
                crossed = calendar.rollovers_crossed(prev_ts, q.ts_open)
                if crossed:
                    charges = broker.charge_swaps(crossed, list(pf.positions))
                    result.swaps_charged += len(charges)

            deadline = calendar.last_bar_open_before_rollover(q.ts_open, period_seconds)
            for pos in list(pf.positions):
                res: Resolution | None = resolver.resolve(pos, q)
                reason: ExitReason | None = None
                level: float | None = None
                if res is not None:
                    reason, level = res.reason, res.level
                    if res.ambiguous:
                        result.ambiguous_bars += 1
                elif q.ts_open >= deadline:
                    reason = ExitReason.ROLLOVER
                elif pos.time_expired(i):
                    reason = ExitReason.TIME
                if reason is None:
                    continue
                price, slip = broker.exit_fill_price(pos, q, level, reason)
                pf.close(pos, exit_ts=q.ts_open, exit_price=price, exit_bar=i,
                         exit_spread=q.spread_close, exit_slippage=slip,
                         reason=reason, session=window.session())
                strategy.on_exit(pf.closed[-1])

        # --- 4. strategy sees only bars 0..i, decides for bar i+1 ---
        pending = list(strategy.on_bar(window, pf.view()))

        pf.mark_to_market(q.mid_close)
        result.equity_ts.append(q.ts_open)
        result.equity.append(pf.equity)
        prev_ts = q.ts_open

    # close anything still open at the end of data
    if pf.positions:
        q = _quotes_from(window, zero_spread=zero_cost)
        for pos in list(pf.positions):
            price, slip = broker.exit_fill_price(pos, q, None, ExitReason.END_OF_DATA)
            pf.close(pos, exit_ts=q.ts_open, exit_price=price,
                     exit_bar=window.index, exit_spread=q.spread_close,
                     exit_slippage=slip, reason=ExitReason.END_OF_DATA,
                     session=window.session())

    result.trades = list(pf.closed)
    result.rejections.extend(broker.rejections)
    return result
