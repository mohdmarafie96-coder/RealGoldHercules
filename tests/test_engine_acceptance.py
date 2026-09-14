"""Acceptance tests for the backtest engine.

The calibration test is the loud one. It does not compare against a hardcoded
dollar figure: spread ran from 0.32 to 1.03 USD/oz over the history, so any
constant would be a window artifact. It computes its reference per fill from
the bars actually traded on.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from xau_backtest.data.source import COLUMNS, TableSource
from xau_backtest.engine.broker import Broker, SlippageModel, SwapModel
from xau_backtest.engine.intrabar import PessimisticResolver, Resolution
from xau_backtest.engine.loop import run_backtest
from xau_backtest.engine.position import Portfolio
from xau_backtest.engine.strategy import RandomStrategy
from xau_backtest.types import ExitReason, Order, Quotes, Side
from xau_data.config import load_config
from xau_data.transform.sessions import SessionCalendar

UTC = dt.timezone.utc
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO / "configs/config.yaml", REPO / "configs/sessions.yaml")


@pytest.fixture(scope="module")
def cal(cfg):
    return SessionCalendar(cfg.sessions)


def synth_bars(n: int, seed: int = 0, spread: float = 0.40,
               start: dt.datetime | None = None, step_min: int = 15,
               spread_cv: float = 0.25) -> pa.Table:
    """Driftless random walk. Expectancy before costs is zero by construction,
    which is what makes the calibration reference exact.

    Spread VARIES bar to bar. That is deliberate: with a constant spread the
    calibration test cannot tell a per-fill reference from a hardcoded one, so
    it would pass against exactly the bug it exists to catch. Real spread
    varied by a factor of three across the history.
    """
    rng = np.random.default_rng(seed)
    start = start or dt.datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    mid = 2300.0 + np.cumsum(rng.normal(0, 1.0, n))
    spreads = np.clip(rng.normal(spread, spread * spread_cv, n), spread * 0.25, None)
    half = spreads / 2.0
    wig = np.abs(rng.normal(1.0, 0.3, n))
    ts = [start + dt.timedelta(minutes=step_min * i) for i in range(n)]
    cols = {
        "ts_open": pa.array(ts, type=pa.timestamp("us", tz="UTC")),
        "ts_end_exclusive": pa.array(
            [t + dt.timedelta(minutes=step_min) for t in ts],
            type=pa.timestamp("us", tz="UTC")),
        "bid_open": mid - half, "bid_high": mid + wig - half,
        "bid_low": mid - wig - half, "bid_close": mid - half,
        "ask_open": mid + half, "ask_high": mid + wig + half,
        "ask_low": mid - wig + half, "ask_close": mid + half,
        "mid_open": mid, "mid_close": mid,
        "spread_open": spreads, "spread_close": spreads,
        "bid_volume": np.ones(n), "ask_volume": np.ones(n),
    }
    return pa.table({k: (v if isinstance(v, pa.Array) else pa.array(v, type=pa.float64()))
                     for k, v in cols.items()})


def _run(src, cal, *, seed: int, zero: bool, slip: float, stops: bool = False,
         p: float = 0.06, max_bars: int = 32):
    return run_backtest(
        src, RandomStrategy(np.random.default_rng(seed), entry_probability=p,
                            max_bars=max_bars, use_stops=stops),
        calendar=cal, period_seconds=900,
        slippage=SlippageModel(per_side=0.0 if zero else slip),
        swap=SwapModel(), columns=COLUMNS, zero_cost=zero)


# =========================================================================
# CALIBRATION — the loud one
# =========================================================================


def test_calibration_paired_differential_equals_per_fill_cost(cfg, cal, capsys):
    """Random entries must lose exactly the cost, no more and no less."""
    slip = cfg.costs.slippage_per_side
    # sized so the precision guard below has real power, not to make it pass:
    # ~1,000 paired trades puts the CI half-width well inside 20% of cost.
    src = TableSource(synth_bars(30_000, seed=3, spread=0.40))

    on = _run(src, cal, seed=11, zero=False, slip=slip, p=0.10, max_bars=24)
    off = _run(src, cal, seed=11, zero=True, slip=slip, p=0.10, max_bars=24)

    assert len(on.trades) == len(off.trades) > 500, (
        "pairing broke: identical seeds must produce identical trade counts "
        "when exits are time-based"
    )
    paid = np.array([t.net_pnl for t in on.trades]) - \
        np.array([t.net_pnl for t in off.trades])
    measured = -paid.mean()

    # reference computed from the bars actually filled on, never hardcoded
    reference = float(np.mean([
        (t.entry_spread + t.exit_spread) / 2.0 + 2.0 * slip for t in on.trades
    ]))

    n = len(paid)
    se = paid.std(ddof=1) / np.sqrt(n)
    lo, hi = measured - 1.96 * se, measured + 1.96 * se
    half = 1.96 * se
    price = float(np.mean([t.entry_price for t in on.trades]))

    print("\n" + "=" * 66)
    print("CALIBRATION: RandomStrategy must lose exactly the cost")
    print("=" * 66)
    print(f"  trades                {n}")
    print(f"  measured cost/trade   {measured:.5f} USD/oz")
    print(f"  per-fill reference    {reference:.5f} USD/oz")
    print(f"  95% CI                [{lo:.5f}, {hi:.5f}]  half-width {half:.5f}")
    print(f"  as bps of price       {measured / price * 1e4:.3f} bps")
    print(f"  contains reference    {lo <= reference <= hi}")
    print("=" * 66)

    # 1. sharp: the interval must contain the per-fill reference.
    # A tiny absolute floor absorbs float noise when the interval is very tight.
    tol = max(1e-9, 0.0)
    assert lo - tol <= reference <= hi + tol, (
        f"measured cost {measured:.5f} excludes reference {reference:.5f}"
    )
    # 2. precision guard: without this, a wide interval passes vacuously
    assert half <= 0.20 * reference, (
        f"interval half-width {half:.5f} exceeds 20% of {reference:.5f}; "
        "the test has no power"
    )
    # 3. regime sanity in BPS, not dollars, so it holds at 1700 and at 5000
    assert 0.5 <= measured / price * 1e4 <= 8.0


def test_calibration_fails_loudly_if_costs_are_removed(cfg, cal):
    """Zero the cost model and the calibration must break. That is its job."""
    src = TableSource(synth_bars(4000, seed=5, spread=0.40))
    broken_on = _run(src, cal, seed=11, zero=True, slip=0.0)
    broken_off = _run(src, cal, seed=11, zero=True, slip=0.0)
    paid = np.array([t.net_pnl for t in broken_on.trades]) - \
        np.array([t.net_pnl for t in broken_off.trades])
    assert abs(paid.mean()) < 1e-12
    reference = 0.40 / 1.0 + 2 * cfg.costs.slippage_per_side
    assert abs(-paid.mean() - reference) > 0.10, (
        "a zero-cost engine must not match the real cost reference"
    )


# =========================================================================
# the rest of the acceptance list
# =========================================================================


def test_no_trade_opens_or_holds_through_rollover_without_a_swap(cfg, cal):
    src = TableSource(synth_bars(6000, seed=9, spread=0.40))
    res = run_backtest(
        src, RandomStrategy(np.random.default_rng(4), entry_probability=0.05,
                            max_bars=200, use_stops=False),
        calendar=cal, period_seconds=900,
        slippage=SlippageModel(per_side=cfg.costs.slippage_per_side),
        swap=SwapModel(long_per_unit=-0.5, short_per_unit=-0.3),
        columns=COLUMNS)
    assert res.trades
    for t in res.trades:
        crossed = cal.rollovers_crossed(t.entry_ts, t.exit_ts)
        if crossed:
            assert t.swap_cost != 0.0, (
                f"trade {t.id} spans {len(crossed)} rollover(s) with no swap"
            )


def test_swap_charged_exactly_once_per_rollover(cfg, cal):
    pf = Portfolio(100_000.0, 0.0)
    broker = Broker(pf, SlippageModel(per_side=0.0),
                    SwapModel(long_per_unit=-1.0, short_per_unit=-1.0))
    from xau_backtest.types import Fill
    ts = dt.datetime(2024, 6, 4, 12, tzinfo=UTC)
    pos = pf.open(Fill(ts, Side.LONG, 2.0, 2300.0, 0.4, 0.0, 0),
                  stop_price=None, target_price=None, max_bars=None)
    rolls = cal.rollovers_crossed(ts, ts + dt.timedelta(days=3))
    assert len(rolls) == 3
    first = broker.charge_swaps(rolls, [pos])
    assert len(first) == 3
    assert pos.swap_paid == pytest.approx(3 * 1.0 * 2.0)
    # idempotent: charging the same rollovers again must do nothing
    again = broker.charge_swaps(rolls, [pos])
    assert again == []
    assert pos.swap_paid == pytest.approx(3 * 1.0 * 2.0)


def test_gapped_stop_fills_worse_than_the_stop_price(cfg, cal):
    pf = Portfolio(100_000.0, 0.0)
    broker = Broker(pf, SlippageModel(per_side=0.02), SwapModel())
    from xau_backtest.types import Fill
    ts = dt.datetime(2024, 6, 4, 12, tzinfo=UTC)
    pos = pf.open(Fill(ts, Side.LONG, 1.0, 2300.0, 0.4, 0.0, 0),
                  stop_price=2295.0, target_price=2310.0, max_bars=None)
    # bar OPENS below the stop: a gap through
    gapped = Quotes(ts, ts, bid_open=2288.0, bid_high=2290.0, bid_low=2285.0,
                    bid_close=2289.0, ask_open=2288.4, ask_high=2290.4,
                    ask_low=2285.4, ask_close=2289.4)
    price, slip = broker.exit_fill_price(pos, gapped, 2295.0, ExitReason.STOP)
    assert price < 2295.0, "a gapped stop must not fill at the stop price"
    assert price == pytest.approx(2288.0 - 0.02)

    # and a normal touch fills AT the stop, less slippage
    touched = Quotes(ts, ts, bid_open=2299.0, bid_high=2300.0, bid_low=2294.0,
                     bid_close=2296.0, ask_open=2299.4, ask_high=2300.4,
                     ask_low=2294.4, ask_close=2296.4)
    price2, _ = broker.exit_fill_price(pos, touched, 2295.0, ExitReason.STOP)
    assert price2 == pytest.approx(2295.0 - 0.02)


def test_short_gap_through_stop_fills_worse_too(cfg, cal):
    pf = Portfolio(100_000.0, 0.0)
    broker = Broker(pf, SlippageModel(per_side=0.02), SwapModel())
    from xau_backtest.types import Fill
    ts = dt.datetime(2024, 6, 4, 12, tzinfo=UTC)
    pos = pf.open(Fill(ts, Side.SHORT, 1.0, 2300.0, 0.4, 0.0, 0),
                  stop_price=2305.0, target_price=2290.0, max_bars=None)
    gapped = Quotes(ts, ts, bid_open=2311.6, bid_high=2315.0, bid_low=2311.0,
                    bid_close=2313.0, ask_open=2312.0, ask_high=2315.4,
                    ask_low=2311.4, ask_close=2313.4)
    price, _ = broker.exit_fill_price(pos, gapped, 2305.0, ExitReason.STOP)
    assert price > 2305.0
    assert price == pytest.approx(2312.0 + 0.02)


def test_buy_fills_at_ask_and_sell_fills_at_bid(cfg, cal):
    pf = Portfolio(100_000.0, 0.0)
    broker = Broker(pf, SlippageModel(per_side=0.0), SwapModel())
    ts = dt.datetime(2024, 6, 4, 12, tzinfo=UTC)
    q = Quotes(ts, ts, bid_open=2299.8, bid_high=2301.0, bid_low=2298.0,
               bid_close=2300.0, ask_open=2300.2, ask_high=2301.4,
               ask_low=2298.4, ask_close=2300.4)
    long_fill = broker.submit(Order(Side.LONG, 1.0), q, 0)
    short_fill = broker.submit(Order(Side.SHORT, 1.0), q, 0)
    assert long_fill.price == pytest.approx(2300.2)     # ask
    assert short_fill.price == pytest.approx(2299.8)    # bid


def test_orders_exceeding_margin_are_rejected(cfg, cal):
    pf = Portfolio(1_000.0, margin_per_unit=800.0)
    broker = Broker(pf, SlippageModel(per_side=0.0), SwapModel())
    ts = dt.datetime(2024, 6, 4, 12, tzinfo=UTC)
    q = Quotes(ts, ts, 2299.8, 2301.0, 2298.0, 2300.0, 2300.2, 2301.4, 2298.4, 2300.4)
    ok = broker.submit(Order(Side.LONG, 1.0), q, 0)
    assert not isinstance(ok, type(broker.rejections[0])) if broker.rejections else True
    pf.open(ok, stop_price=None, target_price=None, max_bars=None)
    bad = broker.submit(Order(Side.LONG, 5.0), q, 0)
    from xau_backtest.types import Rejection, RejectReason
    assert isinstance(bad, Rejection)
    assert bad.reason is RejectReason.INSUFFICIENT_MARGIN


def test_pessimistic_resolver_prefers_the_stop(cfg):
    pf = Portfolio(100_000.0, 0.0)
    from xau_backtest.types import Fill
    ts = dt.datetime(2024, 6, 4, 12, tzinfo=UTC)
    pos = pf.open(Fill(ts, Side.LONG, 1.0, 2300.0, 0.4, 0.0, 0),
                  stop_price=2295.0, target_price=2305.0, max_bars=None)
    both = Quotes(ts, ts, bid_open=2300.0, bid_high=2306.0, bid_low=2294.0,
                  bid_close=2301.0, ask_open=2300.4, ask_high=2306.4,
                  ask_low=2294.4, ask_close=2301.4)
    res = PessimisticResolver().resolve(pos, both)
    assert res is not None and res.reason is ExitReason.STOP and res.ambiguous


def test_determinism_same_seed_same_blotter(cfg, cal):
    src = TableSource(synth_bars(3000, seed=2))
    a = _run(src, cal, seed=99, zero=False, slip=cfg.costs.slippage_per_side, stops=True)
    b = _run(src, cal, seed=99, zero=False, slip=cfg.costs.slippage_per_side, stops=True)
    assert len(a.trades) == len(b.trades)
    for x, y in zip(a.trades, b.trades):
        assert (x.entry_ts, x.exit_ts, x.side, x.net_pnl) == \
               (y.entry_ts, y.exit_ts, y.side, y.net_pnl)


def test_strategy_cannot_read_the_next_bar(cfg, cal):
    """A strategy reaching for bar i+1 must raise, not silently succeed."""
    from xau_backtest.engine.window import LookaheadError
    from xau_backtest.engine.strategy import Strategy

    class Peeker(Strategy):
        def on_bar(self, window, portfolio):
            window[window.index + 1]
            return ()

    src = TableSource(synth_bars(200, seed=1))
    with pytest.raises(LookaheadError):
        run_backtest(src, Peeker(), calendar=cal, period_seconds=900,
                     slippage=SlippageModel(per_side=0.0), columns=COLUMNS)


def test_performance_three_years_of_m15_under_budget(cfg, cal):
    import time
    src = TableSource(synth_bars(75_000, seed=8))
    t0 = time.perf_counter()
    res = _run(src, cal, seed=1, zero=False, slip=cfg.costs.slippage_per_side,
               stops=True, p=0.02)
    elapsed = time.perf_counter() - t0
    assert res.bars == 75_000
    print(f"\n75,000 M15 bars in {elapsed:.2f}s ({res.bars/elapsed:,.0f} bars/s), "
          f"{len(res.trades)} trades")
    assert elapsed < 60.0, f"three years took {elapsed:.1f}s, budget is 60s"
