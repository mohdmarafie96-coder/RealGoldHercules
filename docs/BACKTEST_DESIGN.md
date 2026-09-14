# XAUUSD Backtest Engine — Architecture Proposal

Status: **PROPOSAL — awaiting approval. No implementation code written.**

Two things in the brief need to change before this is buildable. Both are in
section 0. Everything after that is the proposed design.

---

## 0. Blocking issues

### 0.1 The data layer this engine is supposed to consume does not exist

The brief says "consumes the existing parquet/DuckDB data layer in this repo."
There is no such layer. The repository contains two files:

```
README.md
docs/DESIGN.md
```

Zero Python files, zero Parquet, zero DuckDB, zero YAML. `docs/DESIGN.md` is the
schema proposal from the previous task, which was never approved and never
implemented. The measured cost baseline quoted in the brief is real, but it came
from decoding raw Dukascopy `.bi5` files in a scratch directory, not from a
pipeline.

So this engine has nothing to read. Options, in the order I would pick them:

1. Build the minimum slice of the data layer first: bar schema, Parquet writer,
   tick-to-bar transform, DuckDB view. That is most of `xau_data` deliverables
   4 and 6 and is a prerequisite either way.
2. Build the engine against a `BarSource` protocol with a fixture-backed
   implementation now, and wire the Parquet implementation in when the layer
   lands. The engine stays honest and testable; only the acceptance runs wait.

I recommend doing both: define `BarSource` so the engine never imports storage
code directly, and build the bars slice of the data layer in parallel.

### 0.2 Calibration test — reference computed per fill, never hardcoded

Superseded by the full 2021-01..2026-09 history. The earlier "-0.45 within 15%"
band, and my own "flat absolute spread" finding behind it, were window artifacts
of a sample that stopped in mid-2024.

| Year | Price | Mean spread | cost_bps | Median M15 range | cost/range |
|---|---:|---:|---:|---:|---:|
| 2021 | 1799.5 | 0.3648 | 2.362 | 1.724 | 0.2464 |
| 2022 | 1801.5 | 0.3813 | 2.456 | 1.890 | 0.2335 |
| 2023 | 1943.0 | 0.3443 | 2.082 | 1.650 | 0.2450 |
| 2024 | 2388.8 | 0.3938 | 1.903 | 2.397 | 0.1893 |
| 2025 | 3442.5 | 0.6327 | 2.033 | 4.450 | 0.1557 |
| 2026 | 4566.5 | 0.7678 | 1.814 | 8.780 | 0.0943 |

Spread scales with price: 0.32 to 1.03 USD/oz across the range, while relative
cost is stable at 1.67-2.20 bps with no trend. **Any cost constant in dollars is
wrong in some regime.** A `-0.45` sanity band calibrated on 2024 would be 40%
low against 2026 spreads.

**The calibration test therefore computes its own reference.** Run
RandomStrategy twice with the identical seed, costs on and costs off. The paths,
entries and exits are identical, so the per-trade difference is the cost itself
and market noise cancels exactly rather than statistically.

The expected value is computed from the bars actually filled on:

```
reference = mean over trades of
    (spread_at_entry_fill + spread_at_exit_fill) / 2 + 2 * slippage_at(price)
```

Verified against real ticks: measured paired difference -0.4987 against a
per-fill reference of 0.4987, matching to four decimals, while the naive "mean
spread plus slippage" gave 0.4539. The gap is entirely composition: forced exit
concentrates exits into wide bars near rollover.

Four assertions, in this order:

1. **Sharp.** The paired differential equals the per-fill reference within
   tolerance.
2. **Precision guard.** The confidence interval half-width is within 20% of the
   reference. Without this, "the interval contains the target" is vacuous,
   because a wide enough interval always does, and a broken high-variance engine
   passes.
3. **Regime sanity.** Absolute expectancy sits within a band expressed in **bps
   of price**, not dollars, so it holds at 1700 and at 5000.
4. **Reporting.** Point estimate, interval, half-width, trade count and the
   computed reference print every run, pass or fail.

Sample size, measured: per-trade sigma is 12.0358 unpaired, needing 140,369
trades for 80% power at a 20% effect. The paired differential cuts sigma to
0.1030, about 10 trades for the same power. Pooling across seeds is valid
because random entry signs decorrelate overlapping trades; checked empirically
at a bootstrap-to-naive standard error ratio of 0.981.

`tests/test_no_hardcoded_costs.py` enforces this structurally, failing on any
dollar-magnitude literal near a cost-shaped identifier anywhere in the package.

### 0.2a Walk-forward must use an EXPANDING window

The volatility regime range across the history is more than 5x, from 8.6% to
44.2% annualised, **and the extremes cluster in time**. The three most volatile
months are consecutive, 2026-01, 2026-02 and 2026-03. The calmest sit in 2021
and 2023.

That clustering makes a fixed rolling window unsafe. A model trained on a
trailing window ending in 2023 would never have seen an M15 bar wider than
about 65 USD before meeting one of 287 USD in January 2026. Its volatility
scaling, stop sizing and position sizing would all be calibrated to a regime
that no longer exists.

**Therefore: walk-forward validation uses an expanding window. Never a fixed
rolling one.** Training always starts at the beginning of history and grows;
only the test fold slides forward. The cost is that early folds are short and
later folds are dominated by old data, which is the right trade when regime
shifts are this large and this abrupt.

This is a hard constraint on every downstream experiment, not a default.

### 0.3 Three smaller corrections

**Rollover is not always 21:00 UTC.** It is 17:00 New York, which is 21:00 UTC
in summer and **22:00 UTC in winter**. The zero-tick hour we measured was June
data, hence 21:00. Hardcoding 21:00 would silently hold positions through the
winter rollover and miss swap charges for five months of the year. The forced
exit must be computed from the exchange calendar, never from a UTC constant.

**Expectancy units.** "-0.45 per trade" is a price move in USD per ounce. Stated
in dollars of P&L it scales with position size. The calibration test must pin
size to one unit, or assert on cost-per-unit rather than P&L.

**Pessimistic intrabar biases the calibration.** Assuming the stop fills first
whenever stop and target both fall inside one bar adds a systematic negative
drift on top of cost. Calibration must run in tick mode, or with a fixture where
same-bar ambiguity cannot occur, or the measured number will be more negative
than -0.45 for a reason that has nothing to do with costs.

Also worth noting: since the strategy force-exits before rollover, no swap is
ever charged in a calibration run. The swap test needs its own fixture with
forced exit disabled.

---

## 1. Module layout

```
xau_backtest/
  __init__.py
  config.py              # YAML -> frozen dataclasses, validated
  types.py               # Side, OrderType, ExitReason, Money aliases
  errors.py              # LookaheadError, InsufficientMarginError, ...

  engine/
    clock.py             # BarClock: iteration, session labels, rollover flags
    window.py            # BarWindow: zero-copy, lookahead-proof view
    broker.py            # fills, slippage, swap, margin
    position.py          # Position, Portfolio, sizing
    intrabar.py          # stop/target-in-one-bar resolution
    strategy.py          # Strategy ABC + RandomStrategy, LondonBreakoutStrategy
    loop.py              # the event loop tying the above together

  data/
    source.py            # BarSource protocol
    parquet_source.py    # reads the xau_data bar tables
    fixture_source.py    # in-memory, for tests and calibration

  reporting/
    blotter.py           # trade records with reason codes
    metrics.py           # all the statistics, with session/year breakdowns
    plots.py             # equity curve, P&L distribution, monthly heatmap
    report.py            # markdown + JSON output

configs/
  backtest.yaml          # costs, slippage, sizing, sessions, intrabar mode
tests/
  ...
```

---

## 2. The lookahead barrier — CORRECTED

The first draft of this section was wrong and has been replaced. It proposed a
`BarWindow` holding `_cols`, the full column arrays, plus an `_end` index with
bounds checks. That is the full dataframe plus an index wearing a seatbelt: any
strategy could read `window._cols` and see the whole future. Bounds checking is
not the same as unreachability.

**Corrected design: an append-only revealed buffer.** `BarHistory` is owned by
the engine and grows by exactly one row per bar. `BarWindow` is handed numpy
slice views truncated to the revealed length. Future rows are not present in
any object the strategy can reach, because they have not been written yet.

Properties, each covered by a test in `tests/test_window.py`:

- Views are truncated to `n`, so out-of-range indexing has nothing to address.
  `BarWindow.__init__` rejects any column whose length is not exactly `n`.
- Positive index at or beyond the next bar raises `LookaheadError`.
- Slice bounds are checked *before* Python clamps them, so `w[0:n+1]` raises
  rather than quietly returning `n` rows.
- Returned arrays are non-writeable, so history cannot be rewritten.
- Capacity beyond the revealed length is zero-filled and never receives future
  values, so reaching through `ndarray.base` yields zeros, not data.
- `window._cols` remains reachable, because Python has no real privacy, but it
  holds only truncated views. A reachability audit test walks every data
  container reachable from the window and asserts no value from an unrevealed
  bar appears in any of them.

Measured: 75,000 bars, roughly three years of M15, append plus window
construction in 1.075 seconds, 14.3 microseconds per bar. Construction is O(1)
and shares memory with the buffer, verified by `np.shares_memory`.

### 2.1 The signature

```python
class Strategy(ABC):
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
```

### 2.2 Construction in the event loop

```python
def run(source: BarSource, strategy: Strategy, portfolio: PortfolioView) -> None:
    history = BarHistory()

    for raw in source.iter_bars():
        history.append(raw)              # 1. reveal exactly one bar
        window = history.window()        # 2. O(1) truncated read-only views
        orders = strategy.on_bar(window, portfolio)   # 3. history is never passed
        ...                              # 4. broker fills against the NEXT bar
```

Ordering is load-bearing. The bar is appended before `on_bar` is called, so the
strategy sees the bar that just closed and the window can never contain a bar
the clock has not reached. `history` itself is never handed to the strategy.

## 3. Interfaces

### 3.1 Clock and sessions

```python
@dataclass(frozen=True)
class BarEvent:
    index: int
    ts_open: datetime           # tz-aware UTC
    ts_end_exclusive: datetime
    session: Session            # ASIAN | LONDON | OVERLAP | NY | OFF
    is_rollover: bool           # bar containing 17:00 exchange-local
    is_last_before_rollover: bool
```

Sessions are configured as local windows in named timezones and converted per
bar, so DST is handled by the tz database rather than by UTC arithmetic:

```yaml
sessions:
  asian:  { tz: Asia/Tokyo,        start: "09:00", end: "15:00" }
  london: { tz: Europe/London,     start: "08:00", end: "16:30" }
  ny:     { tz: America/New_York,  start: "08:00", end: "17:00" }
  overlap: intersection_of: [london, ny]
rollover:
  tz: America/New_York
  at: "17:00"
```

### 3.2 Strategy

```python
class Strategy(ABC):
    @abstractmethod
    def on_bar(self, window: BarWindow, portfolio: PortfolioView) -> Sequence[Order]: ...
    def on_fill(self, fill: Fill) -> None: ...
    def on_exit(self, trade: Trade) -> None: ...
```

`PortfolioView` is read-only: open positions, equity, free margin. A strategy
cannot mutate portfolio state directly; it returns orders and the engine applies
them.

### 3.3 Broker and cost model

```python
class Broker:
    def submit(self, order: Order, bar: BarEvent, quotes: Quotes) -> Fill | Rejection: ...
    def mark_to_market(self, bar: BarEvent) -> None: ...
    def charge_swap(self, bar: BarEvent) -> Sequence[SwapCharge]: ...
```

Fill rules:

- Buy fills at `ask`, sell fills at `bid`, with no exception anywhere.
- Slippage is `fixed + k * volatility`, where volatility is a per-bar measure
  from the window (ATR or realised range), both terms from config.
- Spread is read per bar from the bar's own spread columns. There is no default
  spread constant in the codebase; a missing spread column is an error.

Stop trigger side matters and is easy to get wrong: a long position's stop and
target are evaluated against the **bid** series, because that is the side you
exit on. A short's are evaluated against the **ask**. Triggering on mid and
filling on bid/ask double-counts or under-counts half a spread.

Gap-through: if the bar opens beyond the stop, the fill is at the gapped open
price plus slippage, always worse than the stop, never at it.

Swap: charged once per open position per rollover bar crossed, with separate
long and short rates from config. Idempotency is enforced by recording the
rollover date on the position, so a re-entrant call cannot double-charge.

Margin: orders that would exceed free margin are rejected with a typed
`Rejection`, recorded in the blotter rather than silently dropped.

### 3.4 Intrabar resolution

```python
class IntrabarResolver(Protocol):
    def resolve(self, position: Position, bar: BarEvent,
                quotes: Quotes) -> ExitEvent | None: ...
```

Two implementations, chosen by config:

- `PessimisticResolver` (default): if both levels are inside the bar, the stop
  wins.
- `TickResolver`: replays that bar's ticks from the tick store in order and
  takes whichever level is touched first.

`TickResolver` needs tick access for the bars in question. I propose lazy
per-bar tick loading rather than holding a tick frame, so memory stays bounded.

### 3.5 Determinism

A single `numpy.random.Generator(PCG64(seed))` is created by the runner and
passed explicitly to anything that needs randomness. No module-level RNG, no
calls to `random` or `np.random` global state. Given a seed, config and data
hash, two runs produce byte-identical blotters, and the runner asserts that in a
test.

---

## 4. Metrics

Blotter row: entry and exit time, side, size, entry and exit price, entry and
exit side used, gross P&L, spread cost, slippage cost, swap cost, net P&L,
`ExitReason` in {STOP, TARGET, TIME, ROLLOVER, MARGIN, END_OF_DATA}, session at
entry, bars held.

Metrics: total return, CAGR, Sharpe, Sortino, max drawdown, drawdown duration,
profit factor, win rate, average win, average loss, expectancy per trade in both
dollars and price units, trade count, exposure time, and cost drag as a fraction
of gross P&L. Broken down by session and by year.

Two conventions I want to pin down now rather than argue about later: Sharpe and
Sortino are computed on the bar-level equity curve, annualised by the number of
tradeable bars per year rather than 252, and the risk-free rate comes from
config, defaulting to zero.

---

## 5. Test plan

| Acceptance test | How it is enforced |
|---|---|
| Calibration | Paired differential, costs on versus off, identical seed, time-based exits. Asserts the difference equals the configured spread plus slippage within tolerance. Printed as a banner in the run report. |
| Lookahead raises | Strategy reads `window[window.index + 1]`; test asserts `LookaheadError`. Repeated for out-of-range slices and for writes to a returned array. |
| Gap-through fills worse | Fixture where the bar opens through the stop; asserts fill is strictly worse than the stop price. |
| Swap once per rollover | Position held across N rollovers; asserts exactly N charges at the configured rate, and that a repeated call does not double-charge. |
| Zero-cost hand calculation | Ten-trade fixture with all costs zero; asserts P&L matches a table of hand-computed numbers exactly. |
| No unaccounted rollover hold | Asserts no position spans a rollover bar without a matching swap charge, and that forced exit fires on the last bar before rollover, with the boundary computed in exchange-local time so it is checked in both DST regimes. |

Plus: determinism under a fixed seed, and a performance test asserting three
years of M15 completes inside the budget.

---

## 6. Data volume reality check

The calibration run wants two or more years. We currently hold one month of
ticks, fetched at roughly 0.4 files per second through this session's proxy.
Three years is 26,280 hourly files, about 18 hours of downloading, and this
container is ephemeral. Two years is about 12 hours.

That is a scheduling problem worth deciding before the acceptance runs, not
during them. It is another argument for the synthetic fixture carrying the sharp
calibration test, with the real-data run as a slower confirmation.

---

## 7. Open questions

1. Approve replacing the absolute calibration assertion with the paired
   differential? This is the one I would most like a yes on.
2. Position sizing model: fixed units, fixed fractional risk per trade, or both
   behind a `Sizer` protocol? I lean to a protocol with fixed-risk as default.
3. Should multiple concurrent positions be allowed to net, or stay independent
   with independent stops? The brief says independent; confirming, because it
   affects margin accounting.
4. Is the 60 second budget for the engine loop alone, or does it include
   loading three years of bars from Parquet?
5. Swap convention: charged per rollover crossing at a flat rate, or the triple
   Wednesday convention that most brokers use?
