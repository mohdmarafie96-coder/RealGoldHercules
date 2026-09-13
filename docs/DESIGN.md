# XAUUSD Data Layer — Schema & Layout Proposal

Status: **PROPOSAL — awaiting approval. No implementation code written yet.**

This document specifies the directory layout, on-disk schemas, time semantics,
and configuration surface for the `xau_data` package. It is the contract that
implementation and tests will be written against.

Design priority is stated once and applied everywhere: **correctness over
speed, and lookahead bias is a build-breaking defect, not a warning.**

---

## 1. Time semantics (read this first)

Every rule below is enforced in code, not by convention.

### 1.1 Universal UTC invariant

- Every timestamp in memory is a timezone-aware instant in UTC.
- Every timestamp on disk is Parquet `timestamp[us, tz=UTC]`.
- Naive datetimes are rejected at every public boundary by a single helper,
  `xau_data.timeutils.require_utc()`. There is no code path that infers a
  timezone from the ambient system clock.
- Microsecond resolution throughout. Dukascopy ticks are millisecond-resolution,
  so microseconds are lossless and leave headroom for tie-breaking.
- A repository-wide test walks every Parquet schema and asserts each timestamp
  field carries `tz="UTC"`, so a new column cannot silently be added naive.

### 1.2 Two clocks, deliberately separated

The system uses exactly two time references and never mixes them:

| Clock | Anchored to | Used for |
|---|---|---|
| **UTC epoch grid** | UTC | Bar boundaries at M1/M5/M15/H1 |
| **Exchange local (America/New_York)** | ET, DST-observing | Session open/close, daily rollover, holidays |

Bars are binned by integer arithmetic on epoch microseconds:

```
bin_open_us = ts_us - (ts_us % period_us)
```

M1, M5, M15 and H1 all divide an hour evenly, so this grid is unambiguous and
completely DST-independent. **DST never moves a bar boundary.**

What DST *does* move is the session calendar. The trading week and the daily
rollover break are defined in `America/New_York`, so their UTC projections shift
by one hour twice a year. Every expected-bar calculation therefore materialises
the session calendar in ET, converts to UTC, and only then intersects with the
UTC bar grid. This is the single place DST is handled, and it is the subject of
the DST acceptance test.

### 1.3 Bar timestamp convention — explicit

Bars are **left-closed, right-open**: `[ts_open, ts_end_exclusive)`.

- `ts_open` is the **BAR OPEN time**, in UTC. It is the label of the bar.
- `ts_end_exclusive` is `ts_open + period`. A tick belongs to the bar iff
  `ts_open <= tick.ts < ts_end_exclusive`.
- A tick landing exactly on `ts_end_exclusive` belongs to the **next** bar.

This is stored as two real columns rather than left implicit, so acceptance
test 1 is a direct assertion against persisted data rather than a re-derivation.

### 1.4 No forward fill

Bars are only emitted for intervals that actually contain at least one tick.
Gaps are **absent rows**, never carried-forward prices. The quality layer
reports gaps against the expected session calendar; it does not fill them.
There is no `ffill`, no `reindex(...).pad()`, and no `resample().last()` on
prices anywhere in the transform package.

### 1.5 Publication time is the only join key for context

Context series carry two timestamps:

- `observation_time` — the instant the observation *refers to*.
- `publication_time` — the instant the value first became publicly knowable.

For a 10-year real yield observed on Monday, `observation_time` is Monday's
date-stamp and `publication_time` is when the H.15 release actually hit the
wire. Joining on `observation_time` would leak the future. The join function
therefore **only accepts `publication_time`** — `observation_time` is not a
permitted right-hand key, enforced by signature, not by comment.

Default join direction is backward with **strict inequality**:

```
publication_time < bar.ts_open
```

Strict, not `<=`: a value published at exactly the bar open instant is treated
as not yet usable at that instant. `allow_exact_matches` is exposed in config
and defaults to `false`. This is deliberately the conservative choice.

Revisions fall out for free: a revised value is a new row with a later
`publication_time`, so an as-of backward join naturally returns the vintage that
was current as of the bar, not today's revised number.

---

## 2. Directory layout

```
xau_data/
  __init__.py
  config.py              # YAML -> validated settings objects
  timeutils.py           # require_utc, epoch binning, tz conversion
  errors.py              # typed exceptions
  cli.py                 # ingest / build / validate entrypoints

  ingest/
    __init__.py
    _http.py             # shared rate-limited, retrying HTTP client
    _manifest.py         # resume-safety ledger
    dukascopy.py         # XAUUSD tick download + .bi5 decode
    context.py           # DXY, SPX, VIX, US10Y, DFII10, WTI
    calendar.py          # economic calendar events

  transform/
    __init__.py
    bars.py              # ticks -> OHLCV at M1/M5/M15/H1
    sessions.py          # ET-anchored session calendar -> expected UTC bars
    context_join.py      # as-of join on publication_time only

  quality/
    __init__.py
    checks.py            # individual check functions, each returns findings
    validate.py          # orchestration
    report.py            # markdown rendering

  storage/
    __init__.py
    schemas.py           # pyarrow schemas — single source of truth
    paths.py             # partition path construction
    parquet_store.py     # read/write with explicit schema enforcement
    duckdb_views.py      # view DDL over the parquet tree
    README.md            # schema documentation (deliverable 6)

configs/
  config.yaml            # roots, symbol, timeframes, rate limits, thresholds
  context_series.yaml    # per-series source + publication rule
  sessions.yaml          # trading hours, rollover break, holidays

tests/
  conftest.py
  fixtures/              # small synthetic + recorded tick samples
  test_timeutils.py
  test_sessions_dst.py           # acceptance: DST alignment
  test_bars.py
  test_bars_no_lookahead.py      # acceptance: no post-close data in a bar
  test_bars_roundtrip.py         # acceptance: ticks -> bars -> OHLC
  test_context_join.py           # acceptance: no future publication leaks
  test_schemas_utc.py            # every timestamp column is tz-aware UTC
  test_validate.py

data/                    # gitignored, path comes from config
  raw/dukascopy/...      # untouched .bi5 payloads
  ticks/ bars/ context/ calendar/
  _manifests/
  _reports/

docs/DESIGN.md
pyproject.toml
```

---

## 3. On-disk schemas

Partitioning is Hive-style `year=YYYY/month=MM` throughout, as required.
Schemas live in `storage/schemas.py` as explicit `pyarrow.Schema` objects.
Nothing is written via pandas dtype inference.

### 3.1 `ticks`

Path: `data/ticks/symbol=XAUUSD/year=YYYY/month=MM/part-*.parquet`

| Column | Type | Notes |
|---|---|---|
| `ts` | `timestamp[us, UTC]` | Tick instant as reported by the feed |
| `bid` | `float64` | |
| `ask` | `float64` | |
| `bid_volume` | `float64` | Feed-native units, opaque; documented as not shares/lots |
| `ask_volume` | `float64` | |

Sorted by `ts` within each file. `spread` is **not** stored; it is derived
(`ask - bid`) in the DuckDB view to guarantee it can never disagree with its
inputs.

Dukascopy `.bi5` records are 20 bytes big-endian, decoded as
`(ms_offset_from_hour: uint32, ask: int32, bid: int32, ask_volume: float32,
bid_volume: float32)`. Note the wire order is **ask before bid**; getting this
backwards is a silent correctness bug, so decode is covered by a byte-level
fixture test. Integer prices are divided by `10 ** point_digits`, taken from
config rather than hardcoded.

### 3.2 `bars`

Path: `data/bars/symbol=XAUUSD/timeframe=M5/year=YYYY/month=MM/part-*.parquet`

| Column | Type | Notes |
|---|---|---|
| `ts_open` | `timestamp[us, UTC]` | **BAR OPEN time.** The bar's label. Inclusive left edge. |
| `ts_end_exclusive` | `timestamp[us, UTC]` | `ts_open + period`. Exclusive right edge. |
| `bid_open` `bid_high` `bid_low` `bid_close` | `float64` | |
| `ask_open` `ask_high` `ask_low` `ask_close` | `float64` | |
| `mid_open` `mid_high` `mid_low` `mid_close` | `float64` | See note below |
| `spread_mean` | `float64` | Tick-count-weighted mean of `ask - bid` |
| `spread_max` | `float64` | |
| `spread_min` | `float64` | |
| `spread_close` | `float64` | Spread of the last tick in the bar |
| `tick_count` | `int32` | |
| `bid_volume` `ask_volume` | `float64` | Sums over the bar |
| `first_tick_ts` | `timestamp[us, UTC]` | Enables the no-lookahead test directly |
| `last_tick_ts` | `timestamp[us, UTC]` | |

**Mid OHLC note.** Mid is computed per tick as `(bid + ask) / 2` and *then*
reduced to OHLC. It is **not** `(bid_high + ask_high) / 2`, which is wrong
whenever the spread widens at an extreme. This is a common and quiet error; it
gets its own test.

Timeframes M1, M5, M15, H1 are each a separate partitioned dataset, all derived
from ticks directly rather than by cascading M1 to M5. Cascading is cheaper but
compounds edge-case errors, and a cross-check that cascaded and direct agree is
a cheap extra test.

### 3.3 `context`

Path: `data/context/series=DGS10/year=YYYY/month=MM/part-*.parquet`

Partitioned by `observation_time`'s year/month.

| Column | Type | Notes |
|---|---|---|
| `series_id` | `string` | `DXY`, `SPX`, `VIX`, `US10Y`, `DFII10`, `WTI` |
| `observation_time` | `timestamp[us, UTC]` | What the value refers to. **Never a join key.** |
| `publication_time` | `timestamp[us, UTC]` | When it became knowable. **The only join key.** |
| `value` | `float64` | |
| `source` | `string` | e.g. `fred` |
| `source_series_id` | `string` | e.g. `DFII10` |
| `vintage_seq` | `int32` | 0 = first print, increments per revision |
| `is_revision` | `bool` | `vintage_seq > 0` |
| `ingested_at` | `timestamp[us, UTC]` | Provenance |

Primary key is `(series_id, observation_time, vintage_seq)`. Multiple rows per
observation are expected and correct.

**Where `publication_time` comes from.** This is the part that is easy to fake
and I would rather be explicit about it. Two mechanisms, chosen per series in
`configs/context_series.yaml`:

1. **Vintage-derived (preferred, FRED/ALFRED).** FRED's `realtime_start` is the
   vintage date on which that value first appeared. It is a date, not an
   instant, so we combine it with a configured release time-of-day in the
   release's own timezone and convert to UTC. Example: the H.15 series
   (`DGS10`, `DFII10`) publish at 16:15 `America/New_York`.
2. **Market-close-derived.** For price series whose "publication" is simply the
   market printing a close (`SPX`, `VIX`, `WTI`, `DXY` spot), publication time
   is the session close in the venue's timezone plus a dissemination lag.

Both paths then add a configurable `safety_lag` (default 0, but available to be
set positive when a feed's exact timing is uncertain). The rule is: when in
doubt, publish **later**, because a late publication time can only make a
feature less informative, while an early one leaks.

Every series config entry is required to declare its rule explicitly. There is
no default that silently sets `publication_time = observation_time`; that
combination is rejected at config load.

### 3.4 `calendar`

Path: `data/calendar/year=YYYY/month=MM/part-*.parquet`

| Column | Type | Notes |
|---|---|---|
| `event_id` | `string` | Stable hash of (currency, title, scheduled_time) |
| `event_time` | `timestamp[us, UTC]` | Scheduled/actual release instant |
| `publication_time` | `timestamp[us, UTC]` | When `actual` became knowable; `= event_time` for scheduled releases |
| `currency` | `string` | Focus on `USD`, plus `XAU`-relevant (`EUR`, `CNY`, `CHF`) |
| `country` | `string` | |
| `title` | `string` | |
| `impact` | `string` | `low` / `medium` / `high`, normalised at ingest |
| `actual` / `forecast` / `previous` | `float64` | Parsed numeric, nullable |
| `actual_raw` / `forecast_raw` / `previous_raw` | `string` | Original text (`"3.2%"`, `"-"`, `"250K"`) kept verbatim |
| `unit` | `string` | `percent`, `count`, `index`, ... |
| `revised_previous` | `float64` | Nullable |
| `snapshot_time` | `timestamp[us, UTC]` | When this row was captured |
| `source` | `string` | |

**Lookahead subtlety.** `forecast` and `previous` are knowable *before*
`event_time`; `actual` is knowable only *at* `event_time`. Storing them in one
row is convenient but dangerous, so the join helper exposes them separately:
pre-event features may read forecast/previous from `snapshot_time`, while
`actual` is gated on `publication_time`. The schema keeps both timestamps so
this distinction is representable rather than assumed.

**Source honesty.** There is no reliable free historical economic-calendar API.
The design is a `CalendarSource` protocol with two concrete implementations: a
CSV/JSON importer for user-supplied history (the guaranteed path), and an
incremental weekly-feed archiver that accumulates snapshots going forward.
Backfill quality is a data-sourcing problem, not something the code can paper
over, and the validator reports calendar coverage gaps rather than hiding them.

### 3.5 Manifests (resume safety)

Path: `data/_manifests/dukascopy.parquet`

| Column | Type | Notes |
|---|---|---|
| `symbol` | `string` | |
| `hour_utc` | `timestamp[us, UTC]` | One row per source file |
| `status` | `string` | `ok` / `empty` / `missing` / `failed` |
| `bytes` | `int64` | |
| `sha256` | `string` | Of the raw `.bi5` payload |
| `tick_count` | `int32` | |
| `fetched_at` | `timestamp[us, UTC]` | |
| `error` | `string` | Nullable |

Resume logic: hours with `ok` or `empty` are skipped. `failed` and `missing` are
retried on the next run. `empty` is a real, meaningful state (market closed) and
is distinguished from `missing` (server had no file) and `failed` (transport
error) so a weekend hole is never mistaken for a download bug.

Raw `.bi5` payloads are cached under `data/raw/` so that a decoder fix can be
replayed over history without re-downloading anything.

---

## 4. Ingestion behaviour

### 4.1 Shared HTTP client (`ingest/_http.py`)

- Token-bucket rate limiter, `requests_per_second` and `max_concurrency` from
  config.
- Retries on connection errors, timeouts, 429 and 5xx: exponential backoff with
  full jitter, `base_delay * 2**attempt`, capped, `max_attempts` from config.
- `Retry-After` honoured when present.
- 404 is **not** an error for Dukascopy hourly files; it is recorded as
  `missing` and moves on.
- Every request logs symbol, URL, attempt, and outcome at debug level.

### 4.2 Dukascopy specifics

- URL shape: `.../{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5`, where the month
  component is **zero-indexed** (January = `00`). This off-by-one is a classic
  source of silently-shifted data and is covered by a URL-construction test.
- Payload is LZMA-compressed; a zero-byte body means "no ticks this hour".
- Files are hour-aligned in UTC. The validator cross-checks this by confirming
  no decoded tick falls outside its own source hour.

---

## 5. Quality report (`quality/validate.py`)

Report-only. **Nothing is auto-fixed, nothing is silently dropped.** Every check
returns findings with a severity (`info` / `warn` / `error`) and a row sample.

Checks, matching the brief:

1. **Missing bars** vs the expected session calendar (ET-anchored, holiday-aware).
2. **Duplicate timestamps** in ticks and bars.
3. **Zero or negative spreads** (`ask <= bid`), reported per tick and per bar.
4. **Price spikes** beyond N sigma, N from config. Sigma is a rolling robust
   estimate (median absolute deviation) on log returns, so one spike does not
   inflate the threshold that is supposed to catch it.
5. **Thin bars** with `tick_count` below a configured threshold.
6. **Weekend / holiday data that should not exist** — ticks outside any session.
7. **Daily rollover gaps** — coverage around the ET rollover break, reported
   separately from ordinary gaps because a rollover gap is expected and an
   unexpectedly *absent* rollover gap is itself a finding.

Additional checks worth having, proposed:

8. **Monotonicity** — timestamps non-decreasing within each partition.
9. **OHLC consistency** — `low <= open, close <= high` on both sides.
10. **Cross-side sanity** — `ask_low >= bid_low` violations flagged.
11. **Context staleness** — a series whose latest `publication_time` is older
    than a configured tolerance.

Output: a markdown report to `data/_reports/validation_<range>.md`, plus a
machine-readable JSON sidecar so CI can gate on it. The CLI exits non-zero when
any `error`-severity finding is present, and that behaviour is itself
configurable.

---

## 6. Storage and query layer

- Parquet, zstd compression, partitioned `year=YYYY/month=MM`.
- Writes go through `parquet_store.py`, which enforces the declared pyarrow
  schema on every write. A column with an unexpected type fails the write rather
  than being coerced.
- Atomic writes: build to a temp path, then rename, so an interrupted run never
  leaves a half-written partition that later reads as valid.
- `duckdb_views.py` creates views over the tree:

```sql
CREATE OR REPLACE VIEW ticks AS
  SELECT *, ask - bid AS spread
  FROM read_parquet('${root}/ticks/**/*.parquet', hive_partitioning := true);

CREATE OR REPLACE VIEW bars_m5 AS
  SELECT * FROM read_parquet('${root}/bars/symbol=*/timeframe=M5/**/*.parquet',
                             hive_partitioning := true);

CREATE OR REPLACE VIEW context AS
  SELECT * FROM read_parquet('${root}/context/**/*.parquet',
                             hive_partitioning := true);
```

DuckDB preserves `TIMESTAMP WITH TIME ZONE` from Parquet, so the UTC invariant
survives into SQL. `storage/README.md` documents every column, every unit, the
bar-open convention, and the publication-time join rule.

---

## 7. Configuration

All of it YAML, no hardcoded paths or dates. Sketch of `configs/config.yaml`:

```yaml
paths:
  data_root: ./data          # overridable by XAU_DATA_ROOT env var

symbol:
  name: XAUUSD
  point_digits: 3            # Dukascopy integer price scaling

timeframes: [M1, M5, M15, H1]

ingest:
  dukascopy:
    base_url: https://datafeed.dukascopy.com/datafeed
    requests_per_second: 4
    max_concurrency: 4
    max_attempts: 5
    backoff_base_seconds: 1.0
    backoff_max_seconds: 60.0

join:
  context:
    key: publication_time    # only legal value; present to make it explicit
    allow_exact_matches: false

quality:
  spike_sigma: 8.0
  min_tick_count:
    M1: 1
    M5: 3
    M15: 5
    H1: 20
  context_staleness_days: 5
  fail_on: error
```

Date ranges are CLI arguments, never config constants, so no date is baked in
anywhere. Config is parsed into frozen dataclasses with validation at load time;
an unknown key is an error, not a silent no-op.

Sessions live in `configs/sessions.yaml`, expressed in ET:

```yaml
timezone: America/New_York
week_open:  { weekday: sunday, time: "18:00" }
week_close: { weekday: friday, time: "17:00" }
daily_break: { start: "17:00", end: "18:00" }
holidays_file: configs/holidays.csv
```

---

## 8. Acceptance tests — how each is satisfied

| Requirement | Test | Mechanism |
|---|---|---|
| No bar contains data after its own close | `test_bars_no_lookahead.py` | Asserts `ts_open <= first_tick_ts <= last_tick_ts < ts_end_exclusive` on generated and recorded data, including ticks placed exactly on both edges |
| Context join never pulls a future publication | `test_context_join.py` | Property test: for every joined row, `publication_time < ts_open`. Includes a value published one microsecond after bar open, which must not be selected, and a revision case where the newer vintage must be ignored |
| Round trip ticks -> bars -> OHLC | `test_bars_roundtrip.py` | Independent slow-path reference implementation recomputes OHLC from ticks and compares with `math.isclose` at a stated tolerance; also checks `tick_count` sums back to the tick total |
| DST transitions align correctly | `test_sessions_dst.py` | Spring-forward and fall-back dates in both US and EU windows. Asserts the UTC bar grid is unbroken across the transition, that session open/close shift by exactly one hour in UTC, and that the ambiguous fall-back hour resolves without duplicate or dropped bars |

Plus the standing guards: `test_schemas_utc.py` (no naive timestamp column can
be introduced), a decoder fixture test pinning the ask-before-bid wire order,
and a URL-construction test pinning the zero-indexed month.

---

## 9. Dependencies

| Package | Role |
|---|---|
| `polars` | Tick and bar transforms; native tz-aware datetimes |
| `pyarrow` | Explicit Parquet schemas and IO |
| `duckdb` | Views and ad-hoc query surface |
| `httpx` | HTTP with timeouts and connection pooling |
| `pydantic` + `pyyaml` | Config parsing and validation |
| `pytest` + `hypothesis` | Tests, including property tests for the join |
| `mypy` (strict) + `ruff` | Type checking and lint in CI |

Binning is done with explicit integer arithmetic on epoch microseconds rather
than relying on any library's resample semantics. That keeps the bar-edge
behaviour a property of our own tested code rather than of a dependency's
default.

---

## 10. Open questions for you

1. **Symbol scaling.** Dukascopy XAUUSD point digits have varied by era. Should
   `point_digits` be a single config value, or a date-ranged table? I default to
   a single value with a validator check that flags implausible price levels.
2. **Calendar source.** Do you have an existing historical calendar export
   (CSV) to import, or should I build the forward-accumulating archiver and
   accept that history starts from first run?
3. **DXY specifically.** True ICE DXY futures are licensed. Options are a FRED
   trade-weighted dollar index (free, but a different basket and daily-only) or
   a synthetic DXY from its six FX legs. Which do you want?
4. **Mid OHLC.** I propose storing it since it is cheap and easy to get wrong
   downstream. Say the word if you would rather keep the bar schema minimal.
5. **Tick storage precision.** Floats at 3dp are exact in float64. I could also
   store the raw integer points for byte-exact reproducibility. Worth it?

---

## 11. What happens on approval

In order: schemas and timeutils first (everything depends on them), then the
session calendar, then bars plus its four acceptance tests, then storage and
DuckDB views, then the three ingesters, then the quality report. Tests land in
the same commit as the code they cover.
