# Storage layout and schema

## Where data is written

Everything lives under a single **`data_root`**, resolved in this order:

1. The `XAU_DATA_ROOT` environment variable, if set.
2. `paths.data_root` in `configs/config.yaml` (default `./data`).

With the default config, the resolved root in this repo is
`/home/user/RealGoldHercules/data`, and the tree is:

```
<data_root>/
  raw/dukascopy/XAUUSD/YYYY/MM/DD/HHh_ticks.bi5   untouched source payloads
  ticks/symbol=XAUUSD/year=YYYY/month=MM/*.parquet
  bars/symbol=XAUUSD/timeframe=<TF>/year=YYYY/month=MM/*.parquet
  context/series=<ID>/year=YYYY/month=MM/*.parquet
  calendar/year=YYYY/month=MM/*.parquet
  _manifests/dukascopy.parquet                    resume ledger
  _reports/validation_<label>.md and .json
  xau.duckdb                                      views over the parquet tree
```

`data_root` is in `.gitignore`. Raw payloads are kept so a decoder fix can be
replayed over history without re-downloading anything.

## Durability — read this

**This execution container is ephemeral. Everything under `data_root` is lost
when the session ends, including the repository working directory.** The only
durable store reachable from inside the container is the git remote.

Size guide, measured from June 2024:

| Dataset | One month | 42 months (2021-01 to 2024-06) |
|---|---|---|
| Raw `.bi5` | 20 MB | ~840 MB |
| `ticks` Parquet | 33 MB | ~1.4 GB |
| `bars` all four timeframes | 3.5 MB | ~150 MB |
| `bars` M15 and H1 only | 0.6 MB | ~25 MB |

To make data durable, pick one:

1. **Mount a persistent volume** and point the root at it:
   `export XAU_DATA_ROOT=/mnt/persistent/xau` before running any command. This
   is the only option that keeps ticks.
2. **Commit derived bars to git.** M15 and H1 for 42 months is roughly 25 MB,
   which git handles. Ticks at 1.4 GB are not git-appropriate; use git-lfs or
   object storage instead.
3. **Run the ingest outside this container**, on a machine or VM with real
   storage. The package has no dependency on this environment: set
   `XAU_DATA_ROOT` and run the same CLI.

Nothing is written outside `data_root` except the reports the CLI prints paths
for.

## Timestamp conventions

Every timestamp column is Parquet `timestamp[us, tz=UTC]`. Naive timestamps are
rejected at write time rather than coerced, because Arrow will otherwise
silently reinterpret local wall-clock as UTC.

**Bar timestamps are BAR OPEN time**, in UTC. Intervals are left-closed and
right-open: a tick belongs to a bar iff `ts_open <= tick.ts < ts_end_exclusive`.
A tick landing exactly on `ts_end_exclusive` belongs to the next bar.

Bar boundaries sit on a UTC epoch grid computed with integer arithmetic, so DST
never moves a bar edge. Session boundaries are different: they are defined at
17:00/18:00 `America/New_York` and therefore shift by one hour in UTC across DST.
The rollover is 21:00 UTC in summer and **22:00 UTC in winter**.

**Gaps are absent rows.** Intervals with no ticks produce no bar. Prices are
never forward-filled across a session gap.

## Tables

### `ticks`

| Column | Type | Notes |
|---|---|---|
| `ts` | timestamp[us, UTC] | Tick instant from the feed |
| `bid` | float64 | |
| `ask` | float64 | |
| `bid_volume` | float64 | Feed-native, opaque; not lots and not ounces |
| `ask_volume` | float64 | |

`spread` is not stored. It is derived as `ask - bid` in the DuckDB view so it
can never disagree with its inputs.

Source records are 20 bytes big-endian: `uint32` ms-into-hour, `uint32` **ask**,
`uint32` **bid**, `float32` ask volume, `float32` bid volume. Ask precedes bid.
Integer prices divide by `10 ** point_digits` from config.

### `bars`

| Column | Type | Notes |
|---|---|---|
| `ts_open` | timestamp[us, UTC] | **BAR OPEN TIME.** The bar's label, inclusive left edge |
| `ts_end_exclusive` | timestamp[us, UTC] | `ts_open + period`, exclusive right edge |
| `bid_open/high/low/close` | float64 | |
| `ask_open/high/low/close` | float64 | |
| `mid_open/high/low/close` | float64 | See note below |
| `spread_mean/max/min/close` | float64 | `spread_close` is the last tick's spread |
| `tick_count` | int32 | |
| `bid_volume`, `ask_volume` | float64 | Sums over the bar |
| `first_tick_ts`, `last_tick_ts` | timestamp[us, UTC] | Makes containment directly assertable |

**Mid is computed per tick** as `(bid + ask) / 2` and then reduced to OHLC. It is
not `(bid_high + ask_high) / 2`, which is wrong whenever the spread widens at an
extreme. Measured on real June 2024 ticks, the naive form disagrees on 24 of 60
minute bars, worst error 0.0615 USD.

Each timeframe is derived directly from ticks, not cascaded from M1.

### `context`

Carries both `observation_time` and `publication_time`.
**`publication_time` is the only legal join key**, enforced at config load.
`observation_time` is what the value refers to; joining on it leaks the future.
A revision is a new row with a later `publication_time`, so a backward as-of
join naturally returns the vintage current as of the bar.

Note on `DXY`: ICE DXY is licensed, so the configured series is the FRED
trade-weighted broad dollar index. Different basket, daily only.

### `calendar`

`event_time` is the release instant; `publication_time` is when `actual` became
knowable. `forecast` and `previous` are knowable from `snapshot_time`, which is
why both timestamps exist.

### `_manifests/dukascopy.parquet`

One row per source file. `status` is `ok`, `empty`, `missing` or `failed`.
`ok` and `empty` are skipped on resume; `missing` and `failed` are retried.
`empty` (market closed) is deliberately distinguished from `missing` (no file on
the server) and `failed` (transport error), so a weekend hole is never mistaken
for a download bug.

## DuckDB views

`xau.duckdb` carries `ticks`, `bars_m1`, `bars_m5`, `bars_m15`, `bars_h1`, and
`context`/`calendar` when present. DuckDB preserves `TIMESTAMP WITH TIME ZONE`
from Parquet, so the UTC invariant survives into SQL.

```sql
SELECT date_trunc('hour', ts_open) AS h,
       avg(spread_mean), count(*)
FROM bars_m15 GROUP BY 1 ORDER BY 1;
```

## Known model gaps

- **Early closes are not modelled.** The session calendar knows full holidays
  and the daily break, but not half-days. June 19 2024 (Juneteenth) closed early
  and shows up as 152 missing M1 bars in the validation report. That finding is
  correct; the calendar is incomplete.
- `holidays.csv` ships with placeholder entries only.
- Context and calendar ingest are implemented against their schemas but no
  source is configured yet.
