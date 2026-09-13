# Data quality report — XAUUSD

Generated 2026-09-13 20:59:23 UTC

**0 error-severity findings, 8 warnings.** This report does not modify data. Every gap listed below is reported, not filled.

## Coverage

| Property | Value |
|---|---|
| timeframes | M1, M5, M15, H1 |
| bars per timeframe | M1=33,087, M5=6,618, M15=2,206, H1=552 |
| first bar open (UTC) | 2021-02-17 00:00:00+00:00 |
| last bar open (UTC) | 2024-06-30 23:59:00+00:00 |
| total ticks in bars | 5,424,793 |

## M1

| Check | Severity | Count | Detail |
|---|---|---:|---|
| `missing_bars` | WARN | 1175133 | 1175133 expected bars absent (1208220 expected, 33087 present) |
| `price_spikes` | WARN | 93 | 93 bar-to-bar returns beyond 8.0 robust sigma (sigma=1.55e-04) |
| `bar_tick_containment` | ok | 0 | 0 bars contain a tick outside [ts_open, ts_end_exclusive) |
| `duplicate_timestamps` | ok | 0 | 0 duplicated bar open timestamps |
| `monotonic_timestamps` | ok | 0 | 0 bar timestamps out of order |
| `ohlc_consistency` | ok | 0 | 0 bars violate low <= open,close <= high |
| `non_positive_spread` | ok | 0 | 0 bars contain a tick where ask <= bid |
| `implausible_spread` | ok | 0 | 0 bars with max spread above 25.0 |
| `thin_bars` | ok | 0 | 0 bars with tick_count below 1 |
| `out_of_session` | ok | 0 | 0 bars fall outside the trading session (weekend, holiday or daily break) |
| `rollover_gap` | ok | 0 | 0 bars inside the daily rollover break across 33 session days |

### M1 samples

**missing_bars** (1175133)

- `2021-02-18 00:00:00`
- `2021-02-18 00:01:00`
- `2021-02-18 00:02:00`
- `2021-02-18 00:03:00`
- `2021-02-18 00:04:00`
- `2021-02-18 00:05:00`
- `2021-02-18 00:06:00`
- `2021-02-18 00:07:00`
- `2021-02-18 00:08:00`
- `2021-02-18 00:09:00`
- ... and 1175123 more

**price_spikes** (93)

- `2021-02-17 08:03:00 z=10.5`
- `2021-02-17 10:30:00 z=8.2`
- `2021-02-17 13:21:00 z=11.4`
- `2021-02-17 13:30:00 z=9.2`
- `2021-02-17 13:33:00 z=8.1`
- `2021-02-17 13:35:00 z=10.5`
- `2021-02-17 13:41:00 z=12.1`
- `2021-02-17 13:54:00 z=9.8`
- `2021-02-17 14:16:00 z=8.9`
- `2021-02-17 14:22:00 z=8.2`
- ... and 83 more

## M5

| Check | Severity | Count | Detail |
|---|---|---:|---|
| `missing_bars` | WARN | 235026 | 235026 expected bars absent (241644 expected, 6618 present) |
| `price_spikes` | WARN | 18 | 18 bar-to-bar returns beyond 8.0 robust sigma (sigma=3.55e-04) |
| `bar_tick_containment` | ok | 0 | 0 bars contain a tick outside [ts_open, ts_end_exclusive) |
| `duplicate_timestamps` | ok | 0 | 0 duplicated bar open timestamps |
| `monotonic_timestamps` | ok | 0 | 0 bar timestamps out of order |
| `ohlc_consistency` | ok | 0 | 0 bars violate low <= open,close <= high |
| `non_positive_spread` | ok | 0 | 0 bars contain a tick where ask <= bid |
| `implausible_spread` | ok | 0 | 0 bars with max spread above 25.0 |
| `thin_bars` | ok | 0 | 0 bars with tick_count below 3 |
| `out_of_session` | ok | 0 | 0 bars fall outside the trading session (weekend, holiday or daily break) |
| `rollover_gap` | ok | 0 | 0 bars inside the daily rollover break across 33 session days |

### M5 samples

**missing_bars** (235026)

- `2021-02-18 00:00:00`
- `2021-02-18 00:05:00`
- `2021-02-18 00:10:00`
- `2021-02-18 00:15:00`
- `2021-02-18 00:20:00`
- `2021-02-18 00:25:00`
- `2021-02-18 00:30:00`
- `2021-02-18 00:35:00`
- `2021-02-18 00:40:00`
- `2021-02-18 00:45:00`
- ... and 235016 more

**price_spikes** (18)

- `2021-02-17 13:50:00 z=8.1`
- `2021-05-19 00:00:00 z=140.6`
- `2021-05-19 13:00:00 z=8.9`
- `2021-05-19 13:35:00 z=17.9`
- `2021-05-19 14:20:00 z=8.8`
- `2021-05-19 18:00:00 z=8.1`
- `2021-08-18 00:00:00 z=123.8`
- `2021-08-18 18:00:00 z=12.5`
- `2021-11-17 00:00:00 z=96.6`
- `2024-06-02 22:00:00 z=623.4`
- ... and 8 more

## M15

| Check | Severity | Count | Detail |
|---|---|---:|---|
| `missing_bars` | WARN | 78342 | 78342 expected bars absent (80548 expected, 2206 present) |
| `price_spikes` | WARN | 10 | 10 bar-to-bar returns beyond 8.0 robust sigma (sigma=6.30e-04) |
| `bar_tick_containment` | ok | 0 | 0 bars contain a tick outside [ts_open, ts_end_exclusive) |
| `duplicate_timestamps` | ok | 0 | 0 duplicated bar open timestamps |
| `monotonic_timestamps` | ok | 0 | 0 bar timestamps out of order |
| `ohlc_consistency` | ok | 0 | 0 bars violate low <= open,close <= high |
| `non_positive_spread` | ok | 0 | 0 bars contain a tick where ask <= bid |
| `implausible_spread` | ok | 0 | 0 bars with max spread above 25.0 |
| `thin_bars` | ok | 0 | 0 bars with tick_count below 5 |
| `out_of_session` | ok | 0 | 0 bars fall outside the trading session (weekend, holiday or daily break) |
| `rollover_gap` | ok | 0 | 0 bars inside the daily rollover break across 33 session days |

### M15 samples

**missing_bars** (78342)

- `2021-02-18 00:00:00`
- `2021-02-18 00:15:00`
- `2021-02-18 00:30:00`
- `2021-02-18 00:45:00`
- `2021-02-18 01:00:00`
- `2021-02-18 01:15:00`
- `2021-02-18 01:30:00`
- `2021-02-18 01:45:00`
- `2021-02-18 02:00:00`
- `2021-02-18 02:15:00`
- ... and 78332 more

**price_spikes** (10)

- `2021-05-19 00:00:00 z=78.6`
- `2021-05-19 13:30:00 z=13.5`
- `2021-05-19 18:00:00 z=9.5`
- `2021-08-18 00:00:00 z=69.2`
- `2021-11-17 00:00:00 z=56.2`
- `2024-06-02 22:00:00 z=348.6`
- `2024-06-07 08:00:00 z=15.1`
- `2024-06-07 12:30:00 z=9.7`
- `2024-06-12 12:30:00 z=9.7`
- `2024-06-12 13:30:00 z=8.2`

## H1

| Check | Severity | Count | Detail |
|---|---|---:|---|
| `missing_bars` | WARN | 19585 | 19585 expected bars absent (20137 expected, 552 present) |
| `price_spikes` | WARN | 6 | 6 bar-to-bar returns beyond 8.0 robust sigma (sigma=1.18e-03) |
| `bar_tick_containment` | ok | 0 | 0 bars contain a tick outside [ts_open, ts_end_exclusive) |
| `duplicate_timestamps` | ok | 0 | 0 duplicated bar open timestamps |
| `monotonic_timestamps` | ok | 0 | 0 bar timestamps out of order |
| `ohlc_consistency` | ok | 0 | 0 bars violate low <= open,close <= high |
| `non_positive_spread` | ok | 0 | 0 bars contain a tick where ask <= bid |
| `implausible_spread` | ok | 0 | 0 bars with max spread above 25.0 |
| `thin_bars` | ok | 0 | 0 bars with tick_count below 20 |
| `out_of_session` | ok | 0 | 0 bars fall outside the trading session (weekend, holiday or daily break) |
| `rollover_gap` | ok | 0 | 0 bars inside the daily rollover break across 33 session days |

### H1 samples

**missing_bars** (19585)

- `2021-02-18 00:00:00`
- `2021-02-18 01:00:00`
- `2021-02-18 02:00:00`
- `2021-02-18 03:00:00`
- `2021-02-18 04:00:00`
- `2021-02-18 05:00:00`
- `2021-02-18 06:00:00`
- `2021-02-18 07:00:00`
- `2021-02-18 08:00:00`
- `2021-02-18 09:00:00`
- ... and 19575 more

**price_spikes** (6)

- `2021-05-19 00:00:00 z=42.3`
- `2021-08-18 00:00:00 z=37.0`
- `2021-11-17 00:00:00 z=31.0`
- `2024-06-02 22:00:00 z=186.0`
- `2024-06-07 08:00:00 z=9.4`
- `2024-06-12 12:00:00 z=8.3`
