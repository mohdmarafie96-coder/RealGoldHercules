# Data quality report — XAUUSD

Generated 2026-09-13 20:37:14 UTC

**0 error-severity findings, 8 warnings.** This report does not modify data. Every gap listed below is reported, not filled.

## Coverage

| Property | Value |
|---|---|
| timeframes | M1, M5, M15, H1 |
| bars per timeframe | M1=27,567, M5=5,514, M15=1,838, H1=460 |
| first bar open (UTC) | 2024-06-02 22:00:00+00:00 |
| last bar open (UTC) | 2024-06-30 23:59:00+00:00 |
| total ticks in bars | 4,639,089 |

## M1

| Check | Severity | Count | Detail |
|---|---|---:|---|
| `missing_bars` | WARN | 153 | 153 expected bars absent (27720 expected, 27567 present) |
| `price_spikes` | WARN | 58 | 58 bar-to-bar returns beyond 8.0 robust sigma (sigma=1.54e-04) |
| `bar_tick_containment` | ok | 0 | 0 bars contain a tick outside [ts_open, ts_end_exclusive) |
| `duplicate_timestamps` | ok | 0 | 0 duplicated bar open timestamps |
| `monotonic_timestamps` | ok | 0 | 0 bar timestamps out of order |
| `ohlc_consistency` | ok | 0 | 0 bars violate low <= open,close <= high |
| `non_positive_spread` | ok | 0 | 0 bars contain a tick where ask <= bid |
| `implausible_spread` | ok | 0 | 0 bars with max spread above 25.0 |
| `thin_bars` | ok | 0 | 0 bars with tick_count below 1 |
| `out_of_session` | ok | 0 | 0 bars fall outside the trading session (weekend, holiday or daily break) |
| `rollover_gap` | ok | 0 | 0 bars inside the daily rollover break across 25 session days |

### M1 samples

**missing_bars** (153)

- `2024-06-19 18:30:00`
- `2024-06-19 18:31:00`
- `2024-06-19 18:32:00`
- `2024-06-19 18:33:00`
- `2024-06-19 18:34:00`
- `2024-06-19 18:35:00`
- `2024-06-19 18:36:00`
- `2024-06-19 18:37:00`
- `2024-06-19 18:38:00`
- `2024-06-19 18:39:00`
- ... and 143 more

**price_spikes** (58)

- `2024-06-03 14:00:00 z=21.0`
- `2024-06-04 13:57:00 z=9.5`
- `2024-06-04 14:00:00 z=17.7`
- `2024-06-04 14:05:00 z=8.8`
- `2024-06-04 14:22:00 z=9.6`
- `2024-06-05 14:16:00 z=8.4`
- `2024-06-05 15:00:00 z=8.1`
- `2024-06-06 01:31:00 z=10.2`
- `2024-06-06 13:37:00 z=9.2`
- `2024-06-06 13:38:00 z=11.5`
- ... and 48 more

## M5

| Check | Severity | Count | Detail |
|---|---|---:|---|
| `missing_bars` | WARN | 30 | 30 expected bars absent (5544 expected, 5514 present) |
| `price_spikes` | WARN | 8 | 8 bar-to-bar returns beyond 8.0 robust sigma (sigma=3.49e-04) |
| `bar_tick_containment` | ok | 0 | 0 bars contain a tick outside [ts_open, ts_end_exclusive) |
| `duplicate_timestamps` | ok | 0 | 0 duplicated bar open timestamps |
| `monotonic_timestamps` | ok | 0 | 0 bar timestamps out of order |
| `ohlc_consistency` | ok | 0 | 0 bars violate low <= open,close <= high |
| `non_positive_spread` | ok | 0 | 0 bars contain a tick where ask <= bid |
| `implausible_spread` | ok | 0 | 0 bars with max spread above 25.0 |
| `thin_bars` | ok | 0 | 0 bars with tick_count below 3 |
| `out_of_session` | ok | 0 | 0 bars fall outside the trading session (weekend, holiday or daily break) |
| `rollover_gap` | ok | 0 | 0 bars inside the daily rollover break across 25 session days |

### M5 samples

**missing_bars** (30)

- `2024-06-19 18:30:00`
- `2024-06-19 18:35:00`
- `2024-06-19 18:40:00`
- `2024-06-19 18:45:00`
- `2024-06-19 18:50:00`
- `2024-06-19 18:55:00`
- `2024-06-19 19:00:00`
- `2024-06-19 19:05:00`
- `2024-06-19 19:10:00`
- `2024-06-19 19:15:00`
- ... and 20 more

**price_spikes** (8)

- `2024-06-05 14:00:00 z=8.4`
- `2024-06-07 08:00:00 z=18.5`
- `2024-06-07 12:30:00 z=16.6`
- `2024-06-12 12:30:00 z=12.0`
- `2024-06-12 13:35:00 z=11.4`
- `2024-06-13 12:30:00 z=15.3`
- `2024-06-21 13:50:00 z=13.7`
- `2024-06-21 14:00:00 z=8.3`

## M15

| Check | Severity | Count | Detail |
|---|---|---:|---|
| `missing_bars` | WARN | 10 | 10 expected bars absent (1848 expected, 1838 present) |
| `price_spikes` | WARN | 4 | 4 bar-to-bar returns beyond 8.0 robust sigma (sigma=6.11e-04) |
| `bar_tick_containment` | ok | 0 | 0 bars contain a tick outside [ts_open, ts_end_exclusive) |
| `duplicate_timestamps` | ok | 0 | 0 duplicated bar open timestamps |
| `monotonic_timestamps` | ok | 0 | 0 bar timestamps out of order |
| `ohlc_consistency` | ok | 0 | 0 bars violate low <= open,close <= high |
| `non_positive_spread` | ok | 0 | 0 bars contain a tick where ask <= bid |
| `implausible_spread` | ok | 0 | 0 bars with max spread above 25.0 |
| `thin_bars` | ok | 0 | 0 bars with tick_count below 5 |
| `out_of_session` | ok | 0 | 0 bars fall outside the trading session (weekend, holiday or daily break) |
| `rollover_gap` | ok | 0 | 0 bars inside the daily rollover break across 25 session days |

### M15 samples

**missing_bars** (10)

- `2024-06-19 18:30:00`
- `2024-06-19 18:45:00`
- `2024-06-19 19:00:00`
- `2024-06-19 19:15:00`
- `2024-06-19 19:30:00`
- `2024-06-19 19:45:00`
- `2024-06-19 20:00:00`
- `2024-06-19 20:15:00`
- `2024-06-19 20:30:00`
- `2024-06-19 20:45:00`

**price_spikes** (4)

- `2024-06-07 08:00:00 z=15.5`
- `2024-06-07 12:30:00 z=10.0`
- `2024-06-12 12:30:00 z=10.0`
- `2024-06-12 13:30:00 z=8.4`

## H1

| Check | Severity | Count | Detail |
|---|---|---:|---|
| `missing_bars` | WARN | 2 | 2 expected bars absent (462 expected, 460 present) |
| `price_spikes` | WARN | 2 | 2 bar-to-bar returns beyond 8.0 robust sigma (sigma=1.09e-03) |
| `bar_tick_containment` | ok | 0 | 0 bars contain a tick outside [ts_open, ts_end_exclusive) |
| `duplicate_timestamps` | ok | 0 | 0 duplicated bar open timestamps |
| `monotonic_timestamps` | ok | 0 | 0 bar timestamps out of order |
| `ohlc_consistency` | ok | 0 | 0 bars violate low <= open,close <= high |
| `non_positive_spread` | ok | 0 | 0 bars contain a tick where ask <= bid |
| `implausible_spread` | ok | 0 | 0 bars with max spread above 25.0 |
| `thin_bars` | ok | 0 | 0 bars with tick_count below 20 |
| `out_of_session` | ok | 0 | 0 bars fall outside the trading session (weekend, holiday or daily break) |
| `rollover_gap` | ok | 0 | 0 bars inside the daily rollover break across 25 session days |

### H1 samples

**missing_bars** (2)

- `2024-06-19 19:00:00`
- `2024-06-19 20:00:00`

**price_spikes** (2)

- `2024-06-07 08:00:00 z=10.2`
- `2024-06-12 12:00:00 z=9.0`
