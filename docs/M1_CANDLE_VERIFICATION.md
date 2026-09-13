# Step 1 verification: Dukascopy M1 candles

Status: **VERIFIED. M1 candles are available, carry both sides, and match
tick-derived bars exactly.** One caveat requires handling, documented below.

## What was checked

URL shape, one file per day per side, month component zero-indexed as with ticks:

```
https://datafeed.dukascopy.com/datafeed/XAUUSD/{YYYY}/{MM-1:02d}/{DD:02d}/BID_candles_min_1.bi5
https://datafeed.dukascopy.com/datafeed/XAUUSD/{YYYY}/{MM-1:02d}/{DD:02d}/ASK_candles_min_1.bi5
```

Both sides return HTTP 200. Each file is LZMA-compressed, decompressing to
34,560 bytes: 1,440 records of 24 bytes, one per minute of the day.

## Record format

24 bytes, big-endian, `>5if`:

```
int32   seconds from the start of the UTC day
int32   open       int32  close      int32  low       int32  high
float32 volume
```

**Field order is open, close, low, high**, not the conventional OHLC. This was
determined empirically, not assumed: decoding as open/high/low/close produces
1,315 OHLC constraint violations out of 1,440 bars, while open/close/low/high
produces zero. Integer prices divide by `10 ** point_digits` as for ticks.

## Agreement with tick-derived bars

Compared against M1 bars built from the same day's ticks through the existing
pipeline, on two days from different years:

| Day | Candle records | Tick-derived bars | Common minutes | Max abs OHLC diff | Exact matches |
|---|---:|---:|---:|---:|---|
| 2024-06-05 | 1,440 | 1,380 | 1,380 | 0.00000 | 5,520 / 5,520 |
| 2022-02-16 | 1,440 | 1,380 | 1,380 | 0.00000 | 5,520 / 5,520 |

Not "close" — bit-for-bit identical on every open, high, low and close. The
candle files are the same aggregation our transform performs.

## The caveat: synthetic forward-filled bars

Candle files always contain all 1,440 minutes. Minutes with no trading are
filled with the last price carried forward:

- 2024-06-05: 60 synthetic minutes, all at **21:00 UTC** (EDT rollover)
- 2022-02-16: 60 synthetic minutes, all at **22:00 UTC** (EST rollover)
- 2024-06-08 (Saturday): **all 1,440 minutes synthetic**

This is exactly the forward-fill the data layer forbids, so it must be stripped
at ingest. Every synthetic bar carries `volume == 0.0` **and**
`open == high == low == close`.

The filter was checked for safety on real data:

| Property | Count |
|---|---:|
| Real minutes with `volume == 0` | 0 |
| Real minutes with flat OHLC | 0 |
| Real minutes matching both (would be lost) | 0 |
| Synthetic minutes matching both (correctly caught) | 60 / 60 |

Lowest tick count on any real minute that day was 4, and no real minute had a
single tick, so the filter has margin. Requiring both conditions rather than
`volume == 0` alone is belt and braces.

Note that the synthetic minutes independently re-confirm the DST rollover shift:
they sit at 21:00 UTC in EDT and 22:00 UTC in EST.

## Download economics

Covering 2021-01-01 to 2026-09-13, skipping Saturdays because they are pure fill:

| Plan | Requests | Hours at the observed 0.075 files/s | Compressed size |
|---|---:|---:|---:|
| M1 candles | 3,568 | 13.2 | ~57 MB |
| Full ticks | 49,968 | 185.1 | ~240 MB |

Fourteen times fewer requests, and small enough that the whole M1 history fits
comfortably in object storage or even git.

## Proposed 6-month tick validation slice

Chosen to span distinct volatility regimes and, importantly, distinct *causes*
of volatility, since intrabar path shape differs between a slow trend and a
liquidity gap.

| Month | Regime | Why it earns a slot |
|---|---|---|
| 2021-08 | Flash crash | The 9 August gap down in thin Asian hours. Worst case for intrabar ordering; if M1 is going to fail anywhere, it fails here. |
| 2022-03 | Geopolitical spike | Invasion of Ukraine, spike to ~2070 and collapse. Highest realised volatility of the period. |
| 2022-09 | Trending, moderate vol | Strong dollar, gold grinding to lows. Directional without shock, a regime the others miss. |
| 2023-03 | Financial-stress spike | Regional banking crisis. High vol driven by rates rather than geopolitics. |
| 2024-06 | Baseline | **Already downloaded**, 4,639,089 ticks, fully validated. Free slot. |
| 2026-08 | Current | Most recent complete month, so the slice covers today's spread and liquidity regime rather than only history. |

Keeping 2024-06 means only five months need fetching: 3,650 files, roughly 13.5
hours, comparable to the M1 history itself.

I would still finalise this list from data. The M1 history arrives first and is
cheap, so ranking every month by realised volatility from M1 and confirming
these six sit at the intended spread of the distribution costs nothing and
replaces my recollection of these events with measurement.

## Recommendation

Proceed with the M1 candle plan. HistData is not needed as a fallback.
