# Phase 5 feature proposal

Status: **PROPOSAL — awaiting approval before implementation.**

54 features across the six groups. Every one is scale-free by construction:
a ratio, a log return, an index bounded 0-100, a z-score, or a percentile rank.
No feature carries a dollar value, because price ran 1722 to 5013 and mean ATR
ran 2.66 to 10.41 over the sample, so any absolute magnitude is a date stamp.

---

## 0. Four things to settle first

### 0.1 `tick_count` is not available and cannot be a feature

Group 4 asks for "tick count vs normal". The full history is built from
Dukascopy **M1 candle** files, which carry a volume float and no tick count.
Checked directly: neither `BARS_M1` nor the resampled schema has the column, and
tick counts exist only for the six-month tick slice.

A feature computable on 4% of the sample cannot go into a model that trades the
other 96%. I propose:

- **Substitute `volume_rel_median`** as the activity proxy on the full history.
  Volume is present on every bar, mean 0.5150, median 0.3968, no zeros after
  synthetic stripping.
- Keep tick count as a **validation-only** quantity on the tick slice, to check
  how well volume proxies for it. If the correlation is poor, that is worth
  knowing before trusting volume-derived features at all.

### 0.2 Dukascopy volume is opaque, so VWAP inherits that

The schema documents volume as "feed-native and opaque; not lots and not
ounces". Group 3 asks for distance from VWAP. A VWAP weighted by a quantity we
cannot interpret is a weaker object than it looks.

I propose building **both** and keeping them until phase 7 decides:
`vwap_dist_atr` weighted by volume, and `twap_dist_atr` weighted equally. If
they are near-identical, the volume weighting is adding nothing and we drop one.

### 0.3 Multi-timeframe features have a causality trap

Group 2 asks for M15/H1/H4 agreement. At M15 bar *i*, the current H1 bar is
**incomplete**: three quarters of its high, low and close have not happened yet.
Computing an H1 indicator from a partially-formed bar is lookahead in the most
ordinary disguise.

Rule: higher-timeframe features use **only completed higher-timeframe bars**.
At 09:15 the most recent usable H1 bar is the one that closed at 09:00, and the
most recent usable H4 bar closed at 08:00. The current partial bar is excluded.
This costs up to one full higher-timeframe period of staleness and is the price
of correctness. The causality test catches it if I get it wrong.

### 0.4 A one-year percentile rank nulls 18% of the sample

Group 6 asks for a trailing 1-2 year rank. Measured: 23,629 M15 bars per year
against 134,687 total, so a one-year trailing window leaves the first 18% of
history without regime features, and a two-year window leaves 35%.

Three options, and I recommend the third:

1. **Fixed 1y trailing**, nulls the first year. Clean, stationary, expensive.
2. **Expanding rank** from the start of history. No nulls after a short warm-up,
   but the rank's meaning drifts as the sample grows, and early ranks are
   computed against very little.
3. **Fixed 1y trailing with an expanding fallback** below the threshold: use an
   expanding rank once at least 60 days of history exist, switch to the 1y
   trailing rank once available, and expose a `regime_rank_is_expanding` flag so
   a model can tell the two regimes of the feature apart. Nulls only the first
   60 days.

Option 3 keeps the sample while being honest that the early values are a
different object. Say which you want.

---

## 1. Volatility (10)

| Feature | Formula | Lookback | Why for gold |
|---|---|---|---|
| `atr_pct` | ATR(14) / close | 14 | Scale-free volatility level. The only absolute-to-relative conversion needed. |
| `atr_ratio_14_50` | ATR(14) / ATR(50) | 50 | Short vs medium vol. Gold's regime shifts are abrupt: the three most volatile months in the sample are consecutive. |
| `atr_ratio_14_200` | ATR(14) / ATR(200) | 200 | Same, longer baseline. |
| `atr_ratio_50_200` | ATR(50) / ATR(200) | 200 | Medium vs long, slower regime signal. |
| `gk_vol_14` | Garman-Klass over 14, annualised | 14 | Uses the whole bar, roughly 5x more efficient than close-to-close. Already log-based. |
| `parkinson_14` | Parkinson HL estimator over 14 | 14 | High-low only. Diverges from GK when opens gap, which for gold is the Asian open and post-rollover. |
| `gk_to_rv_ratio` | gk_vol_14 / rv_20 | 20 | Intrabar vs close-to-close vol. A high ratio means the range is being made and given back, which is where intrabar ambiguity lives. |
| `rv_20` | sd of log returns, 20, annualised | 20 | Standard realised vol. |
| `rv_vs_median_500` | rv_20 / median(rv_20, 500) | 500 | Is current vol high *for this market*, normalised by its own recent norm. |
| `vol_of_vol_100` | sd(rv_20, 100) / mean(rv_20, 100) | 100 | Coefficient of variation, so scale-free. Gold's vol clusters; this measures how unstable the clustering is. |

## 2. Trend (12)

| Feature | Formula | Lookback | Why for gold |
|---|---|---|---|
| `ema_dist_20_atr` | (close - EMA20) / ATR(14) | 20 | Displacement in volatility units, not dollars. |
| `ema_dist_50_atr` | (close - EMA50) / ATR(14) | 50 | |
| `ema_dist_200_atr` | (close - EMA200) / ATR(14) | 200 | |
| `ema_slope_20_atr` | (EMA20 - EMA20[-5]) / ATR(14) | 25 | Slope per 5 bars in ATR units. |
| `ema_slope_50_atr` | (EMA50 - EMA50[-10]) / ATR(14) | 60 | |
| `ema_slope_200_atr` | (EMA200 - EMA200[-20]) / ATR(14) | 220 | |
| `ema_stack` | ordered agreement of EMA20/50/200, in {-1, -1/3, 1/3, 1} | 200 | Cheap trend-structure summary. |
| `adx_14` | Wilder ADX / 100 | 28 | Trend strength without direction. Bounded by construction. |
| `di_diff_14` | (+DI - -DI) / 100 | 28 | Directional component, bounded. |
| `donchian_pos_20` | (close - low20) / (high20 - low20) | 20 | Position in range, 0-1. |
| `donchian_pos_96` | same over 96 bars, one trading day | 96 | Daily-scale position. |
| `mtf_trend_agreement` | mean of sign(ema_slope) at M15, H1, H4 | 220 H4 bars | **Completed higher-TF bars only.** Gold trends persist across timeframes; disagreement marks transitions. |

## 3. Momentum (9)

| Feature | Formula | Lookback | Why for gold |
|---|---|---|---|
| `rsi_14` | Wilder RSI / 100 | 14 | Bounded. |
| `rsi_50` | Wilder RSI / 100 | 50 | Slower, less whipsaw. |
| `roc_atr_4` | (close - close[-4]) / ATR(14) | 4 | One hour of move in ATR units. |
| `roc_atr_16` | over 16 bars, four hours | 16 | Bottom of the 4-12h holding window. |
| `roc_atr_48` | over 48 bars, twelve hours | 48 | Top of the holding window. |
| `roc_atr_96` | over 96 bars, one day | 96 | |
| `ret_z_20` | log return / rv_20 | 20 | Last bar's move in its own vol units. |
| `vwap_dist_atr` | (close - session VWAP) / ATR(14) | session | Volume-weighted, with the caveat in 0.2. |
| `twap_dist_atr` | (close - session TWAP) / ATR(14) | session | Equal-weighted control for the above. |

## 4. Microstructure (9)

| Feature | Formula | Lookback | Why for gold |
|---|---|---|---|
| `spread_rel_median_500` | spread / median(spread, 500) | 500 | Spread scales with price, so only the relative level is portable across the 0.32-1.03 range. |
| `spread_z_500` | (spread - mean) / sd over 500 | 500 | Same signal, different shape; both kept until phase 7. |
| `spread_to_atr` | spread / ATR(14) | 14 | What a round trip costs relative to the move available. |
| `volume_rel_median_500` | volume / median(volume, 500) | 500 | Activity proxy, standing in for tick count per 0.1. |
| `volume_z_500` | z-score of volume over 500 | 500 | |
| `bar_range_rel_median_500` | (high - low) / median(range, 500) | 500 | **The wide% concept from the ranking, computed causally per bar.** |
| `is_wide_bar` | `bar_range_rel_median_500 > 3` | 500 | The exact threshold the month ranking used, as a per-bar flag. This is where intrabar ambiguity concentrates. |
| `vol_imbalance` | (ask_vol - bid_vol) / (ask_vol + bid_vol) | 1 | Crude order-flow tilt. Bounded -1 to 1. |
| `bar_completeness` | m1_count / 15 | 1 | 1.0 is a full M15 bar. Below that means an internal gap, which matters near rollover and holidays. |

## 5. Session and time (10)

All bounded or cyclic. Nothing here encodes an absolute date: the model must not
be able to learn "2026 was volatile", only "this is the London open".

| Feature | Formula | Why for gold |
|---|---|---|
| `session_asian` / `session_london` / `session_overlap` / `session_ny` | one-hot from SessionCalendar | Gold's behaviour differs sharply by session; this is the only categorical set. |
| `session_progress` | minutes into session / session length, 0-1 | Opens and closes behave differently from the middle. |
| `mins_to_rollover` | minutes to next rollover / 1440 | Structural, and DST-correct via the calendar. Directly relevant with a forced exit before rollover. |
| `mins_since_rollover` | minutes since last rollover / 1440 | The thin post-rollover hours are where spread widens most. |
| `dow_sin`, `dow_cos` | cyclic day of week | Cyclic encoding avoids a false ordering between Friday and Monday. |
| `tod_sin`, `tod_cos` | cyclic minute of UTC day | |

## 6. Regime (7)

| Feature | Formula | Lookback | Why for gold |
|---|---|---|---|
| `cost_to_atr_ratio` | existing, round-trip cost / ATR | 14 | Already built. Fell 0.2464 to 0.0943 across the sample; the model should see how expensive trading was. |
| `rv_pct_rank_1y` | percentile rank of rv_20 in trailing 1y | 23,629 | Where the current regime sits historically. |
| `atr_pct_rank_1y` | percentile rank of atr_pct in trailing 1y | 23,629 | |
| `spread_pct_rank_1y` | percentile rank of spread in trailing 1y | 23,629 | |
| `range_pct_rank_1y` | percentile rank of bar range in trailing 1y | 23,629 | |
| `regime_rank_is_expanding` | flag, per 0.4 | — | Marks rows where ranks came from the expanding fallback rather than a full year. |
| `asian_range_atr` | Asian session range / ATR(14) | session | **Width only, no position.** Breakout and fade both failed independently, so direction from this range is a dead end; width remains legitimate volatility context. |

---

## Deliverables on approval

- `xau_backtest/features/` with one module per group, all causal.
- Causality tests reusing the `cost_to_atr_ratio` pattern: truncating future bars
  must not change an earlier value, and appending a bar must not rewrite history.
  Applied to all 54, not a sample.
- `docs/FEATURE_CATALOGUE.md` generated from the code so it cannot drift.
- Correlation matrix with pairs above 0.9 flagged. I expect the ATR ratios, the
  two spread encodings and the RSI pair to cluster; flagging is all that happens
  now.
- Coverage report: null rate and warm-up bars per feature.
- `features_m15` extended and materialised to Parquet, partitioned year/month.

No selection, no importance ranking, no target correlation. That waits for
phase 7 inside folds.

## Open questions

1. Percentile-rank policy: option 1, 2 or 3 from section 0.4?
2. Keep both `vwap_dist_atr` and `twap_dist_atr`, or drop the volume-weighted
   one given the opacity in 0.2?
3. Session one-hot as four booleans, or a single ordinal? One-hot is safer for
   linear models, an ordinal is cheaper for trees.
4. Should I add H4 features at all? At 4-12h holding periods an H4 bar is nearly
   the whole trade, so its slope may be closer to the target than to a feature.
