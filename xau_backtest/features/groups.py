"""The six feature groups. One function per group, every feature causal.

No feature carries a dollar value. Price ran 1722 to 5013 and mean ATR ran 2.66
to 10.41 across the sample, so any absolute magnitude would be a date stamp and
the model would learn the calendar instead of the market.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .core import (ema, log_return, rma, rolling_max, rolling_mean,
                   rolling_median, rolling_min, rolling_rank, rolling_std,
                   safe_div, shift)

__all__ = ["volatility", "trend", "momentum", "microstructure", "session_time",
           "regime", "GROUPS", "BARS_PER_YEAR"]

F = npt.NDArray[np.float64]
BARS_PER_YEAR = 23_629          # measured: 134,687 M15 bars over 5.7 years
_ANN = np.sqrt(BARS_PER_YEAR)


def _tr(h: F, l: F, c: F) -> F:
    pc = shift(c, 1)
    return np.where(np.isnan(pc), h - l,
                    np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)]))


def _atr(h: F, l: F, c: F, p: int) -> F:
    return rma(_tr(h, l, c), p)


# ---------------------------------------------------------------- 1. volatility


def volatility(b: Mapping[str, F]) -> dict[str, F]:
    h, l, c, o = b["bid_high"], b["bid_low"], b["bid_close"], b["bid_open"]
    atr14, atr50, atr200 = _atr(h, l, c, 14), _atr(h, l, c, 50), _atr(h, l, c, 200)
    r1 = log_return(c, 1)
    rv20 = rolling_std(r1, 20) * _ANN
    rv100 = rolling_std(r1, 100) * _ANN

    with np.errstate(divide="ignore", invalid="ignore"):
        lhl = np.where((h > 0) & (l > 0), np.log(h / l), np.nan)
        lco = np.where((c > 0) & (o > 0), np.log(c / o), np.nan)
    gk_bar = 0.5 * lhl ** 2 - (2 * np.log(2) - 1) * lco ** 2
    gk14 = np.sqrt(np.maximum(rolling_mean(gk_bar, 14), 0.0)) * _ANN
    park14 = np.sqrt(np.maximum(rolling_mean(lhl ** 2 / (4 * np.log(2)), 14), 0.0)) * _ANN

    return {
        "atr_pct": safe_div(atr14, c),
        "atr_ratio_14_50": safe_div(atr14, atr50),
        "atr_ratio_14_200": safe_div(atr14, atr200),
        "atr_ratio_50_200": safe_div(atr50, atr200),
        "gk_vol_14": gk14,
        "parkinson_14": park14,
        "gk_to_rv_ratio": safe_div(gk14, rv20),
        "rv_20": rv20,
        "rv_vs_median_500": safe_div(rv20, rolling_median(rv20, 500)),
        "vol_of_vol_100": safe_div(rolling_std(rv20, 100), rolling_mean(rv20, 100)),
        "_atr14": atr14, "_rv20": rv20,        # internal, stripped before output
    }


# --------------------------------------------------------------------- 2. trend


def _adx(h: F, l: F, c: F, p: int = 14) -> tuple[F, F]:
    up = h - shift(h, 1)
    dn = shift(l, 1) - l
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(_tr(h, l, c), p)
    pdi = 100.0 * safe_div(rma(plus, p), tr)
    mdi = 100.0 * safe_div(rma(minus, p), tr)
    dx = 100.0 * safe_div(np.abs(pdi - mdi), pdi + mdi)
    return rma(dx, p), pdi - mdi


def trend(b: Mapping[str, F], atr14: F, higher: Mapping[str, F] | None = None) -> dict[str, F]:
    h, l, c = b["bid_high"], b["bid_low"], b["bid_close"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    adx, didiff = _adx(h, l, c, 14)

    def stack(a: F, bb: F, cc: F) -> F:
        up = ((a > bb).astype(float) + (bb > cc).astype(float))
        dn = ((a < bb).astype(float) + (bb < cc).astype(float))
        out = (up - dn) / 2.0
        out[np.isnan(a) | np.isnan(bb) | np.isnan(cc)] = np.nan
        return out

    def donch(w: int) -> F:
        hi, lo = rolling_max(h, w), rolling_min(l, w)
        return safe_div(c - lo, hi - lo)

    out = {
        "ema_dist_20_atr": safe_div(c - e20, atr14),
        "ema_dist_50_atr": safe_div(c - e50, atr14),
        "ema_dist_200_atr": safe_div(c - e200, atr14),
        "ema_slope_20_atr": safe_div(e20 - shift(e20, 5), atr14),
        "ema_slope_50_atr": safe_div(e50 - shift(e50, 10), atr14),
        "ema_slope_200_atr": safe_div(e200 - shift(e200, 20), atr14),
        "ema_stack": stack(e20, e50, e200),
        "adx_14": adx / 100.0,
        "di_diff_14": didiff / 100.0,
        "donchian_pos_20": donch(20),
        "donchian_pos_96": donch(96),
        "donchian_width_20_atr": safe_div(rolling_max(h, 20) - rolling_min(l, 20), atr14),
    }
    if higher is not None:
        sgn = lambda x: np.sign(np.nan_to_num(x, nan=0.0)) * np.where(np.isnan(x), np.nan, 1.0)
        parts = [sgn(out["ema_slope_20_atr"]), sgn(higher["h1_slope"]), sgn(higher["h4_slope"])]
        stacked = np.vstack(parts)
        with np.errstate(invalid="ignore"):
            allnan = np.all(np.isnan(stacked), axis=0)
            agg = np.full(stacked.shape[1], np.nan)
            agg[~allnan] = np.nanmean(stacked[:, ~allnan], axis=0)
        out["mtf_trend_agreement"] = agg
        out["h1_ema_slope_atr"] = higher["h1_slope"]
        out["h4_ema_slope_atr"] = higher["h4_slope"]
    return out


# ------------------------------------------------------------------ 3. momentum


def _rsi(c: F, p: int) -> F:
    d = c - shift(c, 1)
    gain = rma(np.where(d > 0, d, 0.0), p)
    loss = rma(np.where(d < 0, -d, 0.0), p)
    rs = safe_div(gain, loss)
    return np.where(np.isnan(rs), np.nan, 100.0 - 100.0 / (1.0 + rs))


def momentum(b: Mapping[str, F], atr14: F, rv20: F,
             session_id: npt.NDArray[np.int64]) -> dict[str, F]:
    c = b["bid_close"]
    vol = b["bid_volume"] + b["ask_volume"]
    typical = (b["bid_high"] + b["bid_low"] + b["bid_close"]) / 3.0

    # session-anchored VWAP/TWAP: reset whenever the session id changes.
    n = len(c)
    vwap = np.full(n, np.nan)
    twap = np.full(n, np.nan)
    cum_pv = cum_v = cum_p = 0.0
    cnt = 0
    prev = -1
    for i in range(n):
        if session_id[i] != prev:
            cum_pv = cum_v = cum_p = 0.0
            cnt = 0
            prev = int(session_id[i])
        v = vol[i] if np.isfinite(vol[i]) else 0.0
        cum_pv += typical[i] * v
        cum_v += v
        cum_p += typical[i]
        cnt += 1
        vwap[i] = cum_pv / cum_v if cum_v > 0 else np.nan
        twap[i] = cum_p / cnt

    return {
        "rsi_14": _rsi(c, 14) / 100.0,
        "rsi_50": _rsi(c, 50) / 100.0,
        "roc_atr_4": safe_div(c - shift(c, 4), atr14),
        "roc_atr_16": safe_div(c - shift(c, 16), atr14),
        "roc_atr_48": safe_div(c - shift(c, 48), atr14),
        "roc_atr_96": safe_div(c - shift(c, 96), atr14),
        "ret_z_20": safe_div(log_return(c, 1) * _ANN, rv20),
        "vwap_dist_atr": safe_div(c - vwap, atr14),
        "twap_dist_atr": safe_div(c - twap, atr14),
    }


# ------------------------------------------------------------ 4. microstructure


def microstructure(b: Mapping[str, F], atr14: F) -> dict[str, F]:
    spread = (b["spread_open"] + b["spread_close"]) / 2.0
    vol = b["bid_volume"] + b["ask_volume"]
    rng = b["bid_high"] - b["bid_low"]
    av, bv = b["ask_volume"], b["bid_volume"]
    return {
        "spread_rel_median_500": safe_div(spread, rolling_median(spread, 500)),
        "spread_z_500": safe_div(spread - rolling_mean(spread, 500),
                                 rolling_std(spread, 500)),
        "spread_to_atr": safe_div(spread, atr14),
        "volume_rel_median_500": safe_div(vol, rolling_median(vol, 500)),
        "volume_z_500": safe_div(vol - rolling_mean(vol, 500), rolling_std(vol, 500)),
        "bar_range_rel_median_500": safe_div(rng, rolling_median(rng, 500)),
        "vol_imbalance": safe_div(av - bv, av + bv),
        "bar_completeness": b["m1_count"] / 15.0,
    }


# ----------------------------------------------------------- 5. session and time


def session_time(ts: Sequence[datetime], calendar) -> dict[str, F]:
    n = len(ts)
    names = ("asian", "london", "overlap", "ny")
    out = {f"session_{k}": np.zeros(n) for k in names}
    prog = np.full(n, np.nan)
    to_roll = np.full(n, np.nan)
    dow_s = np.empty(n); dow_c = np.empty(n)
    tod_s = np.empty(n); tod_c = np.empty(n)
    sid = np.zeros(n, dtype=np.int64)

    run = 0
    prev_label = None
    run_start: datetime | None = None
    for i, t in enumerate(ts):
        lab = calendar.session_label(t).value
        if lab != prev_label:
            run += 1
            run_start = t
            prev_label = lab
        sid[i] = run
        if f"session_{lab}" in out:
            out[f"session_{lab}"][i] = 1.0
        # denominator comes from the CONFIGURED session span, not a hardcoded
        # 8h and not the observed run length. An 8h constant gave asian a max
        # of 0.7188 (6h/8h) and london 0.5938, because overlap carves the
        # middle out of london and ny so labelled spans are shorter than the
        # configured windows.
        prog[i] = calendar.session_progress(t)
        nxt = calendar.next_rollover(t)
        to_roll[i] = (nxt - t).total_seconds() / 86400.0
        ang = 2 * np.pi * t.weekday() / 7.0
        dow_s[i], dow_c[i] = np.sin(ang), np.cos(ang)
        m = t.hour * 60 + t.minute
        a2 = 2 * np.pi * m / 1440.0
        tod_s[i], tod_c[i] = np.sin(a2), np.cos(a2)

    out.update({
        "session_progress": prog,
        # mins_since_rollover is deliberately absent. Rollovers are exactly
        # 24h apart, so it would be 1 - mins_to_rollover: a perfectly collinear
        # duplicate (measured r = -1.0000), not an independent signal. This is
        # arithmetic, not feature selection.
        "mins_to_rollover": to_roll,
        "dow_sin": dow_s, "dow_cos": dow_c,
        "tod_sin": tod_s, "tod_cos": tod_c,
        "_session_id": sid.astype(np.float64),
    })
    return out


# -------------------------------------------------------------------- 6. regime


def regime(b: Mapping[str, F], atr14: F, rv20: F, cost: F,
           window: int = BARS_PER_YEAR) -> dict[str, F]:
    c = b["bid_close"]
    spread = (b["spread_open"] + b["spread_close"]) / 2.0
    rng = b["bid_high"] - b["bid_low"]
    return {
        "cost_to_atr_ratio": safe_div(cost, atr14),
        "rv_pct_rank_1y": rolling_rank(rv20, window),
        "atr_pct_rank_1y": rolling_rank(safe_div(atr14, c), window),
        "spread_pct_rank_1y": rolling_rank(spread, window),
        "range_pct_rank_1y": rolling_rank(rng, window),
    }


GROUPS = ("volatility", "trend", "momentum", "microstructure",
          "session_time", "regime")
