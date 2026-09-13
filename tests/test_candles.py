"""Candle decoding, synthetic stripping and resampling.

The field-order test is the important one. The Dukascopy candle record is
open, CLOSE, LOW, HIGH, not the conventional OHLC, and decoding it wrongly
produces plausible-looking bars with silently wrong highs and lows.
"""
from __future__ import annotations

import lzma
import struct
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pytest

from xau_data.ingest.dukascopy_candles import (
    Candle,
    SIDES,
    candle_url,
    decode_candles,
    is_synthetic,
    merge_sides,
    rows_to_table,
    trading_days,
)
from xau_data.transform.resample import bar_range, resample_m1

UTC = timezone.utc
BASE = "https://datafeed.dukascopy.com/datafeed"


def build_payload(records: list[tuple[int, float, float, float, float, float]],
                  digits: int = 3) -> bytes:
    """Encode in the REAL wire order: time, open, close, low, high, volume."""
    scale = 10 ** digits
    raw = b"".join(
        struct.pack(">5if", secs, int(round(o * scale)), int(round(c * scale)),
                    int(round(lo * scale)), int(round(hi * scale)), vol)
        for secs, o, hi, lo, c, vol in records
    )
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


# --- URL ------------------------------------------------------------------


def test_candle_url_month_is_zero_indexed() -> None:
    assert candle_url(BASE, "XAUUSD", date(2024, 1, 5), "BID").endswith(
        "/XAUUSD/2024/00/05/BID_candles_min_1.bi5")
    assert candle_url(BASE, "XAUUSD", date(2024, 12, 31), "ASK").endswith(
        "/XAUUSD/2024/11/31/ASK_candles_min_1.bi5")


def test_candle_url_rejects_bad_side() -> None:
    with pytest.raises(ValueError, match="side must be"):
        candle_url(BASE, "XAUUSD", date(2024, 6, 5), "MID")


# --- the field order ------------------------------------------------------


def test_decode_uses_open_close_low_high_not_ohlc() -> None:
    """Decoding must satisfy OHLC constraints; the conventional order does not.

    The record below has a high well above both open and close and a low well
    below. Read as open/high/low/close it would claim high=2000.5 (the true
    close) and close=2010.0 (the true high), inverting the bar.
    """
    recs = [(0, 2000.0, 2010.0, 1995.0, 2000.5, 1.25)]   # secs, o, hi, lo, c, vol
    got = decode_candles(build_payload(recs), date(2024, 6, 5), point_digits=3)
    assert len(got) == 1
    c = got[0]
    assert c.open == pytest.approx(2000.0)
    assert c.high == pytest.approx(2010.0)
    assert c.low == pytest.approx(1995.0)
    assert c.close == pytest.approx(2000.5)
    assert c.volume == pytest.approx(1.25)
    assert c.low <= min(c.open, c.close) <= max(c.open, c.close) <= c.high


def test_decode_satisfies_ohlc_on_many_random_bars() -> None:
    rng = np.random.default_rng(7)
    recs = []
    for i in range(500):
        o = 2000 + rng.normal(0, 5)
        c = o + rng.normal(0, 2)
        hi = max(o, c) + abs(rng.normal(0, 1))
        lo = min(o, c) - abs(rng.normal(0, 1))
        recs.append((i * 60, round(o, 3), round(hi, 3), round(lo, 3), round(c, 3), 1.0))
    out = decode_candles(build_payload(recs), date(2024, 6, 5), point_digits=3)
    bad = [x for x in out if not (x.low <= min(x.open, x.close)
                                  and max(x.open, x.close) <= x.high)]
    assert not bad, f"{len(bad)} of {len(out)} bars violate OHLC constraints"


def test_timestamps_are_utc_and_offset_from_day_start() -> None:
    recs = [(0, 1.0, 1.0, 1.0, 1.0, 1.0), (3600, 1.0, 1.0, 1.0, 1.0, 1.0)]
    out = decode_candles(build_payload(recs), date(2024, 6, 5), point_digits=3)
    assert out[0].ts == datetime(2024, 6, 5, 0, 0, tzinfo=UTC)
    assert out[1].ts == datetime(2024, 6, 5, 1, 0, tzinfo=UTC)
    assert all(c.ts.utcoffset() == timedelta(0) for c in out)


def test_empty_payload_and_bad_length() -> None:
    from xau_data.errors import DecodeError
    assert decode_candles(b"", date(2024, 6, 5), point_digits=3) == []
    bad = lzma.compress(b"\x00" * 23, format=lzma.FORMAT_ALONE)
    with pytest.raises(DecodeError, match="multiple"):
        decode_candles(bad, date(2024, 6, 5), point_digits=3)


# --- synthetic stripping --------------------------------------------------


def c(ts_min: int, o: float, hi: float, lo: float, cl: float, vol: float) -> Candle:
    return Candle(datetime(2024, 6, 5, tzinfo=UTC) + timedelta(minutes=ts_min),
                  o, hi, lo, cl, vol)


def test_is_synthetic_requires_both_conditions() -> None:
    assert is_synthetic(c(0, 2000, 2000, 2000, 2000, 0.0))          # fill
    assert not is_synthetic(c(0, 2000, 2000, 2000, 2000, 0.5))      # flat but traded
    assert not is_synthetic(c(0, 2000, 2001, 1999, 2000, 0.0))      # zero vol, not flat
    assert not is_synthetic(c(0, 2000, 2001, 1999, 2000.5, 1.0))    # normal


def test_merge_strips_synthetic_and_counts_it() -> None:
    bid = [c(0, 2000, 2001, 1999, 2000.5, 1.0),
           c(1, 2000.5, 2000.5, 2000.5, 2000.5, 0.0),          # fill
           c(2, 2000.5, 2002, 2000, 2001.5, 2.0)]
    ask = [c(0, 2000.4, 2001.4, 1999.4, 2000.9, 1.0),
           c(1, 2000.9, 2000.9, 2000.9, 2000.9, 0.0),          # fill
           c(2, 2000.9, 2002.4, 2000.4, 2001.9, 2.0)]
    rows, stripped = merge_sides(bid, ask)
    assert stripped == 1
    assert len(rows) == 2
    assert [r["ts_open"].minute for r in rows] == [0, 2]
    assert rows[0]["spread_open"] == pytest.approx(0.4)
    assert rows[0]["mid_open"] == pytest.approx(2000.2)


def test_minute_synthetic_on_one_side_only_is_still_stripped() -> None:
    bid = [c(0, 2000, 2001, 1999, 2000.5, 1.0)]
    ask = [c(0, 2000.4, 2000.4, 2000.4, 2000.4, 0.0)]     # ask side is fill
    rows, stripped = merge_sides(bid, ask)
    assert rows == [] and stripped == 1


def test_unpaired_minutes_are_dropped_and_counted() -> None:
    bid = [c(0, 2000, 2001, 1999, 2000.5, 1.0), c(1, 2000, 2001, 1999, 2000.5, 1.0)]
    ask = [c(0, 2000.4, 2001.4, 1999.4, 2000.9, 1.0)]
    rows, stripped = merge_sides(bid, ask)
    assert len(rows) == 1 and stripped == 1


def test_rows_to_table_matches_schema_and_keeps_utc() -> None:
    from xau_data.storage.schemas import BARS_M1
    bid = [c(0, 2000, 2001, 1999, 2000.5, 1.0)]
    ask = [c(0, 2000.4, 2001.4, 1999.4, 2000.9, 1.0)]
    rows, _ = merge_sides(bid, ask)
    t = rows_to_table(rows)
    assert t.schema.equals(BARS_M1)
    assert t.schema.field("ts_open").type.tz == "UTC"


# --- day selection --------------------------------------------------------


def test_trading_days_skips_saturdays_only() -> None:
    days = list(trading_days(date(2024, 6, 3), date(2024, 6, 9)))
    assert date(2024, 6, 8) not in days          # Saturday
    assert date(2024, 6, 9) in days              # Sunday opens the week
    assert len(days) == 6


# --- resampling -----------------------------------------------------------


def make_m1(n: int, start: datetime | None = None) -> pa.Table:
    start = start or datetime(2024, 6, 5, tzinfo=UTC)
    rows = []
    for i in range(n):
        base = 2000 + i * 0.1
        rows.append({
            "ts_open": start + timedelta(minutes=i),
            "ts_end_exclusive": start + timedelta(minutes=i + 1),
            "bid_open": base, "bid_high": base + 0.5,
            "bid_low": base - 0.5, "bid_close": base + 0.1,
            "ask_open": base + 0.4, "ask_high": base + 0.9,
            "ask_low": base - 0.1, "ask_close": base + 0.5,
            "mid_open": base + 0.2, "mid_close": base + 0.3,
            "spread_open": 0.4, "spread_close": 0.4,
            "bid_volume": 1.0, "ask_volume": 1.0,
        })
    return rows_to_table(rows)


def test_resample_m1_to_m15_aggregates_exactly() -> None:
    m1 = make_m1(30)
    m15 = resample_m1(m1, timeframe="M15")
    assert m15.num_rows == 2
    assert m15["m1_count"].to_pylist() == [15, 15]
    b_hi = m1["bid_high"].to_numpy(zero_copy_only=False)
    assert m15["bid_high"][0].as_py() == pytest.approx(b_hi[:15].max())
    assert m15["bid_low"][0].as_py() == pytest.approx(
        m1["bid_low"].to_numpy(zero_copy_only=False)[:15].min())
    assert m15["bid_open"][0].as_py() == pytest.approx(m1["bid_open"][0].as_py())
    assert m15["bid_close"][0].as_py() == pytest.approx(m1["bid_close"][14].as_py())


def test_resample_labels_are_bar_open_on_the_utc_grid() -> None:
    m15 = resample_m1(make_m1(30, datetime(2024, 6, 5, 9, 7, tzinfo=UTC)),
                      timeframe="M15")
    for ts in m15["ts_open"].to_pylist():
        assert ts.minute % 15 == 0 and ts.second == 0
    for o, e in zip(m15["ts_open"].to_pylist(), m15["ts_end_exclusive"].to_pylist()):
        assert e - o == timedelta(minutes=15)


def test_resample_produces_no_row_for_an_empty_window() -> None:
    """A gap must stay a gap, never a filled bar."""
    a = make_m1(15, datetime(2024, 6, 5, 0, 0, tzinfo=UTC))
    b = make_m1(15, datetime(2024, 6, 5, 1, 0, tzinfo=UTC))
    m15 = resample_m1(pa.concat_tables([a, b]), timeframe="M15")
    assert m15.num_rows == 2                        # not 5
    got = [t.hour * 60 + t.minute for t in m15["ts_open"].to_pylist()]
    assert got == [0, 60]


def test_bar_range_refuses_mid() -> None:
    m15 = resample_m1(make_m1(15), timeframe="M15")
    assert bar_range(m15, "bid")[0] == pytest.approx(
        m15["bid_high"][0].as_py() - m15["bid_low"][0].as_py())
    with pytest.raises(ValueError, match="mid extremes are not exact"):
        bar_range(m15, "mid")


def test_resampled_schema_has_no_mid_extremes() -> None:
    from xau_data.transform.resample import RESAMPLED
    assert "mid_high" not in RESAMPLED.names
    assert "mid_low" not in RESAMPLED.names
