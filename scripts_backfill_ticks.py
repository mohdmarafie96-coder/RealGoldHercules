"""Tick slice backfill for the approved six months. Resume-safe, one month at a
time so memory stays bounded. Watchdog exits non-zero on stall."""
from __future__ import annotations

import json, os, threading, time
from datetime import datetime, timezone

from xau_data.config import load_config
from xau_data.ingest.dukascopy import ingest_range

MONTHS = [(2026, 1), (2022, 2), (2021, 8), (2024, 3), (2025, 10), (2025, 7)]
STALL = int(os.environ.get("XAU_STALL_SECONDS", "900"))
UTC = timezone.utc


def _bounds(y: int, m: int):
    return (datetime(y, m, 1, tzinfo=UTC),
            datetime(y + (m == 12), (m % 12) + 1, 1, tzinfo=UTC))


def main() -> int:
    cfg = load_config()
    counter = {"n": 0}
    stop = threading.Event()

    def watchdog() -> None:
        last, last_t = 0, time.monotonic()
        while not stop.wait(60):
            if counter["n"] != last:
                last, last_t = counter["n"], time.monotonic()
            elif time.monotonic() - last_t > STALL:
                print(f"WATCHDOG: no progress in {STALL}s, exiting 75", flush=True)
                os._exit(75)

    threading.Thread(target=watchdog, daemon=True).start()
    results = []
    for y, m in MONTHS:
        start, end = _bounds(y, m)
        print(f"=== {y}-{m:02d} ===", flush=True)
        t0 = time.time()
        s = ingest_range(cfg, start, end, progress=True)
        counter["n"] += s.hours_considered
        r = {"month": f"{y}-{m:02d}", "hours": s.hours_considered,
             "fetched": s.fetched, "skipped": s.skipped, "empty": s.empty,
             "missing": s.missing, "failed": s.failed, "ticks": s.ticks,
             "elapsed_s": round(time.time() - t0)}
        results.append(r)
        print("MONTH_DONE " + json.dumps(r), flush=True)
    stop.set()
    print("TICKS_DONE " + json.dumps(results), flush=True)
    (cfg.data_root / "_tick_backfill_result.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
