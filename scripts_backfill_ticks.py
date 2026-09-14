"""Tick slice backfill for the approved six months.

Resume-safe, one month at a time so memory stays bounded.

The stall watchdog counts RAW FILES ON DISK, not completed months. An earlier
version incremented only when a whole month finished, so it fired on a healthy
job that simply had not finished a 744-hour month within the timeout. A
watchdog that kills working jobs is worse than no watchdog.
"""
from __future__ import annotations

import json, os, threading, time
from datetime import datetime, timezone
from pathlib import Path

from xau_data.config import load_config
from xau_data.ingest.dukascopy import ingest_range

MONTHS = [(2026, 1), (2022, 2), (2021, 8), (2024, 3), (2025, 10), (2025, 7)]
STALL = int(os.environ.get("XAU_STALL_SECONDS", "900"))
UTC = timezone.utc


def _bounds(y: int, m: int):
    return (datetime(y, m, 1, tzinfo=UTC),
            datetime(y + (m == 12), (m % 12) + 1, 1, tzinfo=UTC))


def _raw_count(root: Path) -> int:
    try:
        return sum(1 for _ in (root / "raw" / "dukascopy").rglob("*.bi5"))
    except OSError:
        return -1


def main() -> int:
    cfg = load_config()
    stop = threading.Event()

    def watchdog() -> None:
        last, last_t = _raw_count(cfg.data_root), time.monotonic()
        while not stop.wait(60):
            n = _raw_count(cfg.data_root)
            if n != last:
                last, last_t = n, time.monotonic()
                continue
            idle = time.monotonic() - last_t
            if idle > STALL:
                print(f"WATCHDOG: no new raw file in {idle:.0f}s at {n} files; "
                      "exiting 75 for the supervisor to restart", flush=True)
                os._exit(75)

    threading.Thread(target=watchdog, daemon=True).start()

    results = []
    for y, m in MONTHS:
        start, end = _bounds(y, m)
        print(f"=== {y}-{m:02d} === (raw files now {_raw_count(cfg.data_root)})", flush=True)
        t0 = time.time()
        s = ingest_range(cfg, start, end, progress=True)
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
