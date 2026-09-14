"""Unattended M1 candle backfill.

Safe to kill and restart: resume is decided by the store, so nothing already
persisted is refetched.

A stall watchdog exits non-zero if no day completes within a deadline. That is
deliberate: under systemd with Restart=on-failure, a wedged job restarts itself
instead of hanging silently. A sequential downloader with a leaked connection
pool can otherwise sit blocked for hours while looking alive.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import date, datetime, timezone

from xau_data.config import load_config
from xau_data.ingest.dukascopy_candles import ingest_days
from xau_data.storage.store import open_store

START = date.fromisoformat(os.environ.get("XAU_START", "2021-01-01"))
END = date.fromisoformat(
    os.environ.get("XAU_END", datetime.now(timezone.utc).date().isoformat())
)
STALL_SECONDS = int(os.environ.get("XAU_STALL_SECONDS", "600"))


def _watchdog(cfg, store, stop: threading.Event) -> None:
    """Exit the process if no day completes within STALL_SECONDS."""
    last_count = len(store.completed_days(cfg.symbol))
    last_change = time.monotonic()
    while not stop.wait(60):
        try:
            n = len(store.completed_days(cfg.symbol))
        except Exception:
            continue
        if n != last_count:
            last_count, last_change = n, time.monotonic()
            continue
        idle = time.monotonic() - last_change
        if idle > STALL_SECONDS:
            print(
                f"WATCHDOG: no day completed in {idle:.0f}s (limit {STALL_SECONDS}s); "
                f"{n} days done. Exiting non-zero so the supervisor restarts us.",
                flush=True,
            )
            os._exit(75)          # EX_TEMPFAIL


def main() -> int:
    cfg = load_config()
    store = open_store(cfg.data_root)
    print(f"store={store.uri}", flush=True)
    print(f"range={START}..{END}  stall_limit={STALL_SECONDS}s", flush=True)
    done = store.completed_days(cfg.symbol)
    print(f"already complete: {len(done)} days", flush=True)

    stop = threading.Event()
    wd = threading.Thread(target=_watchdog, args=(cfg, store, stop), daemon=True)
    wd.start()

    t0 = time.time()
    try:
        s = ingest_days(cfg, START, END, store=store, progress_every=10,
                        flush_every_days=10)
    finally:
        stop.set()
    el = time.time() - t0
    out = {
        "considered": s.days_considered, "written": s.days_written,
        "skipped": s.days_skipped, "failed": s.days_failed,
        "rows": s.rows, "stripped": s.stripped,
        "elapsed_s": round(el),
        "days_per_hour": round(s.days_written / max(el, 1) * 3600, 1),
    }
    print("BACKFILL_M1_DONE " + json.dumps(out), flush=True)
    (cfg.data_root / "_m1_backfill_result.json").write_text(
        json.dumps({**out, "failures": s.failures[:50]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
