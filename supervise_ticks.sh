#!/usr/bin/env bash
cd "$(dirname "$0")"
for i in $(seq 1 200); do
  grep -q 'TICKS_DONE' data/_tick_backfill.log 2>/dev/null && { echo "SUPERVISOR: done"; break; }
  echo "SUPERVISOR: attempt $i $(date -u +%H:%M:%S)" >> data/_tick_backfill.log
  python3 scripts_backfill_ticks.py >> data/_tick_backfill.log 2>&1
  rc=$?; echo "SUPERVISOR: rc=$rc" >> data/_tick_backfill.log
  [ "$rc" -eq 0 ] && break
  sleep 20
done
