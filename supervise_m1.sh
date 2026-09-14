#!/usr/bin/env bash
# Restart the backfill whenever it exits non-zero (watchdog stall or crash).
# Mirrors systemd Restart=on-failure for this container.
cd "$(dirname "$0")"
for i in $(seq 1 200); do
  if grep -q 'BACKFILL_M1_DONE' data/_m1_backfill.log 2>/dev/null; then
    echo "SUPERVISOR: backfill complete"; break
  fi
  echo "SUPERVISOR: start attempt $i at $(date -u +%H:%M:%S)" >> data/_m1_backfill.log
  python3 scripts_backfill_m1.py >> data/_m1_backfill.log 2>&1
  rc=$?
  echo "SUPERVISOR: exited rc=$rc" >> data/_m1_backfill.log
  [ "$rc" -eq 0 ] && break
  sleep 15
done
