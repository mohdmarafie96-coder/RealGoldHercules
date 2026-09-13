# Durability snapshot

The execution container is ephemeral: `data/` does not survive the session.
These are the derived M15 and H1 bars plus the validation report, committed so
the month of ingest is not lost to container reclamation.

Not committed: raw `.bi5` payloads (20 MB/month) and tick Parquet (33 MB/month).
For those, set `XAU_DATA_ROOT` to a persistent volume. See
`xau_data/storage/README.md`.

Rebuild the full tree from bars with:

    export XAU_DATA_ROOT=/your/persistent/path
    python -m xau_data.cli views
