"""YAML configuration, parsed into frozen dataclasses and validated at load."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time as dtime
from pathlib import Path
from typing import Any, Final, Mapping
from zoneinfo import ZoneInfo

import yaml

from .errors import ConfigError

__all__ = ["Config", "load_config", "DukascopyConfig", "SessionConfig",
           "SessionWindow", "ContextSeries", "CostConfig"]

_ENV_ROOT: Final[str] = "XAU_DATA_ROOT"


def _req(d: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing required key {key!r} in {where}")
    return d[key]


@dataclass(frozen=True, slots=True)
class DukascopyConfig:
    base_url: str
    month_zero_indexed: bool
    requests_per_second: float
    max_concurrency: int
    max_attempts: int
    backoff_base_seconds: float
    backoff_max_seconds: float
    connect_timeout_seconds: float
    read_timeout_seconds: float
    keep_raw: bool


@dataclass(frozen=True, slots=True)
class ContextSeries:
    series_id: str
    source: str
    source_series_id: str
    rule: str
    tz: str
    at: dtime

    def __post_init__(self) -> None:
        if self.rule not in {"vintage", "market_close"}:
            raise ConfigError(
                f"series {self.series_id}: unknown publication rule {self.rule!r}"
            )
        try:
            ZoneInfo(self.tz)
        except Exception as exc:
            raise ConfigError(f"series {self.series_id}: bad timezone {self.tz!r}") from exc


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Cost model inputs. Spread is never here: it is read per fill."""

    slippage_per_side: float
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.slippage_per_side < 0 or self.slippage_bps < 0:
            raise ConfigError("slippage must be non-negative")

    def slippage_at(self, price: float) -> float:
        """Adverse slippage per side at this price level."""
        return max(self.slippage_per_side, price * self.slippage_bps / 10_000.0)


@dataclass(frozen=True, slots=True)
class SessionWindow:
    name: str
    tz: str
    start: dtime
    end: dtime


@dataclass(frozen=True, slots=True)
class SessionConfig:
    timezone: str
    week_open_weekday: str
    week_open_time: dtime
    week_close_weekday: str
    week_close_time: dtime
    break_start: dtime
    break_end: dtime
    holidays_file: Path | None
    windows: tuple[SessionWindow, ...] = ()


@dataclass(frozen=True, slots=True)
class Config:
    data_root: Path
    symbol: str
    point_digits: int
    plausible_price_range: tuple[float, float]
    timeframes: tuple[str, ...]
    dukascopy: DukascopyConfig
    context_series: tuple[ContextSeries, ...]
    context_safety_lag_seconds: int
    join_key: str
    allow_exact_matches: bool
    quality: Mapping[str, Any]
    sessions: SessionConfig
    costs: CostConfig
    source_files: tuple[Path, ...] = field(default=())

    # ---- derived paths ----
    @property
    def raw_dir(self) -> Path: return self.data_root / "raw" / "dukascopy" / self.symbol
    @property
    def ticks_dir(self) -> Path: return self.data_root / "ticks"
    @property
    def bars_dir(self) -> Path: return self.data_root / "bars"
    @property
    def context_dir(self) -> Path: return self.data_root / "context"
    @property
    def calendar_dir(self) -> Path: return self.data_root / "calendar"
    @property
    def manifest_path(self) -> Path: return self.data_root / "_manifests" / "dukascopy.parquet"
    @property
    def reports_dir(self) -> Path: return self.data_root / "_reports"
    @property
    def duckdb_path(self) -> Path: return self.data_root / "xau.duckdb"


def _parse_time(s: str, where: str) -> dtime:
    try:
        hh, mm = s.split(":")
        return dtime(int(hh), int(mm))
    except Exception as exc:
        raise ConfigError(f"{where}: bad time {s!r}, expected HH:MM") from exc


def load_config(
    path: str | Path = "configs/config.yaml",
    sessions_path: str | Path = "configs/sessions.yaml",
) -> Config:
    """Load and validate configuration. XAU_DATA_ROOT overrides data_root."""
    path, sessions_path = Path(path), Path(sessions_path)
    for p in (path, sessions_path):
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
    raw = yaml.safe_load(path.read_text()) or {}
    sess = yaml.safe_load(sessions_path.read_text()) or {}

    root = os.environ.get(_ENV_ROOT) or _req(_req(raw, "paths", "config"), "data_root", "paths")
    sym = _req(raw, "symbol", "config")
    dk = _req(_req(raw, "ingest", "config"), "dukascopy", "ingest")

    ctx_cfg = raw.get("ingest", {}).get("context", {}) or {}
    series: list[ContextSeries] = []
    for sid, spec in (ctx_cfg.get("series") or {}).items():
        pub = _req(spec, "publication", f"context series {sid}")
        series.append(ContextSeries(
            series_id=sid,
            source=_req(spec, "source", f"context series {sid}"),
            source_series_id=_req(spec, "source_series_id", f"context series {sid}"),
            rule=_req(pub, "rule", f"context series {sid} publication"),
            tz=_req(pub, "tz", f"context series {sid} publication"),
            at=_parse_time(_req(pub, "at", f"context series {sid} publication"),
                           f"context series {sid}"),
        ))

    join = raw.get("join", {}).get("context", {}) or {}
    key = join.get("key", "publication_time")
    if key != "publication_time":
        raise ConfigError(
            f"join.context.key must be 'publication_time', got {key!r}. "
            "Joining context on observation_time leaks the future."
        )

    tfs = tuple(raw.get("timeframes") or ())
    if not tfs:
        raise ConfigError("timeframes must list at least one timeframe")

    db = sess.get("daily_break") or {}
    hol = sess.get("holidays_file")
    windows: list[SessionWindow] = []
    for wname, w in (sess.get("windows") or {}).items():
        windows.append(SessionWindow(
            name=wname,
            tz=_req(w, "tz", f"session window {wname}"),
            start=_parse_time(_req(w, "start", f"session window {wname}"), wname),
            end=_parse_time(_req(w, "end", f"session window {wname}"), wname),
        ))
        try:
            ZoneInfo(windows[-1].tz)
        except Exception as exc:
            raise ConfigError(f"session window {wname}: bad timezone") from exc
    session = SessionConfig(
        timezone=_req(sess, "timezone", "sessions"),
        week_open_weekday=_req(_req(sess, "week_open", "sessions"), "weekday", "week_open"),
        week_open_time=_parse_time(_req(_req(sess, "week_open", "sessions"), "time", "week_open"), "week_open"),
        week_close_weekday=_req(_req(sess, "week_close", "sessions"), "weekday", "week_close"),
        week_close_time=_parse_time(_req(_req(sess, "week_close", "sessions"), "time", "week_close"), "week_close"),
        break_start=_parse_time(_req(db, "start", "daily_break"), "daily_break"),
        break_end=_parse_time(_req(db, "end", "daily_break"), "daily_break"),
        holidays_file=Path(hol) if hol else None,
        windows=tuple(windows),
    )
    try:
        ZoneInfo(session.timezone)
    except Exception as exc:
        raise ConfigError(f"sessions.timezone invalid: {session.timezone!r}") from exc

    pr = sym.get("plausible_price_range", [0.0, 1e9])
    return Config(
        data_root=Path(root).expanduser().resolve(),
        symbol=_req(sym, "name", "symbol"),
        point_digits=int(_req(sym, "point_digits", "symbol")),
        plausible_price_range=(float(pr[0]), float(pr[1])),
        timeframes=tfs,
        dukascopy=DukascopyConfig(
            base_url=_req(dk, "base_url", "dukascopy"),
            month_zero_indexed=bool(dk.get("month_zero_indexed", True)),
            requests_per_second=float(dk.get("requests_per_second", 1.0)),
            max_concurrency=int(dk.get("max_concurrency", 2)),
            max_attempts=int(dk.get("max_attempts", 6)),
            backoff_base_seconds=float(dk.get("backoff_base_seconds", 2.0)),
            backoff_max_seconds=float(dk.get("backoff_max_seconds", 45.0)),
            connect_timeout_seconds=float(dk.get("connect_timeout_seconds", 25.0)),
            read_timeout_seconds=float(dk.get("read_timeout_seconds", 60.0)),
            keep_raw=bool(dk.get("keep_raw", True)),
        ),
        context_series=tuple(series),
        context_safety_lag_seconds=int(ctx_cfg.get("safety_lag_seconds", 0)),
        join_key=key,
        allow_exact_matches=bool(join.get("allow_exact_matches", False)),
        quality=raw.get("quality", {}) or {},
        sessions=session,
        costs=CostConfig(
            slippage_per_side=float(_req(raw.get("costs", {}) or {},
                                         "slippage_per_side", "costs")),
            slippage_bps=float((raw.get("costs", {}) or {}).get("slippage_bps", 0.0)),
        ),
        source_files=(path, sessions_path),
    )
