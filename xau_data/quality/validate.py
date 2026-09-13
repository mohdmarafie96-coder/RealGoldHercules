"""Validation orchestration and markdown reporting.

Report only. Nothing here fixes, fills, drops or reorders data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa

from ..config import Config
from ..timeutils import UTC
from ..transform.sessions import SessionCalendar
from .checks import Finding, run_checks

__all__ = ["validate_bars", "ValidationReport", "write_report"]

_SEV_ORDER = {"error": 0, "warn": 1, "info": 2}
_ICON = {"error": "FAIL", "warn": "WARN", "info": "ok"}


@dataclass(slots=True)
class ValidationReport:
    symbol: str
    generated_at: datetime
    per_timeframe: dict[str, list[Finding]]
    coverage: dict[str, Any]

    @property
    def errors(self) -> int:
        return sum(1 for fs in self.per_timeframe.values()
                   for f in fs if f.severity == "error" and f.count)

    @property
    def warnings(self) -> int:
        return sum(1 for fs in self.per_timeframe.values()
                   for f in fs if f.severity == "warn" and f.count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "generated_at": self.generated_at.isoformat(),
            "errors": self.errors,
            "warnings": self.warnings,
            "coverage": self.coverage,
            "findings": {tf: [f.as_dict() for f in fs]
                         for tf, fs in self.per_timeframe.items()},
        }


def validate_bars(
    cfg: Config,
    bars_by_timeframe: Mapping[str, pa.Table],
    *,
    coverage: dict[str, Any] | None = None,
) -> ValidationReport:
    cal = SessionCalendar(cfg.sessions)
    q = cfg.quality
    out: dict[str, list[Finding]] = {}
    for tf, tbl in bars_by_timeframe.items():
        out[tf] = run_checks(
            tbl,
            calendar=cal,
            timeframe=tf,
            spike_sigma=float(q.get("spike_sigma", 8.0)),
            min_tick_count=q.get("min_tick_count", {}),
            max_spread=float(q.get("max_spread", 25.0)),
        )
    return ValidationReport(cfg.symbol, datetime.now(UTC), out, coverage or {})


def render_markdown(rep: ValidationReport) -> str:
    L: list[str] = []
    L.append(f"# Data quality report — {rep.symbol}")
    L.append("")
    L.append(f"Generated {rep.generated_at:%Y-%m-%d %H:%M:%S} UTC")
    L.append("")
    L.append(f"**{rep.errors} error-severity findings, {rep.warnings} warnings.** "
             "This report does not modify data. Every gap listed below is reported, "
             "not filled.")
    L.append("")
    if rep.coverage:
        L.append("## Coverage")
        L.append("")
        L.append("| Property | Value |")
        L.append("|---|---|")
        for k, v in rep.coverage.items():
            L.append(f"| {k} | {v} |")
        L.append("")
    for tf, findings in rep.per_timeframe.items():
        L.append(f"## {tf}")
        L.append("")
        L.append("| Check | Severity | Count | Detail |")
        L.append("|---|---|---:|---|")
        for f in sorted(findings, key=lambda x: (_SEV_ORDER[x.severity], -x.count)):
            sev = _ICON[f.severity] if f.count else "ok"
            L.append(f"| `{f.check}` | {sev} | {f.count} | {f.message} |")
        L.append("")
        detailed = [f for f in findings if f.count and f.samples]
        if detailed:
            L.append(f"### {tf} samples")
            L.append("")
            for f in detailed:
                L.append(f"**{f.check}** ({f.count})")
                L.append("")
                for s in f.samples[:10]:
                    L.append(f"- `{s}`")
                if f.count > len(f.samples[:10]):
                    L.append(f"- ... and {f.count - len(f.samples[:10])} more")
                L.append("")
    return "\n".join(L)


def write_report(cfg: Config, rep: ValidationReport, label: str) -> tuple[Path, Path]:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    md = cfg.reports_dir / f"validation_{label}.md"
    js = cfg.reports_dir / f"validation_{label}.json"
    md.write_text(render_markdown(rep))
    js.write_text(json.dumps(rep.as_dict(), indent=2, default=str))
    return md, js
