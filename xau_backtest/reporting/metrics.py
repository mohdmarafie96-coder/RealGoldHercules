"""Performance metrics over a blotter and equity curve.

Two conventions pinned here rather than argued about later:

- Sharpe and Sortino are computed on the BAR-level equity curve and annualised
  by the number of tradeable bars per year, not by 252. The bar grid is what the
  engine actually steps on.
- Cost drag is expressed against GROSS P&L, and is reported as undefined rather
  than as a misleading percentage when gross is near zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from ..engine.position import ClosedTrade

__all__ = ["Metrics", "compute_metrics", "render_metrics", "by_session", "by_year"]


@dataclass(frozen=True, slots=True)
class Metrics:
    trades: int
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_duration_bars: int
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    expectancy_per_unit: float
    exposure: float
    gross_pnl: float
    cost_total: float
    cost_drag: float
    spread_cost: float
    slippage_cost: float
    swap_cost: float

    def as_dict(self) -> dict[str, float | int]:
        return {f: getattr(self, f) for f in self.__slots__}


def _annualisation(ts: Sequence[datetime]) -> float:
    """Bars per year, measured from the series rather than assumed."""
    if len(ts) < 3:
        return 1.0
    span = (ts[-1] - ts[0]).total_seconds()
    if span <= 0:
        return 1.0
    per_second = (len(ts) - 1) / span
    return per_second * 365.25 * 24 * 3600


def compute_metrics(
    trades: Sequence[ClosedTrade],
    equity_ts: Sequence[datetime],
    equity: Sequence[float],
    starting_cash: float,
    *,
    risk_free_rate: float = 0.0,
) -> Metrics:
    eq = np.asarray(equity, dtype=float)
    n = len(eq)
    net = np.array([t.net_pnl for t in trades], dtype=float) if trades else np.zeros(0)
    gross = float(sum(t.gross_pnl for t in trades))
    spread_c = float(sum(t.spread_cost for t in trades))
    slip_c = float(sum(t.slippage_cost for t in trades))
    swap_c = float(sum(t.swap_cost for t in trades))
    cost_total = spread_c + slip_c + swap_c

    total_return = (eq[-1] / starting_cash - 1.0) if n else 0.0

    years = 0.0
    if len(equity_ts) >= 2:
        years = (equity_ts[-1] - equity_ts[0]).total_seconds() / (365.25 * 24 * 3600)
    cagr = ((eq[-1] / starting_cash) ** (1 / years) - 1.0) if years > 0 and eq[-1] > 0 else 0.0

    if n > 2:
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
        bars_per_year = _annualisation(equity_ts)
        excess = rets - risk_free_rate / max(bars_per_year, 1.0)
        sd = float(np.std(excess, ddof=1))
        sharpe = float(np.mean(excess) / sd * math.sqrt(bars_per_year)) if sd > 0 else 0.0
        downside = excess[excess < 0]
        dsd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sortino = float(np.mean(excess) / dsd * math.sqrt(bars_per_year)) if dsd > 0 else 0.0
    else:
        sharpe = sortino = 0.0

    if n:
        peak = np.maximum.accumulate(eq)
        dd = eq / np.maximum(peak, 1e-9) - 1.0
        max_dd = float(dd.min())
        under = dd < 0
        longest = cur = 0
        for flag in under:
            cur = cur + 1 if flag else 0
            longest = max(longest, cur)
    else:
        max_dd, longest = 0.0, 0

    wins = net[net > 0]
    losses = net[net < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    bars_in = sum(t.bars_held for t in trades)
    exposure = bars_in / n if n else 0.0

    per_unit = float(np.mean([t.net_pnl / t.size for t in trades])) if trades else 0.0

    return Metrics(
        trades=len(trades),
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_bars=longest,
        profit_factor=profit_factor,
        win_rate=float(len(wins) / len(net)) if len(net) else 0.0,
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        expectancy=float(net.mean()) if len(net) else 0.0,
        expectancy_per_unit=per_unit,
        exposure=exposure,
        gross_pnl=gross,
        cost_total=cost_total,
        cost_drag=(cost_total / abs(gross)) if abs(gross) > 1e-9 else float("nan"),
        spread_cost=spread_c,
        slippage_cost=slip_c,
        swap_cost=swap_c,
    )


def by_session(trades: Sequence[ClosedTrade]) -> dict[str, dict[str, float]]:
    out: dict[str, list[ClosedTrade]] = {}
    for t in trades:
        out.setdefault(t.session, []).append(t)
    return {k: _summary(v) for k, v in sorted(out.items())}


def by_year(trades: Sequence[ClosedTrade]) -> dict[str, dict[str, float]]:
    out: dict[str, list[ClosedTrade]] = {}
    for t in trades:
        out.setdefault(str(t.entry_ts.year), []).append(t)
    return {k: _summary(v) for k, v in sorted(out.items())}


def _summary(ts: Sequence[ClosedTrade]) -> dict[str, float]:
    net = np.array([t.net_pnl for t in ts], dtype=float)
    return {
        "trades": len(ts),
        "expectancy": float(net.mean()),
        "net_pnl": float(net.sum()),
        "win_rate": float((net > 0).mean()),
        "cost": float(sum(t.cost_total for t in ts)),
    }


def render_metrics(m: Metrics, title: str = "Backtest") -> str:
    L = [f"{title}", "-" * len(title)]
    rows = [
        ("trades", f"{m.trades:,}"),
        ("total return", f"{m.total_return*100:+.2f}%"),
        ("CAGR", f"{m.cagr*100:+.2f}%"),
        ("Sharpe", f"{m.sharpe:.2f}"),
        ("Sortino", f"{m.sortino:.2f}"),
        ("max drawdown", f"{m.max_drawdown*100:.2f}%"),
        ("drawdown duration", f"{m.max_drawdown_duration_bars:,} bars"),
        ("profit factor", f"{m.profit_factor:.3f}"),
        ("win rate", f"{m.win_rate*100:.1f}%"),
        ("avg win / avg loss", f"{m.avg_win:+.3f} / {m.avg_loss:+.3f}"),
        ("expectancy per trade", f"{m.expectancy:+.4f}"),
        ("expectancy per unit", f"{m.expectancy_per_unit:+.4f}"),
        ("exposure", f"{m.exposure*100:.1f}%"),
        ("gross P&L", f"{m.gross_pnl:+,.2f}"),
        ("cost total", f"{m.cost_total:,.2f}"),
        ("cost drag vs gross", "undefined" if math.isnan(m.cost_drag)
         else f"{m.cost_drag*100:.1f}%"),
        ("  spread", f"{m.spread_cost:,.2f}"),
        ("  slippage", f"{m.slippage_cost:,.2f}"),
        ("  swap", f"{m.swap_cost:,.2f}"),
    ]
    for k, v in rows:
        L.append(f"  {k:<22} {v:>16}")
    return "\n".join(L)
