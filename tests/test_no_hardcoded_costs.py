"""Structural guard: no dollar cost may be hardcoded anywhere in the package.

Measured 2021-01 to 2026-09, mean M15 spread runs 0.32 to 1.03 USD/oz as gold
goes from 1700 to 5000, while relative cost stays at 1.67-2.20 bps with no
trend. Spread scales with price. Any constant in dollars is a window artifact
of whatever sample it was taken from, and the earlier "flat spread" conclusion
in this repo was exactly that mistake.

Every cost reference must be computed per fill from the bar being filled on.
"""
from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ("xau_data", "xau_backtest")

#: Literals in this range are suspicious near cost-shaped identifiers: they are
#: the observed spread and slippage magnitudes in USD/oz.
_LOW, _HIGH = 0.005, 5.0

_COST_WORDS = (
    "spread", "slippage", "slip", "cost", "commission", "fee",
    "expectancy", "tick_value", "pip",
)


def _cost_literals(path: Path) -> list[str]:
    src = path.read_text()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:  # pragma: no cover
        return []
    code = [
        tk for tk in toks
        if tk.type not in (tokenize.COMMENT, tokenize.STRING, tokenize.NL,
                           tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
    ]
    out: list[str] = []
    for i, tk in enumerate(code):
        if tk.type != tokenize.NUMBER:
            continue
        try:
            val = float(tk.string)
        except ValueError:
            continue
        if not (_LOW <= abs(val) <= _HIGH) or val == int(val):
            continue
        window = "".join(t.string for t in code[max(0, i - 8):i + 4]).lower()
        if any(w in window for w in _COST_WORDS):
            out.append(f"{path.relative_to(REPO)}:{tk.start[0]}: "
                       f"{tk.string} near {window[:70]!r}")
    return out


@pytest.mark.parametrize("package", PACKAGES)
def test_no_hardcoded_dollar_cost(package: str) -> None:
    base = REPO / package
    if not base.exists():
        pytest.skip(f"{package} not present")
    offenders: list[str] = []
    for path in base.rglob("*.py"):
        offenders += _cost_literals(path)
    assert not offenders, (
        "hardcoded dollar cost in executable code; costs must be computed per "
        "fill from the data:\n" + "\n".join(offenders)
    )


def test_slippage_comes_from_config_and_has_no_default() -> None:
    """add_cost_features must REQUIRE slippage, never default it."""
    import inspect

    from xau_data.transform import features

    sig = inspect.signature(features.add_cost_features)
    param = sig.parameters["slippage_per_side"]
    assert param.default is inspect.Parameter.empty, (
        "slippage_per_side has a default; a cost assumption must not be able "
        "to hide as a module constant"
    )
    assert not hasattr(features, "DEFAULT_SLIPPAGE_PER_SIDE")


def test_config_exposes_costs_and_scales_with_price() -> None:
    from xau_data.config import load_config

    cfg = load_config(REPO / "configs/config.yaml", REPO / "configs/sessions.yaml")
    assert cfg.costs.slippage_per_side >= 0
    # the relative floor exists so slippage can track the price regime
    assert cfg.costs.slippage_at(5000.0) >= cfg.costs.slippage_at(1800.0)


def test_spread_quality_threshold_is_relative_not_absolute() -> None:
    """A fixed dollar spread cap is tight at 1800 and useless at 5000."""
    import inspect

    from xau_data.quality import checks

    sig = inspect.signature(checks.check_wide_spread)
    assert "max_spread_bps" in sig.parameters
    assert "max_spread" not in sig.parameters
