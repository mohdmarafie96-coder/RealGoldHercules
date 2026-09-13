"""Rollover placement across every DST transition in the dataset's span.

The session calendar is the single source of truth for rollover timing. These
tests pin that 17:00 New York lands on 21:00 UTC in EDT and 22:00 UTC in EST,
on both transition weeks of every year, and that nothing is duplicated or
dropped across the change.
"""
from __future__ import annotations

import re
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from xau_data.config import load_config
from xau_data.transform.sessions import SessionCalendar, SessionLabel

UTC = timezone.utc
ET = ZoneInfo("America/New_York")
REPO = Path(__file__).resolve().parents[1]

#: Every year the dataset spans, 2021-01 through the present.
YEARS = tuple(range(2021, 2027))


@pytest.fixture(scope="module")
def cal() -> SessionCalendar:
    return SessionCalendar(
        load_config(REPO / "configs/config.yaml", REPO / "configs/sessions.yaml").sessions
    )


def nth_sunday(year: int, month: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)      # first Sunday
    return d + timedelta(weeks=n - 1)


def spring_forward(year: int) -> date:
    """US DST starts the second Sunday in March."""
    return nth_sunday(year, 3, 2)


def fall_back(year: int) -> date:
    """US DST ends the first Sunday in November."""
    return nth_sunday(year, 11, 1)


def is_edt(d: date) -> bool:
    return datetime(d.year, d.month, d.day, 12, tzinfo=ET).utcoffset() == timedelta(hours=-4)


def transition_week(d: date) -> list[date]:
    return [d + timedelta(days=k) for k in range(-3, 5)]


# --- the core assertion ---------------------------------------------------


@pytest.mark.parametrize("year", YEARS)
@pytest.mark.parametrize("which", ["march", "november"])
def test_rollover_utc_hour_across_transition_week(
    cal: SessionCalendar, year: int, which: str
) -> None:
    """17:00 New York is 21:00 UTC in EDT and 22:00 UTC in EST, every day."""
    pivot = spring_forward(year) if which == "march" else fall_back(year)
    seen_edt = seen_est = False
    for d in transition_week(pivot):
        roll = cal.rollover_utc(d)
        assert roll.tzinfo is not None and roll.utcoffset() == timedelta(0)
        expected = 21 if is_edt(d) else 22
        assert roll.hour == expected and roll.minute == 0, (
            f"{d} ({'EDT' if is_edt(d) else 'EST'}): rollover landed at "
            f"{roll:%H:%M} UTC, expected {expected:02d}:00"
        )
        # and it really is 17:00 local, which is the definition
        assert roll.astimezone(ET).hour == 17
        seen_edt |= is_edt(d)
        seen_est |= not is_edt(d)
    assert seen_edt and seen_est, f"{which} {year} week did not straddle the change"


@pytest.mark.parametrize("year", YEARS)
@pytest.mark.parametrize("which", ["march", "november"])
def test_rollover_shifts_by_exactly_one_hour(
    cal: SessionCalendar, year: int, which: str
) -> None:
    pivot = spring_forward(year) if which == "march" else fall_back(year)
    before = cal.rollover_utc(pivot - timedelta(days=1))
    after = cal.rollover_utc(pivot + timedelta(days=1))
    delta = after.hour - before.hour
    expected = -1 if which == "march" else +1
    assert delta == expected, (
        f"{which} {year}: rollover hour moved {delta:+d}, expected {expected:+d}"
    )


@pytest.mark.parametrize("year", YEARS)
@pytest.mark.parametrize("which", ["march", "november"])
def test_no_duplicate_or_missing_rollover_across_transition(
    cal: SessionCalendar, year: int, which: str
) -> None:
    """Exactly one rollover per calendar day, strictly increasing."""
    pivot = spring_forward(year) if which == "march" else fall_back(year)
    days = transition_week(pivot)
    start = datetime.combine(days[0], datetime.min.time(), tzinfo=UTC)
    end = datetime.combine(days[-1] + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    crossed = cal.rollovers_crossed(start, end)
    assert len(crossed) == len(set(crossed)), "duplicate rollover instants"
    assert crossed == sorted(crossed), "rollovers out of order"
    assert len(crossed) == len(days), (
        f"{which} {year}: {len(crossed)} rollovers across {len(days)} days"
    )
    # spacing is 24h everywhere except the transition itself
    gaps = {(b - a) for a, b in zip(crossed, crossed[1:])}
    odd = gaps - {timedelta(hours=24)}
    expected_odd = {timedelta(hours=23)} if which == "march" else {timedelta(hours=25)}
    assert odd == expected_odd, f"{which} {year}: unexpected gaps {odd}"


@pytest.mark.parametrize("year", YEARS)
@pytest.mark.parametrize("which", ["march", "november"])
@pytest.mark.parametrize("period", [60, 300, 900, 3600])
def test_forced_exit_deadline_tracks_the_rollover(
    cal: SessionCalendar, year: int, which: str, period: int
) -> None:
    """The forced-exit bar must close exactly at the rollover, in both regimes."""
    pivot = spring_forward(year) if which == "march" else fall_back(year)
    for d in transition_week(pivot):
        noon = datetime(d.year, d.month, d.day, 12, tzinfo=UTC)
        roll = cal.next_rollover(noon)
        last_open = cal.last_bar_open_before_rollover(noon, period)
        close = last_open + timedelta(seconds=period)
        assert close == roll, (
            f"{d} period={period}: forced-exit bar closes {close:%H:%M}, "
            f"rollover is {roll:%H:%M}"
        )
        assert last_open < roll


@pytest.mark.parametrize("year", YEARS)
def test_swap_charged_once_per_rollover_crossed(cal: SessionCalendar, year: int) -> None:
    """A position held across N rollovers crosses exactly N, at either DST regime."""
    for pivot in (spring_forward(year), fall_back(year)):
        start = datetime.combine(pivot - timedelta(days=2), datetime.min.time(), tzinfo=UTC)
        for days in (1, 2, 3, 5):
            end = start + timedelta(days=days)
            crossed = cal.rollovers_crossed(start, end)
            assert len(crossed) == days, (
                f"{year} around {pivot}: held {days}d, crossed {len(crossed)} rollovers"
            )
            # idempotent: asking twice must not double count
            assert cal.rollovers_crossed(start, end) == crossed


def test_rollover_boundary_is_half_open(cal: SessionCalendar) -> None:
    """(start, end]: a rollover exactly at start is not crossed, at end it is."""
    roll = cal.rollover_utc(date(2024, 6, 5))
    assert cal.rollovers_crossed(roll, roll + timedelta(hours=1)) == []
    assert cal.rollovers_crossed(roll - timedelta(hours=1), roll) == [roll]


@pytest.mark.parametrize("year", YEARS)
@pytest.mark.parametrize("which", ["march", "november"])
def test_session_labels_survive_the_transition(
    cal: SessionCalendar, year: int, which: str
) -> None:
    """Labels stay well-formed across the change; no hour is unlabelled mid-session."""
    pivot = spring_forward(year) if which == "march" else fall_back(year)
    for d in transition_week(pivot):
        if d.weekday() >= 5:
            continue
        for h in range(24):
            ts = datetime(d.year, d.month, d.day, h, tzinfo=UTC)
            label = cal.session_label(ts)
            assert isinstance(label, SessionLabel)
            if label is not SessionLabel.OFF:
                assert cal.is_tradeable(ts)


# --- structural guard -----------------------------------------------------


def test_no_hardcoded_rollover_hour_in_codebase() -> None:
    """No module may encode the rollover as a UTC hour constant.

    Comments and docstrings are excluded by tokenising rather than by pattern,
    because prose legitimately names the hours when explaining why they must
    not be hardcoded.
    """
    import io
    import tokenize

    offenders: list[str] = []
    suspicious = re.compile(r"^2[123]$")
    for path in (REPO / "xau_data").rglob("*.py"):
        src = path.read_text()
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except tokenize.TokenError:  # pragma: no cover
            continue
        code = [tk for tk in toks
                if tk.type not in (tokenize.COMMENT, tokenize.STRING,
                                   tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                   tokenize.DEDENT)]
        for i, tk in enumerate(code):
            if tk.type != tokenize.NUMBER or not suspicious.match(tk.string):
                continue
            window = "".join(t.string for t in code[max(0, i - 6):i + 3]).lower()
            if any(w in window for w in ("rollover", "hour", "utc", "roll")):
                offenders.append(
                    f"{path.relative_to(REPO)}:{tk.start[0]}: "
                    f"literal {tk.string} near {window[:60]!r}"
                )
    assert not offenders, (
        "hardcoded rollover hour in executable code:\n" + "\n".join(offenders)
    )


def test_session_calendar_is_the_only_rollover_source() -> None:
    """Any module computing rollover must go through SessionCalendar."""
    for path in (REPO / "xau_data").rglob("*.py"):
        if path.name == "sessions.py":
            continue
        text = path.read_text()
        if "rollover" in text.lower() and "def " in text:
            assert ("SessionCalendar" in text or "calendar." in text
                    or path.name in {"schemas.py", "config.py"}), (
                f"{path.relative_to(REPO)} mentions rollover without using "
                "SessionCalendar"
            )
