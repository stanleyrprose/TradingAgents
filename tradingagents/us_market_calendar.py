"""Minimal deterministic US equity-market calendar helpers for daily operations."""

from __future__ import annotations

from datetime import date, timedelta

from dateutil.easter import easter


def _observed(day: date) -> date:
    if day.weekday() == 5:  # Saturday -> Friday
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday -> Monday
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    candidate = next_month - timedelta(days=1)
    return candidate - timedelta(days=(candidate.weekday() - weekday) % 7)


def us_equity_market_holidays(year: int) -> frozenset[date]:
    """Return standard full-day US equity-market holidays for one nominal year.

    This intentionally models recurring NYSE-style closures only. Rare one-off
    national mourning/emergency closures are not predicted and therefore remain
    fail-closed operational events.
    """

    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))  # Juneteenth
    return frozenset(holidays)


def is_us_equity_market_day(day: date) -> bool:
    """Return whether a date is a normal full/partial US equity trading day."""

    if not isinstance(day, date):
        raise TypeError("day must be a date")
    if day.weekday() >= 5:
        return False
    # Include adjacent nominal years because an observed New Year's Day can land
    # on Dec 31 of the preceding calendar year.
    holidays = (
        us_equity_market_holidays(day.year - 1)
        | us_equity_market_holidays(day.year)
        | us_equity_market_holidays(day.year + 1)
    )
    return day not in holidays
