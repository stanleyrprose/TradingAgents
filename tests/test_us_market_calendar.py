from datetime import date

from tradingagents.us_market_calendar import is_us_equity_market_day, us_equity_market_holidays


def test_standard_2026_us_equity_holidays_are_closed():
    expected = {
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    }
    assert expected <= set(us_equity_market_holidays(2026))
    assert all(not is_us_equity_market_day(day) for day in expected)


def test_weekends_are_closed_and_regular_weekday_is_open():
    assert not is_us_equity_market_day(date(2026, 9, 12))
    assert not is_us_equity_market_day(date(2026, 9, 13))
    assert is_us_equity_market_day(date(2026, 9, 11))


def test_early_close_day_remains_a_market_day():
    assert is_us_equity_market_day(date(2026, 11, 27))


def test_observed_new_year_can_close_prior_calendar_year():
    assert not is_us_equity_market_day(date(2021, 12, 31))
