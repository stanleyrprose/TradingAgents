import io
from unittest.mock import patch

import pytest

from tradingagents.dataflows import eia_inventory as eia


@pytest.fixture(autouse=True)
def clear_source_cache():
    eia._load_source_page.cache_clear()
    yield
    eia._load_source_page.cache_clear()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _html(months):
    parts = ["<html><table>"]
    for month, weeks in months:
        parts.append(f'<tr><td class="B6">{month}</td></tr>')
        parts.append("<tr>" + "".join(f'<td class="B5">{day}</td>' for day, _ in weeks) + "</tr>")
        parts.append("<tr>" + "".join(f'<td class="B3">{value}</td>' for _, value in weeks) + "</tr>")
    parts.append("</table></html>")
    return "".join(parts)


def _five_crude_rows():
    return _html(
        [
            ("2026-Aug", [("08/07", "100,000"), ("08/14", "101,000"), ("08/21", "103,000"), ("08/28", "102,000")]),
            ("2026-Sep", [("09/04", "105,000")]),
        ]
    )


def test_parser_extracts_multiple_rows_with_blanks_and_commas():
    payload = _html(
        [
            ("2026-Aug", [("08/21", "1,234"), ("&nbsp;", "&nbsp;"), ("08/28", "1,250")]),
            ("2026-Sep", [("09/04", "NA")]),
        ]
    )

    assert eia._parse_eia_html(payload) == (
        (eia.date(2026, 8, 21), 1234.0),
        (eia.date(2026, 8, 28), 1250.0),
    )


def test_crude_latest_weekly_and_four_observation_changes():
    rows = eia._parse_eia_html(_five_crude_rows())
    with patch.object(eia, "_load_source_page", return_value=rows):
        result = eia.fetch_eia_inventory("CL=F", "2026-09-10")

    assert "Weekly U.S. Ending Stocks excluding SPR of Crude Oil" in result
    assert "Latest eligible inventory: 105,000 Thousand Barrels" in result
    assert "1-week change: +3,000 Thousand Barrels (+2.94%)" in result
    assert "4-observation change: +5,000 Thousand Barrels" in result
    assert "U.S. commercial crude stock measure" in result


def test_wti_alias_routes_to_crude():
    rows = eia._parse_eia_html(_five_crude_rows())
    with patch.object(eia, "_load_source_page", return_value=rows) as load:
        result = eia.fetch_eia_inventory("WTI", "2026-09-10")

    load.assert_called_once_with(eia._CRUDE_URL, 12.0)
    assert "— CL=F" in result


def test_brent_is_prominently_labeled_as_us_proxy():
    rows = eia._parse_eia_html(_five_crude_rows())
    with patch.object(eia, "_load_source_page", return_value=rows):
        result = eia.fetch_eia_inventory("BZ=F", "2026-09-10")

    assert "U.S. crude inventory proxy for Brent/global crude" in result
    assert "PROXY WARNING" in result
    assert "not direct Brent/global inventory" in result


def test_natural_gas_routes_to_gas_source_and_calculates():
    rows = eia._parse_eia_html(
        _html([("2026-Aug", [("08/21", "3,000"), ("08/28", "3,100")])])
    )
    with patch.object(eia, "_load_source_page", return_value=rows) as load:
        result = eia.fetch_eia_inventory("NG=F", "2026-09-10")

    load.assert_called_once_with(eia._GAS_URL, 12.0)
    assert "Lower 48 States Natural Gas Working Underground Storage" in result
    assert "Latest eligible inventory: 3,100 Billion Cubic Feet" in result
    assert "1-week change: +100 Billion Cubic Feet (+3.33%)" in result


def test_non_energy_commodity_is_unavailable_without_network():
    with patch.object(eia, "urlopen") as urlopen:
        result = eia.fetch_eia_inventory("GC=F", "2026-09-10")

    urlopen.assert_not_called()
    assert result.startswith("<eia inventory unavailable: not available/applicable")


@pytest.mark.parametrize(
    ("end_date", "expected", "excluded"),
    [
        ("2026-09-09", "2026-08-28", "2026-09-04"),
        ("2026-09-10", "2026-09-04", "2026-08-28"),
    ],
)
def test_fixed_six_day_point_in_time_lag(end_date, expected, excluded):
    rows = eia._parse_eia_html(
        _html(
            [
                ("2026-Aug", [("08/21", "100"), ("08/28", "110")]),
                ("2026-Sep", [("09/04", "120")]),
            ]
        )
    )
    with patch.object(eia, "_load_source_page", return_value=rows):
        result = eia.fetch_eia_inventory("CL=F", end_date)

    assert f"observation date {expected}" in result
    assert f"observation date {excluded}" not in result
    assert "minus 6 calendar days" in result


@pytest.mark.parametrize("end_date", ["bad-date", "2026-02-30", None])
def test_invalid_date_degrades_gracefully(end_date):
    assert eia.fetch_eia_inventory("CL=F", end_date).startswith(
        "<eia inventory unavailable: invalid end_date"
    )


@pytest.mark.parametrize("failure", [OSError("network down"), ValueError("bad page")])
def test_network_and_parser_failures_degrade_gracefully(failure):
    with patch.object(eia, "_load_source_page", side_effect=failure):
        result = eia.fetch_eia_inventory("CL=F", "2026-09-10")

    assert result.startswith("<eia inventory unavailable: ")


def test_cached_source_does_not_redownload():
    payload = _five_crude_rows().encode()
    with patch.object(eia, "urlopen", return_value=_Response(payload)) as urlopen:
        first = eia.fetch_eia_inventory("CL=F", "2026-09-10")
        second = eia.fetch_eia_inventory("CL=F", "2026-09-10")

    assert first == second
    urlopen.assert_called_once()
