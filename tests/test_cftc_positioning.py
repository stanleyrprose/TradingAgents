import csv
import io
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from tradingagents.dataflows import cftc_positioning as cftc


@pytest.fixture(autouse=True)
def clear_archive_cache():
    cftc._load_annual_archive.cache_clear()
    yield
    cftc._load_annual_archive.cache_clear()


def _row(report_type, code, report_date="2026-09-01", **values):
    fields = cftc._TFF_FIELDS if report_type == "tff" else cftc._DISAGGREGATED_FIELDS
    row = dict.fromkeys(fields, "0")
    row.update(
        {
            cftc._DATE_FIELD: report_date,
            cftc._CODE_FIELD: code,
            "Open_Interest_All": "1000",
        }
    )
    row.update({key: str(value) for key, value in values.items()})
    return row


def _zip_bytes(report_type, rows, encoding="utf-8-sig"):
    fields = cftc._TFF_FIELDS if report_type == "tff" else cftc._DISAGGREGATED_FIELDS
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    archive_buffer = io.BytesIO()
    with ZipFile(archive_buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("report.txt", csv_buffer.getvalue().encode(encoding))
    return archive_buffer.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _tff_row(code, report_date="2026-09-01", **values):
    return _row("tff", code, report_date, **values)


def _disaggregated_row(code, report_date="2026-09-01", **values):
    return _row("disaggregated", code, report_date, **values)


def test_latin1_zip_csv_parser_and_exact_headers():
    row = _tff_row("099741é")
    payload = _zip_bytes("tff", [row], encoding="latin-1")

    with patch.object(cftc, "urlopen", return_value=_Response(payload)):
        parsed = cftc._load_annual_archive("tff", 2026)

    assert parsed[0][cftc._CODE_FIELD] == "099741é"


def test_eurusd_uses_only_eur_leg_and_correct_proxy():
    rows = (
        _tff_row(
            "099741",
            Lev_Money_Positions_Long_All=300,
            Lev_Money_Positions_Short_All=100,
            Asset_Mgr_Positions_Long_All=250,
            Asset_Mgr_Positions_Short_All=150,
            Change_in_Lev_Money_Long_All=30,
            Change_in_Lev_Money_Short_All=10,
        ),
    )
    with patch.object(cftc, "_load_annual_archive", return_value=rows) as load:
        result = cftc.fetch_cftc_positioning("EURUSD", "2026-09-11")

    load.assert_called_once_with("tff", 2026, 12.0)
    assert "EUR currency futures vs USD (099741)" in result
    assert "USD currency futures" not in result
    assert "Leveraged money: net 200 contracts; +20.00% of OI" in result
    assert "differential proxy: +20.00 percentage points" in result


def test_usdjpy_inverts_jpy_leg():
    rows = (_tff_row("097741", Lev_Money_Positions_Long_All=300),)
    with patch.object(cftc, "_load_annual_archive", return_value=rows):
        result = cftc.fetch_cftc_positioning("USDJPY", "2026-09-11")

    assert "JPY currency futures vs USD (097741)" in result
    assert "differential proxy: -30.00 percentage points" in result


def test_eurjpy_is_eur_pct_minus_jpy_pct():
    rows = (
        _tff_row("099741", Lev_Money_Positions_Long_All=300),
        _tff_row("097741", Lev_Money_Positions_Long_All=100),
    )
    with patch.object(cftc, "_load_annual_archive", return_value=rows):
        result = cftc.fetch_cftc_positioning("EURJPY", "2026-09-11")

    assert "differential proxy: +20.00 percentage points" in result


def test_cnhusd_is_explicitly_unsupported_without_network():
    with patch.object(cftc, "_load_annual_archive") as load:
        result = cftc.fetch_cftc_positioning("CNHUSD", "2026-09-11")

    load.assert_not_called()
    assert "unsupported CFTC forex mapping for CNH; no contract code guessed" in result


def test_gold_managed_money_and_producer_metrics():
    rows = (
        _disaggregated_row(
            "088691",
            M_Money_Positions_Long_All=400,
            M_Money_Positions_Short_All=150,
            Change_in_M_Money_Long_All=30,
            Change_in_M_Money_Short_All=50,
            Prod_Merc_Positions_Long_All=100,
            Prod_Merc_Positions_Short_All=350,
        ),
    )
    with patch.object(cftc, "_load_annual_archive", return_value=rows):
        result = cftc.fetch_cftc_positioning("GC=F", "2026-09-11")

    assert "GC=F (088691)" in result
    assert "Managed money: net 250 contracts; +25.00% of OI" in result
    assert "weekly net change -20 contracts" in result
    assert "Producer/merchant: net -250 contracts; -25.00% of OI" in result


def test_xauusd_alias_routes_to_gold_contract():
    rows = (_disaggregated_row("088691"),)
    with patch.object(cftc, "_load_annual_archive", return_value=rows):
        result = cftc.fetch_cftc_positioning("XAUUSD", "2026-09-11")

    assert "GC=F (088691)" in result


def test_crude_oil_maps_to_067651():
    rows = (_disaggregated_row("067651"),)
    with patch.object(cftc, "_load_annual_archive", return_value=rows):
        result = cftc.fetch_cftc_positioning("CL=F", "2026-09-11")

    assert "CL=F (067651)" in result


@pytest.mark.parametrize(
    ("end_date", "expected", "excluded"),
    [
        ("2026-09-10", "2026-09-01", "2026-09-08"),
        ("2026-09-11", "2026-09-08", "2026-09-01"),
    ],
)
def test_point_in_time_release_lag(end_date, expected, excluded):
    rows = (
        _tff_row("099741", "2026-09-08"),
        _tff_row("099741", "2026-09-01"),
    )
    with patch.object(cftc, "_load_annual_archive", return_value=rows):
        result = cftc.fetch_cftc_positioning("EURUSD", end_date)

    assert f"Report date (positions as of): {expected}" in result
    assert f"Report date (positions as of): {excluded}" not in result


@pytest.mark.parametrize(
    "failure",
    [ValueError("bad ZIP"), OSError("network down")],
)
def test_archive_failures_degrade_to_unavailable(failure):
    with patch.object(cftc, "_load_annual_archive", side_effect=failure):
        result = cftc.fetch_cftc_positioning("EURUSD", "2026-09-11")

    assert result.startswith("<cftc positioning unavailable: ")


def test_non_supported_asset_class_does_not_use_network():
    with patch.object(cftc, "_load_annual_archive") as load:
        result = cftc.fetch_cftc_positioning("AAPL", "2026-09-11")

    load.assert_not_called()
    assert result == "<cftc positioning unavailable: unsupported asset class for AAPL>"
