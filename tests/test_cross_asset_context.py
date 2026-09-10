from unittest.mock import call, patch

import pytest

from tradingagents.agents.utils.macro_data_tools import get_cross_asset_context


def _invoke(ticker: str) -> str:
    return get_cross_asset_context.invoke(
        {"ticker": ticker, "curr_date": "2026-09-10", "look_back_days": 180}
    )


@pytest.fixture
def routed():
    with (
        patch(
            "tradingagents.agents.utils.macro_data_tools.route_to_vendor",
            side_effect=lambda _method, series_id, _date, _days: f"REPORT:{series_id}",
        ) as route,
        patch(
            "tradingagents.agents.utils.macro_data_tools.fetch_cftc_positioning",
            return_value="MOCK_CFTC_POSITIONING",
        ),
        patch(
            "tradingagents.agents.utils.macro_data_tools.fetch_eia_inventory",
            side_effect=lambda ticker, _date: f"MOCK_EIA_INVENTORY:{ticker}",
        ),
    ):
        yield route


def _series_requested(route) -> list[str]:
    return [args.args[1] for args in route.call_args_list]


def test_eurusd_requests_both_rate_proxies_and_dollar_context(routed):
    result = _invoke("EURUSD")

    assert _series_requested(routed) == ["ECBMRRFR", "FEDFUNDS", "DTWEXBGS"]
    assert "rate proxies, not exact forward carry" in result
    assert "REPORT:ECBMRRFR" in result
    assert "## CFTC positioning\nMOCK_CFTC_POSITIONING" in result


def test_usdcad_requests_canadian_proxy_without_duplicate_fedfunds(routed):
    _invoke("USDCAD")

    requested = _series_requested(routed)
    assert requested == ["FEDFUNDS", "IR3TIB01CAM156N", "DTWEXBGS"]
    assert requested.count("FEDFUNDS") == 1


def test_cnhusd_uses_cny_proxy_with_explicit_note(routed):
    result = _invoke("CNHUSD")

    assert _series_requested(routed) == ["IRSTCI01CNM156N", "FEDFUNDS", "DTWEXBGS"]
    assert "CNH uses the CNY rate proxy" in result


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("CL=F", ["DCOILWTICO", "DTWEXBGS", "DGS10", "INDPRO"]),
        ("GC=F", ["DTWEXBGS", "DGS10", "T10YIE"]),
    ],
)
def test_commodity_driver_baskets(ticker, expected, routed):
    result = _invoke(ticker)

    assert _series_requested(routed) == expected
    assert "term structure remain unavailable" in result
    assert f"REPORT:{expected[0]}" in result
    assert f"## EIA inventory\nMOCK_EIA_INVENTORY:{ticker}" in result
    if ticker == "GC=F":
        assert "## CFTC positioning\nMOCK_CFTC_POSITIONING" in result


def test_non_cross_asset_instrument_is_not_applicable(routed):
    result = _invoke("AAPL")

    assert result.startswith("NOT_APPLICABLE")
    routed.assert_not_called()


def test_one_series_exception_does_not_abort_remaining_series():
    reports = {
        "DCOILWTICO": "oil report",
        "DGS10": "yield report",
        "INDPRO": "production report",
    }

    def route(_method, series_id, _date, _days):
        if series_id == "DTWEXBGS":
            raise RuntimeError("vendor failure")
        return reports[series_id]

    with (
        patch(
            "tradingagents.agents.utils.macro_data_tools.route_to_vendor",
            side_effect=route,
        ) as routed,
        patch(
            "tradingagents.agents.utils.macro_data_tools.fetch_cftc_positioning",
            return_value="MOCK_CFTC_POSITIONING",
        ),
        patch(
            "tradingagents.agents.utils.macro_data_tools.fetch_eia_inventory",
            return_value="MOCK_EIA_INVENTORY:CL=F",
        ),
    ):
        result = _invoke("CL=F")

    assert routed.call_args_list == [
        call("get_macro_indicators", series_id, "2026-09-10", 180)
        for series_id in ("DCOILWTICO", "DTWEXBGS", "DGS10", "INDPRO")
    ]
    assert "### Trade-weighted" not in result
    assert "### Driver (DTWEXBGS)\nDATA_UNAVAILABLE" in result
    assert "yield report" in result
