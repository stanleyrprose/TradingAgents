from types import SimpleNamespace
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
        ) as cftc,
        patch(
            "tradingagents.agents.utils.macro_data_tools.fetch_eia_inventory",
            side_effect=lambda ticker, _date: f"MOCK_EIA_INVENTORY:{ticker}",
        ) as eia,
        patch(
            "tradingagents.agents.utils.macro_data_tools.fetch_futures_curve",
            return_value="MOCK_FUTURES_CURVE",
        ) as curve,
    ):
        yield SimpleNamespace(route=route, cftc=cftc, eia=eia, curve=curve)


def _series_requested(route) -> list[str]:
    return [args.args[1] for args in route.call_args_list]


def test_eurusd_requests_both_rate_proxies_and_dollar_context(routed):
    result = _invoke("EURUSD")

    assert _series_requested(routed.route) == ["ECBMRRFR", "FEDFUNDS", "DTWEXBGS"]
    assert "rate proxies, not exact forward carry" in result
    assert "REPORT:ECBMRRFR" in result
    assert "## CFTC positioning\nMOCK_CFTC_POSITIONING" in result


def test_usdcad_requests_canadian_proxy_without_duplicate_fedfunds(routed):
    _invoke("USDCAD")

    requested = _series_requested(routed.route)
    assert requested == ["FEDFUNDS", "IR3TIB01CAM156N", "DTWEXBGS"]
    assert requested.count("FEDFUNDS") == 1


def test_cnhusd_uses_cny_proxy_with_explicit_note(routed):
    result = _invoke("CNHUSD")

    assert _series_requested(routed.route) == [
        "IRSTCI01CNM156N",
        "FEDFUNDS",
        "DTWEXBGS",
    ]
    assert "CNH uses the CNY rate proxy" in result


def test_fx_future_uses_rate_proxies_positioning_and_curve(routed):
    result = _invoke("6E=F")

    assert _series_requested(routed.route) == ["ECBMRRFR", "FEDFUNDS", "DTWEXBGS"]
    assert "macro proxies, not exact OTC forward points or realized carry" in result
    assert "## CFTC positioning\nMOCK_CFTC_POSITIONING" in result
    assert "## Futures curve\nMOCK_FUTURES_CURVE" in result


def test_index_future_uses_macro_context_and_curve_only(routed):
    result = _invoke("ES=F")

    assert _series_requested(routed.route) == ["FEDFUNDS", "DGS10", "VIXCLS"]
    assert "## Futures curve\nMOCK_FUTURES_CURVE" in result
    routed.cftc.assert_not_called()
    routed.eia.assert_not_called()


def test_treasury_future_uses_rate_context_and_curve_only(routed):
    result = _invoke("ZN=F")

    assert _series_requested(routed.route) == [
        "FEDFUNDS",
        "DGS2",
        "DGS10",
        "DGS30",
    ]
    assert "delivery-basket" in result
    assert "cheapest-to-deliver (CTD)" in result
    assert "## Futures curve\nMOCK_FUTURES_CURVE" in result
    routed.cftc.assert_not_called()
    routed.eia.assert_not_called()


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("CL=F", ["DCOILWTICO", "DTWEXBGS", "DGS10", "INDPRO"]),
        ("GC=F", ["DTWEXBGS", "DGS10", "T10YIE"]),
    ],
)
def test_commodity_driver_baskets(ticker, expected, routed):
    result = _invoke(ticker)

    assert _series_requested(routed.route) == expected
    assert "Yahoo futures-curve snapshot may be provided" in result
    assert "cash basis, and cost-of-carry fair value remain unavailable" in result
    assert f"REPORT:{expected[0]}" in result
    assert f"## EIA inventory\nMOCK_EIA_INVENTORY:{ticker}" in result
    assert "## Futures curve\nMOCK_FUTURES_CURVE" in result
    if ticker == "GC=F":
        assert "## CFTC positioning\nMOCK_CFTC_POSITIONING" in result


def test_non_cross_asset_instrument_is_not_applicable(routed):
    result = _invoke("AAPL")

    assert result.startswith("NOT_APPLICABLE")
    routed.route.assert_not_called()


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
        patch(
            "tradingagents.agents.utils.macro_data_tools.fetch_futures_curve",
            return_value="MOCK_FUTURES_CURVE",
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
