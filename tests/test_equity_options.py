from datetime import date, datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest
import requests

from tradingagents.agents.utils.options_data_tools import get_equity_option_context
from tradingagents.dataflows.equity_options import (
    _parse_occ,
    current_us_option_market_date,
    fetch_equity_option_context,
    fetch_equity_option_snapshot,
    fetch_equity_option_snapshots,
)

TODAY = date(2026, 9, 10)
SYMBOL = "AAPL260918C00300000"


def _option(symbol, *, iv, delta, oi, volume, **values):
    return {
        "option": symbol,
        "iv": iv,
        "delta": delta,
        "open_interest": oi,
        "volume": volume,
        **values,
    }


def _payload():
    return {
        "data": {
            "timestamp": "2026-09-10 15:45:00",
            "current_price": 315,
            "iv30": 25.673,
            "options": [
                _option(
                    SYMBOL,
                    iv=0.30,
                    delta=0.75,
                    oi=100,
                    volume=10,
                    bid=16,
                    ask=18,
                    last_trade_price=16.5,
                    last_trade_time="2026-09-10 15:44:00",
                    gamma=0.0123,
                    vega=0.2345,
                    theta=-0.1234,
                    rho=0.0456,
                    theo=17.1,
                ),
                _option(
                    "AAPL260918C00315000",
                    iv=0.22,
                    delta=0.45,
                    oi=200,
                    volume=20,
                ),
                _option(
                    "AAPL260918P00315000",
                    iv=0.28,
                    delta=-0.55,
                    oi=300,
                    volume=30,
                ),
                _option(
                    "AAPL260918C00325000",
                    iv=0.20,
                    delta=0.24,
                    oi=50,
                    volume=5,
                ),
                _option(
                    "AAPL260918P00305000",
                    iv=0.32,
                    delta=-0.26,
                    oi=150,
                    volume=15,
                ),
                _option(
                    "AAPL260925P00315000",
                    iv=0.99,
                    delta=-0.25,
                    oi=9999,
                    volume=9999,
                ),
            ],
        }
    }


def _response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _fetch(payload=None):
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.option_scenarios.resolve_scenario_risk_free_rate",
            return_value=(0.04, "test rate"),
        ),
        patch(
            "tradingagents.dataflows.equity_options.requests.get",
            return_value=_response(payload or _payload()),
        ) as get,
    ):
        result = fetch_equity_option_context(SYMBOL, TODAY.isoformat())
    get.assert_called_once()
    return result


def test_us_option_market_date_uses_new_york_not_host_calendar_date():
    yangon = timezone(timedelta(hours=6, minutes=30))
    host_time = datetime(2026, 9, 12, 0, 30, tzinfo=yangon)

    assert current_us_option_market_date(host_time) == date(2026, 9, 11)
    with pytest.raises(ValueError, match="timezone-aware"):
        current_us_option_market_date(datetime(2026, 9, 12, 0, 30))


def test_parse_occ_extracts_fields_and_rejects_invalid_dates_and_format():
    contract = _parse_occ(SYMBOL)

    assert contract is not None
    assert (contract.root, contract.expiry, contract.right, contract.strike) == (
        "AAPL",
        date(2026, 9, 18),
        "C",
        300.0,
    )
    assert _parse_occ("AAPL260231C00300000") is None
    assert _parse_occ("aapl260918c00300000") is None
    assert _parse_occ(f" {SYMBOL}") is None
    assert _parse_occ(f"{SYMBOL}X") is None


def test_exact_selected_contract_report_and_all_chain_metrics():
    result = _fetch()

    legacy_context = """Cboe delayed equity option context — AAPL260918C00300000
Contract: AAPL260918C00300000
Underlying: AAPL
Expiry: 2026-09-18
Type: call
Strike: 300
DTE: 8
Source timestamp: 2026-09-10 15:45:00
Underlying price: 315
Bid: 16 | Ask: 18
Midpoint: 17
Bid-ask spread as % of midpoint: 11.76%
Last: 16.5
Last trade time: 2026-09-10 15:44:00
Open interest: 100
Volume: 10
IV: 30.00%
Greeks: delta 0.75; gamma 0.0123; vega 0.2345; theta -0.1234; rho 0.0456
Theoretical value: 17.1
Moneyness (S/K - 1): 5.00%
Intrinsic value: 15
Time-value proxy: 2
Same-expiry chain totals: call OI 350; put OI 450; call volume 35; put volume 45
Same-expiry P/C OI ratio: 1.286
Same-expiry P/C volume ratio: 1.286
ATM IV proxy: 25.00%
25-delta put-minus-call IV skew proxy: 12 percentage points (put AAPL260918P00305000, delta -0.26; call AAPL260918C00325000, delta 0.24)
Cboe IV30: 25.67%
Cautions: this is a delayed snapshot, not realtime; Greeks and IV are vendor-calculated; positioning ratios are descriptive, not directional; stale last trades and wide or zero markets reduce reliability."""
    assert result.startswith(legacy_context + "\n\n")
    assert "# Deterministic option scenario engine" in result
    assert "## Current premium and expiry breakeven" in result
    assert "## Theta burn (unchanged spot and IV)" in result
    assert "## Required underlying to preserve current premium" in result
    assert "## Spot × IV matrix" in result
    assert "## Expiry payoff" in result
    assert "## Delta/gamma local stress" in result
    assert "not market fair value" in result


def test_zero_bid_and_ask_use_last_only_for_time_value():
    payload = _payload()
    payload["data"]["options"][0].update(bid=0, ask=0)

    result = _fetch(payload)

    assert "Midpoint: DATA_UNAVAILABLE" in result
    assert "Bid-ask spread as % of midpoint: DATA_UNAVAILABLE" in result
    assert "Intrinsic value: 15" in result
    assert "Time-value proxy: 1.5" in result
    assert "Current premium: $16.50 per share" in result


def test_missing_positive_iv_preserves_context_and_marks_scenario_unavailable():
    payload = _payload()
    payload["data"]["options"][0]["iv"] = 0

    result = _fetch(payload)

    assert "Cboe delayed equity option context" in result
    assert "Bid: 16 | Ask: 18" in result
    assert "Cautions: this is a delayed snapshot" in result
    assert result.endswith(
        "<deterministic option scenario unavailable: missing positive IV>"
    )


def test_date_and_symbol_guards_do_not_make_network_requests():
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch("tradingagents.dataflows.equity_options.requests.get") as get,
        patch(
            "tradingagents.dataflows.option_scenarios.resolve_scenario_risk_free_rate"
        ) as resolve_rate,
    ):
        historical = fetch_equity_option_context(SYMBOL, "2026-09-09")
        future = fetch_equity_option_context(SYMBOL, "2026-09-11")
        invalid = fetch_equity_option_context("aapl260918c00300000", TODAY.isoformat())

    assert "historical/future" in historical
    assert "historical/future" in future
    assert "invalid OCC option symbol" in invalid
    get.assert_not_called()
    resolve_rate.assert_not_called()


def test_exact_contract_missing_is_unavailable():
    payload = _payload()
    payload["data"]["options"] = payload["data"]["options"][1:]

    assert _fetch(payload) == (
        "<equity option context unavailable: exact contract "
        "AAPL260918C00300000 not found>"
    )


def test_zero_call_denominator_makes_both_ratios_unavailable():
    payload = _payload()
    for row in payload["data"]["options"]:
        if row["option"].endswith(("C00300000", "C00315000", "C00325000")):
            row["open_interest"] = 0
            row["volume"] = 0

    result = _fetch(payload)

    assert "Same-expiry P/C OI ratio: DATA_UNAVAILABLE" in result
    assert "Same-expiry P/C volume ratio: DATA_UNAVAILABLE" in result


def test_missing_one_skew_side_is_unavailable():
    payload = _payload()
    for row in payload["data"]["options"]:
        if row["option"].startswith("AAPL260918P"):
            row["iv"] = None

    assert (
        "25-delta put-minus-call IV skew proxy: DATA_UNAVAILABLE"
        in _fetch(payload)
    )


def test_requests_exception_is_fail_soft():
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.equity_options.requests.get",
            side_effect=requests.Timeout("late"),
        ),
    ):
        result = fetch_equity_option_context(SYMBOL, TODAY.isoformat())

    assert result == "<equity option context unavailable: Timeout>"


def test_options_data_tool_delegates_correctly():
    with patch(
        "tradingagents.agents.utils.options_data_tools.fetch_equity_option_context",
        return_value="context",
    ) as fetch:
        result = get_equity_option_context.invoke(
            {"ticker": SYMBOL, "curr_date": TODAY.isoformat()}
        )

    assert result == "context"
    fetch.assert_called_once_with(SYMBOL, TODAY.isoformat())


def test_structured_snapshot_returns_exact_contract_without_markdown_parsing():
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.equity_options.requests.get",
            return_value=_response(_payload()),
        ) as get,
    ):
        result = fetch_equity_option_snapshot(SYMBOL, TODAY.isoformat())

    assert result.available
    assert result.unavailable_reason is None
    assert result.snapshot is not None
    snapshot = result.snapshot
    assert snapshot.symbol == SYMBOL
    assert snapshot.underlying == "AAPL"
    assert snapshot.expiry == date(2026, 9, 18)
    assert snapshot.right == "C"
    assert snapshot.strike == 300
    assert snapshot.as_of == TODAY
    assert snapshot.dte == 8
    assert snapshot.source_timestamp == "2026-09-10 15:45:00"
    assert snapshot.underlying_spot == 315
    assert snapshot.bid == 16
    assert snapshot.ask == 18
    assert snapshot.midpoint == 17
    assert snapshot.spread_pct == pytest.approx(11.7647058824)
    assert snapshot.last == 16.5
    assert snapshot.iv == 0.30
    assert snapshot.delta == 0.75
    assert snapshot.gamma == 0.0123
    assert snapshot.vega == 0.2345
    assert snapshot.theta == -0.1234
    assert snapshot.open_interest == 100
    assert snapshot.volume == 10
    get.assert_called_once()


def test_structured_snapshot_fails_closed_when_cboe_source_market_date_is_stale():
    payload = _payload()
    payload["data"]["timestamp"] = "2026-09-09 16:15:00"
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.equity_options.requests.get",
            return_value=_response(payload),
        ),
    ):
        result = fetch_equity_option_snapshot(SYMBOL, TODAY.isoformat())

    assert not result.available
    assert result.snapshot is None
    assert "source market date 2026-09-09" in result.unavailable_reason
    assert "requested US market date 2026-09-10" in result.unavailable_reason


def test_context_fails_closed_when_cboe_source_market_date_is_stale():
    payload = _payload()
    payload["data"]["timestamp"] = "2026-09-09 16:15:00"

    result = _fetch(payload)

    assert "equity option context unavailable" in result
    assert "source market date 2026-09-09" in result
    assert "requested US market date 2026-09-10" in result


def test_structured_snapshot_preserves_zero_bid_as_liquidation_input():
    payload = _payload()
    payload["data"]["options"][0].update(bid=0, ask=0)
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.equity_options.requests.get",
            return_value=_response(payload),
        ),
    ):
        result = fetch_equity_option_snapshot(SYMBOL, TODAY.isoformat())

    assert result.available
    assert result.snapshot is not None
    assert result.snapshot.bid == 0
    assert result.snapshot.ask == 0
    assert result.snapshot.midpoint is None
    assert result.snapshot.spread_pct is None


def test_structured_snapshot_preflight_rejects_date_and_symbol_without_network():
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch("tradingagents.dataflows.equity_options.requests.get") as get,
    ):
        historical = fetch_equity_option_snapshot(SYMBOL, "2026-09-09")
        invalid = fetch_equity_option_snapshot("bad", TODAY.isoformat())

    assert not historical.available
    assert "historical/future" in historical.unavailable_reason
    assert not invalid.available
    assert invalid.unavailable_reason == "invalid OCC option symbol"
    get.assert_not_called()


def test_structured_snapshot_missing_exact_contract_is_unavailable():
    payload = _payload()
    payload["data"]["options"] = payload["data"]["options"][1:]
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.equity_options.requests.get",
            return_value=_response(payload),
        ),
    ):
        result = fetch_equity_option_snapshot(SYMBOL, TODAY.isoformat())

    assert not result.available
    assert result.snapshot is None
    assert result.unavailable_reason == f"exact contract {SYMBOL} not found"


def test_structured_snapshot_request_exception_is_fail_soft():
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.equity_options.requests.get",
            side_effect=requests.Timeout("late"),
        ),
    ):
        result = fetch_equity_option_snapshot(SYMBOL, TODAY.isoformat())

    assert not result.available
    assert result.snapshot is None
    assert result.unavailable_reason == "Timeout"


def test_batch_snapshots_reuse_one_chain_request_for_same_underlying():
    second = "AAPL260918C00315000"
    payload = _payload()
    payload["data"]["options"][1].update(
        bid=4.0,
        ask=4.5,
        gamma=0.02,
        vega=0.18,
        theta=-0.09,
    )
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.equity_options.requests.get",
            return_value=_response(payload),
        ) as get,
    ):
        results = fetch_equity_option_snapshots([SYMBOL, second], TODAY.isoformat())

    assert results[SYMBOL].available
    assert results[second].available
    assert results[second].snapshot is not None
    assert results[second].snapshot.bid == 4.0
    get.assert_called_once()


def test_batch_snapshots_use_one_request_per_underlying():
    msft_symbol = "MSFT260918C00400000"
    msft_payload = {
        "data": {
            "timestamp": "2026-09-10 15:45:00",
            "current_price": 410,
            "options": [
                _option(
                    msft_symbol,
                    iv=0.25,
                    delta=0.60,
                    oi=50,
                    volume=5,
                    bid=12,
                    ask=13,
                    gamma=0.01,
                    vega=0.20,
                    theta=-0.10,
                )
            ],
        }
    }

    def fake_get(url, **_kwargs):
        return _response(msft_payload if "MSFT" in url else _payload())

    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.equity_options.requests.get",
            side_effect=fake_get,
        ) as get,
    ):
        results = fetch_equity_option_snapshots([SYMBOL, msft_symbol], TODAY.isoformat())

    assert results[SYMBOL].available
    assert results[msft_symbol].available
    assert get.call_count == 2


def test_batch_snapshot_preflight_fails_without_network_and_preserves_invalid_symbol_reason():
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch("tradingagents.dataflows.equity_options.requests.get") as get,
    ):
        results = fetch_equity_option_snapshots([SYMBOL, "bad"], "2026-09-09")

    assert not results[SYMBOL].available
    assert "historical/future" in results[SYMBOL].unavailable_reason
    assert results["bad"].unavailable_reason == "invalid OCC option symbol"
    get.assert_not_called()


def test_batch_snapshot_missing_one_contract_does_not_hide_other_available_contracts():
    missing = "AAPL260918C00999000"
    with (
        patch("tradingagents.dataflows.equity_options._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.equity_options.requests.get",
            return_value=_response(_payload()),
        ) as get,
    ):
        results = fetch_equity_option_snapshots([SYMBOL, missing], TODAY.isoformat())

    assert results[SYMBOL].available
    assert not results[missing].available
    assert results[missing].unavailable_reason == f"exact contract {missing} not found"
    get.assert_called_once()
