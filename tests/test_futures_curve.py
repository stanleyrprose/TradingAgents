from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from tradingagents.dataflows.futures_curve import (
    _classification,
    _Contract,
    _contract_observation,
    _fetch_contract_history,
    _listed_contracts,
    _Observation,
    fetch_futures_curve,
)

END_DATE = date(2026, 9, 10)


def _observation(
    symbol: str,
    delivery: tuple[int, int],
    close: float,
    volume: float,
    last_bar: date = END_DATE,
) -> _Observation:
    return _Observation(
        contract=_Contract(symbol, date(*delivery, 1)),
        close=close,
        last_bar=last_bar,
        volume_5bar=volume,
    )


def _symbols(contracts: list[_Contract]) -> list[str]:
    return [contract.symbol for contract in contracts]


def test_gc_listed_contracts_start_with_oct_dec_feb_and_are_capped_at_six():
    contracts = _listed_contracts("GC=F", END_DATE)

    assert _symbols(contracts)[:3] == ["GCV26.CMX", "GCZ26.CMX", "GCG27.CMX"]
    assert len(contracts) <= 6


def test_cl_september_listed_contracts_use_exact_month_codes():
    contracts = _listed_contracts("CL=F", END_DATE)

    assert _symbols(contracts)[:4] == [
        "CLU26.NYM",
        "CLV26.NYM",
        "CLX26.NYM",
        "CLZ26.NYM",
    ]


def test_gc_rollover_anchors_on_high_volume_december_then_february_and_april():
    observations = {
        "GCV26.CMX": _observation("GCV26.CMX", (2026, 10), 100, 10),
        "GCZ26.CMX": _observation("GCZ26.CMX", (2026, 12), 101, 1_000),
        "GCG27.CMX": _observation("GCG27.CMX", (2027, 2), 102, 500),
        "GCJ27.CMX": _observation("GCJ27.CMX", (2027, 4), 103, 400),
    }

    with patch(
        "tradingagents.dataflows.futures_curve._contract_observation",
        side_effect=lambda contract, _end: observations.get(contract.symbol),
    ):
        result = fetch_futures_curve("GC=F", "2026-09-10")

    assert result.index("GCZ26.CMX") < result.index("GCG27.CMX") < result.index("GCJ27.CMX")
    assert "GCV26.CMX" not in result


def test_cl_rejects_stale_september_and_anchors_on_october_then_november_december():
    observations = {
        "CLU26.NYM": None,
        "CLV26.NYM": _observation("CLV26.NYM", (2026, 10), 70, 900),
        "CLX26.NYM": _observation("CLX26.NYM", (2026, 11), 71, 500),
        "CLZ26.NYM": _observation("CLZ26.NYM", (2026, 12), 72, 400),
    }

    with patch(
        "tradingagents.dataflows.futures_curve._contract_observation",
        side_effect=lambda contract, _end: observations.get(contract.symbol),
    ):
        result = fetch_futures_curve("CL=F", "2026-09-10")

    assert "CLU26.NYM" not in result
    assert result.index("CLV26.NYM") < result.index("CLX26.NYM") < result.index("CLZ26.NYM")


@pytest.mark.parametrize(
    ("spread", "expected"),
    [
        (0.11, "Contango"),
        (-0.11, "Backwardation"),
        (0.10, "Flat"),
        (-0.10, "Flat"),
    ],
)
def test_classification_boundaries(spread, expected):
    assert _classification(spread) == expected


@pytest.mark.parametrize(
    ("prices", "classification", "spread_sign", "slope_sign"),
    [
        ((100, 102, 104), "Contango", "+2.00%", "+"),
        ((100, 98, 96), "Backwardation", "-2.00%", "-"),
    ],
)
def test_deterministic_prices_render_spread_classification_and_annualized_slope(
    prices, classification, spread_sign, slope_sign
):
    observations = {
        "GCZ26.CMX": _observation("GCZ26.CMX", (2026, 12), prices[0], 1_000),
        "GCG27.CMX": _observation("GCG27.CMX", (2027, 2), prices[1], 500),
        "GCJ27.CMX": _observation("GCJ27.CMX", (2027, 4), prices[2], 400),
    }

    with patch(
        "tradingagents.dataflows.futures_curve._contract_observation",
        side_effect=lambda contract, _end: observations.get(contract.symbol),
    ):
        result = fetch_futures_curve("GC=F", "2026-09-10")

    assert f"Next vs front spread: {spread_sign}" in result
    assert f"Front-to-next curve classification: {classification}" in result
    assert f"Annualized slope proxy (next vs front): {slope_sign}" in result


def test_xauusd_warns_that_gc_f_is_used_as_a_proxy():
    observations = {
        "GCZ26.CMX": _observation("GCZ26.CMX", (2026, 12), 100, 1_000),
        "GCG27.CMX": _observation("GCG27.CMX", (2027, 2), 101, 500),
        "GCJ27.CMX": _observation("GCJ27.CMX", (2027, 4), 102, 400),
    }

    with patch(
        "tradingagents.dataflows.futures_curve._contract_observation",
        side_effect=lambda contract, _end: observations.get(contract.symbol),
    ):
        result = fetch_futures_curve("XAUUSD", "2026-09-10")

    assert "PROXY WARNING" in result
    assert "using the GC=F futures curve proxy" in result
    assert result.startswith("Commodity futures curve — GC=F")


def test_brent_is_explicitly_unsupported_without_observing_contracts():
    with patch("tradingagents.dataflows.futures_curve._contract_observation") as observe:
        result = fetch_futures_curve("BZ=F", "2026-09-10")

    assert "BZ=F/Brent is explicitly unsupported" in result
    observe.assert_not_called()


def test_contract_observation_rejects_a_last_bar_older_than_seven_days():
    history = pd.DataFrame(
        {"Close": [100.0], "Volume": [500.0]},
        index=pd.to_datetime(["2026-09-02"]),
    )
    contract = _Contract("GCV26.CMX", date(2026, 10, 1))

    with patch(
        "tradingagents.dataflows.futures_curve._fetch_contract_history",
        return_value=history,
    ):
        assert _contract_observation(contract, END_DATE) is None


def test_history_request_uses_exclusive_next_day_without_requesting_future_bars():
    history = pd.DataFrame(
        {"Close": [100.0], "Volume": [500.0]},
        index=pd.to_datetime(["2026-09-10"]),
    )
    ticker = Mock()
    ticker.history.return_value = history
    _fetch_contract_history.cache_clear()

    with patch("tradingagents.dataflows.futures_curve.yf.Ticker", return_value=ticker) as factory:
        result = _fetch_contract_history("GCV26.CMX", END_DATE)

    assert result is history
    factory.assert_called_once_with("GCV26.CMX")
    ticker.history.assert_called_once_with(
        start="2026-08-31", end="2026-09-11", auto_adjust=False
    )
    _fetch_contract_history.cache_clear()


def test_bars_after_end_date_are_filtered_before_close_and_volume_selection():
    history = pd.DataFrame(
        {
            "Close": [99.0, 100.0, 999.0],
            "Volume": [10.0, 20.0, 10_000.0],
        },
        index=pd.to_datetime(["2026-09-09", "2026-09-10", "2026-09-11"]),
    )
    contract = _Contract("GCV26.CMX", date(2026, 10, 1))

    with patch(
        "tradingagents.dataflows.futures_curve._fetch_contract_history",
        return_value=history,
    ):
        observation = _contract_observation(contract, END_DATE)

    assert observation is not None
    assert observation.close == 100.0
    assert observation.last_bar == END_DATE
    assert observation.volume_5bar == 30.0


def test_invalid_date_degrades_gracefully_without_observing_contracts():
    with patch("tradingagents.dataflows.futures_curve._contract_observation") as observe:
        result = fetch_futures_curve("GC=F", "not-a-date")

    assert result == "<futures curve unavailable: invalid end_date; expected YYYY-MM-DD>"
    observe.assert_not_called()


@pytest.mark.parametrize("valid_count", [0, 2])
def test_no_or_insufficient_valid_contracts_degrade_gracefully(valid_count):
    observations = [
        _observation("GCV26.CMX", (2026, 10), 100, 100),
        _observation("GCZ26.CMX", (2026, 12), 101, 200),
    ][:valid_count]

    with patch(
        "tradingagents.dataflows.futures_curve._contract_observation",
        side_effect=[*observations, *([None] * (6 - valid_count))],
    ):
        result = fetch_futures_curve("GC=F", "2026-09-10")

    assert result.startswith("<futures curve unavailable:")
    assert "fewer than three fresh, positive-price contracts" in result
