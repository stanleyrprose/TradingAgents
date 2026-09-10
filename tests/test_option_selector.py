from datetime import date
from unittest.mock import Mock, patch

import pytest

from tradingagents.dataflows.option_selector import (
    OptionCandidate,
    OptionSelectionResult,
    _eligible_candidates,
    _points,
    _rank_candidates,
    rank_equity_option_contracts,
    select_equity_option_contract,
)

TODAY = date(2026, 9, 10)


def _symbol(expiry: str, right: str, strike: int) -> str:
    return f"AAPL{expiry}{right}{strike * 1000:08d}"


def _option(
    expiry: str,
    right: str,
    strike: int,
    *,
    bid: float = 4.8,
    ask: float = 5.2,
    iv: float = 0.3,
    delta: float | None = None,
    oi: float = 500,
    volume: float = 100,
    theta: float | None = -0.05,
    gamma: float | None = None,
    vega: float | None = None,
) -> dict:
    row = {
        "option": _symbol(expiry, right, strike),
        "bid": bid,
        "ask": ask,
        "iv": iv,
        "delta": (0.55 if right == "C" else -0.55) if delta is None else delta,
        "open_interest": oi,
        "volume": volume,
    }
    if theta is not None:
        row["theta"] = theta
    if gamma is not None:
        row["gamma"] = gamma
    if vega is not None:
        row["vega"] = vega
    return row


def _payload(options: list[dict]) -> dict:
    return {
        "data": {
            "timestamp": "2026-09-10 15:45:00",
            "current_price": 100,
            "options": options,
        }
    }


def _response(payload: object) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def _select(options: list[dict], direction: str = "bullish", **kwargs) -> str:
    with (
        patch("tradingagents.dataflows.option_selector._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.option_selector.requests.get",
            return_value=_response(_payload(options)),
        ) as get,
    ):
        result = select_equity_option_contract(
            "aapl", direction, TODAY.isoformat(), **kwargs
        )
    get.assert_called_once_with(
        "https://cdn.cboe.com/api/global/delayed_quotes/options/AAPL.json",
        headers={
            "User-Agent": "tradingagents/0.4 (+https://github.com/TauricResearch/TradingAgents)"
        },
        timeout=15.0,
    )
    return result


def test_bullish_selects_calls_and_bearish_selects_puts():
    options = [_option("260925", "C", 100), _option("260925", "P", 100)]

    bullish = _select(options, "BuLlIsH")
    bearish = _select(options, "BEARISH")

    assert f"Recommended contract:\n\n**{_symbol('260925', 'C', 100)}**" in bullish
    assert f"Recommended contract:\n\n**{_symbol('260925', 'P', 100)}**" in bearish


def test_ranking_rewards_fit_spread_and_liquidity_and_is_deterministic():
    weak = _option(
        "260925", "C", 102, bid=4, ask=5, delta=0.35, oi=1, volume=0
    )
    good = _option(
        "260925", "C", 101, bid=4.95, ask=5.05, delta=0.55, oi=2000, volume=500
    )
    tied_higher_strike = _option(
        "260925", "C", 103, bid=4.95, ask=5.05, delta=0.55, oi=2000, volume=500
    )

    report = _select([weak, tied_higher_strike, good])

    assert f"Recommended contract:\n\n**{good['option']}**" in report
    assert report.index(good["option"]) < report.index(tied_higher_strike["option"])
    assert report.index(tied_higher_strike["option"]) < report.index(weak["option"])


def test_components_are_bounded_and_sum_to_the_rounded_total():
    raw = _eligible_candidates(
        [_option("261001", "C", 100)],
        underlying="AAPL",
        right="C",
        spot=100,
        today=TODAY,
        min_dte=7,
        max_dte=45,
    )
    candidate = _rank_candidates(
        raw, target_delta=0.55, target_dte=21, min_dte=7, max_dte=45
    )[0]

    assert len(candidate.components) == 8
    assert all(0 <= points <= maximum for _, points, maximum in candidate.components)
    assert sum(maximum for _, _, maximum in candidate.components) == 100
    assert candidate.score == round(
        sum(points for _, points, _ in candidate.components), 2
    )


def test_ineligible_rows_are_excluded():
    valid_min = _option("260917", "C", 100)
    valid_max = _option("261025", "C", 101)
    rows = [
        valid_min,
        valid_max,
        _option("260916", "C", 102),  # Below the inclusive DTE window.
        _option("261026", "C", 103),  # Above the inclusive DTE window.
        _option("260925", "P", 104),
        _option("260925", "C", 105, bid=0),
        _option("260925", "C", 106, ask=0),
        _option("260925", "C", 107, bid=5.1, ask=5.0),
        _option("260925", "C", 108, bid=3, ask=5),  # 50% spread.
        _option("260925", "C", 109, iv=0),
        _option("260925", "C", 110, delta=-0.5),
        _option("260925", "C", 111, delta=0.09),
        _option("260925", "C", 112, delta=0.91),
        _option("260925", "C", 113, oi=-1),
        _option("260925", "C", 114, volume=-1),
    ]

    report = _select(rows)

    assert "Total eligible count: 2" in report
    assert valid_min["option"] in report
    assert valid_max["option"] in report
    assert all(row["option"] not in report for row in rows[2:])


def test_missing_theta_is_eligible_and_receives_neutral_five_points():
    report = _select([_option("260925", "C", 100, theta=None)])

    assert "theta unavailable (neutral 5/10 theta points)" in report
    assert "| Theta burden | 5.00 | 10.00 |" in report


def test_gamma_and_vega_are_carried_without_affecting_eligibility():
    raw = _eligible_candidates(
        [
            _option("260925", "C", 100, gamma=0.025, vega=0.12),
            _option("260925", "C", 101),
        ],
        underlying="AAPL",
        right="C",
        spot=100,
        today=TODAY,
        min_dte=7,
        max_dte=45,
    )
    ranked = _rank_candidates(
        raw, target_delta=0.55, target_dte=21, min_dte=7, max_dte=45
    )
    by_symbol = {candidate.symbol: candidate for candidate in ranked}

    with_greeks = by_symbol[_symbol("260925", "C", 100)]
    without_greeks = by_symbol[_symbol("260925", "C", 101)]
    assert with_greeks.gamma == 0.025
    assert with_greeks.vega == 0.12
    assert without_greeks.gamma is None
    assert without_greeks.vega is None


@pytest.mark.parametrize(("iv", "expected"), [(0.15, 5), (0.30, 2.5), (0.45, 0)])
def test_iv_relative_points_at_half_median_median_and_one_and_a_half(iv, expected):
    raw = {
        "abs_delta": 0.55,
        "spread_pct": 2.0,
        "oi": 500,
        "volume": 100,
        "dte": 21,
        "theta_burden": 0.01,
        "required_move": 0.05,
        "iv": iv,
    }

    components = _points(
        raw,
        target_delta=0.55,
        target_dte=21,
        min_dte=7,
        max_dte=45,
        median_iv=0.30,
    )

    assert all(0 <= points <= maximum for _, points, maximum in components)
    assert sum(maximum for _, _, maximum in components) == 100
    assert next(points for name, points, _ in components if name == "IV-relative") == expected


def test_equal_score_ties_use_delta_distance_then_spread(monkeypatch):
    rows = [
        _option("260925", "C", 100, delta=0.50, bid=4.9, ask=5.1),
        _option("260925", "C", 101, delta=0.54, bid=4.5, ask=5.5),
        _option("260925", "C", 102, delta=0.54, bid=4.9, ask=5.1),
    ]
    raw = _eligible_candidates(
        rows,
        underlying="AAPL",
        right="C",
        spot=100,
        today=TODAY,
        min_dte=7,
        max_dte=45,
    )
    monkeypatch.setattr(
        "tradingagents.dataflows.option_selector._points",
        lambda *args, **kwargs: (("fixed", 1.0, 1.0),),
    )

    ranked = _rank_candidates(
        raw, target_delta=0.55, target_dte=21, min_dte=7, max_dte=45
    )

    assert [item.symbol for item in ranked] == [
        _symbol("260925", "C", 102),
        _symbol("260925", "C", 101),
        _symbol("260925", "C", 100),
    ]


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (("AAPL", "sideways", TODAY.isoformat()), {}, "unsupported direction"),
        (("AAPL1", "bullish", TODAY.isoformat()), {}, "invalid underlying"),
        (("AAPL", "bullish", "2026/09/10"), {}, "invalid end_date"),
        (("AAPL", "bullish", "2026-09-09"), {}, "historical/future"),
        (("AAPL", "bullish", "2026-09-11"), {}, "historical/future"),
        (("AAPL", "bullish", TODAY.isoformat()), {"min_dte": 0}, "invalid DTE"),
        (("AAPL", "bullish", TODAY.isoformat()), {"min_dte": 30, "max_dte": 20}, "invalid DTE"),
        (("AAPL", "bullish", TODAY.isoformat()), {"target_delta": 0.09}, "invalid target_delta"),
        (("AAPL", "bullish", TODAY.isoformat()), {"target_delta": float("nan")}, "invalid target_delta"),
        (("AAPL", "bullish", TODAY.isoformat()), {"top_n": 0}, "invalid top_n"),
    ],
)
def test_current_only_and_malformed_guards_make_zero_requests(args, kwargs, message):
    with (
        patch("tradingagents.dataflows.option_selector._today", return_value=TODAY),
        patch("tradingagents.dataflows.option_selector.requests.get") as get,
    ):
        result = select_equity_option_contract(*args, **kwargs)

    assert message in result
    get.assert_not_called()


def test_report_has_required_sections_weights_and_caveats():
    report = _select(
        [_option("260925", "C", 100), _option("261002", "C", 101)], top_n=1
    )

    for text in (
        "Cboe delayed equity option contract selection",
        "Recommended contract:",
        "## Top candidates",
        "## Score breakdown",
        "## Best candidate by expiry",
        "Scoring weights (100 points)",
        "Total eligible count: 2",
        "transparent heuristic ranking",
        "profit probability",
        "Cboe data is delayed",
        "current snapshot only",
        "midpoint is not an executable price",
        "American early exercise",
        "single-leg long calls and puts only",
        "does not assert that lower IV is cheap or fair value",
    ):
        assert text in report


def test_bad_payload_fails_soft_without_details_and_only_one_request():
    with (
        patch("tradingagents.dataflows.option_selector._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.option_selector.requests.get",
            return_value=_response({"secret": "must-not-leak"}),
        ) as get,
    ):
        result = select_equity_option_contract("AAPL", "bullish", TODAY.isoformat())

    get.assert_called_once()
    assert result == (
        "<equity option selection unavailable: Cboe delayed chain request or payload failed>"
    )
    assert "secret" not in result


def test_request_exception_fails_soft():
    with (
        patch("tradingagents.dataflows.option_selector._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.option_selector.requests.get",
            side_effect=__import__("requests").RequestException("private detail"),
        ) as get,
    ):
        result = select_equity_option_contract("AAPL", "bullish", TODAY.isoformat())

    get.assert_called_once()
    assert result == (
        "<equity option selection unavailable: Cboe delayed chain request or payload failed>"
    )
    assert "private detail" not in result


def test_best_candidate_by_expiry_has_one_row_each_and_caps_at_eight():
    expiries = [f"2609{day:02d}" for day in range(17, 26)]
    report = _select(
        [_option(expiry, "C", 100 + index) for index, expiry in enumerate(expiries)]
    )
    section = report.split("## Best candidate by expiry", 1)[1].split("## Caveats", 1)[0]

    assert section.count("| 2026-09-") == 8
    for day in range(17, 25):
        assert section.count(f"| 2026-09-{day:02d} |") == 1
    assert "| 2026-09-25 |" not in section


def test_structured_result_retains_full_ranking_and_matches_report():
    options = [
        _option("260925", "C", 100 + index, delta=0.55 - index * 0.02)
        for index in range(4)
    ]
    with (
        patch("tradingagents.dataflows.option_selector._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.option_selector.requests.get",
            return_value=_response(_payload(options)),
        ) as get,
    ):
        result = rank_equity_option_contracts(
            "aapl", "bullish", TODAY.isoformat(), top_n=1
        )

    assert isinstance(result, OptionSelectionResult)
    assert result.available
    assert len(result.candidates) == 4
    assert result.underlying_spot == 100
    assert all(isinstance(candidate, OptionCandidate) for candidate in result.candidates)
    best = result.candidates[0]
    assert f"**{best.symbol}** — score {best.score:.2f}/100" in result.report
    top_section = result.report.split("## Top candidates", 1)[1].split(
        "## Score breakdown", 1
    )[0]
    assert top_section.count("| 1 |") == 1
    assert result.candidates[1].symbol not in top_section
    get.assert_called_once()


def test_structured_unavailable_matches_legacy_sentinel():
    with patch("tradingagents.dataflows.option_selector.requests.get") as get:
        structured = rank_equity_option_contracts(
            "AAPL", "sideways", TODAY.isoformat()
        )
        legacy = select_equity_option_contract("AAPL", "sideways", TODAY.isoformat())

    assert structured.candidates == ()
    assert not structured.available
    assert structured.unavailable_reason
    assert structured.report == legacy
    get.assert_not_called()


def test_legacy_wrapper_returns_structured_report_with_one_request():
    options = [_option("260925", "C", 100)]
    with (
        patch("tradingagents.dataflows.option_selector._today", return_value=TODAY),
        patch(
            "tradingagents.dataflows.option_selector.requests.get",
            return_value=_response(_payload(options)),
        ) as get,
    ):
        legacy = select_equity_option_contract("AAPL", "bullish", TODAY.isoformat())

    assert "# Cboe delayed equity option contract selection" in legacy
    assert options[0]["option"] in legacy
    get.assert_called_once()
