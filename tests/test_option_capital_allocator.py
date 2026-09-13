from dataclasses import FrozenInstanceError, replace
from datetime import date

import pytest

from tradingagents.dataflows.option_selector import OptionCandidate
from tradingagents.option_capital_allocator import (
    allocate_long_option_premium_risk,
    resolve_long_option_risk_budget,
)


def _candidate(
    symbol: str = "AAPL261016C00100000",
    *,
    ask: float = 5.0,
    midpoint: float = 4.0,
    score: float = 90.0,
) -> OptionCandidate:
    return OptionCandidate(
        symbol=symbol,
        expiry=date(2026, 10, 16),
        right="C",
        strike=100.0,
        dte=36,
        bid=3.0,
        ask=ask,
        midpoint=midpoint,
        spread_pct=10.0,
        delta=0.55,
        abs_delta=0.55,
        iv=0.3,
        oi=500.0,
        volume=100.0,
        theta=-0.05,
        theta_burden=0.01,
        breakeven=104.0,
        required_move=0.04,
        components=(),
        score=score,
    )


def test_premium_budget_alone_sets_risk_budget():
    result = allocate_long_option_premium_risk([_candidate()], premium_budget=1_200)

    assert result.risk_budget == 1_200
    assert result.account_equity is None
    assert result.max_risk_pct is None
    assert result.premium_budget == 1_200
    assert result.positions[0].contracts == 2


def test_account_percentage_alone_sets_risk_budget():
    result = allocate_long_option_premium_risk(
        [_candidate()], account_equity=50_000, max_risk_pct=2
    )

    assert result.risk_budget == 1_000
    assert result.premium_budget is None
    assert result.positions[0].contracts == 2


@pytest.mark.parametrize(
    ("premium_budget", "expected"),
    [(600, 600), (2_000, 1_000)],
)
def test_both_budget_methods_use_the_smaller_hard_cap(premium_budget, expected):
    result = allocate_long_option_premium_risk(
        [_candidate()],
        account_equity=50_000,
        max_risk_pct=2,
        premium_budget=premium_budget,
    )

    assert result.risk_budget == expected


def test_public_budget_resolver_uses_same_normalization_and_smaller_cap():
    assert resolve_long_option_risk_budget(
        account_equity=50_000,
        max_risk_pct=2,
        premium_budget=600,
    ) == (600, 50_000, 2, 600)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"account_equity": 10_000},
        {"max_risk_pct": 2},
        {"premium_budget": 0},
        {"account_equity": 10_000, "max_risk_pct": 101},
    ],
)
def test_public_budget_resolver_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        resolve_long_option_risk_budget(
            account_equity=kwargs.get("account_equity"),
            max_risk_pct=kwargs.get("max_risk_pct"),
            premium_budget=kwargs.get("premium_budget"),
        )


def test_whole_contract_rounding_retains_fractional_remainder():
    result = allocate_long_option_premium_risk(
        [_candidate(ask=3.0)], premium_budget=1_000
    )

    assert result.positions[0].contracts == 3
    assert result.total_premium_at_risk == 900
    assert result.unused_budget == 100


def test_all_zero_contracts_is_available_but_has_no_position():
    result = allocate_long_option_premium_risk(
        [_candidate(ask=10.01)], premium_budget=1_000
    )

    assert result.available
    assert not result.has_position
    assert result.positions[0].contracts == 0
    assert result.total_premium_at_risk == 0
    assert result.unused_budget == 1_000


def test_two_candidates_receive_equal_buckets_in_input_order():
    candidates = [_candidate("FIRST", ask=2), _candidate("SECOND", ask=2)]

    result = allocate_long_option_premium_risk(candidates, premium_budget=1_000)

    assert [position.symbol for position in result.positions] == ["FIRST", "SECOND"]
    assert [position.rank for position in result.positions] == [1, 2]
    assert [position.target_budget for position in result.positions] == [500, 500]
    assert [position.contracts for position in result.positions] == [2, 2]


def test_three_candidates_receive_equal_buckets():
    candidates = [
        _candidate("ONE", ask=1),
        _candidate("TWO", ask=1),
        _candidate("THREE", ask=1),
    ]

    result = allocate_long_option_premium_risk(candidates, premium_budget=900)

    assert [position.target_budget for position in result.positions] == [300, 300, 300]
    assert [position.contracts for position in result.positions] == [3, 3, 3]


def test_unaffordable_bucket_is_not_reallocated_to_another_candidate():
    candidates = [_candidate("EXPENSIVE", ask=6), _candidate("CHEAP", ask=1)]

    result = allocate_long_option_premium_risk(candidates, premium_budget=1_000)

    assert [position.contracts for position in result.positions] == [0, 5]
    assert result.total_premium_at_risk == 500
    assert result.unused_budget == 500


def test_selector_score_changes_do_not_change_sizes():
    low_score = _candidate("LOW", ask=2.5, score=1)
    high_score = _candidate("HIGH", ask=2.5, score=99)

    result = allocate_long_option_premium_risk(
        [low_score, high_score], premium_budget=1_000
    )

    assert [position.contracts for position in result.positions] == [2, 2]
    swapped_scores = allocate_long_option_premium_risk(
        [replace(low_score, score=100), replace(high_score, score=0)],
        premium_budget=1_000,
    )
    assert [position.contracts for position in swapped_scores.positions] == [2, 2]


def test_ask_not_midpoint_drives_premium_and_contract_count():
    result = allocate_long_option_premium_risk(
        [_candidate(ask=6, midpoint=2)], premium_budget=1_000
    )

    position = result.positions[0]
    assert position.ask_price_proxy == 6
    assert position.premium_per_contract == 600
    assert position.contracts == 1
    assert position.premium_at_risk == 600


def test_position_and_total_portfolio_risk_percentages_use_equity():
    result = allocate_long_option_premium_risk(
        [_candidate("ONE", ask=2), _candidate("TWO", ask=3)],
        account_equity=50_000,
        max_risk_pct=2,
    )

    assert [position.portfolio_risk_pct for position in result.positions] == [0.8, 0.6]
    assert result.portfolio_risk_pct == pytest.approx(1.4)
    assert result.total_premium_at_risk == 700


def test_portfolio_risk_percentage_is_none_without_equity():
    result = allocate_long_option_premium_risk([_candidate()], premium_budget=1_000)

    assert result.positions[0].portfolio_risk_pct is None
    assert result.portfolio_risk_pct is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"account_equity": 10_000},
        {"max_risk_pct": 2},
        {},
    ],
)
def test_missing_budget_or_invalid_equity_percentage_pairing_raises(kwargs):
    with pytest.raises(ValueError):
        allocate_long_option_premium_risk([_candidate()], **kwargs)


@pytest.mark.parametrize("max_risk_pct", [0, -1, 100.01, float("nan"), float("inf"), True])
def test_invalid_max_risk_percentage_raises(max_risk_pct):
    with pytest.raises(ValueError):
        allocate_long_option_premium_risk(
            [_candidate()], account_equity=10_000, max_risk_pct=max_risk_pct
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_equity", 0),
        ("account_equity", -1),
        ("account_equity", float("nan")),
        ("account_equity", float("inf")),
        ("account_equity", True),
        ("premium_budget", 0),
        ("premium_budget", -1),
        ("premium_budget", float("nan")),
        ("premium_budget", float("inf")),
        ("premium_budget", False),
    ],
)
def test_invalid_equity_and_premium_budget_values_raise(field, value):
    kwargs = (
        {"account_equity": value, "max_risk_pct": 2}
        if field == "account_equity"
        else {"premium_budget": value}
    )
    with pytest.raises(ValueError):
        allocate_long_option_premium_risk([_candidate()], **kwargs)


@pytest.mark.parametrize(
    ("candidate", "reason_fragment"),
    [
        (_candidate(symbol=" "), "symbol"),
        (_candidate(ask=0), "ask"),
        (_candidate(ask=float("nan")), "ask"),
        (_candidate(ask=float("inf")), "ask"),
        (_candidate(ask=True), "ask"),
    ],
)
def test_malformed_candidate_makes_result_unavailable(candidate, reason_fragment):
    result = allocate_long_option_premium_risk([candidate], premium_budget=1_000)

    assert not result.available
    assert not result.has_position
    assert result.positions == ()
    assert reason_fragment in result.unavailable_reason
    assert reason_fragment in result.report


def test_one_malformed_candidate_invalidates_entire_input_without_skipping():
    result = allocate_long_option_premium_risk(
        [_candidate("VALID"), _candidate("BAD", ask=-1)], premium_budget=1_000
    )

    assert not result.available
    assert result.positions == ()
    assert "candidate 2 (BAD)" in result.unavailable_reason
    assert result.total_premium_at_risk == 0


def test_empty_candidates_return_unavailable_result_with_reason():
    result = allocate_long_option_premium_risk([], premium_budget=1_000)

    assert not result.available
    assert not result.has_position
    assert result.unavailable_reason == "no approved option candidates were supplied"
    assert result.risk_budget == 1_000
    assert result.unused_budget == 1_000


def test_report_contains_required_risk_and_method_caveats():
    report = allocate_long_option_premium_risk(
        [_candidate()], premium_budget=1_000
    ).report.lower()

    assert "ask" in report and "not a guaranteed fill" in report
    assert "premium can go to zero" in report
    assert "top3 is not diversification" in report
    assert "selector score is intentionally not a sizing weight" in report
    for excluded in ("commissions", "slippage", "taxes", "greeks", "correlation", "margin"):
        assert excluded in report
    assert "exercise or assignment complications" in report


def test_result_and_position_dataclasses_are_frozen():
    result = allocate_long_option_premium_risk([_candidate()], premium_budget=1_000)

    with pytest.raises(FrozenInstanceError):
        result.risk_budget = 2_000
    with pytest.raises(FrozenInstanceError):
        result.positions[0].contracts = 99
