from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from tradingagents.dataflows.equity_options import EquityOptionSnapshot
from tradingagents.dataflows.option_selector import OptionCandidate, OptionSelectionResult
from tradingagents.option_roll_planner import plan_long_option_roll
from tradingagents.option_thesis_gate import LongOptionThesisRefresh


def _snapshot(**overrides):
    values = {
        "symbol": "AAPL260925C00320000",
        "underlying": "AAPL",
        "expiry": date(2026, 9, 25),
        "right": "C",
        "strike": 320.0,
        "as_of": date(2026, 9, 11),
        "dte": 14,
        "source_timestamp": "2026-09-11 08:00:00",
        "underlying_spot": 324.0,
        "bid": 9.25,
        "ask": 9.60,
        "midpoint": 9.425,
        "spread_pct": 3.71,
        "last": 9.40,
        "iv": 0.26,
        "delta": 0.61,
        "gamma": 0.02,
        "vega": 0.25,
        "theta": -0.22,
        "open_interest": 1000.0,
        "volume": 500.0,
    }
    values.update(overrides)
    return EquityOptionSnapshot(**values)


def _refresh(status="CONFIRMED", *, option_direction="bullish", current_direction="bullish"):
    return LongOptionThesisRefresh(
        option_direction=option_direction,
        current_direction=current_direction,
        status=status,
        portfolio_rating="Overweight" if current_direction == "bullish" else "Underweight",
        trader_action="Buy" if current_direction == "bullish" else "Sell",
        reason=f"{status} test",
    )


def _candidate(symbol="AAPL261016C00320000", **overrides):
    values = {
        "symbol": symbol,
        "expiry": date(2026, 10, 16),
        "right": "C",
        "strike": 320.0,
        "dte": 35,
        "bid": 14.0,
        "ask": 14.2,
        "midpoint": 14.1,
        "spread_pct": 1.42,
        "delta": 0.62,
        "abs_delta": 0.62,
        "iv": 0.25,
        "oi": 5000.0,
        "volume": 1000.0,
        "theta": -0.15,
        "theta_burden": 0.0106,
        "breakeven": 334.1,
        "required_move": 0.031,
        "components": (),
        "score": 90.0,
        "gamma": 0.015,
        "vega": 0.30,
    }
    values.update(overrides)
    return OptionCandidate(**values)


def _selection(*candidates, reason=None):
    return OptionSelectionResult(
        tuple(candidates),
        "selection report",
        unavailable_reason=reason,
        underlying_spot=324.0,
    )


def test_invalidated_thesis_never_offers_roll_candidates():
    result = plan_long_option_roll(
        _snapshot(),
        _refresh("INVALIDATED", current_direction="bearish"),
        contracts=2,
        selection=_selection(_candidate()),
    )

    assert result.status == "NO_ROLL"
    assert result.candidates == ()
    assert "opposite" in result.reason
    assert "No replacement contract is proposed" in result.report


def test_neutral_thesis_is_no_roll_not_invalidation():
    refresh = LongOptionThesisRefresh(
        option_direction="bullish",
        current_direction=None,
        status="NEUTRAL",
        portfolio_rating="Hold",
        trader_action="Hold",
        reason="no consensus",
    )
    result = plan_long_option_roll(_snapshot(), refresh, contracts=1)

    assert result.status == "NO_ROLL"
    assert "lacks directional confirmation" in result.reason


def test_confirmed_thesis_requires_replacement_selection():
    result = plan_long_option_roll(_snapshot(), _refresh(), contracts=1)

    assert result.status == "REVIEW"
    assert "selection is unavailable" in result.reason


def test_unavailable_selection_reason_is_preserved():
    result = plan_long_option_roll(
        _snapshot(),
        _refresh(),
        contracts=1,
        selection=_selection(reason="no eligible contracts"),
    )

    assert result.status == "REVIEW"
    assert result.reason == "no eligible contracts"


def test_missing_current_bid_is_review_not_midpoint_substitution():
    result = plan_long_option_roll(
        _snapshot(bid=None),
        _refresh(),
        contracts=1,
        selection=_selection(_candidate()),
    )

    assert result.status == "REVIEW"
    assert "current delayed bid is unavailable" in result.reason


def test_only_same_right_strictly_later_expiry_candidates_are_compared():
    same_contract = _candidate(
        symbol="AAPL260925C00320000",
        expiry=date(2026, 9, 25),
        dte=14,
    )
    same_expiry_other_strike = _candidate(
        symbol="AAPL260925C00325000",
        expiry=date(2026, 9, 25),
        dte=14,
        strike=325.0,
    )
    later_put = _candidate(
        symbol="AAPL261016P00320000",
        right="P",
        delta=-0.55,
        abs_delta=0.55,
    )
    valid = _candidate()
    result = plan_long_option_roll(
        _snapshot(),
        _refresh(),
        contracts=1,
        selection=_selection(same_contract, same_expiry_other_strike, later_put, valid),
    )

    assert result.status == "COMPARE"
    assert [item.symbol for item in result.candidates] == [valid.symbol]


def test_roll_cash_math_uses_old_bid_and_new_ask():
    result = plan_long_option_roll(
        _snapshot(bid=9.25, delta=0.61, theta=-0.22),
        _refresh(),
        contracts=2,
        selection=_selection(_candidate(ask=14.2, delta=0.62, theta=-0.15)),
    )
    item = result.candidates[0]

    assert item.close_credit_per_share == 9.25
    assert item.open_debit_per_share == 14.2
    assert item.net_debit_per_share == pytest.approx(4.95)
    assert item.net_cash_outflow == pytest.approx(990)
    assert item.new_gross_premium_at_risk == pytest.approx(2840)
    assert item.delta_shares_after_roll == pytest.approx(124)
    assert item.delta_shares_change == pytest.approx(2)
    assert item.theta_dollars_per_day_after_roll == pytest.approx(-30)
    assert item.theta_dollars_per_day_change == pytest.approx(14)
    assert item.dte_extension == 21


def test_net_credit_is_negative_cash_outflow_but_new_premium_risk_stays_gross():
    result = plan_long_option_roll(
        _snapshot(bid=15),
        _refresh(),
        contracts=1,
        selection=_selection(_candidate(ask=12)),
    )
    item = result.candidates[0]

    assert item.net_debit_per_share == -3
    assert item.net_cash_outflow == -300
    assert item.new_gross_premium_at_risk == 1200
    assert "not the same as the roll net debit" in result.report


def test_top_n_limits_output_without_reordering_selector_ranking():
    first = _candidate("AAPL261016C00320000", score=90)
    second = _candidate(
        "AAPL261023C00320000",
        expiry=date(2026, 10, 23),
        dte=42,
        score=89,
    )
    result = plan_long_option_roll(
        _snapshot(),
        _refresh(),
        contracts=1,
        selection=_selection(first, second),
        top_n=1,
    )

    assert [item.symbol for item in result.candidates] == [first.symbol]


def test_long_put_uses_bearish_confirmation_and_signed_delta():
    snapshot = _snapshot(
        symbol="AAPL260925P00320000",
        right="P",
        delta=-0.58,
        theta=-0.20,
    )
    refresh = _refresh(
        option_direction="bearish",
        current_direction="bearish",
    )
    candidate = _candidate(
        "AAPL261016P00320000",
        right="P",
        delta=-0.56,
        abs_delta=0.56,
    )
    result = plan_long_option_roll(
        snapshot,
        refresh,
        contracts=2,
        selection=_selection(candidate),
    )

    assert result.status == "COMPARE"
    assert result.candidates[0].delta_shares_after_roll == pytest.approx(-112)


def test_refresh_direction_must_match_option_right():
    with pytest.raises(ValueError, match="does not match"):
        plan_long_option_roll(
            _snapshot(),
            _refresh(option_direction="bearish", current_direction="bearish"),
            contracts=1,
        )


@pytest.mark.parametrize("contracts", [0, -1, True, 1.5])
def test_contract_count_must_be_positive_integer(contracts):
    with pytest.raises(ValueError, match="positive whole number"):
        plan_long_option_roll(_snapshot(), _refresh(), contracts=contracts)


@pytest.mark.parametrize("top_n", [0, 21, True, 1.5])
def test_top_n_is_bounded(top_n):
    with pytest.raises(ValueError, match="between 1 and 20"):
        plan_long_option_roll(_snapshot(), _refresh(), contracts=1, top_n=top_n)


def test_no_strictly_later_candidate_is_review():
    result = plan_long_option_roll(
        _snapshot(),
        _refresh(),
        contracts=1,
        selection=_selection(
            _candidate(
                symbol="AAPL260925C00325000",
                expiry=date(2026, 9, 25),
                dte=14,
            )
        ),
    )

    assert result.status == "REVIEW"
    assert "strictly later-expiry" in result.reason


def test_report_states_planning_and_execution_boundaries():
    result = plan_long_option_roll(
        _snapshot(),
        _refresh(),
        contracts=1,
        selection=_selection(_candidate()),
    )

    assert "planning comparison, not an order" in result.report
    assert "not guaranteed fills" in result.report
    assert "not expected return" in result.report
    assert "NEUTRAL or INVALIDATED produces NO_ROLL" in result.report


def test_result_and_candidates_are_frozen():
    result = plan_long_option_roll(
        _snapshot(),
        _refresh(),
        contracts=1,
        selection=_selection(_candidate()),
    )

    with pytest.raises(FrozenInstanceError):
        result.status = "NO_ROLL"
    with pytest.raises(FrozenInstanceError):
        result.candidates[0].symbol = "OTHER"
