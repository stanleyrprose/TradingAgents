from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from tradingagents.dataflows.equity_options import EquityOptionSnapshot
from tradingagents.option_hold_roll_decision import compare_hold_vs_roll
from tradingagents.option_roll_planner import OptionRollCandidate, OptionRollPlanResult


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
        "bid": 11.0,
        "ask": 11.3,
        "midpoint": 11.15,
        "spread_pct": 2.69,
        "last": 11.1,
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


def _roll_candidate(**overrides):
    values = {
        "symbol": "AAPL261016C00320000",
        "expiry": "2026-10-16",
        "dte": 35,
        "dte_extension": 21,
        "strike": 320.0,
        "delta": 0.62,
        "iv": 0.25,
        "spread_pct": 1.5,
        "selector_score": 90.0,
        "close_credit_per_share": 11.0,
        "open_debit_per_share": 15.0,
        "net_debit_per_share": 4.0,
        "net_cash_outflow": 800.0,
        "new_gross_premium_at_risk": 3000.0,
        "delta_shares_after_roll": 124.0,
        "delta_shares_change": 2.0,
        "theta_dollars_per_day_after_roll": -30.0,
        "theta_dollars_per_day_change": 14.0,
    }
    values.update(overrides)
    return OptionRollCandidate(**values)


def _plan(*candidates, status="COMPARE"):
    return OptionRollPlanResult(
        status=status,
        reason="test",
        candidates=tuple(candidates),
        report="roll report",
    )


def test_call_hold_and_roll_lifecycle_breakevens_use_original_trade_economics():
    result = compare_hold_vs_roll(
        _snapshot(),
        _plan(_roll_candidate()),
        entry_premium=8.25,
        contracts=2,
    )

    assert result.status == "COMPARE"
    assert result.hold is not None
    assert result.hold.lifecycle_breakeven == pytest.approx(328.25)
    assert result.hold.forward_opportunity_breakeven == pytest.approx(331.0)
    item = result.alternatives[0]
    assert item.lifecycle_net_premium_per_share == pytest.approx(12.25)
    assert item.lifecycle_breakeven == pytest.approx(332.25)
    assert item.lifecycle_breakeven_improvement == pytest.approx(-4.0)


def test_hold_forward_capital_at_risk_uses_current_bid_not_original_entry_cost():
    result = compare_hold_vs_roll(
        _snapshot(bid=11.0),
        _plan(_roll_candidate()),
        entry_premium=8.25,
        contracts=2,
    )

    assert result.hold is not None
    assert result.hold.capital_at_risk_from_now == pytest.approx(2200)
    assert "current liquidation value at risk" in result.report
    assert "sunk historical cost" in result.report


def test_roll_keeps_additional_cash_separate_from_new_gross_premium_risk():
    result = compare_hold_vs_roll(
        _snapshot(),
        _plan(_roll_candidate(net_cash_outflow=800, new_gross_premium_at_risk=3000)),
        entry_premium=8.25,
        contracts=2,
    )
    item = result.alternatives[0]

    assert item.additional_cash_outflow == 800
    assert item.new_gross_premium_at_risk == 3000
    assert "They are not interchangeable" in result.report


def test_theta_relief_and_delta_drift_are_relative_to_hold():
    result = compare_hold_vs_roll(
        _snapshot(delta=0.61, theta=-0.22),
        _plan(
            _roll_candidate(
                delta_shares_after_roll=124,
                delta_shares_change=2,
                theta_dollars_per_day_after_roll=-30,
            )
        ),
        entry_premium=8.25,
        contracts=2,
    )
    item = result.alternatives[0]

    assert result.hold is not None
    assert result.hold.delta_shares == pytest.approx(122)
    assert result.hold.theta_dollars_per_day == pytest.approx(-44)
    assert item.theta_relief_dollars_per_day == pytest.approx(14)
    assert "reduces daily theta drag by about $14.00" in item.summary
    assert "changes delta exposure by +2.00 shares" in item.summary


def test_put_breakeven_directionality_marks_higher_roll_breakeven_as_improvement():
    snapshot = _snapshot(
        symbol="AAPL260925P00320000",
        right="P",
        delta=-0.58,
        strike=320,
    )
    candidate = _roll_candidate(
        symbol="AAPL261016P00325000",
        strike=325,
        delta=-0.56,
        delta_shares_after_roll=-112,
        delta_shares_change=4,
        close_credit_per_share=11,
        open_debit_per_share=15,
    )
    result = compare_hold_vs_roll(
        snapshot,
        _plan(candidate),
        entry_premium=8.25,
        contracts=2,
    )

    assert result.hold is not None
    assert result.hold.lifecycle_breakeven == pytest.approx(311.75)
    item = result.alternatives[0]
    assert item.lifecycle_net_premium_per_share == pytest.approx(12.25)
    assert item.lifecycle_breakeven == pytest.approx(312.75)
    assert item.lifecycle_breakeven_improvement == pytest.approx(1.0)
    assert "improves lifecycle breakeven by 1.00" in item.summary


def test_net_credit_can_improve_lifecycle_breakeven_without_erasing_new_premium_risk():
    candidate = _roll_candidate(
        strike=315,
        close_credit_per_share=18,
        open_debit_per_share=12,
        net_debit_per_share=-6,
        net_cash_outflow=-1200,
        new_gross_premium_at_risk=2400,
    )
    result = compare_hold_vs_roll(
        _snapshot(bid=18),
        _plan(candidate),
        entry_premium=8.25,
        contracts=2,
    )
    item = result.alternatives[0]

    assert item.lifecycle_net_premium_per_share == pytest.approx(2.25)
    assert item.lifecycle_breakeven == pytest.approx(317.25)
    assert item.new_gross_premium_at_risk == 2400
    assert "releases about $1,200.00 cash" in item.summary


def test_non_compare_roll_plan_does_not_invent_a_decision():
    result = compare_hold_vs_roll(
        _snapshot(),
        _plan(status="NO_ROLL"),
        entry_premium=8.25,
        contracts=2,
    )

    assert result.status == "NO_COMPARE"
    assert result.hold is None
    assert result.alternatives == ()
    assert "requires a COMPARE roll plan" in result.reason


@pytest.mark.parametrize(
    ("premium", "contracts", "match"),
    [
        (0, 1, "entry_premium"),
        (-1, 1, "entry_premium"),
        (float("nan"), 1, "entry_premium"),
        (8.25, 0, "contracts"),
        (8.25, True, "contracts"),
    ],
)
def test_invalid_inputs_fail_before_comparison(premium, contracts, match):
    with pytest.raises(ValueError, match=match):
        compare_hold_vs_roll(
            _snapshot(),
            _plan(_roll_candidate()),
            entry_premium=premium,
            contracts=contracts,
        )


def test_missing_bid_keeps_hold_forward_metrics_unavailable_but_lifecycle_comparison_valid():
    result = compare_hold_vs_roll(
        _snapshot(bid=None),
        _plan(_roll_candidate()),
        entry_premium=8.25,
        contracts=2,
    )

    assert result.hold is not None
    assert result.hold.capital_at_risk_from_now is None
    assert result.hold.forward_opportunity_breakeven is None
    assert result.status == "COMPARE"


def test_prior_roll_lifecycle_basis_is_not_replaced_by_current_leg_entry_cost():
    result = compare_hold_vs_roll(
        _snapshot(strike=320.0, bid=16.0),
        _plan(
            _roll_candidate(
                strike=320.0,
                close_credit_per_share=16.0,
                open_debit_per_share=18.0,
                net_debit_per_share=2.0,
                net_cash_outflow=400.0,
            )
        ),
        entry_premium=15.0,
        lifecycle_net_premium_per_share=12.25,
        contracts=2,
    )

    assert result.hold is not None
    assert result.hold.lifecycle_breakeven == pytest.approx(332.25)
    assert result.alternatives[0].lifecycle_net_premium_per_share == pytest.approx(14.25)
    assert result.alternatives[0].lifecycle_breakeven == pytest.approx(334.25)
    assert "Current-leg entry premium: $15.00" in result.report
    assert "Lifecycle net premium basis: $12.25" in result.report


def test_lifecycle_basis_can_be_zero_or_negative_after_net_credit_rolls():
    zero = compare_hold_vs_roll(
        _snapshot(strike=320.0),
        _plan(_roll_candidate(strike=320.0)),
        entry_premium=15.0,
        lifecycle_net_premium_per_share=0.0,
        contracts=1,
    )
    credit = compare_hold_vs_roll(
        _snapshot(strike=320.0),
        _plan(_roll_candidate(strike=320.0)),
        entry_premium=15.0,
        lifecycle_net_premium_per_share=-2.0,
        contracts=1,
    )

    assert zero.hold is not None and zero.hold.lifecycle_breakeven == pytest.approx(320.0)
    assert credit.hold is not None and credit.hold.lifecycle_breakeven == pytest.approx(318.0)


def test_nonfinite_lifecycle_basis_is_rejected():
    with pytest.raises(ValueError, match="lifecycle_net_premium_per_share"):
        compare_hold_vs_roll(
            _snapshot(),
            _plan(_roll_candidate()),
            entry_premium=8.25,
            lifecycle_net_premium_per_share=float("nan"),
            contracts=2,
        )


def test_report_explicitly_refuses_automatic_hold_or_roll_label():
    result = compare_hold_vs_roll(
        _snapshot(),
        _plan(_roll_candidate()),
        entry_premium=8.25,
        contracts=2,
    )

    assert "not a recommendation or order" in result.report
    assert "No alternative is automatically labeled better" in result.report
    assert "would be arbitrary" in result.report


def test_result_objects_are_frozen():
    result = compare_hold_vs_roll(
        _snapshot(),
        _plan(_roll_candidate()),
        entry_premium=8.25,
        contracts=2,
    )

    with pytest.raises(FrozenInstanceError):
        result.status = "ROLL"
    with pytest.raises(FrozenInstanceError):
        result.alternatives[0].symbol = "OTHER"
