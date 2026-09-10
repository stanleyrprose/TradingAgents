from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from tradingagents.dataflows.equity_options import EquityOptionSnapshot
from tradingagents.option_position_manager import (
    evaluate_long_option_position,
    resolve_long_option_position_inputs,
    resolve_option_exit_policy,
)


def _snapshot(**overrides):
    values = {
        "symbol": "AAPL260918C00300000",
        "underlying": "AAPL",
        "expiry": date(2026, 9, 18),
        "right": "C",
        "strike": 300.0,
        "as_of": date(2026, 9, 10),
        "dte": 8,
        "source_timestamp": "2026-09-10 15:45:00",
        "underlying_spot": 315.0,
        "bid": 16.0,
        "ask": 18.0,
        "midpoint": 17.0,
        "spread_pct": 11.7647058824,
        "last": 16.5,
        "iv": 0.30,
        "delta": 0.75,
        "gamma": 0.0123,
        "vega": 0.2345,
        "theta": -0.1234,
        "open_interest": 100.0,
        "volume": 10.0,
    }
    values.update(overrides)
    return EquityOptionSnapshot(**values)


def test_position_input_resolver_normalizes_and_rejects_invalid_values():
    assert resolve_long_option_position_inputs(entry_premium=5, contracts=2) == (5.0, 2)

    for premium in (0, -1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="entry_premium"):
            resolve_long_option_position_inputs(entry_premium=premium, contracts=1)
    for contracts in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive whole number"):
            resolve_long_option_position_inputs(entry_premium=5, contracts=contracts)


def test_exit_policy_has_no_defaults_and_validates_boundaries():
    assert resolve_option_exit_policy() == {
        "take_profit_pct": None,
        "stop_loss_pct": None,
        "exit_at_dte": None,
        "max_theta_burn_pct_per_day": None,
    }
    assert resolve_option_exit_policy(
        take_profit_pct=50,
        stop_loss_pct=100,
        exit_at_dte=0,
        max_theta_burn_pct_per_day=2,
    ) == {
        "take_profit_pct": 50.0,
        "stop_loss_pct": 100.0,
        "exit_at_dte": 0,
        "max_theta_burn_pct_per_day": 2.0,
    }
    with pytest.raises(ValueError, match="less than or equal to 100"):
        resolve_option_exit_policy(stop_loss_pct=100.1)
    with pytest.raises(ValueError, match="nonnegative whole number"):
        resolve_option_exit_policy(exit_at_dte=True)
    with pytest.raises(ValueError, match="between 0 and 3650"):
        resolve_option_exit_policy(exit_at_dte=-1)


def test_report_only_uses_delayed_bid_as_conservative_liquidation_proxy():
    result = evaluate_long_option_position(
        _snapshot(), entry_premium=10, contracts=2
    )

    assert result.status == "REPORT_ONLY"
    assert result.should_exit is None
    assert result.entry_cost == 2_000
    assert result.liquidation_price_proxy == 16
    assert result.current_liquidation_value == 3_200
    assert result.pnl_dollars == 1_200
    assert result.pnl_pct == 60
    assert result.theta_burn_pct_of_midpoint_per_day == pytest.approx(
        abs(-0.1234) / 17 * 100
    )
    assert result.delta_shares == 150
    assert result.delta_notional_proxy == 47_250
    assert result.gamma_delta_shares_per_dollar == pytest.approx(2.46)
    assert result.vega_dollars_per_vol_point == pytest.approx(46.9)
    assert result.theta_dollars_per_day == pytest.approx(-24.68)
    assert "No exit policy supplied" in result.report


def test_take_profit_equality_or_above_triggers_exit():
    result = evaluate_long_option_position(
        _snapshot(bid=15, ask=17, midpoint=16),
        entry_premium=10,
        contracts=1,
        take_profit_pct=50,
    )

    assert result.pnl_pct == 50
    assert result.status == "EXIT"
    assert result.should_exit is True
    assert result.triggers[0].triggered


def test_stop_loss_equality_triggers_exit_for_long_premium():
    result = evaluate_long_option_position(
        _snapshot(bid=8, ask=9, midpoint=8.5),
        entry_premium=10,
        contracts=1,
        stop_loss_pct=20,
    )

    assert result.pnl_pct == -20
    assert result.status == "EXIT"
    assert result.triggers[0].triggered


def test_zero_bid_is_valid_minus_100_percent_liquidation_proxy():
    result = evaluate_long_option_position(
        _snapshot(bid=0, ask=0.5, midpoint=0.25),
        entry_premium=10,
        contracts=1,
        stop_loss_pct=100,
    )

    assert result.liquidation_price_proxy == 0
    assert result.current_liquidation_value == 0
    assert result.pnl_pct == -100
    assert result.status == "EXIT"


def test_exit_at_dte_equality_triggers_exit():
    result = evaluate_long_option_position(
        _snapshot(dte=5),
        entry_premium=10,
        contracts=1,
        exit_at_dte=5,
    )

    assert result.status == "EXIT"
    assert result.triggers[0].actual == 5
    assert result.triggers[0].triggered


def test_theta_burn_threshold_uses_current_midpoint():
    result = evaluate_long_option_position(
        _snapshot(theta=-0.17, midpoint=17),
        entry_premium=10,
        contracts=1,
        max_theta_burn_pct_per_day=1,
    )

    assert result.theta_burn_pct_of_midpoint_per_day == pytest.approx(1)
    assert result.status == "EXIT"


def test_explicit_policy_holds_when_all_evaluable_thresholds_are_clear():
    result = evaluate_long_option_position(
        _snapshot(),
        entry_premium=10,
        contracts=1,
        take_profit_pct=100,
        stop_loss_pct=50,
        exit_at_dte=3,
        max_theta_burn_pct_per_day=5,
    )

    assert result.status == "HOLD"
    assert result.should_exit is False
    assert all(trigger.evaluable and not trigger.triggered for trigger in result.triggers)


def test_missing_required_metric_is_review_not_false_hold():
    result = evaluate_long_option_position(
        _snapshot(theta=None),
        entry_premium=10,
        contracts=1,
        max_theta_burn_pct_per_day=2,
    )

    assert result.status == "REVIEW"
    assert result.should_exit is None
    assert not result.triggers[0].evaluable
    assert "not evaluable" in result.triggers[0].reason


def test_missing_bid_makes_pnl_policy_review():
    result = evaluate_long_option_position(
        _snapshot(bid=None),
        entry_premium=10,
        contracts=1,
        stop_loss_pct=25,
    )

    assert result.pnl_pct is None
    assert result.status == "REVIEW"
    assert not result.triggers[0].evaluable


def test_triggered_exit_takes_precedence_over_another_unevaluable_policy():
    result = evaluate_long_option_position(
        _snapshot(dte=2, theta=None),
        entry_premium=10,
        contracts=1,
        exit_at_dte=3,
        max_theta_burn_pct_per_day=2,
    )

    assert result.status == "EXIT"
    assert any(trigger.triggered for trigger in result.triggers)
    assert any(not trigger.evaluable for trigger in result.triggers)


def test_long_put_keeps_signed_delta_and_uses_same_exit_economics():
    result = evaluate_long_option_position(
        _snapshot(
            symbol="AAPL260918P00300000",
            right="P",
            delta=-0.40,
            bid=12,
            ask=13,
            midpoint=12.5,
        ),
        entry_premium=10,
        contracts=2,
        take_profit_pct=20,
    )

    assert result.status == "EXIT"
    assert result.pnl_pct == 20
    assert result.delta_shares == -80
    assert "Type: put" in result.report


def test_report_states_execution_and_model_boundaries():
    report = evaluate_long_option_position(
        _snapshot(), entry_premium=10, contracts=1
    ).report

    assert "not an order" in report
    assert "delayed Cboe bid" in report
    assert "not midpoint" in report
    assert "does not re-run the underlying investment thesis" in report
    assert "rolling into another contract" in report


def test_result_and_triggers_are_frozen():
    result = evaluate_long_option_position(
        _snapshot(), entry_premium=10, contracts=1, take_profit_pct=50
    )

    with pytest.raises(FrozenInstanceError):
        result.status = "HOLD"
    with pytest.raises(FrozenInstanceError):
        result.triggers[0].triggered = False
