from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from tradingagents.dataflows.equity_options import EquityOptionSnapshot
from tradingagents.dataflows.option_scenarios import black_scholes_price
from tradingagents.option_hold_roll_scenarios import compare_hold_vs_roll_scenarios
from tradingagents.option_roll_planner import OptionRollCandidate, OptionRollPlanResult


def _snapshot(**overrides):
    values = {
        "symbol": "AAPL260925C00100000",
        "underlying": "AAPL",
        "expiry": date(2026, 9, 25),
        "right": "C",
        "strike": 100.0,
        "as_of": date(2026, 9, 11),
        "dte": 14,
        "source_timestamp": "2026-09-11 08:00:00",
        "underlying_spot": 100.0,
        "bid": 10.0,
        "ask": 10.2,
        "midpoint": 10.1,
        "spread_pct": 1.98,
        "last": 10.0,
        "iv": 0.25,
        "delta": 0.60,
        "gamma": 0.02,
        "vega": 0.25,
        "theta": -0.20,
        "open_interest": 1000.0,
        "volume": 500.0,
    }
    values.update(overrides)
    return EquityOptionSnapshot(**values)


def _candidate(symbol="AAPL261016C00100000", **overrides):
    values = {
        "symbol": symbol,
        "expiry": "2026-10-16",
        "dte": 35,
        "dte_extension": 21,
        "strike": 100.0,
        "delta": 0.58,
        "iv": 0.30,
        "spread_pct": 1.5,
        "selector_score": 90.0,
        "close_credit_per_share": 10.0,
        "open_debit_per_share": 12.0,
        "net_debit_per_share": 2.0,
        "net_cash_outflow": 200.0,
        "new_gross_premium_at_risk": 1200.0,
        "delta_shares_after_roll": 58.0,
        "delta_shares_change": -2.0,
        "theta_dollars_per_day_after_roll": -15.0,
        "theta_dollars_per_day_change": 5.0,
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


def _point(result, *, horizon, spot_shock, iv_shock):
    return next(
        point
        for point in result.points
        if point.horizon_days == horizon
        and point.spot_change_pct == spot_shock
        and point.iv_shock_pp == iv_shock
    )


def test_default_grid_uses_3d_7d_and_old_expiry_with_27_points():
    result = compare_hold_vs_roll_scenarios(
        _snapshot(),
        _plan(_candidate()),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.04,
    )

    assert result.status == "COMPARE"
    assert result.horizons == (3, 7, 14)
    assert len(result.points) == 27
    assert result.summaries[0].scenario_count == 27


def test_hold_and_roll_lifecycle_pnl_use_same_scenario_and_original_entry_basis():
    snapshot = _snapshot()
    candidate = _candidate()
    result = compare_hold_vs_roll_scenarios(
        snapshot,
        _plan(candidate),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.04,
    )
    point = _point(result, horizon=3, spot_shock=0.0, iv_shock=0.0)

    hold_value = black_scholes_price(100, 100, 11 / 365, 0.04, 0.25, "C", 0.0)
    roll_value = black_scholes_price(100, 100, 32 / 365, 0.04, 0.30, "C", 0.0)
    expected_hold = (hold_value - 8.0) * 100
    expected_roll = ((10.0 - 8.0) + (roll_value - 12.0)) * 100

    assert point.hold_model_value_per_share == pytest.approx(hold_value)
    assert point.hold_lifecycle_pnl_dollars == pytest.approx(expected_hold)
    assert point.hold_forward_pnl_dollars == pytest.approx((hold_value - 10.0) * 100)
    assert point.alternatives[0].model_value_per_share == pytest.approx(roll_value)
    assert point.alternatives[0].lifecycle_pnl_dollars == pytest.approx(expected_roll)
    assert point.alternatives[0].advantage_vs_hold_dollars == pytest.approx(
        expected_roll - expected_hold
    )


def test_old_expiry_horizon_prices_hold_at_intrinsic_but_roll_keeps_remaining_time():
    result = compare_hold_vs_roll_scenarios(
        _snapshot(),
        _plan(_candidate()),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.04,
    )
    point = _point(result, horizon=14, spot_shock=0.05, iv_shock=0.0)

    assert point.hold_model_value_per_share == pytest.approx(5.0)
    assert point.alternatives[0].model_value_per_share > 5.0


def test_each_contract_keeps_its_own_iv_baseline_under_same_shock():
    snapshot = _snapshot(iv=0.20)
    candidate = _candidate(iv=0.40)
    result = compare_hold_vs_roll_scenarios(
        snapshot,
        _plan(candidate),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.0,
    )
    point = _point(result, horizon=3, spot_shock=0.0, iv_shock=0.05)

    expected_hold = black_scholes_price(100, 100, 11 / 365, 0.0, 0.25, "C", 0.0)
    expected_roll = black_scholes_price(100, 100, 32 / 365, 0.0, 0.45, "C", 0.0)
    assert point.hold_model_value_per_share == pytest.approx(expected_hold)
    assert point.alternatives[0].model_value_per_share == pytest.approx(expected_roll)


def test_negative_iv_shock_is_floored_at_one_percent():
    result = compare_hold_vs_roll_scenarios(
        _snapshot(iv=0.03),
        _plan(_candidate(iv=0.02)),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.0,
    )
    point = _point(result, horizon=3, spot_shock=0.0, iv_shock=-0.05)

    expected_hold = black_scholes_price(100, 100, 11 / 365, 0.0, 0.01, "C", 0.0)
    expected_roll = black_scholes_price(100, 100, 32 / 365, 0.0, 0.01, "C", 0.0)
    assert point.hold_model_value_per_share == pytest.approx(expected_hold)
    assert point.alternatives[0].model_value_per_share == pytest.approx(expected_roll)


def test_zero_bid_is_valid_and_roll_realized_leg_uses_zero_credit():
    snapshot = _snapshot(bid=0.0)
    candidate = _candidate(close_credit_per_share=0.0, net_debit_per_share=12.0)
    result = compare_hold_vs_roll_scenarios(
        snapshot,
        _plan(candidate),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.04,
    )

    assert result.status == "COMPARE"
    point = _point(result, horizon=3, spot_shock=0.0, iv_shock=0.0)
    roll_value = point.alternatives[0].model_value_per_share
    assert point.alternatives[0].lifecycle_pnl_dollars == pytest.approx(
        ((0.0 - 8.0) + (roll_value - 12.0)) * 100
    )


def test_roll_plan_close_credit_must_match_snapshot_bid():
    with pytest.raises(ValueError, match="does not match current snapshot bid"):
        compare_hold_vs_roll_scenarios(
            _snapshot(bid=10.0),
            _plan(_candidate(close_credit_per_share=9.5)),
            entry_premium=8.0,
            contracts=1,
            risk_free_rate=0.04,
        )


def test_missing_snapshot_iv_returns_no_compare_instead_of_crashing():
    result = compare_hold_vs_roll_scenarios(
        _snapshot(iv=None),
        _plan(_candidate()),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.04,
    )

    assert result.status == "NO_COMPARE"
    assert "IV is unavailable" in result.reason


def test_non_compare_roll_plan_does_not_create_scenarios():
    result = compare_hold_vs_roll_scenarios(
        _snapshot(),
        _plan(status="NO_ROLL"),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.04,
    )

    assert result.status == "NO_COMPARE"
    assert result.points == ()
    assert "requires a COMPARE roll plan" in result.reason


def test_put_old_expiry_uses_put_intrinsic_and_preserves_signed_lifecycle_math():
    snapshot = _snapshot(
        symbol="AAPL260925P00100000",
        right="P",
        delta=-0.60,
    )
    candidate = _candidate(
        symbol="AAPL261016P00100000",
        delta=-0.58,
        delta_shares_after_roll=-58.0,
    )
    result = compare_hold_vs_roll_scenarios(
        snapshot,
        _plan(candidate),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.04,
    )
    point = _point(result, horizon=14, spot_shock=-0.05, iv_shock=0.0)

    assert point.hold_model_value_per_share == pytest.approx(5.0)
    assert point.hold_lifecycle_pnl_dollars == pytest.approx(-300.0)
    assert point.alternatives[0].model_value_per_share > 5.0


def test_summary_counts_partition_grid_and_are_not_presented_as_probability():
    result = compare_hold_vs_roll_scenarios(
        _snapshot(),
        _plan(_candidate()),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.04,
    )
    summary = result.summaries[0]

    assert summary.outperform_count + summary.tie_count + summary.underperform_count == 27
    assert summary.min_advantage_dollars <= summary.median_advantage_dollars
    assert summary.median_advantage_dollars <= summary.max_advantage_dollars
    assert "not probabilities" in result.report
    assert "does not assign scenario probabilities" in result.report


def test_short_dte_deduplicates_horizons_instead_of_scenario_after_old_expiry():
    result = compare_hold_vs_roll_scenarios(
        _snapshot(dte=5, expiry=date(2026, 9, 16)),
        _plan(_candidate(dte=20, dte_extension=15)),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.04,
    )

    assert result.horizons == (3, 5)
    assert len(result.points) == 18


def test_prior_roll_lifecycle_basis_drives_hold_and_next_roll_scenario_pnl():
    snapshot = _snapshot(bid=10.0)
    candidate = _candidate(
        close_credit_per_share=10.0,
        open_debit_per_share=12.0,
    )
    result = compare_hold_vs_roll_scenarios(
        snapshot,
        _plan(candidate),
        entry_premium=15.0,
        lifecycle_net_premium_per_share=12.25,
        contracts=1,
        risk_free_rate=0.04,
    )
    point = _point(result, horizon=3, spot_shock=0.0, iv_shock=0.0)

    expected_hold = (point.hold_model_value_per_share - 12.25) * 100
    expected_roll = (
        (10.0 - 12.25)
        + (point.alternatives[0].model_value_per_share - 12.0)
    ) * 100
    assert point.hold_lifecycle_pnl_dollars == pytest.approx(expected_hold)
    assert point.alternatives[0].lifecycle_pnl_dollars == pytest.approx(expected_roll)
    assert "Current-leg entry premium: $15.00" in result.report
    assert "Lifecycle net premium basis: $12.25" in result.report


def test_scenario_lifecycle_basis_can_be_negative_after_prior_net_credits():
    result = compare_hold_vs_roll_scenarios(
        _snapshot(),
        _plan(_candidate()),
        entry_premium=15.0,
        lifecycle_net_premium_per_share=-2.0,
        contracts=1,
        risk_free_rate=0.04,
    )
    point = _point(result, horizon=3, spot_shock=0.0, iv_shock=0.0)
    assert point.hold_lifecycle_pnl_dollars == pytest.approx(
        (point.hold_model_value_per_share + 2.0) * 100
    )


def test_nonfinite_scenario_lifecycle_basis_is_rejected():
    with pytest.raises(ValueError, match="lifecycle_net_premium_per_share"):
        compare_hold_vs_roll_scenarios(
            _snapshot(),
            _plan(_candidate()),
            entry_premium=8.0,
            lifecycle_net_premium_per_share=float("inf"),
            contracts=1,
            risk_free_rate=0.04,
        )


def test_rate_resolver_is_called_once_when_rate_not_supplied(monkeypatch):
    calls = []

    def fake_resolver(as_of):
        calls.append(as_of)
        return 0.0375, "test rate"

    monkeypatch.setattr(
        "tradingagents.option_hold_roll_scenarios.resolve_scenario_risk_free_rate",
        fake_resolver,
    )
    result = compare_hold_vs_roll_scenarios(
        _snapshot(),
        _plan(_candidate()),
        entry_premium=8.0,
        contracts=1,
    )

    assert calls == [date(2026, 9, 11)]
    assert result.risk_free_rate == pytest.approx(0.0375)
    assert result.risk_free_provenance == "test rate"


def test_result_objects_are_frozen():
    result = compare_hold_vs_roll_scenarios(
        _snapshot(),
        _plan(_candidate()),
        entry_premium=8.0,
        contracts=1,
        risk_free_rate=0.04,
    )

    with pytest.raises(FrozenInstanceError):
        result.status = "ROLL"
    with pytest.raises(FrozenInstanceError):
        result.points[0].hold_lifecycle_pnl_dollars = 0
