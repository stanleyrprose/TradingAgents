from dataclasses import FrozenInstanceError
from datetime import date
from unittest.mock import MagicMock

import pytest

from tradingagents.dataflows.option_selector import OptionCandidate
from tradingagents.option_capital_allocator import (
    OptionCapitalAllocation,
    OptionCapitalPosition,
)
from tradingagents.option_exposure_guard import (
    evaluate_long_option_exposure,
    resolve_option_exposure_limits,
)


def _candidate(
    symbol: str = "AAPL261016C00100000",
    *,
    right: str = "C",
    delta: float = 0.5,
    gamma: float | None = 0.02,
    vega: float | None = 0.1,
    theta: float | None = -0.04,
    score: float = 90.0,
) -> OptionCandidate:
    return OptionCandidate(
        symbol=symbol,
        expiry=date(2026, 10, 16),
        right=right,
        strike=100.0,
        dte=36,
        bid=4.0,
        ask=5.0,
        midpoint=4.5,
        spread_pct=10.0,
        delta=delta,
        abs_delta=abs(delta),
        iv=0.3,
        oi=500.0,
        volume=100.0,
        theta=theta,
        theta_burden=None if theta is None else abs(theta) / 4.5,
        breakeven=105.0,
        required_move=0.05,
        components=(),
        score=score,
        gamma=gamma,
        vega=vega,
    )


def _position(symbol: str = "AAPL261016C00100000", contracts: int = 1) -> OptionCapitalPosition:
    return OptionCapitalPosition(
        rank=1,
        symbol=symbol,
        ask_price_proxy=5.0,
        premium_per_contract=500.0,
        target_budget=500.0,
        contracts=contracts,
        premium_at_risk=contracts * 500.0,
        bucket_utilization_pct=100.0 if contracts else 0.0,
        portfolio_risk_pct=None,
    )


def _allocation(*positions: OptionCapitalPosition) -> OptionCapitalAllocation:
    return OptionCapitalAllocation(
        positions=positions,
        risk_budget=1_500.0,
        total_premium_at_risk=sum(item.premium_at_risk for item in positions),
        unused_budget=0.0,
        account_equity=None,
        max_risk_pct=None,
        premium_budget=1_500.0,
        portfolio_risk_pct=None,
        report="allocation",
    )


def _evaluate(
    candidates: list[OptionCandidate],
    positions: list[OptionCapitalPosition],
    **kwargs: object,
):
    return evaluate_long_option_exposure(
        candidates, _allocation(*positions), underlying_spot=kwargs.pop("underlying_spot", 100), **kwargs
    )


def test_one_long_call_arithmetic():
    result = _evaluate([_candidate()], [_position()])

    exposure = result.exposures[0]
    assert exposure.delta_shares == 50
    assert exposure.delta_notional_proxy == 5_000


def test_long_put_keeps_signed_delta():
    result = _evaluate(
        [_candidate(right="P", delta=-0.4)],
        [_position()],
    )

    assert result.exposures[0].delta_shares == -40
    assert result.net_delta_shares == -40


@pytest.mark.parametrize(("contracts", "expected_delta"), [(2, 100), (3, 150)])
def test_contract_count_scales_exposure(contracts, expected_delta):
    result = _evaluate([_candidate()], [_position(contracts=contracts)])

    assert result.net_delta_shares == expected_delta


def test_multiple_positions_aggregate_net_and_gross():
    call = _candidate("CALL", delta=0.6, gamma=0.02, vega=0.1, theta=-0.03)
    put = _candidate("PUT", right="P", delta=-0.4, gamma=0.01, vega=0.2, theta=-0.02)

    result = _evaluate([call, put], [_position("CALL", 2), _position("PUT", 3)])

    assert result.net_delta_shares == 0
    assert result.gross_abs_delta_shares == 240
    assert result.net_gamma_delta_shares_per_dollar == 7
    assert result.net_vega_dollars_per_vol_point == 80
    assert result.net_theta_dollars_per_day == -12


def test_gamma_vega_and_theta_are_scaled_by_contract_multiplier():
    result = _evaluate([_candidate()], [_position(contracts=2)])

    exposure = result.exposures[0]
    assert exposure.gamma_delta_shares_per_dollar == 4
    assert exposure.vega_dollars_per_vol_point == 20
    assert exposure.theta_dollars_per_day == -8


def test_plus_and_minus_one_percent_delta_gamma_local_pl():
    result = _evaluate([_candidate(delta=0.5, gamma=0.02)], [_position()], underlying_spot=100)

    exposure = result.exposures[0]
    assert exposure.up_1pct_delta_gamma_pl_proxy == 51
    assert exposure.down_1pct_delta_gamma_pl_proxy == -49
    assert result.up_1pct_delta_gamma_pl_proxy == 51
    assert result.down_1pct_delta_gamma_pl_proxy == -49


def test_zero_contract_allocation_has_zero_exposure_without_requiring_spot():
    result = _evaluate([_candidate()], [_position(contracts=0)], underlying_spot=None)

    assert result.available
    assert result.exposures == ()
    assert result.net_delta_shares == 0


def test_no_limits_is_report_only_and_allowed_is_none():
    result = _evaluate([_candidate()], [_position()])

    assert result.status == "REPORT_ONLY"
    assert result.allowed is None
    assert "No limits supplied" in result.report


def test_public_limit_resolver_normalizes_optional_limits():
    assert resolve_option_exposure_limits(
        max_abs_delta_shares=1,
        max_abs_gamma_delta_shares_per_dollar=2.5,
    ) == {
        "max_abs_delta_shares": 1.0,
        "max_abs_gamma_delta_shares_per_dollar": 2.5,
        "max_abs_vega_dollars_per_vol_point": None,
        "max_abs_theta_dollars_per_day": None,
    }


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), 0, -1])
def test_public_limit_resolver_uses_finite_positive_semantics(value):
    with pytest.raises(ValueError, match="finite number greater than zero"):
        resolve_option_exposure_limits(max_abs_delta_shares=value)


def test_evaluate_reuses_public_limit_resolver(monkeypatch):
    resolved = {
        "max_abs_delta_shares": None,
        "max_abs_gamma_delta_shares_per_dollar": None,
        "max_abs_vega_dollars_per_vol_point": None,
        "max_abs_theta_dollars_per_day": None,
    }
    resolver = MagicMock(return_value=resolved)
    monkeypatch.setattr(
        "tradingagents.option_exposure_guard.resolve_option_exposure_limits", resolver
    )

    result = _evaluate([_candidate()], [_position()])

    assert result.status == "REPORT_ONLY"
    resolver.assert_called_once_with(
        max_abs_delta_shares=None,
        max_abs_gamma_delta_shares_per_dollar=None,
        max_abs_vega_dollars_per_vol_point=None,
        max_abs_theta_dollars_per_day=None,
    )


def test_explicit_limits_pass():
    result = _evaluate(
        [_candidate()],
        [_position()],
        max_abs_delta_shares=51,
        max_abs_gamma_delta_shares_per_dollar=3,
        max_abs_vega_dollars_per_vol_point=11,
        max_abs_theta_dollars_per_day=5,
    )

    assert result.status == "PASS"
    assert result.allowed is True
    assert result.breaches == ()


@pytest.mark.parametrize(
    ("limit_name", "cap"),
    [
        ("max_abs_delta_shares", 50),
        ("max_abs_gamma_delta_shares_per_dollar", 2),
        ("max_abs_vega_dollars_per_vol_point", 10),
        ("max_abs_theta_dollars_per_day", 4),
    ],
)
def test_equality_to_each_cap_passes(limit_name, cap):
    result = _evaluate([_candidate()], [_position()], **{limit_name: cap})

    assert result.status == "PASS"


@pytest.mark.parametrize(
    ("limit_name", "cap"),
    [
        ("max_abs_delta_shares", 49.9),
        ("max_abs_gamma_delta_shares_per_dollar", 1.9),
        ("max_abs_vega_dollars_per_vol_point", 9.9),
        ("max_abs_theta_dollars_per_day", 3.9),
    ],
)
def test_each_limit_breach_blocks(limit_name, cap):
    result = _evaluate([_candidate()], [_position()], **{limit_name: cap})

    assert result.status == "BLOCK"
    assert result.allowed is False
    assert result.breaches[0].limit_name == limit_name


@pytest.mark.parametrize(
    ("greek", "limit_name"),
    [
        ("gamma", "max_abs_gamma_delta_shares_per_dollar"),
        ("vega", "max_abs_vega_dollars_per_vol_point"),
        ("theta", "max_abs_theta_dollars_per_day"),
    ],
)
def test_missing_limited_greek_blocks(greek, limit_name):
    candidate = _candidate(**{greek: None})

    result = _evaluate([candidate], [_position()], **{limit_name: 100})

    assert result.status == "BLOCK"
    assert result.breaches[0].actual is None
    assert "unavailable" in result.breaches[0].reason


@pytest.mark.parametrize("greek", ["gamma", "vega", "theta"])
def test_missing_unlimited_greek_does_not_block_unrelated_delta_limit(greek):
    candidate = _candidate(**{greek: None})

    result = _evaluate([candidate], [_position()], max_abs_delta_shares=100)

    assert result.status == "PASS"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 0, -1])
@pytest.mark.parametrize(
    "limit_name",
    [
        "max_abs_delta_shares",
        "max_abs_gamma_delta_shares_per_dollar",
        "max_abs_vega_dollars_per_vol_point",
        "max_abs_theta_dollars_per_day",
    ],
)
def test_invalid_limit_values_raise(limit_name, value):
    with pytest.raises(ValueError, match="finite number greater than zero"):
        _evaluate([_candidate()], [_position()], **{limit_name: value})


@pytest.mark.parametrize("spot", [None, "100", True, float("nan"), float("inf"), 0, -1])
def test_missing_malformed_or_nonpositive_spot_blocks_positive_contracts(spot):
    result = _evaluate([_candidate()], [_position()], underlying_spot=spot)

    assert result.status == "BLOCK"
    assert result.allowed is False
    assert "underlying_spot" in result.unavailable_reason


def test_invalid_spot_with_explicit_delta_limit_reports_block_not_evaluable():
    result = _evaluate(
        [_candidate()],
        [_position()],
        underlying_spot=None,
        max_abs_delta_shares=75,
    )

    assert result.status == "BLOCK"
    assert result.allowed is False
    assert "cap 75.00 — BLOCK" in result.report
    assert "not evaluable" in result.report
    assert "No limits supplied" not in result.report
    assert result.breaches[0].actual is None


def test_no_limit_malformed_input_report_never_suggests_pass():
    result = _evaluate([_candidate("KNOWN")], [_position("MISSING")])

    assert result.status == "BLOCK"
    assert "Exposure report unavailable; no policy limits were supplied." in result.report
    assert "PASS" not in result.report


def test_positive_allocation_symbol_missing_from_candidates_blocks():
    result = _evaluate([_candidate("KNOWN")], [_position("MISSING")])

    assert result.status == "BLOCK"
    assert "missing from candidates" in result.unavailable_reason


def test_duplicate_candidate_symbols_block():
    result = _evaluate([_candidate("DUP"), _candidate("DUP")], [_position("DUP")])

    assert result.status == "BLOCK"
    assert "duplicate candidate symbol" in result.unavailable_reason


def test_duplicate_allocation_symbols_block():
    result = _evaluate([_candidate("DUP")], [_position("DUP"), _position("DUP")])

    assert result.status == "BLOCK"
    assert "duplicate allocation symbol" in result.unavailable_reason


def test_selector_score_is_irrelevant_to_exposure():
    low = _evaluate([_candidate(score=1)], [_position()])
    high = _evaluate([_candidate(score=100)], [_position()])

    assert low.exposures == high.exposures
    assert low.net_delta_shares == high.net_delta_shares


def test_report_contains_risk_caveats():
    report = _evaluate([_candidate()], [_position()]).report

    assert "vendor-calculated delayed snapshot" in report
    assert "local first/second-order delta-gamma approximation" in report
    assert "not a full portfolio stress engine" in report
    assert "Same-underlying Top3 is not diversification" in report


def test_guard_result_and_nested_dataclasses_are_frozen():
    result = _evaluate(
        [_candidate()], [_position()], max_abs_delta_shares=1
    )

    with pytest.raises(FrozenInstanceError):
        result.status = "PASS"
    with pytest.raises(FrozenInstanceError):
        result.exposures[0].contracts = 2
    with pytest.raises(FrozenInstanceError):
        result.breaches[0].cap = 100
