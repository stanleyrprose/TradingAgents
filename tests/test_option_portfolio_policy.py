import json
import os
import stat
from datetime import date

import pytest

from tradingagents.option_portfolio_dashboard import (
    OptionPortfolioDashboardResult,
    OptionPortfolioRow,
    UnderlyingExposureSummary,
)
from tradingagents.option_portfolio_policy import (
    OptionPortfolioRiskPolicy,
    build_daily_action_queue,
    clear_portfolio_risk_policy,
    default_portfolio_policy_path,
    evaluate_portfolio_risk_policy,
    load_portfolio_risk_policy,
    policy_to_dict,
    render_daily_action_queue,
    resolve_portfolio_risk_policy,
    save_portfolio_risk_policy,
)

TODAY = date(2026, 9, 11)


def _row(
    position_id,
    underlying,
    *,
    status="HOLD",
    liquidation=1000.0,
    pnl=100.0,
    delta=60.0,
    gamma=2.0,
    vega=20.0,
    theta=-15.0,
    reason="no exit trigger",
):
    return OptionPortfolioRow(
        position_id=position_id,
        underlying=underlying,
        symbol=f"{underlying}260925C00320000",
        contracts=1,
        status=status,
        attention_reason=reason,
        dte=14,
        underlying_spot=325.0,
        bid=10.0,
        ask=10.5,
        current_liquidation_value=liquidation,
        current_leg_entry_premium=9.0,
        lifecycle_net_premium_per_share=9.0,
        current_leg_pnl_dollars=pnl,
        current_leg_pnl_pct=11.11,
        lifecycle_pnl_dollars=pnl,
        delta_shares=delta,
        gamma_delta_shares_per_dollar=gamma,
        vega_dollars_per_vol_point=vega,
        theta_dollars_per_day=theta,
        roll_count=0,
        latest_thesis_status="CONFIRMED",
        source_timestamp="2026-09-11 10:00:00",
        unavailable_reason=None,
    )


def _summary(
    underlying,
    *,
    positions=1,
    liquidation=1000.0,
    pnl=100.0,
    net_delta=60.0,
    gross_delta=60.0,
    net_gamma=2.0,
    gross_gamma=2.0,
    net_vega=20.0,
    gross_vega=20.0,
    net_theta=-15.0,
    gross_theta=15.0,
):
    return UnderlyingExposureSummary(
        underlying=underlying,
        position_count=positions,
        liquidation_value=liquidation,
        lifecycle_pnl_dollars=pnl,
        net_delta_shares=net_delta,
        gross_delta_shares=gross_delta,
        net_gamma_delta_shares_per_dollar=net_gamma,
        gross_gamma_delta_shares_per_dollar=gross_gamma,
        net_vega_dollars_per_vol_point=net_vega,
        gross_vega_dollars_per_vol_point=gross_vega,
        net_theta_dollars_per_day=net_theta,
        gross_theta_dollars_per_day=gross_theta,
    )


def _dashboard(rows=None, underlyings=None, *, liquidation=1000.0, pnl=100.0):
    rows = tuple(rows if rows is not None else [_row("aapl", "AAPL")])
    underlyings = tuple(
        underlyings if underlyings is not None else [_summary("AAPL")]
    )
    return OptionPortfolioDashboardResult(
        as_of=TODAY,
        rows=rows,
        underlyings=underlyings,
        total_liquidation_value=liquidation,
        total_current_leg_pnl_dollars=pnl,
        total_lifecycle_pnl_dollars=pnl,
        exit_count=sum(row.status == "EXIT" for row in rows),
        review_count=sum(row.status == "REVIEW" for row in rows),
        hold_count=sum(row.status == "HOLD" for row in rows),
        report_only_count=sum(row.status == "REPORT_ONLY" for row in rows),
        report="DASHBOARD",
    )


def test_policy_validation_has_no_hidden_defaults_and_rejects_invalid_caps():
    empty = resolve_portfolio_risk_policy()
    assert not empty.configured
    assert policy_to_dict(empty) == {}

    policy = resolve_portfolio_risk_policy(
        max_book_liquidation_value=5000,
        max_underlying_abs_delta_shares=150,
    )
    assert policy.configured
    assert policy_to_dict(policy) == {
        "max_book_liquidation_value": 5000.0,
        "max_underlying_abs_delta_shares": 150.0,
    }

    for value in (0, -1, float("nan"), float("inf"), True, "100"):
        with pytest.raises(ValueError):
            resolve_portfolio_risk_policy(max_book_liquidation_value=value)


def test_policy_file_roundtrip_is_private_and_unknown_fields_fail(tmp_path):
    path = tmp_path / "options_policy.json"
    policy = resolve_portfolio_risk_policy(
        max_book_gross_theta_dollars_per_day=100,
        max_underlying_gross_delta_shares=250,
    )
    assert save_portfolio_risk_policy(policy, path) == path
    assert load_portfolio_risk_policy(path) == policy
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    path.write_text(json.dumps({"mystery_limit": 10}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown option portfolio policy"):
        load_portfolio_risk_policy(path)


def test_missing_policy_file_loads_as_not_configured_and_clear_is_idempotent(tmp_path):
    path = tmp_path / "missing.json"
    assert load_portfolio_risk_policy(path) == OptionPortfolioRiskPolicy()
    assert clear_portfolio_risk_policy(path) is False
    save_portfolio_risk_policy(
        resolve_portfolio_risk_policy(max_book_liquidation_value=1000), path
    )
    assert clear_portfolio_risk_policy(path) is True
    assert clear_portfolio_risk_policy(path) is False


def test_default_policy_path_is_sibling_of_registry_and_env_override_wins(tmp_path, monkeypatch):
    db = tmp_path / "custom.sqlite3"
    assert default_portfolio_policy_path(db) == tmp_path / "options_policy.json"
    override = tmp_path / "override.json"
    monkeypatch.setenv("TRADINGAGENTS_OPTION_PORTFOLIO_POLICY", str(override))
    assert default_portfolio_policy_path(db) == override


def test_full_book_policy_can_be_ok_or_breach_without_scores():
    dashboard = _dashboard(
        rows=[_row("aapl", "AAPL", theta=-15), _row("msft", "MSFT", theta=-20)],
        underlyings=[_summary("AAPL"), _summary("MSFT", net_delta=40, gross_delta=40)],
        liquidation=2000,
        pnl=200,
    )
    ok = evaluate_portfolio_risk_policy(
        dashboard,
        resolve_portfolio_risk_policy(
            max_book_liquidation_value=2500,
            max_book_gross_theta_dollars_per_day=40,
            max_underlying_abs_delta_shares=100,
        ),
    )
    assert ok.status == "OK"
    assert all(check.status == "OK" for check in ok.checks)

    breached = evaluate_portfolio_risk_policy(
        dashboard,
        resolve_portfolio_risk_policy(
            max_book_liquidation_value=1500,
            max_book_gross_theta_dollars_per_day=30,
            max_underlying_abs_delta_shares=50,
        ),
    )
    assert breached.status == "BREACH"
    assert breached.breach_count == 3
    assert "hidden weighted risk score" in breached.report


def test_underlying_policy_distinguishes_net_and_gross_delta():
    dashboard = _dashboard(
        rows=[_row("call", "AAPL", delta=100), _row("put", "AAPL", delta=-90)],
        underlyings=[_summary("AAPL", positions=2, net_delta=10, gross_delta=190)],
        liquidation=2000,
        pnl=200,
    )
    result = evaluate_portfolio_risk_policy(
        dashboard,
        resolve_portfolio_risk_policy(
            max_underlying_abs_delta_shares=20,
            max_underlying_gross_delta_shares=150,
        ),
    )
    by_name = {check.policy_name: check for check in result.checks}
    assert by_name["max_underlying_abs_delta_shares"].status == "OK"
    assert by_name["max_underlying_gross_delta_shares"].status == "BREACH"


def test_filtered_dashboard_marks_book_checks_not_evaluable_but_keeps_underlying_checks():
    result = evaluate_portfolio_risk_policy(
        _dashboard(),
        resolve_portfolio_risk_policy(
            max_book_liquidation_value=2000,
            max_underlying_abs_delta_shares=100,
        ),
        book_scope_complete=False,
    )
    by_scope = {(check.scope, check.policy_name): check for check in result.checks}
    assert by_scope[("BOOK", "max_book_liquidation_value")].status == "NOT_EVALUABLE"
    assert by_scope[("UNDERLYING", "max_underlying_abs_delta_shares")].status == "OK"
    assert result.status == "NOT_EVALUABLE"


def test_missing_market_metric_makes_strict_policy_not_evaluable():
    dashboard = _dashboard(
        rows=[_row("aapl", "AAPL", theta=None)],
        underlyings=[_summary("AAPL", gross_theta=None)],
        liquidation=None,
        pnl=None,
    )
    result = evaluate_portfolio_risk_policy(
        dashboard,
        resolve_portfolio_risk_policy(
            max_book_liquidation_value=2000,
            max_book_gross_theta_dollars_per_day=50,
            max_underlying_gross_theta_dollars_per_day=50,
        ),
    )
    assert result.status == "NOT_EVALUABLE"
    assert result.not_evaluable_count == 3


def test_configured_per_underlying_policy_with_empty_book_is_ok():
    result = evaluate_portfolio_risk_policy(
        _dashboard(rows=[], underlyings=[], liquidation=0, pnl=0),
        resolve_portfolio_risk_policy(max_underlying_abs_delta_shares=100),
    )
    assert result.status == "OK"
    assert result.checks == ()


def test_daily_action_queue_is_categorical_stable_and_maps_policy_to_positions():
    rows = [
        _row("exit", "AAPL", status="EXIT", reason="take profit triggered"),
        _row("review", "MSFT", status="REVIEW", reason="quote unavailable"),
        _row("hold", "AAPL", status="HOLD"),
    ]
    dashboard = _dashboard(
        rows=rows,
        underlyings=[
            _summary("AAPL", positions=2, net_delta=120, gross_delta=140),
            _summary("MSFT", net_delta=40, gross_delta=40),
        ],
        liquidation=3000,
        pnl=200,
    )
    policy_result = evaluate_portfolio_risk_policy(
        dashboard,
        resolve_portfolio_risk_policy(
            max_book_liquidation_value=2500,
            max_underlying_abs_delta_shares=100,
        ),
    )
    queue = build_daily_action_queue(dashboard, policy_result)

    assert [item.kind for item in queue] == [
        "POSITION_EXIT",
        "POLICY_BREACH",
        "POLICY_BREACH",
        "POSITION_REVIEW",
    ]
    assert queue[0].position_ids == ("exit",)
    book_item = next(item for item in queue if item.kind == "POLICY_BREACH" and item.target == "BOOK")
    assert book_item.position_ids == ("exit", "review", "hold")
    aapl_item = next(item for item in queue if item.kind == "POLICY_BREACH" and item.target == "AAPL")
    assert aapl_item.position_ids == ("exit", "hold")
    report = render_daily_action_queue(queue)
    assert "not a numeric risk score" in report
    assert "no trade, close, hedge, or roll is executed automatically" in report


def test_not_evaluable_policy_precedes_position_review_and_no_hold_queue_items():
    rows = [
        _row("review", "AAPL", status="REVIEW", reason="market data unavailable"),
        _row("hold", "MSFT", status="HOLD"),
        _row("report", "NVDA", status="REPORT_ONLY"),
    ]
    dashboard = _dashboard(
        rows=rows,
        underlyings=[_summary("AAPL", net_delta=None, gross_delta=None)],
        liquidation=None,
        pnl=None,
    )
    policy_result = evaluate_portfolio_risk_policy(
        dashboard,
        resolve_portfolio_risk_policy(max_book_liquidation_value=5000),
    )
    queue = build_daily_action_queue(dashboard, policy_result)
    assert [item.kind for item in queue] == ["POLICY_NOT_EVALUABLE", "POSITION_REVIEW"]
    policy_item, review_item = queue
    assert policy_item.position_ids == ("review", "hold", "report")
    assert review_item.position_ids == ("review",)


def test_no_configured_policy_still_keeps_explicit_position_exit_queue():
    dashboard = _dashboard(rows=[_row("exit", "AAPL", status="EXIT", reason="stop triggered")])
    policy_result = evaluate_portfolio_risk_policy(dashboard, OptionPortfolioRiskPolicy())
    queue = build_daily_action_queue(dashboard, policy_result)
    assert policy_result.status == "NOT_CONFIGURED"
    assert [item.kind for item in queue] == ["POSITION_EXIT"]
