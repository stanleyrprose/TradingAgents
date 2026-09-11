import json
import os
import stat
from datetime import date

import pytest

from tradingagents.option_daily_ops import (
    build_option_daily_brief,
    delivery_target_hash,
    record_successful_delivery,
    successful_receipt_exists,
)
from tradingagents.option_portfolio_dashboard import OptionPortfolioDashboardResult
from tradingagents.option_portfolio_policy import (
    OptionDailyActionItem,
    OptionPortfolioPolicyResult,
    OptionPortfolioRiskPolicy,
)

TODAY = date(2026, 9, 11)


def _dashboard(*, liquidation=2590.0, pnl=190.0):
    return OptionPortfolioDashboardResult(
        as_of=TODAY,
        rows=(),
        underlyings=(),
        total_liquidation_value=liquidation,
        total_current_leg_pnl_dollars=pnl,
        total_lifecycle_pnl_dollars=pnl,
        exit_count=0,
        review_count=0,
        hold_count=0,
        report_only_count=0,
        report="DASHBOARD",
    )


def _policy(status="BREACH"):
    return OptionPortfolioPolicyResult(
        status=status,
        policy=OptionPortfolioRiskPolicy(),
        checks=(),
        breach_count=0,
        not_evaluable_count=0,
        report="POLICY",
    )


def _actions():
    return (
        OptionDailyActionItem(
            priority=0,
            kind="POSITION_EXIT",
            target="call1",
            position_ids=("call1",),
            reason="take profit 33.33% >= 10.00% threshold",
        ),
        OptionDailyActionItem(
            priority=1,
            kind="POLICY_BREACH",
            target="AAPL",
            position_ids=("call1", "put1"),
            reason="absolute net delta shares 100.69 exceeds cap 90.00",
        ),
        OptionDailyActionItem(
            priority=2,
            kind="POLICY_NOT_EVALUABLE",
            target="BOOK",
            position_ids=("call1", "put1"),
            reason="book gross theta is unavailable",
        ),
        OptionDailyActionItem(
            priority=3,
            kind="POSITION_REVIEW",
            target="put1",
            position_ids=("put1",),
            reason="market data unavailable",
        ),
    )


def test_empty_action_queue_is_silent_delivery_candidate():
    brief = build_option_daily_brief(_dashboard(), _policy("OK"), ())
    assert not brief.send_required
    assert brief.action_count == 0
    assert "No EXIT" in brief.text
    assert len(brief.fingerprint) == 64


def test_brief_is_short_categorical_and_contains_no_hidden_score_language():
    brief = build_option_daily_brief(_dashboard(), _policy(), _actions())
    assert brief.send_required
    assert brief.action_count == 4
    assert brief.exit_count == 1
    assert brief.policy_breach_count == 1
    assert brief.policy_not_evaluable_count == 1
    assert brief.review_count == 1
    assert "Actions 4 | EXIT 1 | BREACH 1 | CHECK 2" in brief.text
    assert "Book value +$2,590" in brief.text
    assert "Lifecycle P/L +$190" in brief.text
    assert "POSITION_EXIT call1" in brief.text
    assert "POLICY_BREACH AAPL [call1,put1]" in brief.text
    assert "No trade/close/hedge/roll was executed" in brief.text
    assert len(brief.text) <= 4000


def test_max_actions_truncates_lines_but_preserves_total_count():
    brief = build_option_daily_brief(_dashboard(), _policy(), _actions(), max_actions=2)
    assert brief.action_count == 4
    assert "1. POSITION_EXIT" in brief.text
    assert "2. POLICY_BREACH" in brief.text
    assert "POSITION_REVIEW" not in brief.text
    assert "+2 more action item(s)" in brief.text


@pytest.mark.parametrize("value", [0, 21, True, 1.5, "8"])
def test_invalid_max_actions_rejected(value):
    with pytest.raises(ValueError, match="max_actions"):
        build_option_daily_brief(_dashboard(), _policy(), _actions(), max_actions=value)


def test_fingerprint_changes_with_action_content_and_date():
    first = build_option_daily_brief(_dashboard(), _policy(), _actions())
    changed_actions = list(_actions())
    changed_actions[0] = OptionDailyActionItem(
        priority=0,
        kind="POSITION_EXIT",
        target="call1",
        position_ids=("call1",),
        reason="take profit 40% >= 10% threshold",
    )
    second = build_option_daily_brief(_dashboard(), _policy(), tuple(changed_actions))
    next_day_dashboard = OptionPortfolioDashboardResult(
        **{**_dashboard().__dict__, "as_of": date(2026, 9, 12)}
    )
    third = build_option_daily_brief(next_day_dashboard, _policy(), _actions())
    assert first.fingerprint != second.fingerprint
    assert first.fingerprint != third.fingerprint


def test_receipt_roundtrip_deduplicates_exact_target_and_fingerprint_only(tmp_path):
    path = tmp_path / "receipts.json"
    brief = build_option_daily_brief(_dashboard(), _policy(), _actions())
    first_target = delivery_target_hash("12345")
    second_target = delivery_target_hash("67890")
    assert first_target != second_target
    assert "12345" not in first_target
    assert not successful_receipt_exists(
        path,
        fingerprint=brief.fingerprint,
        target_hash=first_target,
    )

    result = record_successful_delivery(
        path,
        brief=brief,
        target_hash=first_target,
        provider_message_id=42,
    )
    assert result.status == "SENT"
    assert result.provider_message_id == 42
    assert successful_receipt_exists(
        path,
        fingerprint=brief.fingerprint,
        target_hash=first_target,
    )
    assert not successful_receipt_exists(
        path,
        fingerprint=brief.fingerprint,
        target_hash=second_target,
    )
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    raw = json.loads(path.read_text())
    assert raw["deliveries"][0]["target_hash"] == first_target
    assert "12345" not in path.read_text()


def test_invalid_receipt_file_fails_closed(tmp_path):
    path = tmp_path / "receipts.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid option daily-ops receipt"):
        successful_receipt_exists(path, fingerprint="abc", target_hash="def")
