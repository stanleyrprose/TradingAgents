import json
import os
from datetime import date, datetime, timezone

from tradingagents.option_operations_watchdog import (
    build_watchdog_alert,
    evaluate_watchdog,
    latest_daily_run_for_date,
    record_daily_run,
    record_watchdog_alert,
    successful_watchdog_alert_exists,
    watchdog_fingerprint,
)

MARKET_DATE = date(2026, 9, 11)
STARTED = datetime(2026, 9, 11, 15, 30, tzinfo=timezone.utc)
COMPLETED = datetime(2026, 9, 11, 15, 31, tzinfo=timezone.utc)


def test_run_history_round_trip_and_permissions(tmp_path):
    path = tmp_path / "history.json"
    record = record_daily_run(
        path,
        market_date=MARKET_DATE,
        started_at=STARTED,
        completed_at=COMPLETED,
        status="SUCCESS",
        exit_code=0,
        delivery_status="NO_ACTIONS",
        action_count=0,
    )
    assert record.market_date == "2026-09-11"
    loaded = latest_daily_run_for_date(path, MARKET_DATE)
    assert loaded == record
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0


def test_latest_record_for_same_market_date_wins(tmp_path):
    path = tmp_path / "history.json"
    record_daily_run(
        path,
        market_date=MARKET_DATE,
        started_at=STARTED,
        completed_at=COMPLETED,
        status="FAILED",
        exit_code=2,
        delivery_status=None,
        action_count=None,
        error_type="RuntimeError",
    )
    later = record_daily_run(
        path,
        market_date=MARKET_DATE,
        started_at=STARTED,
        completed_at=datetime(2026, 9, 11, 15, 40, tzinfo=timezone.utc),
        status="SUCCESS",
        exit_code=0,
        delivery_status="SENT",
        action_count=2,
    )
    assert latest_daily_run_for_date(path, MARKET_DATE) == later


def test_watchdog_ok_requires_success_runtime_binding_and_loaded_scheduler(tmp_path):
    record = record_daily_run(
        tmp_path / "history.json",
        market_date=MARKET_DATE,
        started_at=STARTED,
        completed_at=COMPLETED,
        status="SUCCESS",
        exit_code=0,
        delivery_status="NO_ACTIONS",
        action_count=0,
    )
    result = evaluate_watchdog(
        market_date=MARKET_DATE,
        run_record=record,
        runtime_status="PASS",
        launchd_loaded=True,
    )
    assert result.status == "OK"
    assert result.issues == ()


def test_watchdog_alert_aggregates_missing_run_runtime_and_scheduler_issues():
    result = evaluate_watchdog(
        market_date=MARKET_DATE,
        run_record=None,
        runtime_status="FAIL",
        runtime_issues=("runner broken",),
        scheduler_binding_issues=("wrong plist",),
        launchd_loaded=False,
    )
    assert result.status == "ALERT"
    joined = " | ".join(result.issues)
    assert "no successful-or-failed scheduled run record" in joined
    assert "hardened runtime health is not PASS" in joined
    assert "runner broken" in joined
    assert "wrong plist" in joined
    assert "LaunchAgent is not loaded" in joined
    alert = build_watchdog_alert(result)
    assert "TradingAgents Options Ops Watchdog" in alert
    assert "No trade/close/hedge/roll was executed" in alert


def test_market_holiday_is_silent_even_without_run_record():
    result = evaluate_watchdog(
        market_date=date(2026, 11, 26),
        run_record=None,
        runtime_status="FAIL",
        runtime_issues=("ignored on closed market day",),
        launchd_loaded=False,
    )
    assert result.status == "MARKET_CLOSED"
    assert result.issues == ()


def test_failed_scheduled_run_is_alerted():
    from tradingagents.option_operations_watchdog import OptionDailyRunRecord

    record = OptionDailyRunRecord(
        market_date="2026-09-11",
        started_at="2026-09-11T15:30:00+00:00",
        completed_at="2026-09-11T15:31:00+00:00",
        status="FAILED",
        exit_code=2,
        delivery_status=None,
        action_count=None,
        error_type="RuntimeError",
    )
    result = evaluate_watchdog(
        market_date=MARKET_DATE,
        run_record=record,
        runtime_status="PASS",
    )
    assert result.status == "ALERT"
    assert "RuntimeError" in result.issues[0]


def test_watchdog_alert_receipt_deduplicates_without_storing_target(tmp_path):
    path = tmp_path / "watchdog-receipts.json"
    result = evaluate_watchdog(
        market_date=MARKET_DATE,
        run_record=None,
        runtime_status="PASS",
    )
    fingerprint = watchdog_fingerprint(result)
    target_hash = "abc123"
    assert not successful_watchdog_alert_exists(
        path,
        fingerprint=fingerprint,
        target_hash=target_hash,
    )
    record_watchdog_alert(
        path,
        fingerprint=fingerprint,
        target_hash=target_hash,
        provider_message_id=77,
    )
    assert successful_watchdog_alert_exists(
        path,
        fingerprint=fingerprint,
        target_hash=target_hash,
    )
    text = path.read_text()
    assert "abc123" in text
    payload = json.loads(text)
    assert payload["alerts"][0]["provider_message_id"] == 77
