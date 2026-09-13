import importlib.util
from datetime import date
from pathlib import Path
from unittest.mock import patch

from tradingagents.option_daily_scheduler import TelegramCredentialFile
from tradingagents.option_operations_watchdog import OptionDailyRunRecord
from tradingagents.option_runtime import OptionRuntimeHealth
from tradingagents.telegram_delivery import TelegramSendResult

TODAY = date(2026, 9, 11)


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "options_ops_watchdog.py"
    spec = importlib.util.spec_from_file_location("options_ops_watchdog_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(status="SUCCESS", exit_code=0):
    return OptionDailyRunRecord(
        market_date=TODAY.isoformat(),
        started_at="2026-09-11T15:30:00+00:00",
        completed_at="2026-09-11T15:31:00+00:00",
        status=status,
        exit_code=exit_code,
        delivery_status="NO_ACTIONS" if status == "SUCCESS" else None,
        action_count=0 if status == "SUCCESS" else None,
        error_type=None if status == "SUCCESS" else "RuntimeError",
    )


def _health(status="PASS", issues=()):
    return OptionRuntimeHealth(
        status=status,
        current_release="abc",
        issues=issues,
        runner_path="/runtime/runner",
    )


def test_ok_watchdog_is_silent_and_does_not_read_credentials(capsys):
    module = _load_script()
    with (
        patch.object(module, "latest_daily_run_for_date", return_value=_record()),
        patch.object(module, "health_runtime", return_value=_health()),
        patch.object(module, "scheduler_binding_issues", return_value=()),
        patch.object(module, "_launchd_loaded", return_value=True),
        patch.object(module, "load_telegram_credentials") as credentials,
        patch.object(module, "send_telegram_text") as send,
    ):
        rc = module.main(["--date", TODAY.isoformat(), "--send"])
    assert rc == 0
    credentials.assert_not_called()
    send.assert_not_called()
    output = capsys.readouterr().out
    assert "OK" in output
    assert "Delivery: NO_ALERT" in output


def test_alert_preview_does_not_send(capsys):
    module = _load_script()
    with (
        patch.object(module, "latest_daily_run_for_date", return_value=None),
        patch.object(module, "health_runtime", return_value=_health()),
        patch.object(module, "scheduler_binding_issues", return_value=()),
        patch.object(module, "_launchd_loaded", return_value=True),
        patch.object(module, "send_telegram_text") as send,
    ):
        rc = module.main(["--date", TODAY.isoformat()])
    assert rc == 0
    send.assert_not_called()
    output = capsys.readouterr().out
    assert "ALERT" in output
    assert "PREVIEW_ONLY" in output


def test_alert_send_records_receipt_and_deduplicates(tmp_path, capsys):
    module = _load_script()
    receipt = tmp_path / "receipt.json"
    with (
        patch.object(module, "latest_daily_run_for_date", return_value=None),
        patch.object(module, "health_runtime", return_value=_health()),
        patch.object(module, "scheduler_binding_issues", return_value=()),
        patch.object(module, "_launchd_loaded", return_value=True),
        patch.object(
            module,
            "load_telegram_credentials",
            return_value=TelegramCredentialFile(credential="secret", target="12345"),
        ),
        patch.object(
            module,
            "send_telegram_text",
            return_value=TelegramSendResult(provider_message_id=9),
        ) as send,
    ):
        args = ["--date", TODAY.isoformat(), "--send", "--receipt", str(receipt)]
        assert module.main(args) == 0
        assert "Delivery: SENT" in capsys.readouterr().out
        assert module.main(args) == 0
        assert "Delivery: DEDUPLICATED" in capsys.readouterr().out
    assert send.call_count == 1
    text = receipt.read_text()
    assert "secret" not in text
    assert "12345" not in text


def test_market_closed_does_not_read_credentials_or_send(capsys):
    module = _load_script()
    with (
        patch.object(module, "latest_daily_run_for_date", return_value=None),
        patch.object(module, "health_runtime", return_value=_health(status="FAIL")),
        patch.object(module, "scheduler_binding_issues", return_value=("bad",)),
        patch.object(module, "_launchd_loaded", return_value=False),
        patch.object(module, "load_telegram_credentials") as credentials,
        patch.object(module, "send_telegram_text") as send,
    ):
        rc = module.main(["--date", "2026-11-26", "--send"])
    assert rc == 0
    credentials.assert_not_called()
    send.assert_not_called()
    output = capsys.readouterr().out
    assert "MARKET_CLOSED" in output
