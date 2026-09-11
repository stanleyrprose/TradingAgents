import importlib.util
import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

from tradingagents.dataflows.equity_options import (
    EquityOptionSnapshot,
    EquityOptionSnapshotResult,
)
from tradingagents.option_position_registry import OptionPositionRegistry
from tradingagents.telegram_delivery import TelegramSendResult

TODAY = date(2026, 9, 11)
CALL = "AAPL260925C00320000"


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "options_daily_ops.py"
    spec = importlib.util.spec_from_file_location("options_daily_ops_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry(tmp_path):
    return OptionPositionRegistry(tmp_path / "options.sqlite3")


def _open(registry, *, with_exit_policy=True):
    return registry.open_position(
        CALL,
        entry_premium=8.25,
        contracts=2,
        entry_date=TODAY.isoformat(),
        exit_policy={"take_profit_pct": 10.0} if with_exit_policy else {},
        position_id="aapl_call",
    )


def _snapshot(bid=11.0):
    ask = bid + 0.5
    midpoint = (bid + ask) / 2
    return EquityOptionSnapshotResult(
        EquityOptionSnapshot(
            symbol=CALL,
            underlying="AAPL",
            expiry=date(2026, 9, 25),
            right="C",
            strike=320.0,
            as_of=TODAY,
            dte=14,
            source_timestamp="2026-09-11 10:00:00",
            underlying_spot=325.0,
            bid=bid,
            ask=ask,
            midpoint=midpoint,
            spread_pct=(ask - bid) / midpoint * 100,
            last=bid,
            iv=0.25,
            delta=0.60,
            gamma=0.02,
            vega=0.30,
            theta=-0.20,
            open_interest=100,
            volume=10,
        )
    )


def test_default_run_is_preview_only_and_never_sends(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry)
    with (
        patch.object(module, "fetch_equity_option_snapshots", return_value={CALL: _snapshot()}),
        patch.object(module, "send_telegram_text") as send,
    ):
        rc = module.main(["--db", str(registry.path), "--date", TODAY.isoformat()])

    assert rc == 0
    send.assert_not_called()
    output = capsys.readouterr().out
    assert "POSITION_EXIT aapl_call" in output
    assert "Delivery: PREVIEW_ONLY" in output


def test_send_with_no_actions_skips_credentials_and_provider(tmp_path, capsys, monkeypatch):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry, with_exit_policy=False)
    monkeypatch.delenv(module._CREDENTIAL_ENV, raising=False)
    monkeypatch.delenv(module._TARGET_ENV, raising=False)
    with (
        patch.object(module, "fetch_equity_option_snapshots", return_value={CALL: _snapshot()}),
        patch.object(module, "send_telegram_text") as send,
    ):
        rc = module.main(
            ["--db", str(registry.path), "--date", TODAY.isoformat(), "--send"]
        )

    assert rc == 0
    send.assert_not_called()
    assert "Delivery: NO_ACTIONS" in capsys.readouterr().out


def test_send_with_actions_and_missing_credentials_fails_closed(tmp_path, capsys, monkeypatch):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry)
    monkeypatch.delenv(module._CREDENTIAL_ENV, raising=False)
    monkeypatch.delenv(module._TARGET_ENV, raising=False)
    with (
        patch.object(module, "fetch_equity_option_snapshots", return_value={CALL: _snapshot()}),
        patch.object(module, "send_telegram_text") as send,
    ):
        rc = module.main(
            ["--db", str(registry.path), "--date", TODAY.isoformat(), "--send"]
        )

    assert rc == 2
    send.assert_not_called()
    assert "requires environment variables" in capsys.readouterr().out


def test_successful_send_records_receipt_and_identical_second_run_deduplicates(
    tmp_path, capsys, monkeypatch
):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry)
    receipt = tmp_path / "receipts.json"
    monkeypatch.setenv(module._CREDENTIAL_ENV, "dummy-credential")
    monkeypatch.setenv(module._TARGET_ENV, "12345")

    with patch.object(
        module, "fetch_equity_option_snapshots", return_value={CALL: _snapshot()}
    ), patch.object(
        module,
        "send_telegram_text",
        return_value=TelegramSendResult(provider_message_id=77),
    ) as send:
        args = [
            "--db",
            str(registry.path),
            "--date",
            TODAY.isoformat(),
            "--receipt",
            str(receipt),
            "--send",
        ]
        assert module.main(args) == 0
        first_output = capsys.readouterr().out
        assert "Delivery: SENT" in first_output
        assert "Provider message ID: 77" in first_output
        assert send.call_count == 1

        assert module.main(args) == 0
        second_output = capsys.readouterr().out
        assert "Delivery: DEDUPLICATED" in second_output
        assert send.call_count == 1

    payload = json.loads(receipt.read_text())
    assert len(payload["deliveries"]) == 1
    assert payload["deliveries"][0]["provider_message_id"] == 77
    assert "12345" not in receipt.read_text()
    assert "dummy-credential" not in receipt.read_text()


def test_force_bypasses_existing_receipt(tmp_path, capsys, monkeypatch):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry)
    receipt = tmp_path / "receipts.json"
    monkeypatch.setenv(module._CREDENTIAL_ENV, "dummy-credential")
    monkeypatch.setenv(module._TARGET_ENV, "12345")
    with patch.object(
        module, "fetch_equity_option_snapshots", return_value={CALL: _snapshot()}
    ), patch.object(
        module,
        "send_telegram_text",
        side_effect=[
            TelegramSendResult(provider_message_id=1),
            TelegramSendResult(provider_message_id=2),
        ],
    ) as send:
        base = [
            "--db",
            str(registry.path),
            "--date",
            TODAY.isoformat(),
            "--receipt",
            str(receipt),
            "--send",
        ]
        assert module.main(base) == 0
        capsys.readouterr()
        assert module.main([*base, "--force"]) == 0
        assert "Provider message ID: 2" in capsys.readouterr().out
        assert send.call_count == 2


def test_force_without_send_fails_before_network(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry)
    with (
        patch.object(module, "fetch_equity_option_snapshots") as fetch,
        patch.object(module, "send_telegram_text") as send,
    ):
        rc = module.main(["--db", str(registry.path), "--force"])
    assert rc == 2
    fetch.assert_not_called()
    send.assert_not_called()
    assert "--force requires --send" in capsys.readouterr().out


def test_json_preview_is_machine_readable(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry)
    with patch.object(
        module, "fetch_equity_option_snapshots", return_value={CALL: _snapshot()}
    ):
        rc = module.main(
            [
                "--db",
                str(registry.path),
                "--date",
                TODAY.isoformat(),
                "--json",
            ]
        )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["send_required"] is True
    assert payload["action_count"] == 1
    assert payload["counts"]["POSITION_EXIT"] == 1
    assert payload["delivery"] is None
    assert len(payload["fingerprint"]) == 64
