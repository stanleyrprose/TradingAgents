import importlib.util
import plistlib
from pathlib import Path
from unittest.mock import Mock, patch

from tradingagents.option_daily_scheduler import save_telegram_credentials


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "options_daily_scheduler.py"
    spec = importlib.util.spec_from_file_location("options_daily_scheduler_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configure_telegram_prompts_token_and_never_prints_contents(tmp_path, capsys):
    module = _load_script()
    path = tmp_path / "telegram.json"
    with (
        patch.object(module.getpass, "getpass", return_value="dummy-secret"),
        patch("builtins.input", return_value="12345"),
    ):
        rc = module.main(
            [
                "--credential-file",
                str(path),
                "configure-telegram",
            ]
        )
    assert rc == 0
    output = capsys.readouterr().out
    assert str(path) in output
    assert "dummy-secret" not in output
    assert "12345" not in output


def test_production_install_requires_existing_secure_credentials_before_launchctl(
    tmp_path, capsys
):
    module = _load_script()
    with patch.object(module, "_launchctl") as launchctl:
        rc = module.main(
            [
                "--credential-file",
                str(tmp_path / "missing.json"),
                "--launch-agent-dir",
                str(tmp_path / "agents"),
                "--log-dir",
                str(tmp_path / "logs"),
                "install",
                "--source-checkout",
            ]
        )
    assert rc == 2
    launchctl.assert_not_called()
    assert "does not exist" in capsys.readouterr().out


def test_preview_install_writes_non_sending_plist_and_calls_launchctl(tmp_path, capsys):
    module = _load_script()
    agent_dir = tmp_path / "agents"
    log_dir = tmp_path / "logs"
    launchctl_result = Mock(returncode=0, stdout="")
    with patch.object(module, "_launchctl", return_value=launchctl_result) as launchctl:
        rc = module.main(
            [
                "--label",
                "com.test.preview",
                "--launch-agent-dir",
                str(agent_dir),
                "--log-dir",
                str(log_dir),
                "install",
                "--source-checkout",
                "--preview",
                "--time",
                "07:45",
                "--kickstart",
            ]
        )
    assert rc == 0
    plist_path = agent_dir / "com.test.preview.plist"
    payload = plistlib.loads(plist_path.read_bytes())
    assert payload["StartCalendarInterval"] == [
        {"Weekday": weekday, "Hour": 7, "Minute": 45} for weekday in range(1, 6)
    ]
    assert "--send" not in payload["ProgramArguments"]
    assert "--telegram-config" not in payload["ProgramArguments"]
    assert log_dir.exists()
    calls = [call.args for call in launchctl.call_args_list]
    assert calls[0][:2] == ("bootout", f"gui/{module.os.getuid()}")
    assert calls[1][0] == "bootstrap"
    assert calls[2][0] == "enable"
    assert calls[3][0] == "kickstart"
    output = capsys.readouterr().out
    assert "Mode: PREVIEW" in output
    assert "Monday-Friday at 07:45" in output


def test_send_install_plist_contains_path_not_credential_value(tmp_path, capsys):
    module = _load_script()
    credential_path = save_telegram_credentials(
        tmp_path / "telegram.json",
        credential="dummy-secret",
        target="12345",
    )
    agent_dir = tmp_path / "agents"
    with patch.object(module, "_launchctl", return_value=Mock(returncode=0, stdout="")):
        rc = module.main(
            [
                "--label",
                "com.test.send",
                "--credential-file",
                str(credential_path),
                "--launch-agent-dir",
                str(agent_dir),
                "--log-dir",
                str(tmp_path / "logs"),
                "install",
                "--source-checkout",
            ]
        )
    assert rc == 0
    payload = plistlib.loads((agent_dir / "com.test.send.plist").read_bytes())
    args = payload["ProgramArguments"]
    assert "--send" in args
    assert args[args.index("--telegram-config") + 1] == str(credential_path)
    encoded = (agent_dir / "com.test.send.plist").read_text()
    assert "dummy-secret" not in encoded
    assert "12345" not in encoded
    output = capsys.readouterr().out
    assert "dummy-secret" not in output
    assert "12345" not in output


def test_hardened_runtime_install_uses_stable_runner(tmp_path, capsys):
    module = _load_script()
    runtime_root = tmp_path / "runtime"
    runner = runtime_root / "bin" / "options-daily-runner"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    agent_dir = tmp_path / "agents"
    log_dir = tmp_path / "logs"

    with patch.object(module, "_launchctl", return_value=Mock(returncode=0, stdout="")):
        rc = module.main(
            [
                "--label",
                "com.test.runtime",
                "--runtime-root",
                str(runtime_root),
                "--launch-agent-dir",
                str(agent_dir),
                "--log-dir",
                str(log_dir),
                "install",
                "--preview",
            ]
        )

    assert rc == 0
    payload = plistlib.loads((agent_dir / "com.test.runtime.plist").read_bytes())
    assert payload["ProgramArguments"] == [str(runner.absolute())]
    assert payload["WorkingDirectory"] == str(runtime_root.resolve())
    output = capsys.readouterr().out
    assert "Runtime mode: HARDENED" in output
    assert str(runner) in output


def test_hardened_runtime_install_fails_before_launchctl_when_runner_missing(tmp_path, capsys):
    module = _load_script()
    with patch.object(module, "_launchctl") as launchctl:
        rc = module.main(
            [
                "--runtime-root",
                str(tmp_path / "runtime"),
                "--launch-agent-dir",
                str(tmp_path / "agents"),
                "install",
                "--preview",
            ]
        )
    assert rc == 2
    launchctl.assert_not_called()
    assert "hardened option runtime is not installed" in capsys.readouterr().out


def test_status_reports_not_loaded(tmp_path, capsys):
    module = _load_script()
    result = Mock(returncode=1, stdout="", stderr="not found")
    with patch.object(module, "_launchctl", return_value=result):
        rc = module.main(["--label", "com.test.missing", "status"])
    assert rc == 1
    assert "NOT_LOADED" in capsys.readouterr().out


def test_uninstall_boots_out_and_removes_plist(tmp_path, capsys):
    module = _load_script()
    agent_dir = tmp_path / "agents"
    agent_dir.mkdir()
    plist = agent_dir / "com.test.remove.plist"
    plist.write_text("test")
    with patch.object(module, "_launchctl", return_value=Mock(returncode=0, stdout="")) as launchctl:
        rc = module.main(
            [
                "--label",
                "com.test.remove",
                "--launch-agent-dir",
                str(agent_dir),
                "uninstall",
            ]
        )
    assert rc == 0
    assert not plist.exists()
    launchctl.assert_called_once()
    assert "Uninstalled" in capsys.readouterr().out


def test_invalid_time_fails_before_launchctl(tmp_path, capsys):
    module = _load_script()
    with patch.object(module, "_launchctl") as launchctl:
        rc = module.main(
            [
                "--launch-agent-dir",
                str(tmp_path / "agents"),
                "install",
                "--preview",
                "--time",
                "8:30",
            ]
        )
    assert rc == 2
    launchctl.assert_not_called()
    assert "HH:MM" in capsys.readouterr().out
