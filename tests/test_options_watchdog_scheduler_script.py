import importlib.util
import plistlib
from pathlib import Path
from unittest.mock import Mock, patch


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "options_watchdog_scheduler.py"
    spec = importlib.util.spec_from_file_location("options_watchdog_scheduler_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare(tmp_path):
    runtime = tmp_path / "runtime"
    runner = runtime / "bin" / "options-watchdog-runner"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "com.test.daily.plist").write_bytes(plistlib.dumps({"Label": "com.test.daily"}))
    return runtime, agents, runner


def test_preview_install_uses_hardened_watchdog_runner_and_no_send(tmp_path, capsys):
    module = _load_script()
    runtime, agents, runner = _prepare(tmp_path)
    with patch.object(module, "_launchctl", return_value=Mock(returncode=0, stdout="")):
        rc = module.main(
            [
                "--label",
                "com.test.watchdog",
                "--daily-label",
                "com.test.daily",
                "--launch-agent-dir",
                str(agents),
                "--runtime-root",
                str(runtime),
                "install",
                "--preview",
            ]
        )
    assert rc == 0
    payload = plistlib.loads((agents / "com.test.watchdog.plist").read_bytes())
    assert payload["ProgramArguments"][0] == str(runner)
    assert "--send" not in payload["ProgramArguments"]
    assert payload["WorkingDirectory"] == str(runtime.resolve())
    assert "23:00" in capsys.readouterr().out


def test_production_install_validates_credentials_and_adds_send(tmp_path):
    module = _load_script()
    runtime, agents, _ = _prepare(tmp_path)
    credential = tmp_path / "telegram.json"
    credential.write_text("placeholder", encoding="utf-8")
    with (
        patch.object(module, "load_telegram_credentials") as load,
        patch.object(module, "_launchctl", return_value=Mock(returncode=0, stdout="")),
    ):
        rc = module.main(
            [
                "--label",
                "com.test.watchdog",
                "--daily-label",
                "com.test.daily",
                "--launch-agent-dir",
                str(agents),
                "--runtime-root",
                str(runtime),
                "--credential-file",
                str(credential),
                "install",
            ]
        )
    assert rc == 0
    load.assert_called_once_with(credential)
    payload = plistlib.loads((agents / "com.test.watchdog.plist").read_bytes())
    args = payload["ProgramArguments"]
    assert "--send" in args
    assert args[args.index("--telegram-config") + 1] == str(credential)


def test_install_fails_before_launchctl_when_runtime_runner_missing(tmp_path, capsys):
    module = _load_script()
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "com.test.daily.plist").write_bytes(plistlib.dumps({"Label": "com.test.daily"}))
    with patch.object(module, "_launchctl") as launchctl:
        rc = module.main(
            [
                "--daily-label",
                "com.test.daily",
                "--launch-agent-dir",
                str(agents),
                "--runtime-root",
                str(tmp_path / "runtime"),
                "install",
                "--preview",
            ]
        )
    assert rc == 2
    launchctl.assert_not_called()
    assert "watchdog runner is not installed" in capsys.readouterr().out
