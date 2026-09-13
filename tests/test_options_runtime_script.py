import importlib.util
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tradingagents.option_runtime import OptionRuntimeHealth, OptionRuntimeRelease


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "options_runtime.py"
    spec = importlib.util.spec_from_file_location("options_runtime_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release():
    return OptionRuntimeRelease(
        git_sha="a" * 40,
        git_short="a" * 12,
        dependency_fingerprint="deps123",
        env_relpath="envs/deps123/venv",
        created_at="2026-09-12T00:00:00+00:00",
        python_version="3.12.0",
    )


def test_install_delegates_to_runtime_installer(tmp_path, capsys):
    module = _load_script()
    runtime = tmp_path / "runtime"
    with patch.object(module, "install_runtime", return_value=_release()) as install:
        rc = module.main(["--runtime-root", str(runtime), "install"])

    assert rc == 0
    assert install.call_args.kwargs["runtime_root"] == runtime
    assert install.call_args.kwargs["allow_unpushed"] is False
    output = capsys.readouterr().out
    assert "status: INSTALLED" in output
    assert "dependency_fingerprint: deps123" in output


def test_health_combines_runtime_scheduler_binding_and_launchd_state(tmp_path, capsys):
    module = _load_script()
    runtime = tmp_path / "runtime"
    plist = tmp_path / "agent.plist"
    health = OptionRuntimeHealth(
        status="PASS",
        current_release="abc123",
        issues=(),
        runner_path=str(runtime / "bin" / "options-daily-runner"),
        manifest_path=str(runtime / "current" / "release.json"),
    )
    with (
        patch.object(module, "health_runtime", return_value=health),
        patch.object(module, "scheduler_binding_issues", return_value=()),
        patch.object(module, "_launchd_loaded", return_value=True),
    ):
        rc = module.main(
            [
                "--runtime-root",
                str(runtime),
                "--launch-agent",
                str(plist),
                "health",
            ]
        )

    assert rc == 0
    output = capsys.readouterr().out
    assert "status: PASS" in output
    assert "launchd_loaded: True" in output


def test_health_fails_when_scheduler_binding_or_launchd_is_bad(tmp_path, capsys):
    module = _load_script()
    health = OptionRuntimeHealth(
        status="PASS",
        current_release="abc123",
        issues=(),
        runner_path="/runtime/runner",
    )
    with (
        patch.object(module, "health_runtime", return_value=health),
        patch.object(module, "scheduler_binding_issues", return_value=("wrong runner",)),
        patch.object(module, "_launchd_loaded", return_value=False),
    ):
        rc = module.main(["--runtime-root", str(tmp_path), "health"])

    assert rc == 1
    output = capsys.readouterr().out
    assert "status: FAIL" in output
    assert "wrong runner" in output
    assert "LaunchAgent is not loaded" in output


def test_rollback_and_prune_commands_are_explicit(tmp_path, capsys):
    module = _load_script()
    release = _release()
    with patch.object(module, "rollback_runtime", return_value=release) as rollback:
        assert module.main(["--runtime-root", str(tmp_path), "rollback", "--to", "aaaa"]) == 0
    assert rollback.call_args.kwargs["to_release"] == "aaaa"
    assert "ROLLED_BACK" in capsys.readouterr().out

    with patch.object(module, "prune_releases", return_value=("old1", "old2")) as prune:
        assert module.main(["--runtime-root", str(tmp_path), "prune", "--keep", "2"]) == 0
    assert prune.call_args.kwargs["keep"] == 2
    output = capsys.readouterr().out
    assert "old1" in output and "old2" in output


def test_list_json_marks_current_and_previous(tmp_path, capsys):
    module = _load_script()
    current = tmp_path / "releases" / "aaaaaaaaaaaa"
    previous = tmp_path / "releases" / "bbbbbbbbbbbb"
    current.mkdir(parents=True)
    previous.mkdir(parents=True)
    (tmp_path / "current").symlink_to(current)
    (tmp_path / "previous").symlink_to(previous)
    release = _release()
    prior = replace(release, git_short="bbbbbbbbbbbb", git_sha="b" * 40)
    with patch.object(module, "list_releases", return_value=(release, prior)):
        rc = module.main(["--runtime-root", str(tmp_path), "--json", "list"])
    assert rc == 0
    output = capsys.readouterr().out
    assert '"current": "aaaaaaaaaaaa"' in output
    assert '"previous": "bbbbbbbbbbbb"' in output
