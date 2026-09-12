import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

import tradingagents.option_runtime as runtime_module
from tradingagents.option_runtime import (
    OptionRuntimeRelease,
    activate_release,
    dependency_fingerprint,
    health_runtime,
    list_releases,
    prune_releases,
    rollback_runtime,
    runtime_runner_path,
    scheduler_binding_issues,
)


def _release(root: Path, name: str, *, created_at: str) -> Path:
    release = root / "releases" / name
    app = release / "app"
    (app / "scripts").mkdir(parents=True)
    (app / "scripts" / "options_daily_ops.py").write_text("print('ok')\n", encoding="utf-8")
    env_rel = f"envs/deps-{name}/venv"
    env_python = root / env_rel / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.write_text("placeholder", encoding="utf-8")
    manifest = OptionRuntimeRelease(
        git_sha=(name * 4)[:40],
        git_short=name,
        dependency_fingerprint=f"deps-{name}",
        env_relpath=env_rel,
        created_at=created_at,
        python_version="3.12.0",
    )
    (release / "release.json").write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    return release


def _runner(root: Path, *, exit_code: int = 0) -> Path:
    runner = root / "bin" / "options-daily-runner"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
    os.chmod(runner, 0o755)
    return runner


def test_install_refuses_unpushed_head_before_environment_copy(tmp_path):
    with (
        patch.object(runtime_module, "_tracked_clean"),
        patch.object(runtime_module, "_git_sha", return_value=("a" * 40, "b" * 40)),
        patch.object(runtime_module, "_freeze_environment") as freeze,
        pytest.raises(ValueError, match="match its upstream"),
    ):
        runtime_module.install_runtime(
            repo_root=tmp_path,
            source_python=Path("/tmp/python"),
            runtime_root=tmp_path / "runtime",
        )
    freeze.assert_not_called()


def test_dependency_fingerprint_is_deterministic_version_sensitive_and_ignores_app_revision():
    a = dependency_fingerprint(
        python_version="3.12.1",
        freeze_text="requests==1\n-e git+https://example/repo@aaa#egg=tradingagents\n",
    )
    b = dependency_fingerprint(
        python_version="3.12.1",
        freeze_text="requests==1\n-e git+https://example/repo@bbb#egg=tradingagents\n",
    )
    c = dependency_fingerprint(python_version="3.12.2", freeze_text="requests==1\n")
    assert a == b
    assert a != c
    assert len(a) == 20


def test_activate_tracks_previous_and_rollback_swaps_back(tmp_path):
    first = _release(tmp_path, "111111111111", created_at="2026-09-10T00:00:00+00:00")
    second = _release(tmp_path, "222222222222", created_at="2026-09-11T00:00:00+00:00")

    activate_release(tmp_path, "111")
    assert (tmp_path / "current").resolve() == first
    assert not (tmp_path / "previous").exists()

    activate_release(tmp_path, "222")
    assert (tmp_path / "current").resolve() == second
    assert (tmp_path / "previous").resolve() == first

    rolled = rollback_runtime(tmp_path)
    assert rolled.git_short == "111111111111"
    assert (tmp_path / "current").resolve() == first
    assert (tmp_path / "previous").resolve() == second


def test_release_selector_must_be_unique(tmp_path):
    _release(tmp_path, "abc111111111", created_at="2026-09-10T00:00:00+00:00")
    _release(tmp_path, "abc222222222", created_at="2026-09-11T00:00:00+00:00")
    with pytest.raises(ValueError, match="exactly one"):
        activate_release(tmp_path, "abc")


def test_health_passes_with_current_manifest_env_script_and_runner(tmp_path):
    release = _release(tmp_path, "aaaaaaaaaaaa", created_at="2026-09-11T00:00:00+00:00")
    _runner(tmp_path)
    (tmp_path / "current").symlink_to(release)

    health = health_runtime(tmp_path)

    assert health.status == "PASS"
    assert health.current_release == "aaaaaaaaaaaa"
    assert health.issues == ()


def test_health_fails_closed_when_runner_or_current_missing(tmp_path):
    health = health_runtime(tmp_path)
    assert health.status == "FAIL"
    assert "runtime runner missing" in health.issues[0]
    assert "current release symlink is missing" in health.issues[1]


def test_list_and_prune_preserve_current_previous_and_newest(tmp_path):
    r1 = _release(tmp_path, "111111111111", created_at="2026-09-09T00:00:00+00:00")
    r2 = _release(tmp_path, "222222222222", created_at="2026-09-10T00:00:00+00:00")
    r3 = _release(tmp_path, "333333333333", created_at="2026-09-11T00:00:00+00:00")
    _release(tmp_path, "444444444444", created_at="2026-09-12T00:00:00+00:00")
    runtime_module._chmod_readonly_tree(r3 / "app")
    (tmp_path / "current").symlink_to(r2)
    (tmp_path / "previous").symlink_to(r1)

    releases = list_releases(tmp_path)
    assert [item.git_short for item in releases][:2] == ["444444444444", "333333333333"]

    removed = prune_releases(tmp_path, keep=1)
    assert set(removed) == {"333333333333"}
    assert (tmp_path / "releases" / "444444444444").exists()
    assert r2.exists()
    assert r1.exists()


def test_scheduler_binding_requires_hardened_runner_send_and_config(tmp_path):
    runner = runtime_runner_path(tmp_path)
    plist = tmp_path / "agent.plist"
    plist.write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        b'<plist version="1.0"><dict>'
        b'<key>ProgramArguments</key><array>'
        + f"<string>{runner.absolute()}</string><string>--send</string><string>--telegram-config</string><string>/tmp/tg.json</string>".encode()
        + b'</array><key>EnvironmentVariables</key><dict><key>PYTHONUNBUFFERED</key><string>1</string></dict>'
        b'</dict></plist>'
    )

    assert scheduler_binding_issues(runtime_root=tmp_path, launch_agent_path=plist) == ()


def test_scheduler_binding_rejects_source_checkout_or_embedded_secret_env(tmp_path):
    plist = tmp_path / "agent.plist"
    plist.write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        b'<plist version="1.0"><dict>'
        b'<key>ProgramArguments</key><array><string>/repo/.venv/bin/python</string></array>'
        b'<key>EnvironmentVariables</key><dict><key>TRADINGAGENTS_TG_BOT_TOKEN</key><string>x</string></dict>'
        b'</dict></plist>'
    )

    issues = scheduler_binding_issues(runtime_root=tmp_path, launch_agent_path=plist)
    assert "not bound to the hardened runtime runner" in " ".join(issues)
    assert "production SEND mode" in " ".join(issues)
    assert "secure Telegram config" in " ".join(issues)
    assert "must not embed Telegram credentials" in " ".join(issues)


def test_prune_rejects_invalid_keep(tmp_path):
    with pytest.raises(ValueError, match="positive whole"):
        prune_releases(tmp_path, keep=0)
