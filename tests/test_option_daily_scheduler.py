import os
import plistlib
import stat

import pytest

from tradingagents.option_daily_scheduler import (
    default_schedule_time,
    load_telegram_credentials,
    parse_schedule_time,
    render_launchd_plist,
    save_telegram_credentials,
    write_launchd_plist,
)


def test_default_schedule_is_weekday_us_market_overlap_time():
    assert default_schedule_time() == "22:00"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("00:00", (0, 0)), ("08:30", (8, 30)), ("23:59", (23, 59))],
)
def test_parse_schedule_time(value, expected):
    assert parse_schedule_time(value) == expected


@pytest.mark.parametrize("value", ["8:30", "24:00", "12:60", "", "noon", "08:30:00"])
def test_invalid_schedule_time_rejected(value):
    with pytest.raises(ValueError, match="HH:MM"):
        parse_schedule_time(value)


def test_credential_file_roundtrip_is_0600_and_not_symlink(tmp_path):
    path = save_telegram_credentials(
        tmp_path / "telegram.json",
        credential="dummy-secret",
        target="12345",
    )
    loaded = load_telegram_credentials(path)
    assert loaded.credential == "dummy-secret"
    assert loaded.target == "12345"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_credential_file_rejects_insecure_permissions_and_unknown_shape(tmp_path):
    path = save_telegram_credentials(
        tmp_path / "telegram.json",
        credential="dummy-secret",
        target="12345",
    )
    if os.name != "nt":
        os.chmod(path, 0o644)
        with pytest.raises(ValueError, match="permissions"):
            load_telegram_credentials(path)
        os.chmod(path, 0o600)

    path.write_text('{"credential":"x","target":"y","extra":true}', encoding="utf-8")
    if os.name != "nt":
        os.chmod(path, 0o600)
    with pytest.raises(ValueError, match="invalid Telegram credential"):
        load_telegram_credentials(path)


def test_credential_symlink_rejected(tmp_path):
    target = save_telegram_credentials(
        tmp_path / "real.json",
        credential="dummy-secret",
        target="12345",
    )
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        load_telegram_credentials(link)


def test_preview_plist_contains_no_send_or_credential_path(tmp_path):
    repo = tmp_path / "repo"
    python = repo / ".venv" / "bin" / "python"
    data = plistlib.loads(
        render_launchd_plist(
            label="com.test.options",
            repo_root=repo,
            python_path=python,
            schedule_time="08:30",
            log_dir=tmp_path / "logs",
            preview=True,
            credential_path=tmp_path / "secret.json",
            registry_db_path=tmp_path / "options.sqlite3",
        )
    )
    args = data["ProgramArguments"]
    assert args[0] == str(python.resolve())
    assert args[1].endswith("scripts/options_daily_ops.py")
    assert "--db" in args
    assert "--send" not in args
    assert "--telegram-config" not in args
    assert "secret.json" not in " ".join(args)
    assert data["StartCalendarInterval"] == [
        {"Weekday": weekday, "Hour": 8, "Minute": 30} for weekday in range(1, 6)
    ]
    assert data["RunAtLoad"] is False


def test_plist_preserves_virtualenv_python_symlink_path(tmp_path):
    repo = tmp_path / "repo"
    venv_python = repo / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    base_python = tmp_path / "base-python"
    base_python.write_text("placeholder", encoding="utf-8")
    venv_python.symlink_to(base_python)

    data = plistlib.loads(
        render_launchd_plist(
            label="com.test.options",
            repo_root=repo,
            python_path=venv_python,
            schedule_time="22:00",
            log_dir=tmp_path / "logs",
            preview=True,
        )
    )

    assert data["ProgramArguments"][0] == str(venv_python.absolute())
    assert data["ProgramArguments"][0] != str(base_python.resolve())


def test_send_plist_contains_only_credential_path_not_secret(tmp_path):
    repo = tmp_path / "repo"
    credential_path = tmp_path / "telegram_credentials.json"
    data = plistlib.loads(
        render_launchd_plist(
            label="com.test.options",
            repo_root=repo,
            python_path=repo / ".venv" / "bin" / "python",
            schedule_time="21:05",
            log_dir=tmp_path / "logs",
            preview=False,
            credential_path=credential_path,
            policy_path=tmp_path / "policy.json",
        )
    )
    args = data["ProgramArguments"]
    assert "--send" in args
    assert args[args.index("--telegram-config") + 1] == str(credential_path)
    assert "--policy" in args
    encoded = plistlib.dumps(data).decode("utf-8")
    assert "dummy-secret" not in encoded
    assert data["StartCalendarInterval"] == [
        {"Weekday": weekday, "Hour": 21, "Minute": 5} for weekday in range(1, 6)
    ]


def test_production_plist_requires_credential_path(tmp_path):
    with pytest.raises(ValueError, match="credential file"):
        render_launchd_plist(
            label="com.test.options",
            repo_root=tmp_path,
            python_path=tmp_path / "python",
            schedule_time="08:30",
            log_dir=tmp_path / "logs",
            preview=False,
        )


def test_write_plist_is_atomic_and_0644(tmp_path):
    path = write_launchd_plist(tmp_path / "agent.plist", b"hello")
    assert path.read_bytes() == b"hello"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o644
