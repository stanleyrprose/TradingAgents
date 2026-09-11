"""macOS launchd scheduling helpers for option daily operations."""

from __future__ import annotations

import json
import os
import plistlib
import re
import stat
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_LABEL = "com.tradingagents.options-daily"
_DEFAULT_TIME = "22:00"
_DEFAULT_HOME = Path.home() / ".tradingagents" / "options-ops"
_DEFAULT_CREDENTIALS = _DEFAULT_HOME / "telegram_credentials.json"
_DEFAULT_LOG_DIR = _DEFAULT_HOME / "logs"


@dataclass(frozen=True)
class TelegramCredentialFile:
    credential: str
    target: str


def default_scheduler_label() -> str:
    return _DEFAULT_LABEL


def default_schedule_time() -> str:
    return _DEFAULT_TIME


def default_scheduler_home() -> Path:
    return _DEFAULT_HOME


def default_scheduler_credential_path() -> Path:
    return _DEFAULT_CREDENTIALS


def default_scheduler_log_dir() -> Path:
    return _DEFAULT_LOG_DIR


def parse_schedule_time(value: str) -> tuple[int, int]:
    text = str(value).strip()
    match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", text)
    if match is None:
        raise ValueError("schedule time must be HH:MM in 24-hour local time")
    return int(match.group(1)), int(match.group(2))


def _credential_object(credential: str, target: str) -> dict[str, str]:
    secret = str(credential).strip()
    destination = str(target).strip()
    if not secret or not destination:
        raise ValueError("Telegram credential and target must be nonempty")
    return {"credential": secret, "target": destination}


def save_telegram_credentials(
    path: str | os.PathLike[str],
    *,
    credential: str,
    target: str,
) -> Path:
    destination = Path(path).expanduser()
    payload = _credential_object(credential, target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_telegram_credentials(
    path: str | os.PathLike[str],
) -> TelegramCredentialFile:
    source = Path(path).expanduser()
    if not source.exists():
        raise ValueError(f"Telegram credential file does not exist: {source}")
    if source.is_symlink() or not source.is_file():
        raise ValueError("Telegram credential file must be a regular non-symlink file")
    if os.name != "nt":
        mode = stat.S_IMODE(source.stat().st_mode)
        if mode & 0o077:
            raise ValueError("Telegram credential file permissions must be 0600 or stricter")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Telegram credential file") from exc
    if not isinstance(raw, dict) or set(raw) != {"credential", "target"}:
        raise ValueError("invalid Telegram credential file")
    payload = _credential_object(raw["credential"], raw["target"])
    return TelegramCredentialFile(**payload)


def render_launchd_plist(
    *,
    label: str,
    repo_root: str | os.PathLike[str],
    python_path: str | os.PathLike[str],
    schedule_time: str,
    log_dir: str | os.PathLike[str],
    preview: bool,
    credential_path: str | os.PathLike[str] | None = None,
    registry_db_path: str | os.PathLike[str] | None = None,
    policy_path: str | os.PathLike[str] | None = None,
) -> bytes:
    job_label = str(label).strip()
    if not job_label:
        raise ValueError("launchd label must be nonempty")
    hour, minute = parse_schedule_time(schedule_time)
    repo = Path(repo_root).expanduser().resolve()
    # Preserve the virtualenv launcher path. Resolving the symlink would replace
    # .venv/bin/python with the base interpreter and lose the venv site-packages.
    python = Path(python_path).expanduser().absolute()
    script = repo / "scripts" / "options_daily_ops.py"
    logs = Path(log_dir).expanduser().resolve()

    arguments = [str(python), str(script)]
    if registry_db_path is not None:
        arguments.extend(["--db", str(Path(registry_db_path).expanduser())])
    if policy_path is not None:
        arguments.extend(["--policy", str(Path(policy_path).expanduser())])
    if not preview:
        if credential_path is None:
            raise ValueError("production scheduler requires a Telegram credential file")
        arguments.extend(
            [
                "--send",
                "--telegram-config",
                str(Path(credential_path).expanduser()),
            ]
        )

    payload = {
        "Label": job_label,
        "ProgramArguments": arguments,
        "WorkingDirectory": str(repo),
        "StartCalendarInterval": [
            {"Weekday": weekday, "Hour": hour, "Minute": minute}
            for weekday in range(1, 6)
        ],
        "RunAtLoad": False,
        "ProcessType": "Background",
        "StandardOutPath": str(logs / "options-daily.out.log"),
        "StandardErrorPath": str(logs / "options-daily.err.log"),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
    }
    return plistlib.dumps(payload, sort_keys=True)


def write_launchd_plist(
    path: str | os.PathLike[str],
    content: bytes,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(content)
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
