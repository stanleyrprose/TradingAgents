#!/usr/bin/env python3
"""Install, inspect, or remove the macOS launchd job for option daily operations."""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
from pathlib import Path

from tradingagents.option_daily_scheduler import (
    default_schedule_time,
    default_scheduler_credential_path,
    default_scheduler_label,
    default_scheduler_log_dir,
    load_telegram_credentials,
    parse_schedule_time,
    render_launchd_plist,
    save_telegram_credentials,
    write_launchd_plist,
)
from tradingagents.option_runtime import default_runtime_root, runtime_runner_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=default_scheduler_label())
    parser.add_argument(
        "--launch-agent-dir",
        default=str(Path.home() / "Library" / "LaunchAgents"),
        help="launchd user-agent directory",
    )
    parser.add_argument("--log-dir", default=str(default_scheduler_log_dir()))
    parser.add_argument("--credential-file", default=str(default_scheduler_credential_path()))
    parser.add_argument("--runtime-root", default=str(default_runtime_root()))
    parser.add_argument("--db", help="optional registry SQLite override passed to daily ops")
    parser.add_argument("--policy", help="optional portfolio policy JSON override passed to daily ops")
    sub = parser.add_subparsers(dest="command", required=True)

    configure = sub.add_parser("configure-telegram", help="securely store Telegram delivery credentials")
    configure.add_argument("--chat-id", help="chat ID; token is always read interactively")

    install = sub.add_parser("install", help="install or replace the user LaunchAgent")
    install.add_argument(
        "--time",
        default=default_schedule_time(),
        help="weekday Mac-local time in HH:MM, default 22:00",
    )
    install.add_argument(
        "--preview",
        action="store_true",
        help="install a non-sending scheduler for launchd/TCC validation",
    )
    install.add_argument(
        "--kickstart",
        action="store_true",
        help="run the newly installed job immediately once",
    )
    install.add_argument(
        "--source-checkout",
        action="store_true",
        help="use the editable source checkout instead of the hardened production runtime",
    )

    sub.add_parser("status", help="print launchctl state for the configured label")
    sub.add_parser("uninstall", help="boot out and remove the configured LaunchAgent")
    return parser


def _target_plist(args) -> Path:
    return Path(args.launch_agent_dir).expanduser() / f"{args.label}.plist"


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _configure(args) -> int:
    token = getpass.getpass("Telegram bot token: ").strip()
    chat_id = args.chat_id or input("Telegram chat ID: ").strip()
    path = save_telegram_credentials(
        args.credential_file,
        credential=token,
        target=chat_id,
    )
    # Validate permission/content shape without printing either secret value.
    load_telegram_credentials(path)
    print(f"Telegram credential file configured: {path}")
    print("Credential contents were not printed.")
    return 0


def _install(args) -> int:
    parse_schedule_time(args.time)
    repo = _repo_root()
    runtime_root = Path(args.runtime_root).expanduser()
    runner = runtime_runner_path(runtime_root)
    python_path = repo / ".venv" / "bin" / "python"
    daily_script = repo / "scripts" / "options_daily_ops.py"
    if args.source_checkout:
        if not python_path.exists():
            raise ValueError(f"project virtualenv Python not found: {python_path}")
        if not daily_script.exists():
            raise ValueError(f"daily operations script not found: {daily_script}")
    elif not runner.exists():
        raise ValueError(
            f"hardened option runtime is not installed: {runner}; run scripts/options_runtime.py install"
        )

    credential_path = Path(args.credential_file).expanduser()
    if not args.preview:
        load_telegram_credentials(credential_path)

    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    content = render_launchd_plist(
        label=args.label,
        repo_root=repo if args.source_checkout else None,
        python_path=python_path if args.source_checkout else None,
        schedule_time=args.time,
        log_dir=log_dir,
        preview=args.preview,
        credential_path=None if args.preview else credential_path,
        registry_db_path=args.db,
        policy_path=args.policy,
        runner_path=None if args.source_checkout else runner,
        working_directory=repo if args.source_checkout else runtime_root,
    )
    target = write_launchd_plist(_target_plist(args), content)
    domain = _domain()
    _launchctl("bootout", domain, str(target), check=False)
    _launchctl("bootstrap", domain, str(target))
    _launchctl("enable", f"{domain}/{args.label}")
    if args.kickstart:
        _launchctl("kickstart", "-k", f"{domain}/{args.label}")

    print(f"LaunchAgent: {target}")
    print(f"Label: {args.label}")
    print(f"Mode: {'PREVIEW' if args.preview else 'SEND'}")
    print(f"Schedule: Monday-Friday at {args.time} Mac local time")
    print(f"Logs: {log_dir}")
    if args.preview:
        print("Preview mode never requests Telegram credentials and never sends Telegram.")
    else:
        print(f"Credential file: {credential_path} (contents not printed)")
    if args.source_checkout:
        print("Runtime mode: SOURCE_CHECKOUT (development/smoke only).")
    else:
        print(f"Runtime mode: HARDENED ({runtime_root})")
        print(f"Runner: {runner}")
    return 0


def _status(args) -> int:
    target = f"{_domain()}/{args.label}"
    result = _launchctl("print", target, check=False)
    if result.returncode != 0:
        print(f"NOT_LOADED: {args.label}")
        return 1
    print(result.stdout.rstrip())
    return 0


def _uninstall(args) -> int:
    target = _target_plist(args)
    _launchctl("bootout", _domain(), str(target), check=False)
    if target.exists():
        target.unlink()
    print(f"Uninstalled: {args.label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "configure-telegram":
            return _configure(args)
        if args.command == "install":
            return _install(args)
        if args.command == "status":
            return _status(args)
        if args.command == "uninstall":
            return _uninstall(args)
        raise ValueError(f"unsupported command: {args.command}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"<option daily scheduler unavailable: {exc}>")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
