#!/usr/bin/env python3
"""Install, inspect, or remove the macOS launchd watchdog for Options Daily Ops."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from tradingagents.option_daily_scheduler import (
    default_scheduler_credential_path,
    default_scheduler_label,
    default_scheduler_log_dir,
    default_watchdog_label,
    default_watchdog_schedule_time,
    load_telegram_credentials,
    parse_schedule_time,
    render_watchdog_launchd_plist,
    write_launchd_plist,
)
from tradingagents.option_runtime import default_runtime_root, runtime_watchdog_runner_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=default_watchdog_label())
    parser.add_argument("--daily-label", default=default_scheduler_label())
    parser.add_argument(
        "--launch-agent-dir",
        default=str(Path.home() / "Library" / "LaunchAgents"),
    )
    parser.add_argument("--log-dir", default=str(default_scheduler_log_dir()))
    parser.add_argument("--credential-file", default=str(default_scheduler_credential_path()))
    parser.add_argument("--runtime-root", default=str(default_runtime_root()))
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="install or replace the watchdog LaunchAgent")
    install.add_argument("--time", default=default_watchdog_schedule_time())
    install.add_argument("--preview", action="store_true")
    install.add_argument("--kickstart", action="store_true")

    sub.add_parser("status")
    sub.add_parser("uninstall")
    return parser


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _target_plist(args) -> Path:
    return Path(args.launch_agent_dir).expanduser() / f"{args.label}.plist"


def _daily_plist(args) -> Path:
    return Path(args.launch_agent_dir).expanduser() / f"{args.daily_label}.plist"


def _install(args) -> int:
    parse_schedule_time(args.time)
    runtime_root = Path(args.runtime_root).expanduser()
    runner = runtime_watchdog_runner_path(runtime_root)
    if not runner.exists():
        raise ValueError(
            f"hardened watchdog runner is not installed: {runner}; run scripts/options_runtime.py install"
        )
    daily_plist = _daily_plist(args)
    if not daily_plist.exists():
        raise ValueError(f"production Daily Ops LaunchAgent is missing: {daily_plist}")

    credential_path = Path(args.credential_file).expanduser()
    if not args.preview:
        load_telegram_credentials(credential_path)

    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    content = render_watchdog_launchd_plist(
        label=args.label,
        runner_path=runner,
        runtime_root=runtime_root,
        schedule_time=args.time,
        log_dir=log_dir,
        daily_label=args.daily_label,
        daily_launch_agent=daily_plist,
        preview=args.preview,
        credential_path=None if args.preview else credential_path,
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
    print(f"Runner: {runner}")
    if args.preview:
        print("Preview mode never reads Telegram credentials or sends Telegram.")
    else:
        print(f"Credential file: {credential_path} (contents not printed)")
    return 0


def _status(args) -> int:
    result = _launchctl("print", f"{_domain()}/{args.label}", check=False)
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
        if args.command == "install":
            return _install(args)
        if args.command == "status":
            return _status(args)
        if args.command == "uninstall":
            return _uninstall(args)
        raise ValueError(f"unsupported command: {args.command}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"<option watchdog scheduler unavailable: {exc}>")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
