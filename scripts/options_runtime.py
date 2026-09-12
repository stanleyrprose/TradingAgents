#!/usr/bin/env python3
"""Install, inspect, rollback, or prune the hardened option production runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from tradingagents.option_daily_scheduler import default_scheduler_label
from tradingagents.option_runtime import (
    default_runtime_root,
    health_runtime,
    install_runtime,
    list_releases,
    prune_releases,
    rollback_runtime,
    scheduler_binding_issues,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", default=str(default_runtime_root()))
    parser.add_argument("--label", default=default_scheduler_label())
    parser.add_argument(
        "--launch-agent",
        default=str(Path.home() / "Library" / "LaunchAgents" / f"{default_scheduler_label()}.plist"),
    )
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="install and activate the current reviewed Git revision")
    install.add_argument(
        "--allow-unpushed",
        action="store_true",
        help="allow installing HEAD when it does not match the configured upstream",
    )

    sub.add_parser("health", help="validate the active runtime and runner")
    sub.add_parser("list", help="list installed releases")

    rollback = sub.add_parser("rollback", help="activate previous or an explicitly selected release")
    rollback.add_argument("--to", dest="to_release", help="release SHA prefix; defaults to previous")

    prune = sub.add_parser("prune", help="remove old inactive releases")
    prune.add_argument("--keep", type=int, default=3, help="keep at least this many newest releases")
    return parser


def _emit(payload: dict, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")


def _launchd_loaded(label: str) -> bool:
    result = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runtime_root = Path(args.runtime_root).expanduser()
    try:
        if args.command == "install":
            release = install_runtime(
                repo_root=_repo_root(),
                source_python=Path(sys.executable),
                runtime_root=runtime_root,
                allow_unpushed=args.allow_unpushed,
            )
            _emit(
                {
                    "status": "INSTALLED",
                    "runtime_root": str(runtime_root),
                    "release": release.git_short,
                    "git_sha": release.git_sha,
                    "dependency_fingerprint": release.dependency_fingerprint,
                    "env_relpath": release.env_relpath,
                },
                json_output=args.json,
            )
            return 0

        if args.command == "health":
            health = health_runtime(runtime_root)
            issues = list(health.issues)
            launch_agent_path = Path(args.launch_agent).expanduser()
            issues.extend(
                scheduler_binding_issues(
                    runtime_root=runtime_root,
                    launch_agent_path=launch_agent_path,
                )
            )
            launchd_loaded = _launchd_loaded(args.label)
            if not launchd_loaded:
                issues.append(f"LaunchAgent is not loaded: {args.label}")
            payload = asdict(health)
            payload.update(
                {
                    "status": "PASS" if not issues else "FAIL",
                    "issues": issues,
                    "launch_agent": str(launch_agent_path),
                    "launchd_loaded": launchd_loaded,
                }
            )
            _emit(payload, json_output=args.json)
            return 0 if not issues else 1

        if args.command == "list":
            releases = list_releases(runtime_root)
            current = (runtime_root / "current").resolve().name if (runtime_root / "current").is_symlink() else None
            previous = (runtime_root / "previous").resolve().name if (runtime_root / "previous").is_symlink() else None
            if args.json:
                print(
                    json.dumps(
                        {
                            "current": current,
                            "previous": previous,
                            "releases": [asdict(release) for release in releases],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"Current: {current or 'NONE'}")
                print(f"Previous: {previous or 'NONE'}")
                for release in releases:
                    marker = "*" if release.git_short == current else " "
                    print(
                        f"{marker} {release.git_short}  deps={release.dependency_fingerprint}  "
                        f"created={release.created_at}"
                    )
            return 0

        if args.command == "rollback":
            release = rollback_runtime(runtime_root, to_release=args.to_release)
            _emit(
                {"status": "ROLLED_BACK", "release": release.git_short, "git_sha": release.git_sha},
                json_output=args.json,
            )
            return 0

        if args.command == "prune":
            removed = prune_releases(runtime_root, keep=args.keep)
            _emit(
                {"status": "PRUNED", "removed": list(removed), "keep": args.keep},
                json_output=args.json,
            )
            return 0

        raise ValueError(f"unsupported command: {args.command}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"<option runtime unavailable: {exc}>")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
