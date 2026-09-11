#!/usr/bin/env python3
"""Manage one registered open option position without retyping its trade basis."""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
from datetime import date
from pathlib import Path

from tradingagents.option_position_registry import OptionPositionRecord, OptionPositionRegistry

_EXIT_ARG_MAP = {
    "take_profit_pct": "--take-profit-pct",
    "stop_loss_pct": "--stop-loss-pct",
    "exit_at_dte": "--exit-at-dte",
    "max_theta_burn_pct_per_day": "--max-theta-burn-pct-per-day",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="position_id, current OCC symbol, or unique open underlying")
    parser.add_argument("--db", help="override option registry SQLite path")
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (current Cboe day only)")
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="show the resolved registry position and generated manager inputs without external calls",
    )
    parser.add_argument("--refresh-thesis", action="store_true")
    parser.add_argument("--exit-on-thesis-invalidation", action="store_true")
    parser.add_argument("--plan-roll", action="store_true")
    parser.add_argument("--roll-top", type=int)
    parser.add_argument("--roll-min-dte", type=int)
    parser.add_argument("--roll-max-dte", type=int)
    parser.add_argument("--roll-target-delta", type=float)
    return parser


def _load_position_manager():
    path = Path(__file__).with_name("manage_option_position.py")
    spec = importlib.util.spec_from_file_location("registered_option_position_manager", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load position manager: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manager_argv(position: OptionPositionRecord, args) -> list[str]:
    argv = [
        position.current_symbol,
        "--entry-premium",
        str(position.current_leg_entry_premium),
        "--lifecycle-net-premium",
        str(position.lifecycle_net_premium_per_share),
        "--contracts",
        str(position.contracts),
        "--date",
        args.date,
    ]
    for field, flag in _EXIT_ARG_MAP.items():
        value = position.exit_policy.get(field)
        if value is not None:
            argv.extend([flag, str(value)])
    if args.refresh_thesis:
        argv.append("--refresh-thesis")
    if args.exit_on_thesis_invalidation:
        argv.append("--exit-on-thesis-invalidation")
    if args.plan_roll:
        argv.append("--plan-roll")
    for value, flag in (
        (args.roll_top, "--roll-top"),
        (args.roll_min_dte, "--roll-min-dte"),
        (args.roll_max_dte, "--roll-max-dte"),
        (args.roll_target_delta, "--roll-target-delta"),
    ):
        if value is not None:
            argv.extend([flag, str(value)])
    return argv


def _print_context(position: OptionPositionRecord, manager_argv: list[str], *, resolve_only: bool) -> None:
    print("# Registered option position")
    print()
    print(f"Position ID: {position.position_id}")
    print(f"Underlying: {position.underlying} | Current contract: {position.current_symbol}")
    print(f"Contracts: {position.contracts}")
    print(f"Current-leg entry premium: ${position.current_leg_entry_premium:.4f}/share")
    print(f"Lifecycle net premium basis: ${position.lifecycle_net_premium_per_share:.4f}/share")
    print(f"Roll count: {position.roll_count}")
    print(f"Stored exit policy: {position.exit_policy or '{}'}")
    print(f"Stored Greek limits: {position.greek_limits or '{}'}")
    print(f"Latest stored thesis: {position.latest_thesis or '{}'}")
    if position.greek_limits:
        print(
            "Greek-policy note: stored limits are journaled for audit but are not automatically "
            "re-enforced by this v1.8 existing-position manager."
        )
    print(
        "Journal note: analysis, thesis refresh, and roll planning do not record execution events; "
        "use options_registry.py roll/close only after an actual execution."
    )
    if resolve_only:
        print()
        print("# Resolved manager inputs")
        print(" ".join(manager_argv))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        registry = OptionPositionRegistry(args.db)
        position = registry.resolve_open_position(args.query)
        manager_argv = _manager_argv(position, args)
        _print_context(position, manager_argv, resolve_only=args.resolve_only)
        if args.resolve_only:
            return 0
        manager = _load_position_manager()
        print()
        return int(manager.main(manager_argv))
    except (KeyError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"<registered option management unavailable: {exc}>")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
