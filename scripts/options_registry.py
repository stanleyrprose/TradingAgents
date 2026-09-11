#!/usr/bin/env python3
"""Manage the persistent long-equity-option position registry and trade journal."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from datetime import date
from typing import Any

from tradingagents.option_exposure_guard import resolve_option_exposure_limits
from tradingagents.option_position_manager import resolve_option_exit_policy
from tradingagents.option_position_registry import OptionPositionRecord, OptionPositionRegistry

_EXIT_FIELDS = (
    "take_profit_pct",
    "stop_loss_pct",
    "exit_at_dte",
    "max_theta_burn_pct_per_day",
)
_GREEK_FIELDS = (
    "max_abs_delta_shares",
    "max_abs_gamma_delta_shares_per_dollar",
    "max_abs_vega_dollars_per_vol_point",
    "max_abs_theta_dollars_per_day",
)


def _add_exit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--take-profit-pct", type=float)
    parser.add_argument("--stop-loss-pct", type=float)
    parser.add_argument("--exit-at-dte", type=int)
    parser.add_argument("--max-theta-burn-pct-per-day", type=float)


def _add_greek_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-abs-delta-shares", type=float)
    parser.add_argument("--max-abs-gamma-delta-shares-per-dollar", type=float)
    parser.add_argument("--max-abs-vega-dollars-per-vol-point", type=float)
    parser.add_argument("--max-abs-theta-dollars-per-day", type=float)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="override option registry SQLite path")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    open_cmd = sub.add_parser("open", help="record a newly executed long option position")
    open_cmd.add_argument("symbol")
    open_cmd.add_argument("--entry-premium", type=float, required=True)
    open_cmd.add_argument("--contracts", type=int, required=True)
    open_cmd.add_argument("--entry-date", default=date.today().isoformat())
    open_cmd.add_argument("--position-id")
    _add_exit_args(open_cmd)
    _add_greek_args(open_cmd)
    open_cmd.add_argument("--thesis-direction", choices=("bullish", "bearish"))
    open_cmd.add_argument("--thesis-note")
    open_cmd.add_argument("--thesis-report-path")

    list_cmd = sub.add_parser("list", help="list positions")
    list_cmd.add_argument("--underlying")
    state = list_cmd.add_mutually_exclusive_group()
    state.add_argument("--all", action="store_true")
    state.add_argument("--closed", action="store_true")

    show_cmd = sub.add_parser("show", help="show one position by id or unique open symbol/underlying")
    show_cmd.add_argument("query")

    events_cmd = sub.add_parser("events", help="show append-only journal events")
    events_cmd.add_argument("query")

    policy_cmd = sub.add_parser("policy", help="replace stored exit and/or Greek policy groups")
    policy_cmd.add_argument("query")
    policy_cmd.add_argument("--date", default=date.today().isoformat())
    policy_cmd.add_argument("--clear-exit-policy", action="store_true")
    policy_cmd.add_argument("--clear-greek-limits", action="store_true")
    _add_exit_args(policy_cmd)
    _add_greek_args(policy_cmd)

    thesis_cmd = sub.add_parser("thesis", help="record a refreshed underlying thesis fact")
    thesis_cmd.add_argument("query")
    thesis_cmd.add_argument("--date", default=date.today().isoformat())
    thesis_cmd.add_argument("--status", choices=("CONFIRMED", "NEUTRAL", "INVALIDATED"), required=True)
    thesis_cmd.add_argument("--current-direction", choices=("bullish", "bearish", "none"), default="none")
    thesis_cmd.add_argument("--portfolio-rating")
    thesis_cmd.add_argument("--trader-action")
    thesis_cmd.add_argument("--reason")
    thesis_cmd.add_argument("--report-path")

    roll_cmd = sub.add_parser("roll", help="record an actually executed same-direction roll")
    roll_cmd.add_argument("query")
    roll_cmd.add_argument("--new-symbol", required=True)
    roll_cmd.add_argument("--close-credit", type=float, required=True)
    roll_cmd.add_argument("--new-entry-premium", type=float, required=True)
    roll_cmd.add_argument("--date", default=date.today().isoformat())

    close_cmd = sub.add_parser("close", help="record an actually executed close")
    close_cmd.add_argument("query")
    close_cmd.add_argument("--close-premium", type=float, required=True)
    close_cmd.add_argument("--date", default=date.today().isoformat())
    close_cmd.add_argument("--reason")
    return parser


def _compact(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _exit_policy(args) -> dict[str, Any]:
    return _compact(
        resolve_option_exit_policy(
            take_profit_pct=args.take_profit_pct,
            stop_loss_pct=args.stop_loss_pct,
            exit_at_dte=args.exit_at_dte,
            max_theta_burn_pct_per_day=args.max_theta_burn_pct_per_day,
        )
    )


def _greek_limits(args) -> dict[str, Any]:
    return _compact(
        resolve_option_exposure_limits(
            max_abs_delta_shares=args.max_abs_delta_shares,
            max_abs_gamma_delta_shares_per_dollar=args.max_abs_gamma_delta_shares_per_dollar,
            max_abs_vega_dollars_per_vol_point=args.max_abs_vega_dollars_per_vol_point,
            max_abs_theta_dollars_per_day=args.max_abs_theta_dollars_per_day,
        )
    )


def _initial_thesis(args) -> dict[str, Any]:
    thesis = {}
    if args.thesis_direction is not None:
        thesis["direction"] = args.thesis_direction
    if args.thesis_note:
        thesis["note"] = args.thesis_note
    if args.thesis_report_path:
        thesis["report_path"] = args.thesis_report_path
    return thesis


def _resolve_any(registry: OptionPositionRegistry, query: str) -> OptionPositionRecord:
    try:
        return registry.get_position(query)
    except KeyError:
        return registry.resolve_open_position(query)


def _position_dict(position: OptionPositionRecord) -> dict[str, Any]:
    return asdict(position)


def _print_position(position: OptionPositionRecord) -> None:
    print(f"Position ID: {position.position_id}")
    print(f"Status: {position.status} | Underlying: {position.underlying} | Right: {position.option_right}")
    print(f"Original contract: {position.original_symbol} @ ${position.original_entry_premium:.4f}")
    print(f"Current contract:  {position.current_symbol} @ ${position.current_leg_entry_premium:.4f}")
    print(f"Lifecycle net premium/share: ${position.lifecycle_net_premium_per_share:.4f}")
    print(f"Contracts: {position.contracts} | Opened: {position.opened_at} | Current leg: {position.current_leg_opened_at}")
    print(f"Rolls: {position.roll_count}")
    if position.status == "CLOSED":
        print(f"Closed: {position.closed_at} @ ${position.close_premium:.4f}")
    print("Exit policy: " + json.dumps(position.exit_policy, sort_keys=True))
    print("Greek limits: " + json.dumps(position.greek_limits, sort_keys=True))
    print("Initial thesis: " + json.dumps(position.initial_thesis, sort_keys=True))
    print("Latest thesis:  " + json.dumps(position.latest_thesis, sort_keys=True))


def _emit_position(position: OptionPositionRecord, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(_position_dict(position), indent=2, sort_keys=True))
    else:
        _print_position(position)


def _emit_positions(positions: tuple[OptionPositionRecord, ...], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps([_position_dict(item) for item in positions], indent=2, sort_keys=True))
        return
    if not positions:
        print("No matching option positions.")
        return
    print("POSITION_ID | STATUS | UNDERLYING | CURRENT_CONTRACT | CONTRACTS | CURRENT_LEG_ENTRY | LIFECYCLE_NET | ROLLS")
    for item in positions:
        print(
            f"{item.position_id} | {item.status} | {item.underlying} | {item.current_symbol} | "
            f"{item.contracts} | {item.current_leg_entry_premium:.4f} | "
            f"{item.lifecycle_net_premium_per_share:.4f} | {item.roll_count}"
        )


def _policy_group_supplied(args, fields: tuple[str, ...]) -> bool:
    return any(getattr(args, field) is not None for field in fields)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        registry = OptionPositionRegistry(args.db)

        if args.command == "open":
            position = registry.open_position(
                args.symbol,
                entry_premium=args.entry_premium,
                contracts=args.contracts,
                entry_date=args.entry_date,
                exit_policy=_exit_policy(args),
                greek_limits=_greek_limits(args),
                initial_thesis=_initial_thesis(args),
                position_id=args.position_id,
            )
            _emit_position(position, json_output=args.json)
            return 0

        if args.command == "list":
            status = None if args.all else "CLOSED" if args.closed else "OPEN"
            positions = registry.list_positions(underlying=args.underlying, status=status)
            _emit_positions(positions, json_output=args.json)
            return 0

        position = _resolve_any(registry, args.query)

        if args.command == "show":
            _emit_position(position, json_output=args.json)
            return 0

        if args.command == "events":
            events = registry.list_events(position.position_id)
            if args.json:
                print(json.dumps([asdict(event) for event in events], indent=2, sort_keys=True))
            else:
                for event in events:
                    print(
                        f"{event.event_id} | {event.occurred_at} | {event.event_type} | "
                        + json.dumps(event.payload, sort_keys=True)
                    )
            return 0

        if args.command == "policy":
            exit_supplied = _policy_group_supplied(args, _EXIT_FIELDS)
            greek_supplied = _policy_group_supplied(args, _GREEK_FIELDS)
            if args.clear_exit_policy and exit_supplied:
                raise ValueError("--clear-exit-policy cannot be combined with exit policy values")
            if args.clear_greek_limits and greek_supplied:
                raise ValueError("--clear-greek-limits cannot be combined with Greek limit values")
            if not any((exit_supplied, greek_supplied, args.clear_exit_policy, args.clear_greek_limits)):
                raise ValueError("policy command requires an exit/Greek policy replacement or clear flag")
            exit_policy = (
                {}
                if args.clear_exit_policy
                else _exit_policy(args)
                if exit_supplied
                else None
            )
            greek_limits = (
                {}
                if args.clear_greek_limits
                else _greek_limits(args)
                if greek_supplied
                else None
            )
            updated = registry.set_policies(
                position.position_id,
                exit_policy=exit_policy,
                greek_limits=greek_limits,
                occurred_at=args.date,
            )
            _emit_position(updated, json_output=args.json)
            return 0

        if args.command == "thesis":
            thesis = {
                "status": args.status,
                "current_direction": None if args.current_direction == "none" else args.current_direction,
                "portfolio_rating": args.portfolio_rating,
                "trader_action": args.trader_action,
                "reason": args.reason,
                "report_path": args.report_path,
            }
            updated = registry.record_thesis(
                position.position_id,
                _compact(thesis),
                occurred_at=args.date,
            )
            _emit_position(updated, json_output=args.json)
            return 0

        if args.command == "roll":
            updated = registry.record_roll(
                position.position_id,
                args.new_symbol,
                close_credit=args.close_credit,
                new_entry_premium=args.new_entry_premium,
                roll_date=args.date,
            )
            _emit_position(updated, json_output=args.json)
            return 0

        if args.command == "close":
            updated = registry.close_position(
                position.position_id,
                close_premium=args.close_premium,
                close_date=args.date,
                reason=args.reason,
            )
            _emit_position(updated, json_output=args.json)
            return 0

        raise ValueError(f"unsupported command: {args.command}")
    except (KeyError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"<option registry error: {exc}>")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
