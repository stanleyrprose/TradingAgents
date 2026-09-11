#!/usr/bin/env python3
"""Persist or inspect explicit option portfolio risk policy limits."""

from __future__ import annotations

import argparse
import json

from tradingagents.option_portfolio_policy import (
    clear_portfolio_risk_policy,
    default_portfolio_policy_path,
    load_portfolio_risk_policy,
    policy_to_dict,
    resolve_portfolio_risk_policy,
    save_portfolio_risk_policy,
)


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-book-liquidation-value", type=float)
    parser.add_argument("--max-book-gross-theta-dollars-per-day", type=float)
    parser.add_argument("--max-underlying-liquidation-value", type=float)
    parser.add_argument("--max-underlying-abs-delta-shares", type=float)
    parser.add_argument("--max-underlying-gross-delta-shares", type=float)
    parser.add_argument("--max-underlying-gross-theta-dollars-per-day", type=float)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="registry DB used to derive the default policy path")
    parser.add_argument("--policy", help="explicit policy JSON path override")
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="show the currently saved explicit policy")
    set_cmd = sub.add_parser("set", help="replace the saved policy with supplied explicit caps")
    _add_policy_args(set_cmd)
    sub.add_parser("clear", help="delete the saved portfolio policy")
    return parser


def _path(args):
    return args.policy or default_portfolio_policy_path(args.db)


def _policy_from_args(args):
    return resolve_portfolio_risk_policy(
        max_book_liquidation_value=args.max_book_liquidation_value,
        max_book_gross_theta_dollars_per_day=args.max_book_gross_theta_dollars_per_day,
        max_underlying_liquidation_value=args.max_underlying_liquidation_value,
        max_underlying_abs_delta_shares=args.max_underlying_abs_delta_shares,
        max_underlying_gross_delta_shares=args.max_underlying_gross_delta_shares,
        max_underlying_gross_theta_dollars_per_day=args.max_underlying_gross_theta_dollars_per_day,
    )


def _print_policy(policy, path, *, json_output: bool) -> None:
    payload = policy_to_dict(policy)
    if json_output:
        print(json.dumps({"path": str(path), "policy": payload}, indent=2, sort_keys=True))
        return
    print(f"Policy path: {path}")
    if not payload:
        print("No explicit option portfolio risk policy is configured.")
        return
    for name, value in sorted(payload.items()):
        print(f"{name}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = _path(args)
    try:
        if args.command == "show":
            _print_policy(load_portfolio_risk_policy(path), path, json_output=args.json)
            return 0
        if args.command == "set":
            policy = _policy_from_args(args)
            if not policy.configured:
                raise ValueError("set requires at least one explicit portfolio risk cap")
            save_portfolio_risk_policy(policy, path)
            _print_policy(policy, path, json_output=args.json)
            return 0
        removed = clear_portfolio_risk_policy(path)
        if args.json:
            print(json.dumps({"path": str(path), "cleared": removed}, indent=2, sort_keys=True))
        else:
            print(f"Policy {'cleared' if removed else 'already absent'}: {path}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"<option portfolio policy unavailable: {exc}>")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
