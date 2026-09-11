#!/usr/bin/env python3
"""Refresh registered open option positions into a daily portfolio dashboard and action queue."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from datetime import date

from tradingagents.dataflows.equity_options import fetch_equity_option_snapshots
from tradingagents.option_portfolio_dashboard import build_option_portfolio_dashboard
from tradingagents.option_portfolio_policy import (
    OptionPortfolioRiskPolicy,
    build_daily_action_queue,
    default_portfolio_policy_path,
    evaluate_portfolio_risk_policy,
    load_portfolio_risk_policy,
    policy_result_to_dict,
    render_daily_action_queue,
)
from tradingagents.option_position_registry import OptionPositionRegistry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="override option registry SQLite path")
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (current Cboe day only)")
    parser.add_argument("--underlying", help="optional underlying filter, e.g. AAPL")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of Markdown")
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument("--policy", help="explicit portfolio policy JSON path override")
    policy_group.add_argument(
        "--no-policy",
        action="store_true",
        help="skip any saved portfolio policy for this dashboard run",
    )
    return parser


def _json_payload(dashboard, policy_result, actions, policy_path) -> dict:
    return {
        "as_of": dashboard.as_of.isoformat(),
        "counts": {
            "EXIT": dashboard.exit_count,
            "REVIEW": dashboard.review_count,
            "HOLD": dashboard.hold_count,
            "REPORT_ONLY": dashboard.report_only_count,
        },
        "total_liquidation_value": dashboard.total_liquidation_value,
        "total_current_leg_pnl_dollars": dashboard.total_current_leg_pnl_dollars,
        "total_lifecycle_pnl_dollars": dashboard.total_lifecycle_pnl_dollars,
        "rows": [asdict(row) for row in dashboard.rows],
        "underlyings": [asdict(item) for item in dashboard.underlyings],
        "portfolio_policy_path": None if policy_path is None else str(policy_path),
        "portfolio_policy": policy_result_to_dict(policy_result),
        "actions": [asdict(item) for item in actions],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        try:
            as_of = date.fromisoformat(args.date)
        except ValueError as exc:
            raise ValueError("date must be YYYY-MM-DD") from exc

        registry = OptionPositionRegistry(args.db)
        positions = registry.list_positions(
            underlying=None if args.underlying is None else args.underlying.upper(),
            status="OPEN",
        )
        symbols = [position.current_symbol for position in positions]
        snapshots = {} if not symbols else fetch_equity_option_snapshots(symbols, args.date)
        dashboard = build_option_portfolio_dashboard(positions, snapshots, as_of=as_of)

        if args.no_policy:
            policy_path = None
            policy = OptionPortfolioRiskPolicy()
        else:
            policy_path = args.policy or default_portfolio_policy_path(args.db)
            policy = load_portfolio_risk_policy(policy_path)
        policy_result = evaluate_portfolio_risk_policy(
            dashboard,
            policy,
            book_scope_complete=args.underlying is None,
        )
        actions = build_daily_action_queue(dashboard, policy_result)

        if args.json:
            print(
                json.dumps(
                    _json_payload(dashboard, policy_result, actions, policy_path),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(dashboard.report)
            print()
            print(policy_result.report)
            print()
            print(render_daily_action_queue(actions))
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"<option dashboard unavailable: {exc}>")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
