#!/usr/bin/env python3
"""Refresh all registered open option positions into one deterministic daily dashboard."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from datetime import date

from tradingagents.dataflows.equity_options import fetch_equity_option_snapshots
from tradingagents.option_portfolio_dashboard import build_option_portfolio_dashboard
from tradingagents.option_position_registry import OptionPositionRegistry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="override option registry SQLite path")
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (current Cboe day only)")
    parser.add_argument("--underlying", help="optional underlying filter, e.g. AAPL")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of Markdown")
    return parser


def _json_payload(result) -> dict:
    return {
        "as_of": result.as_of.isoformat(),
        "counts": {
            "EXIT": result.exit_count,
            "REVIEW": result.review_count,
            "HOLD": result.hold_count,
            "REPORT_ONLY": result.report_only_count,
        },
        "total_liquidation_value": result.total_liquidation_value,
        "total_current_leg_pnl_dollars": result.total_current_leg_pnl_dollars,
        "total_lifecycle_pnl_dollars": result.total_lifecycle_pnl_dollars,
        "rows": [asdict(row) for row in result.rows],
        "underlyings": [asdict(item) for item in result.underlyings],
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
        snapshots = (
            {}
            if not symbols
            else fetch_equity_option_snapshots(symbols, args.date)
        )
        dashboard = build_option_portfolio_dashboard(
            positions,
            snapshots,
            as_of=as_of,
        )
        if args.json:
            print(json.dumps(_json_payload(dashboard), indent=2, sort_keys=True))
        else:
            print(dashboard.report)
        return 0
    except (ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"<option dashboard unavailable: {exc}>")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
