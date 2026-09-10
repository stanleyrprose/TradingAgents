#!/usr/bin/env python3
"""Monitor one existing long US equity-option position against explicit exit rules."""

from __future__ import annotations

import argparse
from datetime import date

from tradingagents.dataflows.equity_options import fetch_equity_option_snapshot
from tradingagents.option_position_manager import (
    evaluate_long_option_position,
    resolve_long_option_position_inputs,
    resolve_option_exit_policy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="exact OCC equity-option symbol")
    parser.add_argument("--entry-premium", type=float, required=True)
    parser.add_argument("--contracts", type=int, required=True)
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (today only)")
    parser.add_argument("--take-profit-pct", type=float)
    parser.add_argument("--stop-loss-pct", type=float)
    parser.add_argument("--exit-at-dte", type=int)
    parser.add_argument("--max-theta-burn-pct-per-day", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        entry_premium, contracts = resolve_long_option_position_inputs(
            entry_premium=args.entry_premium,
            contracts=args.contracts,
        )
        policy = resolve_option_exit_policy(
            take_profit_pct=args.take_profit_pct,
            stop_loss_pct=args.stop_loss_pct,
            exit_at_dte=args.exit_at_dte,
            max_theta_burn_pct_per_day=args.max_theta_burn_pct_per_day,
        )
    except ValueError as exc:
        print(f"option position policy error: {exc}")
        return 2

    snapshot_result = fetch_equity_option_snapshot(args.symbol, args.date)
    if not snapshot_result.available or snapshot_result.snapshot is None:
        reason = snapshot_result.unavailable_reason or "current option snapshot unavailable"
        print(f"<option position management unavailable: {reason}>")
        return 2

    result = evaluate_long_option_position(
        snapshot_result.snapshot,
        entry_premium=entry_premium,
        contracts=contracts,
        **policy,
    )
    print(result.report)
    return 2 if result.status == "REVIEW" else 0


if __name__ == "__main__":
    raise SystemExit(main())
