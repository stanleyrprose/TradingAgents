#!/usr/bin/env python3
"""Select a current Cboe delayed equity option contract without running the graph."""

import argparse
from datetime import date

from tradingagents.dataflows.option_selector import select_equity_option_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("underlying", help="US equity ticker (1-6 letters)")
    parser.add_argument("--direction", required=True, choices=("bullish", "bearish"))
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (today only)")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=45)
    parser.add_argument("--target-delta", type=float, default=0.55)
    parser.add_argument("--top", type=int, default=5, help="number of ranked candidates to show")
    args = parser.parse_args()
    print(
        select_equity_option_contract(
            args.underlying,
            args.direction,
            args.date,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            target_delta=args.target_delta,
            top_n=args.top,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
