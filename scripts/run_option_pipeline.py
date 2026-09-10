#!/usr/bin/env python3
"""Select current equity option contracts, then analyze the top one or three."""

import argparse
from datetime import date

from tradingagents.dataflows.option_selector import rank_equity_option_contracts
from tradingagents.instrument_router import classify_instrument


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("underlying", help="US equity ticker (1-6 letters)")
    parser.add_argument("--direction", required=True, choices=("bullish", "bearish"))
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (today only)")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=45)
    parser.add_argument("--target-delta", type=float, default=0.55)
    parser.add_argument("--top", type=int, default=5, help="number of ranked candidates to show")
    parser.add_argument("--analyze-top", type=int, choices=(1, 3), default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selection = rank_equity_option_contracts(
        args.underlying,
        args.direction,
        args.date,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        target_delta=args.target_delta,
        top_n=args.top,
    )
    print(selection.report)
    if not selection.available:
        return 2

    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.runner_config import build_codex_oauth_config

    failed = False
    for rank, candidate in enumerate(selection.candidates[: args.analyze_top], 1):
        try:
            profile = classify_instrument(candidate.symbol)
            graph = TradingAgentsGraph(
                selected_analysts=list(profile.analysts),
                config=build_codex_oauth_config(),
                debug=False,
            )
            final_state, decision = graph.propagate(
                profile.canonical_symbol,
                args.date,
                asset_type=profile.pipeline_asset_type,
                analysis_symbol=profile.analysis_symbol,
            )
            report_path = graph.save_reports(final_state, profile.canonical_symbol)
            print(
                f"rank: {rank} | symbol: {candidate.symbol} | selector score: "
                f"{candidate.score:.2f} | decision: {decision} | report: {report_path}"
            )
        except Exception as exc:  # noqa: BLE001 - one contract must not abort the shortlist
            failed = True
            print(
                f"rank: {rank} | symbol: {candidate.symbol} | selector score: "
                f"{candidate.score:.2f} | failed: {exc}"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
