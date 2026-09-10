#!/usr/bin/env python3
"""Select current equity option contracts, then analyze the top one or three."""

import argparse
import re
from datetime import date

from tradingagents.dataflows.option_selector import rank_equity_option_contracts
from tradingagents.instrument_router import classify_instrument
from tradingagents.option_thesis_gate import gate_option_thesis

_CBOE_UNDERLYING_RE = re.compile(r"[A-Z]{1,6}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("underlying", help="US equity ticker (1-6 letters)")
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--direction", choices=("bullish", "bearish"))
    direction.add_argument(
        "--auto-direction",
        action="store_true",
        help="derive direction from an underlying TradingAgents thesis",
    )
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (today only)")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=45)
    parser.add_argument("--target-delta", type=float, default=0.55)
    parser.add_argument("--top", type=int, default=5, help="number of ranked candidates to show")
    parser.add_argument("--analyze-top", type=int, choices=(1, 3), default=1)
    return parser


def _graph_dependencies():
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.runner_config import build_codex_oauth_config

    return TradingAgentsGraph, build_codex_oauth_config


def _analyze_contracts(args, selection, graph_factory, config_factory) -> int:
    """Run the existing fresh-graph analysis loop for selected contracts."""
    failed = False
    for rank, candidate in enumerate(selection.candidates[: args.analyze_top], 1):
        try:
            profile = classify_instrument(candidate.symbol)
            graph = graph_factory(
                selected_analysts=list(profile.analysts),
                config=config_factory(),
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


def _auto_direction(args):
    profile = classify_instrument(args.underlying)
    canonical = profile.canonical_symbol
    supported_equity = (
        profile.asset_class == "equity"
        and profile.primary_type in {"stock", "fund"}
        and profile.instrument_kind in {"stock", "etf", "fund"}
        and profile.can_run
        and _CBOE_UNDERLYING_RE.fullmatch(canonical) is not None
    )
    if not supported_equity:
        print(
            "<underlying thesis unavailable: auto-direction requires a runnable US "
            "equity stock or equity fund with a 1-6 letter Cboe symbol>"
        )
        return None, None, None, None, 2

    graph_factory, config_factory = _graph_dependencies()
    graph = graph_factory(
        selected_analysts=list(profile.analysts),
        config=config_factory(),
        debug=False,
    )
    final_state, decision = graph.propagate(
        profile.canonical_symbol,
        args.date,
        asset_type=profile.pipeline_asset_type,
        analysis_symbol=profile.analysis_symbol,
    )
    report_path = graph.save_reports(final_state, profile.canonical_symbol)
    gate = gate_option_thesis(decision, final_state.get("trader_investment_plan"))
    gate_result = gate.direction or "NO OPTION TRADE"
    print(
        "underlying thesis | "
        f"symbol: {profile.canonical_symbol} | portfolio rating: "
        f"{gate.portfolio_rating or 'unparseable'} | trader action: "
        f"{gate.trader_action or 'unparseable'} | gate: {gate_result} | "
        f"reason: {gate.reason} | report: {report_path}"
    )
    return gate.direction, profile.canonical_symbol, graph_factory, config_factory, 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    graph_factory = config_factory = None
    if args.auto_direction:
        direction, selector_underlying, graph_factory, config_factory, rc = _auto_direction(args)
        if rc or direction is None:
            return rc
    else:
        direction = args.direction
        selector_underlying = args.underlying

    selection = rank_equity_option_contracts(
        selector_underlying,
        direction,
        args.date,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        target_delta=args.target_delta,
        top_n=args.top,
    )
    print(selection.report)
    if not selection.available:
        return 2

    if graph_factory is None or config_factory is None:
        graph_factory, config_factory = _graph_dependencies()
    return _analyze_contracts(args, selection, graph_factory, config_factory)


if __name__ == "__main__":
    raise SystemExit(main())
