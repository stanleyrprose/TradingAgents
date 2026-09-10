#!/usr/bin/env python3
"""Monitor one existing long US equity-option position against explicit exit rules."""

from __future__ import annotations

import argparse
from datetime import date

from tradingagents.dataflows.equity_options import fetch_equity_option_snapshot
from tradingagents.instrument_router import classify_instrument
from tradingagents.option_position_manager import (
    evaluate_long_option_position,
    resolve_long_option_position_inputs,
    resolve_option_exit_policy,
)
from tradingagents.option_thesis_gate import refresh_long_option_thesis


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
    parser.add_argument(
        "--refresh-thesis",
        action="store_true",
        help="rerun the underlying TradingAgents thesis and compare it with this long call/put",
    )
    parser.add_argument(
        "--exit-on-thesis-invalidation",
        action="store_true",
        help="explicitly treat opposite refreshed consensus as EXIT; implies --refresh-thesis",
    )
    return parser


def _graph_dependencies():
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.runner_config import build_codex_oauth_config

    return TradingAgentsGraph, build_codex_oauth_config


def _refresh_underlying_thesis(snapshot, analysis_date: str):
    profile = classify_instrument(snapshot.underlying)
    if not profile.can_run or profile.asset_class != "equity":
        raise ValueError(
            "thesis refresh requires a runnable equity underlying"
        )
    graph_factory, config_factory = _graph_dependencies()
    graph = graph_factory(
        selected_analysts=list(profile.analysts),
        config=config_factory(),
        debug=False,
    )
    final_state, decision = graph.propagate(
        profile.canonical_symbol,
        analysis_date,
        asset_type=profile.pipeline_asset_type,
        analysis_symbol=profile.analysis_symbol,
    )
    report_path = graph.save_reports(final_state, profile.canonical_symbol)
    refresh = refresh_long_option_thesis(
        snapshot.right,
        decision,
        final_state.get("trader_investment_plan", ""),
    )
    return refresh, report_path


def _render_thesis_refresh(refresh, report_path, *, exit_on_invalidation: bool) -> str:
    current = refresh.current_direction or "none"
    policy = (
        "EXIT on explicit opposite consensus"
        if exit_on_invalidation
        else "informational only; does not alter the position-management decision"
    )
    return "\n".join(
        [
            "# Underlying thesis refresh",
            "",
            f"Status: **{refresh.status}**",
            f"Existing option thesis: {refresh.option_direction}",
            f"Refreshed underlying direction: {current}",
            f"Portfolio rating: {refresh.portfolio_rating or 'unparseable'}",
            f"Trader action: {refresh.trader_action or 'unparseable'}",
            f"Policy: {policy}",
            f"Reason: {refresh.reason}",
            f"Underlying report: {report_path}",
        ]
    )


def _final_status(position_status: str, refresh_status: str | None, *, exit_on_invalidation: bool) -> str:
    if position_status == "EXIT":
        return "EXIT"
    if exit_on_invalidation and refresh_status == "INVALIDATED":
        return "EXIT"
    return position_status


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

    snapshot = snapshot_result.snapshot
    result = evaluate_long_option_position(
        snapshot,
        entry_premium=entry_premium,
        contracts=contracts,
        **policy,
    )
    print(result.report)

    refresh_status = None
    refresh_requested = args.refresh_thesis or args.exit_on_thesis_invalidation
    if refresh_requested:
        try:
            refresh, report_path = _refresh_underlying_thesis(snapshot, args.date)
        except Exception as exc:  # noqa: BLE001 - explicit refresh must fail closed
            print(f"<underlying thesis refresh unavailable: {exc}>")
            return 2
        refresh_status = refresh.status
        print()
        print(
            _render_thesis_refresh(
                refresh,
                report_path,
                exit_on_invalidation=args.exit_on_thesis_invalidation,
            )
        )

    final_status = _final_status(
        result.status,
        refresh_status,
        exit_on_invalidation=args.exit_on_thesis_invalidation,
    )
    if final_status != result.status:
        print()
        print(
            "FINAL LIFECYCLE STATUS: EXIT — explicit "
            "--exit-on-thesis-invalidation policy triggered."
        )
    return 2 if final_status == "REVIEW" else 0


if __name__ == "__main__":
    raise SystemExit(main())
