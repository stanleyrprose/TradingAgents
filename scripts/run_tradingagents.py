#!/usr/bin/env python3
import argparse
import datetime
import json
from pathlib import Path

from tradingagents.instrument_router import PRIMARY_TYPES, classify_instrument

DECISION_CONTRACT_VERSION = "tradingagents-decision-v1"


def _default_decision_json_path(report_path: str) -> Path:
    report = Path(report_path)
    if report.suffix:
        return report.with_name("decision.json")
    return report / "decision.json"


def _write_decision_contract(
    *,
    profile,
    analysis_date: str,
    decision,
    report_path: str,
    output_path: str | None = None,
) -> Path:
    destination = Path(output_path) if output_path else _default_decision_json_path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    decision_text = str(decision).strip()
    payload = {
        "contract_version": DECISION_CONTRACT_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "analysis_date": analysis_date,
        "canonical_symbol": profile.canonical_symbol,
        "analysis_symbol": profile.analysis_symbol,
        "primary_type": profile.primary_type,
        "asset_class": profile.asset_class,
        "instrument_kind": profile.instrument_kind,
        "decision": decision_text,
        "decision_normalized": decision_text.upper().replace("_", " "),
        "report_path": str(report_path),
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(
        description="Classify an instrument and run TradingAgents with a safe analysis symbol.",
        epilog=(
            "examples: %(prog)s AAPL --date 2026-09-10; "
            "%(prog)s AAPL260918C00200000 --detect-only; "
            "%(prog)s MYSTERY --type fund"
        ),
    )
    parser.add_argument("symbol", help="ticker, pair, alias, futures symbol, OCC option, or ISIN")
    parser.add_argument(
        "--date",
        default=datetime.date.today().isoformat(),
        help="analysis date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--type",
        choices=PRIMARY_TYPES,
        help="override detected routing type ('bond' is a legacy alias for fixed_income)",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="print routing metadata without constructing or running the graph",
    )
    parser.add_argument(
        "--decision-json",
        help=(
            "write a machine-readable decision contract to this path; "
            "defaults to decision.json beside the generated report"
        ),
    )
    args = parser.parse_args()
    profile = classify_instrument(args.symbol, args.type)
    header_fields = (
        "canonical_symbol",
        "analysis_symbol",
        "primary_type",
        "asset_class",
        "instrument_kind",
        "capability",
        "analysts",
        "can_run",
        "notes",
    )
    for field in header_fields:
        print(f"{field}: {getattr(profile, field)}")
    if args.detect_only:
        return 0
    if not profile.can_run:
        print(
            "Cannot run this instrument: the current graph requires vendor price bars. "
            "Use a Yahoo-compatible price symbol/proxy or run with --detect-only.",
            flush=True,
        )
        return 2
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.runner_config import build_codex_oauth_config

    config = build_codex_oauth_config()
    graph = TradingAgentsGraph(selected_analysts=list(profile.analysts), config=config, debug=False)
    final_state, decision = graph.propagate(
        profile.canonical_symbol,
        args.date,
        asset_type=profile.pipeline_asset_type,
        analysis_symbol=profile.analysis_symbol,
    )
    report_path = graph.save_reports(final_state, profile.canonical_symbol)
    contract_path = _write_decision_contract(
        profile=profile,
        analysis_date=args.date,
        decision=decision,
        report_path=report_path,
        output_path=args.decision_json,
    )
    print(f"decision: {decision}")
    print(f"reports: {report_path}")
    print(f"decision_json: {contract_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
