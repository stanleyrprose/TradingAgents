#!/usr/bin/env python3
import argparse
import datetime
import os

from tradingagents.instrument_router import PRIMARY_TYPES, classify_instrument


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--type", choices=PRIMARY_TYPES)
    parser.add_argument("--detect-only", action="store_true")
    args = parser.parse_args()
    profile = classify_instrument(args.symbol, args.type)
    for field in profile.__dataclass_fields__:
        print(f"{field}: {getattr(profile, field)}")
    if args.detect_only:
        return 0
    if not profile.can_run:
        print("Cannot run: this ISIN-shaped bond requires a supported bond data pipeline.", flush=True)
        return 2
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    config = DEFAULT_CONFIG.copy()
    defaults = {
        "TRADINGAGENTS_QUICK_THINK_LLM_PROVIDER": ("quick_think_llm_provider", "codex_cli"),
        "TRADINGAGENTS_QUICK_THINK_LLM": ("quick_think_llm", "gpt-5.6-luna"),
        "TRADINGAGENTS_CODEX_QUICK_REASONING_EFFORT": ("codex_quick_reasoning_effort", "low"),
        "TRADINGAGENTS_TRADER_THINK_LLM_PROVIDER": ("trader_think_llm_provider", "codex_cli"),
        "TRADINGAGENTS_TRADER_THINK_LLM": ("trader_think_llm", "gpt-5.6-sol"),
        "TRADINGAGENTS_CODEX_TRADER_REASONING_EFFORT": ("codex_trader_reasoning_effort", "medium"),
        "TRADINGAGENTS_DEEP_THINK_LLM_PROVIDER": ("deep_think_llm_provider", "codex_cli"),
        "TRADINGAGENTS_DEEP_THINK_LLM": ("deep_think_llm", "gpt-5.6-sol"),
        "TRADINGAGENTS_CODEX_DEEP_REASONING_EFFORT": ("codex_deep_reasoning_effort", "high"),
        "TRADINGAGENTS_CHECKPOINT_ENABLED": ("checkpoint_enabled", True),
    }
    for env, (key, value) in defaults.items():
        if env not in os.environ:
            config[key] = value
    graph = TradingAgentsGraph(selected_analysts=list(profile.analysts), config=config, debug=False)
    final_state, decision = graph.propagate(profile.canonical_symbol, args.date, asset_type=profile.pipeline_asset_type)
    print(f"decision: {decision}")
    print(f"reports: {graph.save_reports(final_state, profile.canonical_symbol)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
