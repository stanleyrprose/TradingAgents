#!/usr/bin/env python3
"""Build the daily option action brief and optionally deliver it to Telegram."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import asdict
from datetime import date

from tradingagents.dataflows.equity_options import (
    current_us_option_market_date,
    fetch_equity_option_snapshots,
)
from tradingagents.option_daily_ops import (
    build_option_daily_brief,
    deduplicated_delivery_result,
    default_daily_ops_receipt_path,
    delivery_target_hash,
    record_successful_delivery,
    successful_receipt_exists,
)
from tradingagents.option_daily_scheduler import load_telegram_credentials
from tradingagents.option_portfolio_dashboard import build_option_portfolio_dashboard
from tradingagents.option_portfolio_policy import (
    OptionPortfolioRiskPolicy,
    build_daily_action_queue,
    default_portfolio_policy_path,
    evaluate_portfolio_risk_policy,
    load_portfolio_risk_policy,
)
from tradingagents.option_position_registry import OptionPositionRegistry
from tradingagents.telegram_delivery import send_telegram_text

_CREDENTIAL_ENV = "TRADINGAGENTS_" + "TG_" + "BOT_" + "TOKEN"
_TARGET_ENV = "TRADINGAGENTS_" + "TG_" + "CHAT_" + "ID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="override option registry SQLite path")
    parser.add_argument(
        "--date",
        default=current_us_option_market_date().isoformat(),
        help="YYYY-MM-DD US option-market date; defaults to current America/New_York date",
    )
    parser.add_argument("--policy", help="explicit portfolio policy JSON path override")
    parser.add_argument("--no-policy", action="store_true", help="ignore any saved portfolio policy for this run")
    parser.add_argument("--receipt", help="override exact-message delivery receipt JSON path")
    parser.add_argument(
        "--telegram-config",
        help="0600 JSON credential file used when Telegram env vars are not set",
    )
    parser.add_argument("--max-actions", type=int, default=8, help="maximum action lines in the short brief (1-20)")
    parser.add_argument("--send", action="store_true", help="explicitly enable Telegram delivery")
    parser.add_argument("--force", action="store_true", help="with --send, bypass exact-message receipt deduplication")
    parser.add_argument("--json", action="store_true", help="emit machine-readable run result")
    return parser


def _credentials(config_path: str | None = None) -> tuple[str, str]:
    credential = os.getenv(_CREDENTIAL_ENV, "").strip()
    target = os.getenv(_TARGET_ENV, "").strip()
    if credential or target:
        if not credential or not target:
            raise ValueError(
                f"Telegram environment configuration is incomplete; set both {_CREDENTIAL_ENV} and {_TARGET_ENV}"
            )
        return credential, target
    if config_path:
        configured = load_telegram_credentials(config_path)
        return configured.credential, configured.target
    raise ValueError(
        f"Telegram delivery requires environment variables {_CREDENTIAL_ENV} and {_TARGET_ENV} or --telegram-config"
    )


def _payload(brief, delivery, policy_status: str) -> dict:
    return {
        "as_of": brief.as_of.isoformat(),
        "send_required": brief.send_required,
        "action_count": brief.action_count,
        "counts": {
            "POSITION_EXIT": brief.exit_count,
            "POLICY_BREACH": brief.policy_breach_count,
            "POLICY_NOT_EVALUABLE": brief.policy_not_evaluable_count,
            "POSITION_REVIEW": brief.review_count,
        },
        "policy_status": policy_status,
        "fingerprint": brief.fingerprint,
        "brief": brief.text,
        "delivery": None if delivery is None else asdict(delivery),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.force and not args.send:
            raise ValueError("--force requires --send")
        try:
            as_of = date.fromisoformat(args.date)
        except ValueError as exc:
            raise ValueError("date must be YYYY-MM-DD") from exc

        registry = OptionPositionRegistry(args.db)
        positions = registry.list_positions(status="OPEN")
        symbols = [position.current_symbol for position in positions]
        snapshots = {} if not symbols else fetch_equity_option_snapshots(symbols, args.date)
        dashboard = build_option_portfolio_dashboard(positions, snapshots, as_of=as_of)

        if args.no_policy:
            policy = OptionPortfolioRiskPolicy()
        else:
            policy_path = args.policy or default_portfolio_policy_path(args.db)
            policy = load_portfolio_risk_policy(policy_path)
        policy_result = evaluate_portfolio_risk_policy(dashboard, policy, book_scope_complete=True)
        actions = build_daily_action_queue(dashboard, policy_result)
        brief = build_option_daily_brief(
            dashboard,
            policy_result,
            actions,
            max_actions=args.max_actions,
        )

        delivery = None
        if not brief.send_required:
            status_line = "NO_ACTIONS"
        elif not args.send:
            status_line = "PREVIEW_ONLY"
        else:
            credential, target = _credentials(args.telegram_config)
            receipt_path = args.receipt or default_daily_ops_receipt_path(args.db)
            target_hash = delivery_target_hash(target)
            if not args.force and successful_receipt_exists(
                receipt_path,
                fingerprint=brief.fingerprint,
                target_hash=target_hash,
            ):
                delivery = deduplicated_delivery_result(receipt_path)
                status_line = "DEDUPLICATED"
            else:
                sent = send_telegram_text(
                    brief.text,
                    credential=credential,
                    target=target,
                )
                delivery = record_successful_delivery(
                    receipt_path,
                    brief=brief,
                    target_hash=target_hash,
                    provider_message_id=sent.provider_message_id,
                )
                status_line = "SENT"

        if args.json:
            print(json.dumps(_payload(brief, delivery, policy_result.status), indent=2, sort_keys=True))
        else:
            print(brief.text)
            print()
            print(f"Delivery: {status_line}")
            if delivery is not None and delivery.provider_message_id is not None:
                print(f"Provider message ID: {delivery.provider_message_id}")
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"<option daily operations unavailable: {exc}>")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
