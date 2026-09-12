#!/usr/bin/env python3
"""Independently verify scheduled Options Daily Ops and optionally alert via Telegram."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from datetime import date
from pathlib import Path

from tradingagents.dataflows.equity_options import current_us_option_market_date
from tradingagents.option_daily_ops import delivery_target_hash
from tradingagents.option_daily_scheduler import (
    default_scheduler_credential_path,
    default_scheduler_label,
    load_telegram_credentials,
)
from tradingagents.option_operations_watchdog import (
    build_watchdog_alert,
    default_daily_run_history_path,
    default_watchdog_receipt_path,
    evaluate_watchdog,
    latest_daily_run_for_date,
    record_watchdog_alert,
    successful_watchdog_alert_exists,
    watchdog_fingerprint,
)
from tradingagents.option_runtime import (
    default_runtime_root,
    health_runtime,
    scheduler_binding_issues,
)
from tradingagents.telegram_delivery import send_telegram_text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=current_us_option_market_date().isoformat(),
        help="YYYY-MM-DD US option-market date; defaults to current America/New_York date",
    )
    parser.add_argument("--history", default=str(default_daily_run_history_path()))
    parser.add_argument("--receipt", default=str(default_watchdog_receipt_path()))
    parser.add_argument("--runtime-root", default=str(default_runtime_root()))
    parser.add_argument("--daily-label", default=default_scheduler_label())
    parser.add_argument(
        "--daily-launch-agent",
        default=str(Path.home() / "Library" / "LaunchAgents" / f"{default_scheduler_label()}.plist"),
    )
    parser.add_argument("--telegram-config", default=str(default_scheduler_credential_path()))
    parser.add_argument("--send", action="store_true", help="send alert only when watchdog status is ALERT")
    parser.add_argument("--force", action="store_true", help="with --send, bypass alert deduplication")
    parser.add_argument("--json", action="store_true")
    return parser


def _launchd_loaded(label: str) -> bool:
    result = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _payload(evaluation, delivery_status: str, message: str | None) -> dict:
    return {
        "market_date": evaluation.market_date.isoformat(),
        "status": evaluation.status,
        "issues": list(evaluation.issues),
        "run_record": None if evaluation.run_record is None else asdict(evaluation.run_record),
        "delivery": delivery_status,
        "message": message,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.force and not args.send:
            raise ValueError("--force requires --send")
        try:
            market_date = date.fromisoformat(args.date)
        except ValueError as exc:
            raise ValueError("date must be YYYY-MM-DD") from exc

        run_record = latest_daily_run_for_date(args.history, market_date)
        runtime = health_runtime(args.runtime_root)
        binding = scheduler_binding_issues(
            runtime_root=args.runtime_root,
            launch_agent_path=args.daily_launch_agent,
        )
        evaluation = evaluate_watchdog(
            market_date=market_date,
            run_record=run_record,
            runtime_status=runtime.status,
            runtime_issues=runtime.issues,
            scheduler_binding_issues=binding,
            launchd_loaded=_launchd_loaded(args.daily_label),
        )

        message = None
        if evaluation.status == "MARKET_CLOSED":
            delivery_status = "MARKET_CLOSED"
        elif evaluation.status == "OK":
            delivery_status = "NO_ALERT"
        else:
            message = build_watchdog_alert(evaluation)
            if not args.send:
                delivery_status = "PREVIEW_ONLY"
            else:
                credentials = load_telegram_credentials(args.telegram_config)
                target_hash = delivery_target_hash(credentials.target)
                fingerprint = watchdog_fingerprint(evaluation)
                if not args.force and successful_watchdog_alert_exists(
                    args.receipt,
                    fingerprint=fingerprint,
                    target_hash=target_hash,
                ):
                    delivery_status = "DEDUPLICATED"
                else:
                    sent = send_telegram_text(
                        message,
                        credential=credentials.credential,
                        target=credentials.target,
                    )
                    record_watchdog_alert(
                        args.receipt,
                        fingerprint=fingerprint,
                        target_hash=target_hash,
                        provider_message_id=sent.provider_message_id,
                    )
                    delivery_status = "SENT"

        if args.json:
            print(json.dumps(_payload(evaluation, delivery_status, message), indent=2, sort_keys=True))
        else:
            print(f"Options Ops Watchdog {market_date.isoformat()}: {evaluation.status}")
            for issue in evaluation.issues:
                print(f"- {issue}")
            print(f"Delivery: {delivery_status}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"<option operations watchdog unavailable: {exc}>")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
