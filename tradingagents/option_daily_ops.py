"""Daily option-book operations brief and delivery receipt helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.option_portfolio_dashboard import OptionPortfolioDashboardResult
from tradingagents.option_portfolio_policy import (
    OptionDailyActionItem,
    OptionPortfolioPolicyResult,
)
from tradingagents.option_position_registry import default_registry_path

_RECEIPT_BASENAME = "options_daily_ops_receipts.json"
_MAX_MESSAGE_TEXT = 4000
_MAX_RECEIPTS = 100


@dataclass(frozen=True)
class OptionDailyBrief:
    """Concise daily operations message derived from deterministic action items."""

    as_of: date
    send_required: bool
    text: str
    fingerprint: str
    action_count: int
    exit_count: int
    policy_breach_count: int
    policy_not_evaluable_count: int
    review_count: int


@dataclass(frozen=True)
class DeliveryReceiptResult:
    """Structured exact-message receipt lookup/result."""

    status: str
    deduplicated: bool
    receipt_path: str
    provider_message_id: int | None = None
    reason: str | None = None


def default_daily_ops_receipt_path(
    registry_db_path: str | os.PathLike[str] | None = None,
) -> Path:
    registry_path = (
        Path(registry_db_path).expanduser()
        if registry_db_path is not None
        else default_registry_path()
    )
    return registry_path.parent / _RECEIPT_BASENAME


def _money(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}${value:,.0f}"


def _compact_reason(reason: str, limit: int = 140) -> str:
    text = " ".join(reason.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _brief_counts(actions: tuple[OptionDailyActionItem, ...]) -> dict[str, int]:
    return {
        "POSITION_EXIT": sum(item.kind == "POSITION_EXIT" for item in actions),
        "POLICY_BREACH": sum(item.kind == "POLICY_BREACH" for item in actions),
        "POLICY_NOT_EVALUABLE": sum(
            item.kind == "POLICY_NOT_EVALUABLE" for item in actions
        ),
        "POSITION_REVIEW": sum(item.kind == "POSITION_REVIEW" for item in actions),
    }


def brief_fingerprint(
    as_of: date,
    actions: tuple[OptionDailyActionItem, ...],
) -> str:
    canonical = {
        "as_of": as_of.isoformat(),
        "actions": [asdict(item) for item in actions],
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_option_daily_brief(
    dashboard: OptionPortfolioDashboardResult,
    policy_result: OptionPortfolioPolicyResult,
    actions: tuple[OptionDailyActionItem, ...],
    *,
    max_actions: int = 8,
) -> OptionDailyBrief:
    """Render a concise notification-safe brief from deterministic daily action items."""

    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or not 1 <= max_actions <= 20:
        raise ValueError("max_actions must be a whole number from 1 to 20")

    counts = _brief_counts(actions)
    fingerprint = brief_fingerprint(dashboard.as_of, actions)
    if not actions:
        return OptionDailyBrief(
            as_of=dashboard.as_of,
            send_required=False,
            text=(
                f"Options Daily {dashboard.as_of.isoformat()}\n"
                "No EXIT, portfolio-policy breach/unavailable check, or position REVIEW items."
            ),
            fingerprint=fingerprint,
            action_count=0,
            exit_count=0,
            policy_breach_count=0,
            policy_not_evaluable_count=0,
            review_count=0,
        )

    lines = [
        f"⚠️ Options Daily {dashboard.as_of.isoformat()}",
        (
            f"Actions {len(actions)} | EXIT {counts['POSITION_EXIT']} | "
            f"BREACH {counts['POLICY_BREACH']} | "
            f"CHECK {counts['POLICY_NOT_EVALUABLE'] + counts['POSITION_REVIEW']}"
        ),
        (
            f"Book value {_money(dashboard.total_liquidation_value)} | "
            f"Lifecycle P/L {_money(dashboard.total_lifecycle_pnl_dollars)} | "
            f"Policy {policy_result.status}"
        ),
        "",
    ]
    for index, item in enumerate(actions[:max_actions], start=1):
        positions = ",".join(item.position_ids)
        suffix = f" [{positions}]" if positions and positions != item.target else ""
        lines.append(
            f"{index}. {item.kind} {item.target}{suffix} — {_compact_reason(item.reason)}"
        )
    hidden = len(actions) - min(len(actions), max_actions)
    if hidden > 0:
        lines.append(f"… +{hidden} more action item(s) in the full dashboard")
    lines.extend(
        [
            "",
            "Delayed-data monitoring only. No trade/close/hedge/roll was executed.",
        ]
    )
    text = "\n".join(lines)
    if len(text) > _MAX_MESSAGE_TEXT:
        text = text[: _MAX_MESSAGE_TEXT - 1].rstrip() + "…"
    return OptionDailyBrief(
        as_of=dashboard.as_of,
        send_required=True,
        text=text,
        fingerprint=fingerprint,
        action_count=len(actions),
        exit_count=counts["POSITION_EXIT"],
        policy_breach_count=counts["POLICY_BREACH"],
        policy_not_evaluable_count=counts["POLICY_NOT_EVALUABLE"],
        review_count=counts["POSITION_REVIEW"],
    )


def delivery_target_hash(target: str) -> str:
    text = str(target).strip()
    if not text:
        raise ValueError("delivery target must be nonempty")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_receipts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"deliveries": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid option daily-ops receipt file: {path}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("deliveries"), list):
        raise ValueError(f"invalid option daily-ops receipt file: {path}")
    return raw


def _write_receipts(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with temporary.open("r+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def successful_receipt_exists(
    path: str | os.PathLike[str],
    *,
    fingerprint: str,
    target_hash: str,
) -> bool:
    payload = _load_receipts(Path(path).expanduser())
    return any(
        isinstance(item, dict)
        and item.get("fingerprint") == fingerprint
        and item.get("target_hash") == target_hash
        and item.get("status") == "SENT"
        for item in payload["deliveries"]
    )


def record_successful_delivery(
    path: str | os.PathLike[str],
    *,
    brief: OptionDailyBrief,
    target_hash: str,
    provider_message_id: int | None,
) -> DeliveryReceiptResult:
    destination = Path(path).expanduser()
    payload = _load_receipts(destination)
    deliveries = [item for item in payload["deliveries"] if isinstance(item, dict)]
    deliveries.append(
        {
            "fingerprint": brief.fingerprint,
            "as_of": brief.as_of.isoformat(),
            "target_hash": target_hash,
            "status": "SENT",
            "provider_message_id": provider_message_id,
            "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    payload["deliveries"] = deliveries[-_MAX_RECEIPTS:]
    _write_receipts(destination, payload)
    return DeliveryReceiptResult(
        status="SENT",
        deduplicated=False,
        receipt_path=str(destination),
        provider_message_id=provider_message_id,
    )


def deduplicated_delivery_result(
    path: str | os.PathLike[str],
) -> DeliveryReceiptResult:
    return DeliveryReceiptResult(
        status="DEDUPLICATED",
        deduplicated=True,
        receipt_path=str(Path(path).expanduser()),
        reason="matching successful delivery receipt already exists",
    )
