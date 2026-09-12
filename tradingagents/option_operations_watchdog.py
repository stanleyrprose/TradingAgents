"""Scheduled option-operations run history and independent watchdog helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.us_market_calendar import is_us_equity_market_day

_DEFAULT_HOME = Path.home() / ".tradingagents" / "options-ops"
_DEFAULT_HISTORY = _DEFAULT_HOME / "options_daily_run_history.json"
_DEFAULT_ALERT_RECEIPTS = _DEFAULT_HOME / "options_watchdog_receipts.json"
_MAX_HISTORY = 120
_MAX_ALERT_RECEIPTS = 100


@dataclass(frozen=True)
class OptionDailyRunRecord:
    market_date: str
    started_at: str
    completed_at: str
    status: str
    exit_code: int
    delivery_status: str | None
    action_count: int | None
    error_type: str | None = None


@dataclass(frozen=True)
class OptionWatchdogEvaluation:
    market_date: date
    status: str
    issues: tuple[str, ...]
    run_record: OptionDailyRunRecord | None


def default_daily_run_history_path() -> Path:
    return _DEFAULT_HISTORY


def default_watchdog_receipt_path() -> Path:
    return _DEFAULT_ALERT_RECEIPTS


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _load_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid option daily run-history file: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError(f"invalid option daily run-history file: {path}")
    return payload


def record_daily_run(
    path: str | os.PathLike[str],
    *,
    market_date: date,
    started_at: datetime,
    status: str,
    exit_code: int,
    delivery_status: str | None,
    action_count: int | None,
    error_type: str | None = None,
    completed_at: datetime | None = None,
) -> OptionDailyRunRecord:
    state = str(status).strip().upper()
    if state not in {"SUCCESS", "FAILED"}:
        raise ValueError("scheduled run status must be SUCCESS or FAILED")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise ValueError("scheduled run exit_code must be an integer")
    if action_count is not None and (isinstance(action_count, bool) or not isinstance(action_count, int) or action_count < 0):
        raise ValueError("scheduled run action_count must be a nonnegative whole number or None")
    completed = completed_at or datetime.now(timezone.utc)
    record = OptionDailyRunRecord(
        market_date=market_date.isoformat(),
        started_at=started_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        completed_at=completed.astimezone(timezone.utc).isoformat(timespec="seconds"),
        status=state,
        exit_code=exit_code,
        delivery_status=None if delivery_status is None else str(delivery_status),
        action_count=action_count,
        error_type=None if error_type is None else str(error_type),
    )
    destination = Path(path).expanduser()
    payload = _load_history(destination)
    runs = [item for item in payload["runs"] if isinstance(item, dict)]
    runs.append(asdict(record))
    payload["runs"] = runs[-_MAX_HISTORY:]
    _atomic_json(destination, payload)
    return record


def latest_daily_run_for_date(
    path: str | os.PathLike[str],
    market_date: date,
) -> OptionDailyRunRecord | None:
    payload = _load_history(Path(path).expanduser())
    matches: list[OptionDailyRunRecord] = []
    for item in payload["runs"]:
        if not isinstance(item, dict) or item.get("market_date") != market_date.isoformat():
            continue
        try:
            matches.append(OptionDailyRunRecord(**item))
        except TypeError as exc:
            raise ValueError("invalid scheduled run-history record") from exc
    if not matches:
        return None
    return max(matches, key=lambda record: record.completed_at)


def evaluate_watchdog(
    *,
    market_date: date,
    run_record: OptionDailyRunRecord | None,
    runtime_status: str,
    runtime_issues: tuple[str, ...] = (),
    scheduler_binding_issues: tuple[str, ...] = (),
    launchd_loaded: bool = True,
) -> OptionWatchdogEvaluation:
    if not is_us_equity_market_day(market_date):
        return OptionWatchdogEvaluation(
            market_date=market_date,
            status="MARKET_CLOSED",
            issues=(),
            run_record=run_record,
        )

    issues: list[str] = []
    if run_record is None:
        issues.append("no successful-or-failed scheduled run record exists for the US market date")
    else:
        if run_record.status != "SUCCESS" or run_record.exit_code != 0:
            description = f"scheduled Daily Ops status={run_record.status} exit_code={run_record.exit_code}"
            if run_record.error_type:
                description += f" error={run_record.error_type}"
            issues.append(description)
    if str(runtime_status).upper() != "PASS":
        issues.append("hardened runtime health is not PASS")
    issues.extend(f"runtime: {item}" for item in runtime_issues)
    issues.extend(f"scheduler: {item}" for item in scheduler_binding_issues)
    if not launchd_loaded:
        issues.append("production Daily Ops LaunchAgent is not loaded")

    return OptionWatchdogEvaluation(
        market_date=market_date,
        status="ALERT" if issues else "OK",
        issues=tuple(issues),
        run_record=run_record,
    )


def build_watchdog_alert(evaluation: OptionWatchdogEvaluation) -> str:
    if evaluation.status != "ALERT":
        raise ValueError("watchdog alert text requires ALERT evaluation")
    lines = [
        f"⚠️ TradingAgents Options Ops Watchdog {evaluation.market_date.isoformat()}",
        "Production Daily Ops did not pass its independent health check.",
        "",
    ]
    lines.extend(f"- {issue}" for issue in evaluation.issues)
    lines.extend(
        [
            "",
            "Watchdog only. No trade/close/hedge/roll was executed.",
        ]
    )
    return "\n".join(lines)


def watchdog_fingerprint(evaluation: OptionWatchdogEvaluation) -> str:
    canonical = {
        "market_date": evaluation.market_date.isoformat(),
        "status": evaluation.status,
        "issues": list(evaluation.issues),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_alert_receipts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"alerts": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid option watchdog receipt file: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("alerts"), list):
        raise ValueError(f"invalid option watchdog receipt file: {path}")
    return payload


def successful_watchdog_alert_exists(
    path: str | os.PathLike[str],
    *,
    fingerprint: str,
    target_hash: str,
) -> bool:
    payload = _load_alert_receipts(Path(path).expanduser())
    return any(
        isinstance(item, dict)
        and item.get("fingerprint") == fingerprint
        and item.get("target_hash") == target_hash
        and item.get("status") == "SENT"
        for item in payload["alerts"]
    )


def record_watchdog_alert(
    path: str | os.PathLike[str],
    *,
    fingerprint: str,
    target_hash: str,
    provider_message_id: int | None,
) -> None:
    destination = Path(path).expanduser()
    payload = _load_alert_receipts(destination)
    alerts = [item for item in payload["alerts"] if isinstance(item, dict)]
    alerts.append(
        {
            "fingerprint": fingerprint,
            "target_hash": target_hash,
            "status": "SENT",
            "provider_message_id": provider_message_id,
            "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    payload["alerts"] = alerts[-_MAX_ALERT_RECEIPTS:]
    _atomic_json(destination, payload)
