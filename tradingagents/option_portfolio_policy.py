"""Explicit book-level risk policy and daily action queue for option portfolios."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from tradingagents.option_portfolio_dashboard import OptionPortfolioDashboardResult
from tradingagents.option_position_registry import default_registry_path

_ENV_POLICY = "TRADINGAGENTS_OPTION_PORTFOLIO_POLICY"
_POLICY_BASENAME = "options_policy.json"
_POLICY_FIELDS = (
    "max_book_liquidation_value",
    "max_book_gross_theta_dollars_per_day",
    "max_underlying_liquidation_value",
    "max_underlying_abs_delta_shares",
    "max_underlying_gross_delta_shares",
    "max_underlying_gross_theta_dollars_per_day",
)


@dataclass(frozen=True)
class OptionPortfolioRiskPolicy:
    """Explicit limits only; ``None`` means no limit was supplied."""

    max_book_liquidation_value: float | None = None
    max_book_gross_theta_dollars_per_day: float | None = None
    max_underlying_liquidation_value: float | None = None
    max_underlying_abs_delta_shares: float | None = None
    max_underlying_gross_delta_shares: float | None = None
    max_underlying_gross_theta_dollars_per_day: float | None = None

    @property
    def configured(self) -> bool:
        return any(getattr(self, field) is not None for field in _POLICY_FIELDS)


@dataclass(frozen=True)
class OptionPortfolioPolicyCheck:
    """One explicit policy comparison."""

    scope: str
    target: str
    policy_name: str
    label: str
    actual: float | None
    cap: float
    status: str
    reason: str


@dataclass(frozen=True)
class OptionPortfolioPolicyResult:
    """Book/underlying policy evaluation without hidden thresholds."""

    status: str
    policy: OptionPortfolioRiskPolicy
    checks: tuple[OptionPortfolioPolicyCheck, ...]
    breach_count: int
    not_evaluable_count: int
    report: str


@dataclass(frozen=True)
class OptionDailyActionItem:
    """One deterministic queue item requiring attention today."""

    priority: int
    kind: str
    target: str
    position_ids: tuple[str, ...]
    reason: str


def default_portfolio_policy_path(
    registry_db_path: str | os.PathLike[str] | None = None,
) -> Path:
    override = os.getenv(_ENV_POLICY)
    if override:
        return Path(override).expanduser()
    registry_path = (
        Path(registry_db_path).expanduser()
        if registry_db_path is not None
        else default_registry_path()
    )
    return registry_path.parent / _POLICY_BASENAME


def _positive_optional(value: object | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number greater than zero")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return number


def resolve_portfolio_risk_policy(
    *,
    max_book_liquidation_value: object | None = None,
    max_book_gross_theta_dollars_per_day: object | None = None,
    max_underlying_liquidation_value: object | None = None,
    max_underlying_abs_delta_shares: object | None = None,
    max_underlying_gross_delta_shares: object | None = None,
    max_underlying_gross_theta_dollars_per_day: object | None = None,
) -> OptionPortfolioRiskPolicy:
    return OptionPortfolioRiskPolicy(
        max_book_liquidation_value=_positive_optional(
            max_book_liquidation_value, "max_book_liquidation_value"
        ),
        max_book_gross_theta_dollars_per_day=_positive_optional(
            max_book_gross_theta_dollars_per_day,
            "max_book_gross_theta_dollars_per_day",
        ),
        max_underlying_liquidation_value=_positive_optional(
            max_underlying_liquidation_value, "max_underlying_liquidation_value"
        ),
        max_underlying_abs_delta_shares=_positive_optional(
            max_underlying_abs_delta_shares, "max_underlying_abs_delta_shares"
        ),
        max_underlying_gross_delta_shares=_positive_optional(
            max_underlying_gross_delta_shares, "max_underlying_gross_delta_shares"
        ),
        max_underlying_gross_theta_dollars_per_day=_positive_optional(
            max_underlying_gross_theta_dollars_per_day,
            "max_underlying_gross_theta_dollars_per_day",
        ),
    )


def policy_to_dict(policy: OptionPortfolioRiskPolicy) -> dict[str, float]:
    return {
        field: value
        for field in _POLICY_FIELDS
        if (value := getattr(policy, field)) is not None
    }


def save_portfolio_risk_policy(
    policy: OptionPortfolioRiskPolicy,
    path: str | os.PathLike[str],
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(policy_to_dict(policy), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        with temporary.open("r+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_portfolio_risk_policy(
    path: str | os.PathLike[str],
) -> OptionPortfolioRiskPolicy:
    source = Path(path).expanduser()
    if not source.exists():
        return OptionPortfolioRiskPolicy()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid option portfolio policy file: {source}") from exc
    if not isinstance(raw, dict):
        raise ValueError("option portfolio policy must be a JSON object")
    unknown = sorted(set(raw) - set(_POLICY_FIELDS))
    if unknown:
        raise ValueError(f"unknown option portfolio policy field(s): {', '.join(unknown)}")
    return resolve_portfolio_risk_policy(**raw)


def clear_portfolio_risk_policy(path: str | os.PathLike[str]) -> bool:
    target = Path(path).expanduser()
    if not target.exists():
        return False
    target.unlink()
    return True


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.2f}"


def _check(
    *,
    scope: str,
    target: str,
    policy_name: str,
    label: str,
    actual: float | None,
    cap: float,
    forced_unavailable_reason: str | None = None,
) -> OptionPortfolioPolicyCheck:
    if forced_unavailable_reason is not None:
        return OptionPortfolioPolicyCheck(
            scope=scope,
            target=target,
            policy_name=policy_name,
            label=label,
            actual=None,
            cap=cap,
            status="NOT_EVALUABLE",
            reason=forced_unavailable_reason,
        )
    if actual is None:
        return OptionPortfolioPolicyCheck(
            scope=scope,
            target=target,
            policy_name=policy_name,
            label=label,
            actual=None,
            cap=cap,
            status="NOT_EVALUABLE",
            reason=f"{label} is unavailable for the current book snapshot",
        )
    if actual > cap:
        return OptionPortfolioPolicyCheck(
            scope=scope,
            target=target,
            policy_name=policy_name,
            label=label,
            actual=actual,
            cap=cap,
            status="BREACH",
            reason=f"{label} {_fmt(actual)} exceeds cap {_fmt(cap)}",
        )
    return OptionPortfolioPolicyCheck(
        scope=scope,
        target=target,
        policy_name=policy_name,
        label=label,
        actual=actual,
        cap=cap,
        status="OK",
        reason=f"{label} {_fmt(actual)} is within cap {_fmt(cap)}",
    )


def _book_gross_theta(dashboard: OptionPortfolioDashboardResult) -> float | None:
    values = [row.theta_dollars_per_day for row in dashboard.rows]
    if any(value is None for value in values):
        return None
    return sum(abs(value) for value in values if value is not None)


def _render_policy_result(
    status: str,
    policy: OptionPortfolioRiskPolicy,
    checks: tuple[OptionPortfolioPolicyCheck, ...],
    *,
    book_scope_complete: bool,
) -> str:
    lines = [
        "# Option portfolio risk policy",
        "",
        f"Status: **{status}**",
        f"Book scope: {'FULL' if book_scope_complete else 'FILTERED'}",
        "",
    ]
    if not policy.configured:
        lines.extend(
            [
                "No portfolio risk policy is configured. This layer adds no hidden limits.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| Scope | Target | Policy | Actual | Cap | Result |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for item in checks:
            lines.append(
                f"| {item.scope} | {item.target} | {item.label} | {_fmt(item.actual)} | "
                f"{_fmt(item.cap)} | {item.status} |"
            )
        if not checks:
            lines.append("No active positions required a per-underlying comparison.")
        lines.extend(["", "## Reasons", ""])
        for item in checks:
            lines.append(f"- **{item.status} {item.target} / {item.policy_name}**: {item.reason}")
    lines.extend(
        [
            "## Decision boundary",
            "",
            "BREACH is produced only by an explicit saved cap. NOT_EVALUABLE means the required current exposure is unavailable or the dashboard is a filtered partial-book view. OK means the explicit cap was evaluated and not exceeded.",
            "The policy result is deterministic monitoring, not an order, expected-return forecast, probability, or hidden weighted risk score.",
            "Book liquidation value is the current delayed bid-based forward-capital-at-risk proxy for long options; it is not original premium paid, margin, VaR, or a guaranteed executable value.",
        ]
    )
    return "\n".join(lines)


def evaluate_portfolio_risk_policy(
    dashboard: OptionPortfolioDashboardResult,
    policy: OptionPortfolioRiskPolicy,
    *,
    book_scope_complete: bool = True,
) -> OptionPortfolioPolicyResult:
    if not policy.configured:
        return OptionPortfolioPolicyResult(
            status="NOT_CONFIGURED",
            policy=policy,
            checks=(),
            breach_count=0,
            not_evaluable_count=0,
            report=_render_policy_result(
                "NOT_CONFIGURED", policy, (), book_scope_complete=book_scope_complete
            ),
        )

    checks: list[OptionPortfolioPolicyCheck] = []
    filtered_reason = "full-book policy cannot be evaluated from a filtered dashboard"
    if policy.max_book_liquidation_value is not None:
        checks.append(
            _check(
                scope="BOOK",
                target="BOOK",
                policy_name="max_book_liquidation_value",
                label="book delayed liquidation value",
                actual=dashboard.total_liquidation_value,
                cap=policy.max_book_liquidation_value,
                forced_unavailable_reason=None if book_scope_complete else filtered_reason,
            )
        )
    if policy.max_book_gross_theta_dollars_per_day is not None:
        checks.append(
            _check(
                scope="BOOK",
                target="BOOK",
                policy_name="max_book_gross_theta_dollars_per_day",
                label="book gross absolute theta $/day",
                actual=_book_gross_theta(dashboard),
                cap=policy.max_book_gross_theta_dollars_per_day,
                forced_unavailable_reason=None if book_scope_complete else filtered_reason,
            )
        )

    for summary in dashboard.underlyings:
        if policy.max_underlying_liquidation_value is not None:
            checks.append(
                _check(
                    scope="UNDERLYING",
                    target=summary.underlying,
                    policy_name="max_underlying_liquidation_value",
                    label="underlying delayed liquidation value",
                    actual=summary.liquidation_value,
                    cap=policy.max_underlying_liquidation_value,
                )
            )
        if policy.max_underlying_abs_delta_shares is not None:
            checks.append(
                _check(
                    scope="UNDERLYING",
                    target=summary.underlying,
                    policy_name="max_underlying_abs_delta_shares",
                    label="absolute net delta shares",
                    actual=(
                        None
                        if summary.net_delta_shares is None
                        else abs(summary.net_delta_shares)
                    ),
                    cap=policy.max_underlying_abs_delta_shares,
                )
            )
        if policy.max_underlying_gross_delta_shares is not None:
            checks.append(
                _check(
                    scope="UNDERLYING",
                    target=summary.underlying,
                    policy_name="max_underlying_gross_delta_shares",
                    label="gross delta shares",
                    actual=summary.gross_delta_shares,
                    cap=policy.max_underlying_gross_delta_shares,
                )
            )
        if policy.max_underlying_gross_theta_dollars_per_day is not None:
            checks.append(
                _check(
                    scope="UNDERLYING",
                    target=summary.underlying,
                    policy_name="max_underlying_gross_theta_dollars_per_day",
                    label="gross absolute theta $/day",
                    actual=summary.gross_theta_dollars_per_day,
                    cap=policy.max_underlying_gross_theta_dollars_per_day,
                )
            )

    frozen = tuple(checks)
    breach_count = sum(item.status == "BREACH" for item in frozen)
    not_evaluable_count = sum(item.status == "NOT_EVALUABLE" for item in frozen)
    if breach_count:
        status = "BREACH"
    elif not_evaluable_count:
        status = "NOT_EVALUABLE"
    else:
        status = "OK"
    return OptionPortfolioPolicyResult(
        status=status,
        policy=policy,
        checks=frozen,
        breach_count=breach_count,
        not_evaluable_count=not_evaluable_count,
        report=_render_policy_result(
            status, policy, frozen, book_scope_complete=book_scope_complete
        ),
    )


def build_daily_action_queue(
    dashboard: OptionPortfolioDashboardResult,
    policy_result: OptionPortfolioPolicyResult,
) -> tuple[OptionDailyActionItem, ...]:
    """Build a stable queue from explicit position exits/reviews and portfolio policy results."""

    actions: list[OptionDailyActionItem] = []
    for row in dashboard.rows:
        if row.status == "EXIT":
            actions.append(
                OptionDailyActionItem(
                    priority=0,
                    kind="POSITION_EXIT",
                    target=row.position_id,
                    position_ids=(row.position_id,),
                    reason=row.attention_reason,
                )
            )

    rows_by_underlying: dict[str, tuple[str, ...]] = {}
    for summary in dashboard.underlyings:
        rows_by_underlying[summary.underlying] = tuple(
            row.position_id for row in dashboard.rows if row.underlying == summary.underlying
        )
    all_positions = tuple(row.position_id for row in dashboard.rows)

    for check in policy_result.checks:
        if check.status not in {"BREACH", "NOT_EVALUABLE"}:
            continue
        positions = (
            all_positions
            if check.scope == "BOOK"
            else rows_by_underlying.get(check.target, ())
        )
        actions.append(
            OptionDailyActionItem(
                priority=1 if check.status == "BREACH" else 2,
                kind="POLICY_BREACH" if check.status == "BREACH" else "POLICY_NOT_EVALUABLE",
                target=check.target,
                position_ids=positions,
                reason=check.reason,
            )
        )

    for row in dashboard.rows:
        if row.status == "REVIEW":
            actions.append(
                OptionDailyActionItem(
                    priority=3,
                    kind="POSITION_REVIEW",
                    target=row.position_id,
                    position_ids=(row.position_id,),
                    reason=row.attention_reason,
                )
            )

    return tuple(
        sorted(
            actions,
            key=lambda item: (item.priority, item.kind, item.target, item.position_ids),
        )
    )


def render_daily_action_queue(actions: tuple[OptionDailyActionItem, ...]) -> str:
    lines = ["# Option daily action queue", ""]
    if not actions:
        lines.append("No EXIT, policy BREACH/NOT_EVALUABLE, or position REVIEW items today.")
    else:
        lines.extend(
            [
                "| Priority | Kind | Target | Positions | Reason |",
                "|---:|---|---|---|---|",
            ]
        )
        for item in actions:
            lines.append(
                f"| {item.priority} | {item.kind} | {item.target} | "
                f"{', '.join(item.position_ids) or 'N/A'} | {item.reason} |"
            )
    lines.extend(
        [
            "",
            "Priority is categorical and deterministic: position EXIT → explicit policy BREACH → policy NOT_EVALUABLE → position REVIEW. It is not a numeric risk score.",
            "Queue items are review tasks only; no trade, close, hedge, or roll is executed automatically.",
        ]
    )
    return "\n".join(lines)


def policy_result_to_dict(result: OptionPortfolioPolicyResult) -> dict:
    return {
        "status": result.status,
        "policy": policy_to_dict(result.policy),
        "checks": [asdict(item) for item in result.checks],
        "breach_count": result.breach_count,
        "not_evaluable_count": result.not_evaluable_count,
    }
