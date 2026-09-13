"""Deterministic daily dashboard for registered long equity-option positions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from tradingagents.dataflows.equity_options import EquityOptionSnapshotResult
from tradingagents.option_capital_allocator import CONTRACT_MULTIPLIER
from tradingagents.option_position_manager import evaluate_long_option_position
from tradingagents.option_position_registry import OptionPositionRecord

_STATUS_PRIORITY = {"EXIT": 0, "REVIEW": 1, "HOLD": 2, "REPORT_ONLY": 3}
_EXIT_POLICY_FIELDS = (
    "take_profit_pct",
    "stop_loss_pct",
    "exit_at_dte",
    "max_theta_burn_pct_per_day",
)


@dataclass(frozen=True)
class OptionPortfolioRow:
    """One open registered position refreshed against a current delayed snapshot."""

    position_id: str
    underlying: str
    symbol: str
    contracts: int
    status: str
    attention_reason: str
    dte: int | None
    underlying_spot: float | None
    bid: float | None
    ask: float | None
    current_liquidation_value: float | None
    current_leg_entry_premium: float
    lifecycle_net_premium_per_share: float
    current_leg_pnl_dollars: float | None
    current_leg_pnl_pct: float | None
    lifecycle_pnl_dollars: float | None
    delta_shares: float | None
    gamma_delta_shares_per_dollar: float | None
    vega_dollars_per_vol_point: float | None
    theta_dollars_per_day: float | None
    roll_count: int
    latest_thesis_status: str | None
    source_timestamp: str | None
    unavailable_reason: str | None


@dataclass(frozen=True)
class UnderlyingExposureSummary:
    """Strict per-underlying aggregate; fields become unavailable if any leg is unavailable."""

    underlying: str
    position_count: int
    liquidation_value: float | None
    lifecycle_pnl_dollars: float | None
    net_delta_shares: float | None
    gross_delta_shares: float | None
    net_gamma_delta_shares_per_dollar: float | None
    gross_gamma_delta_shares_per_dollar: float | None
    net_vega_dollars_per_vol_point: float | None
    gross_vega_dollars_per_vol_point: float | None
    net_theta_dollars_per_day: float | None
    gross_theta_dollars_per_day: float | None


@dataclass(frozen=True)
class OptionPortfolioDashboardResult:
    """Daily deterministic book view across all open registered option positions."""

    as_of: date
    rows: tuple[OptionPortfolioRow, ...]
    underlyings: tuple[UnderlyingExposureSummary, ...]
    total_liquidation_value: float | None
    total_current_leg_pnl_dollars: float | None
    total_lifecycle_pnl_dollars: float | None
    exit_count: int
    review_count: int
    hold_count: int
    report_only_count: int
    report: str


def _fmt(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:,.{decimals}f}"


def _strict_sum(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _strict_net_gross(values: list[float | None]) -> tuple[float | None, float | None]:
    if any(value is None for value in values):
        return None, None
    concrete = [value for value in values if value is not None]
    return sum(concrete), sum(abs(value) for value in concrete)


def _policy_kwargs(position: OptionPositionRecord) -> dict[str, object | None]:
    return {name: position.exit_policy.get(name) for name in _EXIT_POLICY_FIELDS}


def _latest_thesis_status(position: OptionPositionRecord) -> str | None:
    status = position.latest_thesis.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip().upper()
    direction = position.latest_thesis.get("direction")
    if isinstance(direction, str) and direction.strip():
        return direction.strip().lower()
    return None


def _attention_reason(status: str, management, unavailable_reason: str | None) -> str:
    if unavailable_reason:
        return f"market data unavailable: {unavailable_reason}"
    if status == "EXIT":
        reasons = [trigger.reason for trigger in management.triggers if trigger.triggered]
        return "; ".join(reasons) or "explicit exit policy triggered"
    if status == "REVIEW":
        reasons = [trigger.reason for trigger in management.triggers if not trigger.evaluable]
        return "; ".join(reasons) or "explicit policy cannot be fully evaluated"
    if status == "HOLD":
        return "explicit exit policy supplied; no condition is currently triggered"
    return "no exit policy supplied; informational monitoring only"


def _build_row(
    position: OptionPositionRecord,
    snapshot_result: EquityOptionSnapshotResult | None,
) -> OptionPortfolioRow:
    if snapshot_result is None or not snapshot_result.available or snapshot_result.snapshot is None:
        reason = (
            "snapshot missing from batch result"
            if snapshot_result is None
            else snapshot_result.unavailable_reason or "snapshot unavailable"
        )
        return OptionPortfolioRow(
            position_id=position.position_id,
            underlying=position.underlying,
            symbol=position.current_symbol,
            contracts=position.contracts,
            status="REVIEW",
            attention_reason=f"market data unavailable: {reason}",
            dte=None,
            underlying_spot=None,
            bid=None,
            ask=None,
            current_liquidation_value=None,
            current_leg_entry_premium=position.current_leg_entry_premium,
            lifecycle_net_premium_per_share=position.lifecycle_net_premium_per_share,
            current_leg_pnl_dollars=None,
            current_leg_pnl_pct=None,
            lifecycle_pnl_dollars=None,
            delta_shares=None,
            gamma_delta_shares_per_dollar=None,
            vega_dollars_per_vol_point=None,
            theta_dollars_per_day=None,
            roll_count=position.roll_count,
            latest_thesis_status=_latest_thesis_status(position),
            source_timestamp=None,
            unavailable_reason=reason,
        )

    snapshot = snapshot_result.snapshot
    management = evaluate_long_option_position(
        snapshot,
        entry_premium=position.current_leg_entry_premium,
        contracts=position.contracts,
        **_policy_kwargs(position),
    )
    scale = CONTRACT_MULTIPLIER * position.contracts
    lifecycle_pnl = (
        None
        if snapshot.bid is None
        else (snapshot.bid - position.lifecycle_net_premium_per_share) * scale
    )
    return OptionPortfolioRow(
        position_id=position.position_id,
        underlying=position.underlying,
        symbol=position.current_symbol,
        contracts=position.contracts,
        status=management.status,
        attention_reason=_attention_reason(management.status, management, None),
        dte=snapshot.dte,
        underlying_spot=snapshot.underlying_spot,
        bid=snapshot.bid,
        ask=snapshot.ask,
        current_liquidation_value=management.current_liquidation_value,
        current_leg_entry_premium=position.current_leg_entry_premium,
        lifecycle_net_premium_per_share=position.lifecycle_net_premium_per_share,
        current_leg_pnl_dollars=management.pnl_dollars,
        current_leg_pnl_pct=management.pnl_pct,
        lifecycle_pnl_dollars=lifecycle_pnl,
        delta_shares=management.delta_shares,
        gamma_delta_shares_per_dollar=management.gamma_delta_shares_per_dollar,
        vega_dollars_per_vol_point=management.vega_dollars_per_vol_point,
        theta_dollars_per_day=management.theta_dollars_per_day,
        roll_count=position.roll_count,
        latest_thesis_status=_latest_thesis_status(position),
        source_timestamp=snapshot.source_timestamp,
        unavailable_reason=None,
    )


def _underlying_summary(
    underlying: str,
    rows: tuple[OptionPortfolioRow, ...],
) -> UnderlyingExposureSummary:
    liquidation = _strict_sum([row.current_liquidation_value for row in rows])
    lifecycle_pnl = _strict_sum([row.lifecycle_pnl_dollars for row in rows])
    net_delta, gross_delta = _strict_net_gross([row.delta_shares for row in rows])
    net_gamma, gross_gamma = _strict_net_gross(
        [row.gamma_delta_shares_per_dollar for row in rows]
    )
    net_vega, gross_vega = _strict_net_gross([row.vega_dollars_per_vol_point for row in rows])
    net_theta, gross_theta = _strict_net_gross([row.theta_dollars_per_day for row in rows])
    return UnderlyingExposureSummary(
        underlying=underlying,
        position_count=len(rows),
        liquidation_value=liquidation,
        lifecycle_pnl_dollars=lifecycle_pnl,
        net_delta_shares=net_delta,
        gross_delta_shares=gross_delta,
        net_gamma_delta_shares_per_dollar=net_gamma,
        gross_gamma_delta_shares_per_dollar=gross_gamma,
        net_vega_dollars_per_vol_point=net_vega,
        gross_vega_dollars_per_vol_point=gross_vega,
        net_theta_dollars_per_day=net_theta,
        gross_theta_dollars_per_day=gross_theta,
    )


def _render(
    as_of: date,
    rows: tuple[OptionPortfolioRow, ...],
    underlyings: tuple[UnderlyingExposureSummary, ...],
    total_liquidation: float | None,
    total_current_leg_pnl: float | None,
    total_lifecycle_pnl: float | None,
) -> str:
    counts = {status: sum(row.status == status for row in rows) for status in _STATUS_PRIORITY}
    lines = [
        "# Option portfolio daily dashboard",
        "",
        f"As of: {as_of.isoformat()} | Open positions: {len(rows)}",
        (
            "Attention: "
            f"EXIT {counts['EXIT']} | REVIEW {counts['REVIEW']} | "
            f"HOLD {counts['HOLD']} | REPORT_ONLY {counts['REPORT_ONLY']}"
        ),
        f"Book delayed liquidation value: ${_fmt(total_liquidation)}",
        f"Book current-leg P/L proxy: ${_fmt(total_current_leg_pnl)}",
        f"Book lifecycle P/L proxy: ${_fmt(total_lifecycle_pnl)}",
        "",
        "## Position triage",
        "",
        "| Status | Underlying | Contract | Qty | DTE | Bid | Current value | Current-leg P/L | Lifecycle P/L | Delta sh | Theta $/day | Thesis |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.status} | {row.underlying} | {row.symbol} | {row.contracts} | "
            f"{_fmt(row.dte, 0)} | {_fmt(row.bid)} | {_fmt(row.current_liquidation_value)} | "
            f"{_fmt(row.current_leg_pnl_dollars)} | {_fmt(row.lifecycle_pnl_dollars)} | "
            f"{_fmt(row.delta_shares)} | {_fmt(row.theta_dollars_per_day)} | "
            f"{row.latest_thesis_status or 'N/A'} |"
        )
    lines.extend(["", "## Attention reasons", ""])
    for row in rows:
        lines.append(f"- **{row.status} {row.position_id} / {row.symbol}**: {row.attention_reason}")

    lines.extend(
        [
            "",
            "## Exposure by underlying",
            "",
            "Cross-underlying share deltas are not netted together. Net/gross Greeks below are calculated only within the same underlying.",
            "",
            "| Underlying | Positions | Liquidation value | Lifecycle P/L | Net delta sh | Gross delta sh | Net gamma | Gross gamma | Net vega | Net theta $/day |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in underlyings:
        lines.append(
            f"| {item.underlying} | {item.position_count} | {_fmt(item.liquidation_value)} | "
            f"{_fmt(item.lifecycle_pnl_dollars)} | {_fmt(item.net_delta_shares)} | "
            f"{_fmt(item.gross_delta_shares)} | {_fmt(item.net_gamma_delta_shares_per_dollar)} | "
            f"{_fmt(item.gross_gamma_delta_shares_per_dollar)} | "
            f"{_fmt(item.net_vega_dollars_per_vol_point)} | {_fmt(item.net_theta_dollars_per_day)} |"
        )

    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            "EXIT appears only when an explicit stored exit-policy condition is triggered. REVIEW means current market data or an explicit policy check cannot be evaluated. HOLD means explicit exit policy exists and none is triggered. REPORT_ONLY means no exit policy was stored.",
            "The ordering is deterministic triage (EXIT → REVIEW → HOLD → REPORT_ONLY, then shorter DTE first); it is not a probability, expected-return ranking, or hidden weighted score.",
            "Stored Greek limits and stored thesis are shown for context but are not silently converted into exit decisions by this dashboard.",
            "Cboe data are delayed; bid-based liquidation and P/L are conservative proxies, not executable fills. Commissions, taxes, slippage, and market impact are excluded.",
        ]
    )
    return "\n".join(lines)


def build_option_portfolio_dashboard(
    positions: tuple[OptionPositionRecord, ...] | list[OptionPositionRecord],
    snapshots: Mapping[str, EquityOptionSnapshotResult],
    *,
    as_of: date,
) -> OptionPortfolioDashboardResult:
    """Build a book dashboard from open registry positions and frozen snapshot results."""

    supplied = tuple(positions)
    if any(position.status != "OPEN" for position in supplied):
        raise ValueError("portfolio dashboard accepts OPEN positions only")
    rows = tuple(
        sorted(
            (_build_row(position, snapshots.get(position.current_symbol)) for position in supplied),
            key=lambda row: (
                _STATUS_PRIORITY.get(row.status, 99),
                row.dte if row.dte is not None else 10**9,
                row.underlying,
                row.symbol,
                row.position_id,
            ),
        )
    )

    grouped: dict[str, list[OptionPortfolioRow]] = {}
    for row in rows:
        grouped.setdefault(row.underlying, []).append(row)
    underlyings = tuple(
        _underlying_summary(underlying, tuple(grouped[underlying]))
        for underlying in sorted(grouped)
    )

    total_liquidation = _strict_sum([row.current_liquidation_value for row in rows])
    total_current_leg_pnl = _strict_sum([row.current_leg_pnl_dollars for row in rows])
    total_lifecycle_pnl = _strict_sum([row.lifecycle_pnl_dollars for row in rows])
    report = _render(
        as_of,
        rows,
        underlyings,
        total_liquidation,
        total_current_leg_pnl,
        total_lifecycle_pnl,
    )
    return OptionPortfolioDashboardResult(
        as_of=as_of,
        rows=rows,
        underlyings=underlyings,
        total_liquidation_value=total_liquidation,
        total_current_leg_pnl_dollars=total_current_leg_pnl,
        total_lifecycle_pnl_dollars=total_lifecycle_pnl,
        exit_count=sum(row.status == "EXIT" for row in rows),
        review_count=sum(row.status == "REVIEW" for row in rows),
        hold_count=sum(row.status == "HOLD" for row in rows),
        report_only_count=sum(row.status == "REPORT_ONLY" for row in rows),
        report=report,
    )
