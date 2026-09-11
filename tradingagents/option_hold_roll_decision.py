"""Deterministic hold-vs-roll comparison for an existing long equity option."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tradingagents.dataflows.equity_options import EquityOptionSnapshot
from tradingagents.option_capital_allocator import CONTRACT_MULTIPLIER
from tradingagents.option_roll_planner import OptionRollPlanResult


@dataclass(frozen=True)
class HoldBaseline:
    """Forward-looking baseline for continuing to hold the existing long option."""

    symbol: str
    dte: int
    strike: float
    lifecycle_breakeven: float
    forward_opportunity_breakeven: float | None
    additional_cash_outflow: float
    capital_at_risk_from_now: float | None
    delta_shares: float | None
    theta_dollars_per_day: float | None


@dataclass(frozen=True)
class HoldRollAlternative:
    """One roll candidate expressed relative to the current HOLD baseline."""

    symbol: str
    dte: int
    dte_extension: int
    lifecycle_net_premium_per_share: float
    lifecycle_breakeven: float
    lifecycle_breakeven_improvement: float
    additional_cash_outflow: float
    new_gross_premium_at_risk: float
    delta_shares: float
    delta_shares_change: float | None
    theta_dollars_per_day: float | None
    theta_relief_dollars_per_day: float | None
    summary: str


@dataclass(frozen=True)
class HoldRollDecisionResult:
    """Transparent comparison only; never an execution instruction."""

    status: str
    reason: str
    hold: HoldBaseline | None
    alternatives: tuple[HoldRollAlternative, ...]
    report: str


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive(value: object, name: str) -> float:
    number = _finite(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return number


def _lifecycle_breakeven(right: str, strike: float, net_premium: float) -> float:
    if right == "C":
        return strike + net_premium
    if right == "P":
        return strike - net_premium
    raise ValueError("snapshot right must be C or P")


def _breakeven_improvement(right: str, hold_be: float, roll_be: float) -> float:
    """Positive means the rolled lifecycle breakeven is directionally easier."""
    if right == "C":
        return hold_be - roll_be
    if right == "P":
        return roll_be - hold_be
    raise ValueError("snapshot right must be C or P")


def _fmt(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"


def _build_summary(
    *,
    dte_extension: int,
    additional_cash: float,
    theta_relief: float | None,
    delta_change: float | None,
    breakeven_improvement: float,
) -> str:
    parts = [f"extends DTE by {dte_extension}d"]
    if theta_relief is not None:
        if theta_relief > 0:
            parts.append(f"reduces daily theta drag by about ${theta_relief:,.2f}")
        elif theta_relief < 0:
            parts.append(f"increases daily theta drag by about ${abs(theta_relief):,.2f}")
        else:
            parts.append("leaves daily theta drag unchanged")
    if delta_change is not None:
        parts.append(f"changes delta exposure by {delta_change:+.2f} shares")
    if additional_cash > 0:
        parts.append(f"requires about ${additional_cash:,.2f} additional cash")
    elif additional_cash < 0:
        parts.append(f"releases about ${abs(additional_cash):,.2f} cash")
    else:
        parts.append("requires no additional net cash")
    if breakeven_improvement > 0:
        parts.append(f"improves lifecycle breakeven by {breakeven_improvement:.2f} underlying points")
    elif breakeven_improvement < 0:
        parts.append(f"worsens lifecycle breakeven by {abs(breakeven_improvement):.2f} underlying points")
    else:
        parts.append("leaves lifecycle breakeven unchanged")
    return "; ".join(parts)


def _render(
    snapshot: EquityOptionSnapshot,
    hold: HoldBaseline,
    alternatives: tuple[HoldRollAlternative, ...],
    *,
    entry_premium: float,
    lifecycle_basis: float,
    contracts: int,
) -> str:
    lines = [
        "# Hold vs roll decision comparison",
        "",
        "Status: **COMPARE**",
        f"Current contract: {snapshot.symbol} | Contracts: {contracts} | Current-leg entry premium: ${entry_premium:.2f} | Lifecycle net premium basis: ${lifecycle_basis:.2f}",
        "This is a deterministic trade-off comparison, not a recommendation or order.",
        "",
        "## Same-scale comparison",
        "",
        "| Alternative | DTE | Lifecycle breakeven | Additional cash now | Capital / gross premium risk | Delta shares | Theta $/day |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| HOLD {hold.symbol} | {hold.dte} | {hold.lifecycle_breakeven:.2f} | $0.00 | "
            f"${_fmt(hold.capital_at_risk_from_now)} current liquidation value at risk | "
            f"{_fmt(hold.delta_shares)} | {_fmt(hold.theta_dollars_per_day)} |"
        ),
    ]
    for item in alternatives:
        lines.append(
            f"| ROLL {item.symbol} | {item.dte} (+{item.dte_extension}) | "
            f"{item.lifecycle_breakeven:.2f} | ${item.additional_cash_outflow:,.2f} | "
            f"${item.new_gross_premium_at_risk:,.2f} new gross premium risk | "
            f"{item.delta_shares:.2f} | {_fmt(item.theta_dollars_per_day)} |"
        )
    lines.extend(
        [
            "",
            "## Trade-offs versus HOLD",
            "",
        ]
    )
    for item in alternatives:
        lines.append(f"- **{item.symbol}**: {item.summary}.")
    lines.extend(
        [
            "",
            "## Breakeven semantics",
            "",
            f"HOLD lifecycle breakeven uses the cumulative lifecycle net premium basis: {hold.lifecycle_breakeven:.2f}.",
            (
                "HOLD forward-opportunity breakeven, using the current delayed bid as the value you give up by not selling now: "
                f"{_fmt(hold.forward_opportunity_breakeven)}."
            ),
            "Each ROLL lifecycle breakeven uses cumulative net premium = current lifecycle basis - old delayed bid + new delayed ask, then applies that premium to the replacement strike.",
            "A lower lifecycle breakeven is directionally easier for a call; a higher lifecycle breakeven is directionally easier for a put.",
            "",
            "## Important distinction",
            "",
            "Additional cash outflow is the roll transaction cash difference. New gross premium-at-risk is the replacement option ask × 100 × contracts and is the amount the new long option can lose from its own purchase price. They are not interchangeable.",
            "The HOLD row uses current delayed liquidation value as forward capital at risk. Current-leg purchase cost is sunk historical cost for the forward decision, while prior realized roll cashflows remain embedded in the lifecycle net premium basis rather than being erased.",
            "",
            "## Decision boundary",
            "",
            "No alternative is automatically labeled better. More DTE and lower theta can require materially more cash, change delta exposure, and worsen lifecycle breakeven. Without an explicit utility/risk policy or expected-return model, converting these trade-offs into an automatic HOLD or ROLL choice would be arbitrary.",
        ]
    )
    return "\n".join(lines)


def compare_hold_vs_roll(
    snapshot: EquityOptionSnapshot,
    roll_plan: OptionRollPlanResult,
    *,
    entry_premium: object,
    contracts: object,
    lifecycle_net_premium_per_share: object | None = None,
) -> HoldRollDecisionResult:
    """Compare HOLD with rolls, preserving cumulative lifecycle cost when supplied."""

    premium = _positive(entry_premium, "entry_premium")
    lifecycle_basis = (
        premium
        if lifecycle_net_premium_per_share is None
        else _finite(lifecycle_net_premium_per_share)
    )
    if lifecycle_basis is None:
        raise ValueError("lifecycle_net_premium_per_share must be a finite number")
    if isinstance(contracts, bool) or not isinstance(contracts, int) or contracts <= 0:
        raise ValueError("contracts must be a positive whole number")

    if roll_plan.status != "COMPARE" or not roll_plan.candidates:
        reason = "hold-vs-roll comparison requires a COMPARE roll plan with candidates"
        return HoldRollDecisionResult("NO_COMPARE", reason, None, (), f"<hold-vs-roll comparison unavailable: {reason}>")

    scale = CONTRACT_MULTIPLIER * contracts
    bid = _finite(snapshot.bid)
    delta = _finite(snapshot.delta)
    theta = _finite(snapshot.theta)
    hold_be = _lifecycle_breakeven(snapshot.right, snapshot.strike, lifecycle_basis)
    forward_be = None if bid is None else _lifecycle_breakeven(snapshot.right, snapshot.strike, bid)
    hold = HoldBaseline(
        symbol=snapshot.symbol,
        dte=snapshot.dte,
        strike=snapshot.strike,
        lifecycle_breakeven=hold_be,
        forward_opportunity_breakeven=forward_be,
        additional_cash_outflow=0.0,
        capital_at_risk_from_now=None if bid is None else bid * scale,
        delta_shares=None if delta is None else delta * scale,
        theta_dollars_per_day=None if theta is None else theta * scale,
    )

    alternatives: list[HoldRollAlternative] = []
    for item in roll_plan.candidates:
        lifecycle_net_premium = (
            lifecycle_basis - item.close_credit_per_share + item.open_debit_per_share
        )
        lifecycle_be = _lifecycle_breakeven(snapshot.right, item.strike, lifecycle_net_premium)
        be_improvement = _breakeven_improvement(snapshot.right, hold_be, lifecycle_be)
        theta_relief = (
            None
            if hold.theta_dollars_per_day is None or item.theta_dollars_per_day_after_roll is None
            else item.theta_dollars_per_day_after_roll - hold.theta_dollars_per_day
        )
        alternative = HoldRollAlternative(
            symbol=item.symbol,
            dte=item.dte,
            dte_extension=item.dte_extension,
            lifecycle_net_premium_per_share=lifecycle_net_premium,
            lifecycle_breakeven=lifecycle_be,
            lifecycle_breakeven_improvement=be_improvement,
            additional_cash_outflow=item.net_cash_outflow,
            new_gross_premium_at_risk=item.new_gross_premium_at_risk,
            delta_shares=item.delta_shares_after_roll,
            delta_shares_change=item.delta_shares_change,
            theta_dollars_per_day=item.theta_dollars_per_day_after_roll,
            theta_relief_dollars_per_day=theta_relief,
            summary=_build_summary(
                dte_extension=item.dte_extension,
                additional_cash=item.net_cash_outflow,
                theta_relief=theta_relief,
                delta_change=item.delta_shares_change,
                breakeven_improvement=be_improvement,
            ),
        )
        alternatives.append(alternative)

    result_alternatives = tuple(alternatives)
    return HoldRollDecisionResult(
        "COMPARE",
        "hold baseline and roll candidates are expressed on consistent lifecycle economics",
        hold,
        result_alternatives,
        _render(
            snapshot,
            hold,
            result_alternatives,
            entry_premium=premium,
            lifecycle_basis=lifecycle_basis,
            contracts=contracts,
        ),
    )
