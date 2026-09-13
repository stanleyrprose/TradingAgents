"""Deterministic lifecycle monitoring for an existing long equity-option position."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from tradingagents.dataflows.equity_options import EquityOptionSnapshot
from tradingagents.option_capital_allocator import CONTRACT_MULTIPLIER


@dataclass(frozen=True)
class OptionExitTrigger:
    """One explicit exit-policy check."""

    policy_name: str
    label: str
    actual: float | int | None
    threshold: float | int
    evaluable: bool
    triggered: bool
    reason: str


@dataclass(frozen=True)
class OptionPositionManagementResult:
    """Current long-option lifecycle metrics and explicit-policy decision."""

    status: str
    entry_premium: float
    contracts: int
    entry_cost: float
    liquidation_price_proxy: float | None
    current_liquidation_value: float | None
    pnl_dollars: float | None
    pnl_pct: float | None
    theta_burn_pct_of_midpoint_per_day: float | None
    delta_shares: float | None
    delta_notional_proxy: float | None
    gamma_delta_shares_per_dollar: float | None
    vega_dollars_per_vol_point: float | None
    theta_dollars_per_day: float | None
    triggers: tuple[OptionExitTrigger, ...]
    report: str

    @property
    def should_exit(self) -> bool | None:
        if self.status == "EXIT":
            return True
        if self.status == "HOLD":
            return False
        return None


_POLICY_FIELDS = (
    ("take_profit_pct", "take profit", "pnl_pct"),
    ("stop_loss_pct", "stop loss", "pnl_pct"),
    ("exit_at_dte", "time-to-expiry exit", "dte"),
    (
        "max_theta_burn_pct_per_day",
        "maximum theta burn as % of midpoint/day",
        "theta_burn_pct_of_midpoint_per_day",
    ),
)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number greater than zero")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return number


def resolve_long_option_position_inputs(
    *, entry_premium: object, contracts: object
) -> tuple[float, int]:
    """Validate entry basis and whole-contract count before any network request."""

    premium = _positive_number(entry_premium, "entry_premium")
    if isinstance(contracts, bool) or not isinstance(contracts, int) or contracts <= 0:
        raise ValueError("contracts must be a positive whole number")
    return premium, contracts


def resolve_option_exit_policy(
    *,
    take_profit_pct: object | None = None,
    stop_loss_pct: object | None = None,
    exit_at_dte: object | None = None,
    max_theta_burn_pct_per_day: object | None = None,
) -> dict[str, float | int | None]:
    """Validate optional explicit exit policy without inventing defaults."""

    take_profit = (
        None
        if take_profit_pct is None
        else _positive_number(take_profit_pct, "take_profit_pct")
    )
    stop_loss = (
        None
        if stop_loss_pct is None
        else _positive_number(stop_loss_pct, "stop_loss_pct")
    )
    if stop_loss is not None and stop_loss > 100:
        raise ValueError("stop_loss_pct must be less than or equal to 100 for a long option")

    if exit_at_dte is None:
        dte = None
    else:
        if isinstance(exit_at_dte, bool) or not isinstance(exit_at_dte, int):
            raise ValueError("exit_at_dte must be a nonnegative whole number")
        if not 0 <= exit_at_dte <= 3650:
            raise ValueError("exit_at_dte must be between 0 and 3650")
        dte = exit_at_dte

    theta_cap = (
        None
        if max_theta_burn_pct_per_day is None
        else _positive_number(
            max_theta_burn_pct_per_day, "max_theta_burn_pct_per_day"
        )
    )
    return {
        "take_profit_pct": take_profit,
        "stop_loss_pct": stop_loss,
        "exit_at_dte": dte,
        "max_theta_burn_pct_per_day": theta_cap,
    }


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:,.{decimals}f}"


def _policy_trigger(
    *,
    name: str,
    label: str,
    actual: float | int | None,
    threshold: float | int,
) -> OptionExitTrigger:
    if actual is None:
        return OptionExitTrigger(
            policy_name=name,
            label=label,
            actual=None,
            threshold=threshold,
            evaluable=False,
            triggered=False,
            reason=f"{label} is not evaluable from the current snapshot",
        )

    if name == "take_profit_pct":
        actual_number = float(actual)
        threshold_number = float(threshold)
        triggered = actual_number >= threshold_number or math.isclose(
            actual_number, threshold_number, rel_tol=1e-12, abs_tol=1e-12
        )
        comparison = ">="
    elif name == "stop_loss_pct":
        actual_number = float(actual)
        stop_threshold = -float(threshold)
        triggered = actual_number <= stop_threshold or math.isclose(
            actual_number, stop_threshold, rel_tol=1e-12, abs_tol=1e-12
        )
        comparison = f"<= -{_fmt(threshold)}"
        return OptionExitTrigger(
            policy_name=name,
            label=label,
            actual=actual,
            threshold=threshold,
            evaluable=True,
            triggered=triggered,
            reason=(
                f"P/L {_fmt(actual)}% {comparison}% stop-loss threshold"
                if triggered
                else f"P/L {_fmt(actual)}% remains above -{_fmt(threshold)}% stop-loss threshold"
            ),
        )
    elif name == "exit_at_dte":
        triggered = int(actual) <= int(threshold)
        comparison = "<="
    elif name == "max_theta_burn_pct_per_day":
        actual_number = float(actual)
        threshold_number = float(threshold)
        triggered = actual_number >= threshold_number or math.isclose(
            actual_number, threshold_number, rel_tol=1e-12, abs_tol=1e-12
        )
        comparison = ">="
    else:  # pragma: no cover - internal invariant
        raise ValueError(f"unsupported policy: {name}")

    unit = "%" if name != "exit_at_dte" else " DTE"
    threshold_unit = "%" if name != "exit_at_dte" else " DTE"
    return OptionExitTrigger(
        policy_name=name,
        label=label,
        actual=actual,
        threshold=threshold,
        evaluable=True,
        triggered=triggered,
        reason=(
            f"{label} {_fmt(actual)}{unit} {comparison} {_fmt(threshold)}{threshold_unit} threshold"
            if triggered
            else f"{label} {_fmt(actual)}{unit} has not reached {_fmt(threshold)}{threshold_unit} threshold"
        ),
    )


def _render_report(
    snapshot: EquityOptionSnapshot,
    result: OptionPositionManagementResult,
    policy: dict[str, float | int | None],
) -> str:
    lines = [
        "# Long option position management",
        "",
        f"Status: **{result.status}**",
        f"Contract: {snapshot.symbol} | Underlying: {snapshot.underlying} | "
        f"Type: {'call' if snapshot.right == 'C' else 'put'} | Strike: {_fmt(snapshot.strike)}",
        f"As of: {snapshot.as_of.isoformat()} | DTE: {snapshot.dte} | "
        f"Source timestamp: {snapshot.source_timestamp or 'DATA_UNAVAILABLE'}",
        "",
        "## Current snapshot",
        "",
        f"Underlying spot: ${_fmt(snapshot.underlying_spot)}",
        f"Bid / Ask / Midpoint: ${_fmt(snapshot.bid)} / ${_fmt(snapshot.ask)} / ${_fmt(snapshot.midpoint)}",
        f"Bid-ask spread: {_fmt(snapshot.spread_pct)}%",
        f"IV: {_fmt(None if snapshot.iv is None else snapshot.iv * 100)}%",
        f"Delta / Gamma / Vega / Theta: {_fmt(snapshot.delta, 4)} / {_fmt(snapshot.gamma, 4)} / "
        f"{_fmt(snapshot.vega, 4)} / {_fmt(snapshot.theta, 4)}",
        "",
        "## Position economics",
        "",
        f"Entry premium: ${_fmt(result.entry_premium)} per share | Contracts: {result.contracts}",
        f"Entry premium at risk: ${_fmt(result.entry_cost)}",
        f"Conservative liquidation proxy: ${_fmt(result.liquidation_price_proxy)} per share (current delayed bid)",
        f"Current liquidation value proxy: ${_fmt(result.current_liquidation_value)}",
        f"P/L proxy: ${_fmt(result.pnl_dollars)} | Return on entry premium: {_fmt(result.pnl_pct)}%",
        f"Theta burn proxy: {_fmt(result.theta_burn_pct_of_midpoint_per_day)}% of current midpoint/day",
        "",
        "## Current Greek exposure",
        "",
        f"Delta: {_fmt(result.delta_shares)} shares | Delta notional proxy: ${_fmt(result.delta_notional_proxy)}",
        f"Gamma: {_fmt(result.gamma_delta_shares_per_dollar)} delta-shares per $1 underlying move",
        f"Vega: {_fmt(result.vega_dollars_per_vol_point)} $/vol-point vendor-convention proxy",
        f"Theta: ${_fmt(result.theta_dollars_per_day)} / day vendor-convention proxy",
        "",
        "## Explicit exit policy",
        "",
    ]
    supplied = False
    for name, label, _ in _POLICY_FIELDS:
        threshold = policy[name]
        if threshold is None:
            continue
        supplied = True
        trigger = next(item for item in result.triggers if item.policy_name == name)
        outcome = "EXIT" if trigger.triggered else "REVIEW" if not trigger.evaluable else "HOLD"
        lines.append(
            f"- {label}: actual {_fmt(trigger.actual)} vs threshold {_fmt(threshold)} — {outcome}. "
            f"{trigger.reason}"
        )
    if not supplied:
        lines.append(
            "No exit policy supplied; this is REPORT_ONLY and does not decide whether to close the position."
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "This is deterministic lifecycle monitoring for an existing long option, not an order. EXIT means an explicit user-supplied policy condition was met; it does not submit or simulate an execution.",
            "P/L uses the current delayed Cboe bid as a conservative liquidation proxy, not midpoint, theoretical value, or a guaranteed fill. Commissions, taxes, slippage, market impact, and after-hours changes are excluded.",
            "Theta/Greeks are vendor-calculated delayed snapshot values and their units/conventions can vary. Theta burn is normalized by current midpoint only when both a valid midpoint and theta are available.",
            "This layer does not re-run the underlying investment thesis, forecast volatility, model early exercise/dividends/event gaps, or decide whether rolling into another contract is superior. Those require separate analysis.",
        ]
    )
    return "\n".join(lines)


def evaluate_long_option_position(
    snapshot: EquityOptionSnapshot,
    *,
    entry_premium: object,
    contracts: object,
    take_profit_pct: object | None = None,
    stop_loss_pct: object | None = None,
    exit_at_dte: object | None = None,
    max_theta_burn_pct_per_day: object | None = None,
) -> OptionPositionManagementResult:
    """Evaluate one existing long option against only explicit exit policy."""

    premium, contract_count = resolve_long_option_position_inputs(
        entry_premium=entry_premium, contracts=contracts
    )
    policy = resolve_option_exit_policy(
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        exit_at_dte=exit_at_dte,
        max_theta_burn_pct_per_day=max_theta_burn_pct_per_day,
    )

    bid = _finite(snapshot.bid)
    if bid is not None and bid < 0:
        bid = None
    entry_cost = premium * CONTRACT_MULTIPLIER * contract_count
    current_value = (
        None if bid is None else bid * CONTRACT_MULTIPLIER * contract_count
    )
    pnl_dollars = None if current_value is None else current_value - entry_cost
    pnl_pct = None if bid is None else (bid - premium) / premium * 100

    midpoint = _finite(snapshot.midpoint)
    theta = _finite(snapshot.theta)
    theta_burn = (
        None
        if midpoint is None or midpoint <= 0 or theta is None
        else abs(theta) / midpoint * 100
    )

    scale = CONTRACT_MULTIPLIER * contract_count
    delta = _finite(snapshot.delta)
    gamma = _finite(snapshot.gamma)
    vega = _finite(snapshot.vega)
    spot = _finite(snapshot.underlying_spot)
    delta_shares = None if delta is None else delta * scale
    delta_notional = (
        None
        if delta_shares is None or spot is None
        else delta_shares * spot
    )
    gamma_exposure = None if gamma is None else gamma * scale
    vega_exposure = None if vega is None else vega * scale
    theta_exposure = None if theta is None else theta * scale

    actuals: dict[str, float | int | None] = {
        "take_profit_pct": pnl_pct,
        "stop_loss_pct": pnl_pct,
        "exit_at_dte": snapshot.dte,
        "max_theta_burn_pct_per_day": theta_burn,
    }
    triggers = tuple(
        _policy_trigger(
            name=name,
            label=label,
            actual=actuals[name],
            threshold=threshold,
        )
        for name, label, _ in _POLICY_FIELDS
        if (threshold := policy[name]) is not None
    )

    if not triggers:
        status = "REPORT_ONLY"
    elif any(trigger.triggered for trigger in triggers):
        status = "EXIT"
    elif any(not trigger.evaluable for trigger in triggers):
        status = "REVIEW"
    else:
        status = "HOLD"

    result = OptionPositionManagementResult(
        status=status,
        entry_premium=premium,
        contracts=contract_count,
        entry_cost=entry_cost,
        liquidation_price_proxy=bid,
        current_liquidation_value=current_value,
        pnl_dollars=pnl_dollars,
        pnl_pct=pnl_pct,
        theta_burn_pct_of_midpoint_per_day=theta_burn,
        delta_shares=delta_shares,
        delta_notional_proxy=delta_notional,
        gamma_delta_shares_per_dollar=gamma_exposure,
        vega_dollars_per_vol_point=vega_exposure,
        theta_dollars_per_day=theta_exposure,
        triggers=triggers,
        report="",
    )
    return OptionPositionManagementResult(
        **{**result.__dict__, "report": _render_report(snapshot, result, policy)}
    )
