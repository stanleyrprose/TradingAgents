"""Deterministic comparison of replacement contracts for an existing long option."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tradingagents.dataflows.equity_options import EquityOptionSnapshot
from tradingagents.dataflows.option_selector import OptionCandidate, OptionSelectionResult
from tradingagents.option_capital_allocator import CONTRACT_MULTIPLIER
from tradingagents.option_thesis_gate import LongOptionThesisRefresh


@dataclass(frozen=True)
class OptionRollCandidate:
    """One later-expiry replacement candidate with conservative cash mechanics."""

    symbol: str
    expiry: str
    dte: int
    dte_extension: int
    strike: float
    delta: float
    iv: float
    spread_pct: float
    selector_score: float
    close_credit_per_share: float
    open_debit_per_share: float
    net_debit_per_share: float
    net_cash_outflow: float
    new_gross_premium_at_risk: float
    delta_shares_after_roll: float
    delta_shares_change: float | None
    theta_dollars_per_day_after_roll: float | None
    theta_dollars_per_day_change: float | None


@dataclass(frozen=True)
class OptionRollPlanResult:
    """Thesis-gated roll comparison; never an execution instruction."""

    status: str
    reason: str
    candidates: tuple[OptionRollCandidate, ...]
    report: str



def _valid_nonnegative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None



def _valid_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None



def _right_direction(right: str) -> str:
    if right == "C":
        return "bullish"
    if right == "P":
        return "bearish"
    raise ValueError("snapshot right must be C or P")



def _render_no_roll(
    snapshot: EquityOptionSnapshot,
    refresh: LongOptionThesisRefresh,
    *,
    status: str,
    reason: str,
) -> str:
    return "\n".join(
        [
            "# Long option roll planning",
            "",
            f"Status: **{status}**",
            f"Current contract: {snapshot.symbol} | DTE: {snapshot.dte}",
            f"Existing option thesis: {refresh.option_direction}",
            f"Refreshed thesis: {refresh.status} | Current direction: {refresh.current_direction or 'none'}",
            f"Reason: {reason}",
            "",
            "No replacement contract is proposed. A roll is not a substitute for a confirmed underlying thesis.",
        ]
    )



def _to_roll_candidate(
    snapshot: EquityOptionSnapshot,
    candidate: OptionCandidate,
    *,
    contracts: int,
    close_bid: float,
) -> OptionRollCandidate:
    scale = CONTRACT_MULTIPLIER * contracts
    current_delta = _valid_number(snapshot.delta)
    current_theta = _valid_number(snapshot.theta)
    new_theta = _valid_number(candidate.theta)
    new_delta_shares = candidate.delta * scale
    current_delta_shares = None if current_delta is None else current_delta * scale
    new_theta_dollars = None if new_theta is None else new_theta * scale
    current_theta_dollars = None if current_theta is None else current_theta * scale
    return OptionRollCandidate(
        symbol=candidate.symbol,
        expiry=candidate.expiry.isoformat(),
        dte=candidate.dte,
        dte_extension=candidate.dte - snapshot.dte,
        strike=candidate.strike,
        delta=candidate.delta,
        iv=candidate.iv,
        spread_pct=candidate.spread_pct,
        selector_score=candidate.score,
        close_credit_per_share=close_bid,
        open_debit_per_share=candidate.ask,
        net_debit_per_share=candidate.ask - close_bid,
        net_cash_outflow=(candidate.ask - close_bid) * scale,
        new_gross_premium_at_risk=candidate.ask * scale,
        delta_shares_after_roll=new_delta_shares,
        delta_shares_change=(
            None if current_delta_shares is None else new_delta_shares - current_delta_shares
        ),
        theta_dollars_per_day_after_roll=new_theta_dollars,
        theta_dollars_per_day_change=(
            None
            if current_theta_dollars is None or new_theta_dollars is None
            else new_theta_dollars - current_theta_dollars
        ),
    )



def _fmt(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"



def _render_compare(
    snapshot: EquityOptionSnapshot,
    refresh: LongOptionThesisRefresh,
    candidates: tuple[OptionRollCandidate, ...],
    *,
    contracts: int,
) -> str:
    lines = [
        "# Long option roll planning",
        "",
        "Status: **COMPARE**",
        f"Current contract: {snapshot.symbol} | DTE: {snapshot.dte} | Contracts: {contracts}",
        f"Refreshed thesis: {refresh.status} ({refresh.current_direction})",
        f"Conservative close proxy: current delayed bid ${_fmt(snapshot.bid)} per share",
        "",
        "## Later-expiry replacement candidates",
        "",
        "| Rank | Contract | DTE (+days) | Strike | Delta | Spread | IV | Selector | New ask | Net debit/share | Net cash outflow | New gross premium risk | Delta shares | Theta $/day |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(candidates, 1):
        lines.append(
            f"| {rank} | {item.symbol} | {item.dte} (+{item.dte_extension}) | "
            f"{item.strike:.2f} | {item.delta:.3f} | {item.spread_pct:.2f}% | "
            f"{item.iv * 100:.2f}% | {item.selector_score:.2f} | ${item.open_debit_per_share:.2f} | "
            f"${item.net_debit_per_share:.2f} | ${item.net_cash_outflow:,.2f} | "
            f"${item.new_gross_premium_at_risk:,.2f} | {item.delta_shares_after_roll:.2f} | "
            f"{_fmt(item.theta_dollars_per_day_after_roll)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Positive net cash outflow means the roll requires additional cash; a negative value is a net credit. This is based on closing the old long at its delayed bid and opening the new long at its delayed ask.",
            "New gross premium-at-risk is the replacement option ask × 100 × contracts. It is not the same as the roll net debit: proceeds from closing the old position do not reduce the amount the new long option itself can subsequently lose.",
            "Selector score ranks contract quality/liquidity fit only. It is not expected return, thesis confidence, or a command to roll.",
            "",
            "## Caveats",
            "",
            "This is a planning comparison, not an order. Cboe quotes are delayed and bid/ask are conservative proxies, not guaranteed fills. Commissions, taxes, slippage, market impact, assignment/early exercise, dividends, earnings/event gaps, and broker-specific mechanics are excluded.",
            "Only same-direction single-leg long calls or puts with strictly later expiries are considered. A thesis refresh must be CONFIRMED before candidates are compared; NEUTRAL or INVALIDATED produces NO_ROLL.",
        ]
    )
    return "\n".join(lines)



def plan_long_option_roll(
    snapshot: EquityOptionSnapshot,
    refresh: LongOptionThesisRefresh,
    *,
    contracts: int,
    selection: OptionSelectionResult | None = None,
    top_n: int = 3,
) -> OptionRollPlanResult:
    """Compare later-expiry replacement contracts only after thesis confirmation."""

    if isinstance(contracts, bool) or not isinstance(contracts, int) or contracts <= 0:
        raise ValueError("contracts must be a positive whole number")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 20:
        raise ValueError("top_n must be between 1 and 20")

    expected_direction = _right_direction(snapshot.right)
    if refresh.option_direction != expected_direction:
        raise ValueError("thesis refresh direction does not match the current option right")

    if refresh.status == "INVALIDATED":
        reason = "refreshed underlying consensus is opposite the existing option thesis"
        return OptionRollPlanResult(
            "NO_ROLL",
            reason,
            (),
            _render_no_roll(snapshot, refresh, status="NO_ROLL", reason=reason),
        )
    if refresh.status != "CONFIRMED":
        reason = "refreshed underlying thesis lacks directional confirmation"
        return OptionRollPlanResult(
            "NO_ROLL",
            reason,
            (),
            _render_no_roll(snapshot, refresh, status="NO_ROLL", reason=reason),
        )

    close_bid = _valid_nonnegative(snapshot.bid)
    if close_bid is None:
        reason = "current delayed bid is unavailable, so conservative roll cash mechanics cannot be evaluated"
        return OptionRollPlanResult(
            "REVIEW",
            reason,
            (),
            _render_no_roll(snapshot, refresh, status="REVIEW", reason=reason),
        )
    if selection is None or not selection.available:
        reason = (
            selection.unavailable_reason
            if selection is not None and selection.unavailable_reason
            else "replacement-contract selection is unavailable"
        )
        return OptionRollPlanResult(
            "REVIEW",
            reason,
            (),
            _render_no_roll(snapshot, refresh, status="REVIEW", reason=reason),
        )

    eligible = tuple(
        candidate
        for candidate in selection.candidates
        if candidate.right == snapshot.right
        and candidate.symbol != snapshot.symbol
        and candidate.expiry > snapshot.expiry
        and candidate.dte > snapshot.dte
    )
    if not eligible:
        reason = "no same-direction strictly later-expiry replacement candidates are available"
        return OptionRollPlanResult(
            "REVIEW",
            reason,
            (),
            _render_no_roll(snapshot, refresh, status="REVIEW", reason=reason),
        )

    compared = tuple(
        _to_roll_candidate(snapshot, candidate, contracts=contracts, close_bid=close_bid)
        for candidate in eligible[:top_n]
    )
    return OptionRollPlanResult(
        "COMPARE",
        "thesis is confirmed; later-expiry replacement candidates are available for comparison",
        compared,
        _render_compare(snapshot, refresh, compared, contracts=contracts),
    )
