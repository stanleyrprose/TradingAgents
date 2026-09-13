"""Pure premium-risk sizing for approved long equity-option candidates.

This module deliberately does not interpret selector scores or analysis decisions.
Its inputs are assumed to have already been approved for long, single-leg exposure.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from tradingagents.dataflows.option_selector import OptionCandidate

CONTRACT_MULTIPLIER = 100


@dataclass(frozen=True)
class OptionCapitalPosition:
    """Whole-contract allocation for one approved long option."""

    rank: int
    symbol: str
    ask_price_proxy: float
    premium_per_contract: float
    target_budget: float
    contracts: int
    premium_at_risk: float
    bucket_utilization_pct: float
    portfolio_risk_pct: float | None


@dataclass(frozen=True)
class OptionCapitalAllocation:
    """Deterministic allocation result and a compact human-readable report."""

    positions: tuple[OptionCapitalPosition, ...]
    risk_budget: float
    total_premium_at_risk: float
    unused_budget: float
    account_equity: float | None
    max_risk_pct: float | None
    premium_budget: float | None
    portfolio_risk_pct: float | None
    report: str
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """Whether candidate input was present and valid for calculation."""

        return self.unavailable_reason is None

    @property
    def has_position(self) -> bool:
        """Whether at least one candidate received one or more contracts."""

        return any(position.contracts > 0 for position in self.positions)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number greater than zero")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return number


def resolve_long_option_risk_budget(
    *,
    account_equity: object | None,
    max_risk_pct: object | None,
    premium_budget: object | None,
) -> tuple[float, float | None, float | None, float | None]:
    """Validate sizing inputs and return the effective budget and normalized caps.

    This is public so callers can reject invalid sizing requests before performing
    selector or graph work. The allocator uses the same function, keeping the CLI
    and allocation semantics identical.
    """
    if (account_equity is None) != (max_risk_pct is None):
        raise ValueError("account_equity and max_risk_pct must be supplied together")
    if account_equity is None and premium_budget is None:
        raise ValueError(
            "supply premium_budget, or account_equity together with max_risk_pct"
        )

    equity = (
        None if account_equity is None else _positive_number(account_equity, "account_equity")
    )
    risk_pct = (
        None if max_risk_pct is None else _positive_number(max_risk_pct, "max_risk_pct")
    )
    if risk_pct is not None and risk_pct > 100:
        raise ValueError("max_risk_pct must be less than or equal to 100")
    cash_cap = (
        None if premium_budget is None else _positive_number(premium_budget, "premium_budget")
    )

    percentage_budget = None if equity is None else equity * risk_pct / 100
    risk_budget = (
        cash_cap
        if percentage_budget is None
        else percentage_budget
        if cash_cap is None
        else min(percentage_budget, cash_cap)
    )
    return risk_budget, equity, risk_pct, cash_cap


def _unavailable_report(reason: str, risk_budget: float) -> str:
    return (
        "## Long option premium-risk allocation\n\n"
        f"Unavailable: {reason}\n\n"
        f"Risk budget: ${risk_budget:,.2f}; no premium allocated."
    )


def _allocation_report(
    positions: tuple[OptionCapitalPosition, ...],
    *,
    risk_budget: float,
    total_premium_at_risk: float,
    unused_budget: float,
    portfolio_risk_pct: float | None,
) -> str:
    rows = [
        "| Rank | Symbol | Ask proxy | Target | Contracts | Premium at risk | Bucket used |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    rows.extend(
        f"| {item.rank} | {item.symbol} | ${item.ask_price_proxy:,.2f} | "
        f"${item.target_budget:,.2f} | {item.contracts} | "
        f"${item.premium_at_risk:,.2f} | {item.bucket_utilization_pct:.2f}% |"
        for item in positions
    )
    risk_pct_text = (
        "N/A" if portfolio_risk_pct is None else f"{portfolio_risk_pct:.2f}%"
    )
    return "\n".join(
        [
            "## Long option premium-risk allocation",
            "",
            *rows,
            "",
            f"Risk budget: ${risk_budget:,.2f}; premium at risk: "
            f"${total_premium_at_risk:,.2f}; unused budget: ${unused_budget:,.2f}; "
            f"portfolio risk: {risk_pct_text}.",
            "",
            "Sizing uses each contract's ask as a conservative proxy, not a guaranteed fill. "
            "A long option's premium can go to zero; premium per contract (ask × 100) is the "
            "maximum premium-at-risk proxy. This ignores commissions and slippage and assumes "
            "no exercise or assignment complications before exit.",
            "",
            "Candidates receive equal, non-reallocated risk buckets in input/rank order. "
            "Selector score is intentionally not a sizing weight. Same-underlying, "
            "same-direction Top3 is NOT diversification; it is only a split of premium risk.",
            "",
            "Excludes commissions, slippage, and taxes; does not model portfolio Greeks, "
            "correlation, or margin.",
        ]
    )


def allocate_long_option_premium_risk(
    candidates: Iterable[OptionCandidate],
    *,
    account_equity: float | None = None,
    max_risk_pct: float | None = None,
    premium_budget: float | None = None,
) -> OptionCapitalAllocation:
    """Allocate equal premium-risk buckets across approved long options.

    The conservative entry proxy is ``candidate.ask``; midpoint and selector score
    never affect sizing. Unspent cash in one bucket is not moved to another bucket.
    """

    risk_budget, equity, risk_pct, cash_cap = resolve_long_option_risk_budget(
        account_equity=account_equity,
        max_risk_pct=max_risk_pct,
        premium_budget=premium_budget,
    )

    try:
        supplied = tuple(candidates)
    except TypeError:
        supplied = ()
        candidate_error = "candidates must be an iterable of OptionCandidate values"
    else:
        candidate_error = None

    if not supplied and candidate_error is None:
        candidate_error = "no approved option candidates were supplied"

    validated: list[tuple[str, float]] = []
    if candidate_error is None:
        for rank, candidate in enumerate(supplied, start=1):
            symbol = getattr(candidate, "symbol", None)
            ask = getattr(candidate, "ask", None)
            if not isinstance(symbol, str) or not symbol.strip():
                candidate_error = f"candidate {rank} has an empty or invalid symbol"
                break
            if isinstance(ask, bool) or not isinstance(ask, Real):
                candidate_error = f"candidate {rank} ({symbol}) has an invalid ask"
                break
            ask_number = float(ask)
            if not math.isfinite(ask_number) or ask_number <= 0:
                candidate_error = (
                    f"candidate {rank} ({symbol}) ask must be finite and greater than zero"
                )
                break
            validated.append((symbol, ask_number))

    if candidate_error is not None:
        return OptionCapitalAllocation(
            positions=(),
            risk_budget=risk_budget,
            total_premium_at_risk=0.0,
            unused_budget=risk_budget,
            account_equity=equity,
            max_risk_pct=risk_pct,
            premium_budget=cash_cap,
            portfolio_risk_pct=0.0 if equity is not None else None,
            report=_unavailable_report(candidate_error, risk_budget),
            unavailable_reason=candidate_error,
        )

    target_budget = risk_budget / len(validated)
    positions = tuple(
        OptionCapitalPosition(
            rank=rank,
            symbol=symbol,
            ask_price_proxy=ask,
            premium_per_contract=ask * CONTRACT_MULTIPLIER,
            target_budget=target_budget,
            contracts=math.floor(target_budget / (ask * CONTRACT_MULTIPLIER)),
            premium_at_risk=(
                math.floor(target_budget / (ask * CONTRACT_MULTIPLIER))
                * ask
                * CONTRACT_MULTIPLIER
            ),
            bucket_utilization_pct=(
                math.floor(target_budget / (ask * CONTRACT_MULTIPLIER))
                * ask
                * CONTRACT_MULTIPLIER
                / target_budget
                * 100
            ),
            portfolio_risk_pct=(
                None
                if equity is None
                else math.floor(target_budget / (ask * CONTRACT_MULTIPLIER))
                * ask
                * CONTRACT_MULTIPLIER
                / equity
                * 100
            ),
        )
        for rank, (symbol, ask) in enumerate(validated, start=1)
    )
    total_premium_at_risk = sum(item.premium_at_risk for item in positions)
    unused_budget = risk_budget - total_premium_at_risk
    portfolio_risk_pct = (
        None
        if equity is None
        else total_premium_at_risk / equity * 100
    )
    report = _allocation_report(
        positions,
        risk_budget=risk_budget,
        total_premium_at_risk=total_premium_at_risk,
        unused_budget=unused_budget,
        portfolio_risk_pct=portfolio_risk_pct,
    )
    return OptionCapitalAllocation(
        positions=positions,
        risk_budget=risk_budget,
        total_premium_at_risk=total_premium_at_risk,
        unused_budget=unused_budget,
        account_equity=equity,
        max_risk_pct=risk_pct,
        premium_budget=cash_cap,
        portfolio_risk_pct=portfolio_risk_pct,
        report=report,
    )
