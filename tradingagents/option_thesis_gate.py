"""Deterministic consensus gate between an underlying thesis and option selection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from tradingagents.agents.utils.rating import RATING_REVIEW, extract_rating

OptionDirection = Literal["bullish", "bearish"]
TraderAction = Literal["Buy", "Hold", "Sell"]
ThesisRefreshStatus = Literal["CONFIRMED", "INVALIDATED", "NEUTRAL"]

_FINAL_ACTION_RE = re.compile(
    r"^\s*(?:\*\*)?FINAL TRANSACTION PROPOSAL(?:\*\*)?\s*:\s*"
    r"(?:\*\*)?(BUY|HOLD|SELL)(?:\*\*)?\s*$",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"^\s*(?:\*\*)?Action(?:\*\*)?\s*:\s*"
    r"(?:\*\*)?(BUY|HOLD|SELL)(?:\*\*)?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OptionThesisGate:
    """Immutable result of applying the conservative option-direction gate."""

    portfolio_rating: str | None
    trader_action: TraderAction | None
    direction: OptionDirection | None
    reason: str


@dataclass(frozen=True)
class LongOptionThesisRefresh:
    """Current underlying thesis relative to an existing long call or put."""

    option_direction: OptionDirection
    current_direction: OptionDirection | None
    status: ThesisRefreshStatus
    portfolio_rating: str | None
    trader_action: TraderAction | None
    reason: str


@dataclass(frozen=True)
class LongOptionApproval:
    """Strict post-analysis approval result for buying one option contract."""

    approved: bool
    portfolio_rating: str | None
    trader_action: TraderAction | None


def parse_trader_action(text: str) -> TraderAction | None:
    """Parse only explicit TraderProposal labels, preferring the final label.

    Prose containing buy, hold, or sell is deliberately ignored.
    """
    if not text:
        return None
    lines = unicodedata.normalize("NFKC", text).splitlines()
    for pattern in (_FINAL_ACTION_RE, _ACTION_RE):
        for line in lines:
            match = pattern.fullmatch(line)
            if match:
                return match.group(1).capitalize()  # type: ignore[return-value]
    return None


def _portfolio_rating(decision: str | None) -> str | None:
    if not decision:
        return None
    if unicodedata.normalize("NFKC", decision).strip().upper() == RATING_REVIEW:
        return RATING_REVIEW
    return extract_rating(decision)


def gate_long_option_purchase(
    portfolio_decision: str | None,
    trader_investment_plan: str | None,
) -> LongOptionApproval:
    """Approve long premium only for PM Buy/Overweight plus explicit Trader Buy.

    This gate deliberately does not interpret call/put direction. It applies to the
    deeply analyzed option contract itself, so an approved long put is still a buy.
    """
    portfolio_rating = _portfolio_rating(portfolio_decision)
    trader_action = parse_trader_action(trader_investment_plan or "")
    return LongOptionApproval(
        approved=(
            portfolio_rating in {"Buy", "Overweight"} and trader_action == "Buy"
        ),
        portfolio_rating=portfolio_rating,
        trader_action=trader_action,
    )


def gate_option_thesis(
    portfolio_decision: str | None,
    trader_investment_plan: str | None,
) -> OptionThesisGate:
    """Return a direction only when Portfolio Manager and Trader agree exactly."""
    portfolio_rating = _portfolio_rating(portfolio_decision)
    trader_action = parse_trader_action(trader_investment_plan or "")

    if portfolio_rating in {"Buy", "Overweight"} and trader_action == "Buy":
        return OptionThesisGate(
            portfolio_rating,
            trader_action,
            "bullish",
            "bullish consensus: portfolio is Buy/Overweight and trader action is Buy",
        )
    if portfolio_rating in {"Sell", "Underweight"} and trader_action == "Sell":
        return OptionThesisGate(
            portfolio_rating,
            trader_action,
            "bearish",
            "bearish consensus: portfolio is Sell/Underweight and trader action is Sell",
        )
    return OptionThesisGate(
        portfolio_rating,
        trader_action,
        None,
        "NO OPTION TRADE: portfolio rating and trader action lack directional consensus",
    )


def refresh_long_option_thesis(
    option_right: str,
    portfolio_decision: str | None,
    trader_investment_plan: str | None,
) -> LongOptionThesisRefresh:
    """Compare refreshed underlying consensus with the direction of an existing long option.

    A long call needs bullish consensus and a long put needs bearish consensus.
    Lack of consensus is NEUTRAL, not INVALIDATED. Only an explicit opposite
    consensus invalidates the original directional thesis.
    """

    right = option_right.upper() if isinstance(option_right, str) else ""
    if right == "C":
        option_direction: OptionDirection = "bullish"
    elif right == "P":
        option_direction = "bearish"
    else:
        raise ValueError("option_right must be C or P")

    gate = gate_option_thesis(portfolio_decision, trader_investment_plan)
    if gate.direction is None:
        return LongOptionThesisRefresh(
            option_direction=option_direction,
            current_direction=None,
            status="NEUTRAL",
            portfolio_rating=gate.portfolio_rating,
            trader_action=gate.trader_action,
            reason=(
                "refreshed underlying thesis has no directional PM/Trader consensus; "
                "absence of confirmation is not an opposite thesis"
            ),
        )
    if gate.direction == option_direction:
        return LongOptionThesisRefresh(
            option_direction=option_direction,
            current_direction=gate.direction,
            status="CONFIRMED",
            portfolio_rating=gate.portfolio_rating,
            trader_action=gate.trader_action,
            reason=f"refreshed {gate.direction} consensus confirms the existing long option direction",
        )
    return LongOptionThesisRefresh(
        option_direction=option_direction,
        current_direction=gate.direction,
        status="INVALIDATED",
        portfolio_rating=gate.portfolio_rating,
        trader_action=gate.trader_action,
        reason=(
            f"refreshed {gate.direction} consensus is opposite the existing "
            f"{option_direction} long-option thesis"
        ),
    )
