"""Deterministic consensus gate between an underlying thesis and option selection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from tradingagents.agents.utils.rating import RATING_REVIEW, extract_rating

OptionDirection = Literal["bullish", "bearish"]
TraderAction = Literal["Buy", "Hold", "Sell"]

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
