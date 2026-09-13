"""Regression tests for equity-aware bull and bear debate prompts."""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher


def _capturing_llm(captured):
    llm = MagicMock()
    llm.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or MagicMock(content="argument")
    )
    return llm


def _state(ticker):
    return {
        "company_of_interest": ticker,
        "asset_type": "stock",
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "count": 0,
        },
    }


@pytest.mark.unit
@pytest.mark.parametrize("factory", [create_bull_researcher, create_bear_researcher])
@pytest.mark.parametrize("ticker", ["EURUSD", "GC=F"])
def test_cross_asset_debate_prompt_uses_market_instrument_drivers(factory, ticker):
    captured = {}
    factory(_capturing_llm(captured))(_state(ticker))
    prompt = captured["prompt"].lower()

    assert "revenue projections" not in prompt
    assert "competitive advantages" not in prompt
    assert "macro" in prompt
    assert "flow" in prompt
    assert "catalyst" in prompt
    assert "invalidation" in prompt
    assert "market instrument" in prompt


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory", "expected_wording"),
    [
        (create_bull_researcher, ("revenue projections", "competitive advantages")),
        (create_bear_researcher, ("competitive weaknesses", "financial instability")),
    ],
)
def test_equity_debate_prompt_retains_company_specific_language(
    factory, expected_wording
):
    captured = {}
    factory(_capturing_llm(captured))(_state("AAPL"))
    prompt = captured["prompt"].lower()

    assert "company" in prompt
    assert all(wording in prompt for wording in expected_wording)
