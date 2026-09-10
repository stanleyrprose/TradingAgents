import pytest

from tradingagents.option_thesis_gate import gate_option_thesis, parse_trader_action


@pytest.mark.parametrize(
    ("portfolio", "action", "expected"),
    [
        (portfolio, action, expected)
        for portfolio in ("Buy", "Overweight", "Hold", "Underweight", "Sell", "REVIEW", None)
        for action, expected in (
            ("Buy", "bullish" if portfolio in {"Buy", "Overweight"} else None),
            ("Hold", None),
            ("Sell", "bearish" if portfolio in {"Sell", "Underweight"} else None),
            (None, None),
        )
    ],
)
def test_gate_all_portfolio_and_trader_combinations(portfolio, action, expected):
    plan = None if action is None else f"**Action**: {action}"

    result = gate_option_thesis(portfolio, plan)

    assert result.portfolio_rating == portfolio
    assert result.trader_action == action
    assert result.direction == expected
    if expected is None:
        assert "NO OPTION TRADE" in result.reason


def test_final_transaction_label_has_priority_over_action_and_prose():
    plan = (
        "The trader discusses selling and says BUY in prose.\n"
        "**Action**: Sell\n"
        "FINAL TRANSACTION PROPOSAL: **BUY**"
    )

    assert parse_trader_action(plan) == "Buy"


def test_action_label_is_fallback_when_final_label_is_absent():
    assert parse_trader_action("**Action**: Sell\n\nReasoning: valuation") == "Sell"


@pytest.mark.parametrize(
    "plan",
    [
        "We should buy this stock.",
        "Reasoning: HOLD looks prudent.",
        "The final transaction proposal is SELL.",
        "Actionable view: Buy",
        "FINAL TRANSACTION PROPOSAL: **ACCUMULATE**",
        "",
        None,
    ],
)
def test_parser_never_infers_action_from_prose_or_malformed_labels(plan):
    assert parse_trader_action(plan) is None


def test_gate_result_is_immutable():
    result = gate_option_thesis("Buy", "**Action**: Buy")

    with pytest.raises(AttributeError):
        result.direction = "bearish"
