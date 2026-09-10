import pytest

from tradingagents.option_thesis_gate import (
    gate_long_option_purchase,
    gate_option_thesis,
    parse_trader_action,
    refresh_long_option_thesis,
)


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


@pytest.mark.parametrize("portfolio", ["Buy", "Overweight"])
def test_long_option_purchase_approves_bullish_pm_rating_and_trader_buy(portfolio):
    result = gate_long_option_purchase(portfolio, "**Action**: Buy")

    assert result.approved is True
    assert result.portfolio_rating == portfolio
    assert result.trader_action == "Buy"


def test_long_put_purchase_is_direction_agnostic_at_gate():
    plan = "Contract: long put\n**Action**: Buy"

    result = gate_long_option_purchase("Buy", plan)

    assert result.approved is True
    assert result.trader_action == "Buy"


@pytest.mark.parametrize(
    ("portfolio", "action"),
    [
        ("Underweight", "Buy"),
        ("Sell", "Buy"),
        ("Buy", "Hold"),
        ("REVIEW", "Buy"),
    ],
)
def test_long_option_purchase_rejects_non_approving_consensus(portfolio, action):
    result = gate_long_option_purchase(portfolio, f"**Action**: {action}")

    assert result.approved is False
    assert result.portfolio_rating == portfolio
    assert result.trader_action == action


def test_long_option_purchase_rejects_unparseable_pm_decision():
    result = gate_long_option_purchase(
        "The portfolio manager supplied no rating.",
        "**Action**: Buy",
    )

    assert result.approved is False
    assert result.portfolio_rating is None
    assert result.trader_action == "Buy"


def test_long_option_purchase_rejects_malformed_trader_text():
    result = gate_long_option_purchase("Buy", "The trader would buy this contract.")

    assert result.approved is False
    assert result.portfolio_rating == "Buy"
    assert result.trader_action is None


def test_long_option_purchase_uses_strict_final_transaction_proposal_priority():
    plan = "**Action**: Buy\nFINAL TRANSACTION PROPOSAL: **HOLD**"

    assert parse_trader_action(plan) == "Hold"
    result = gate_long_option_purchase("Buy", plan)
    assert result.approved is False
    assert result.trader_action == "Hold"


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


@pytest.mark.parametrize(
    ("right", "portfolio", "action", "status", "option_direction", "current_direction"),
    [
        ("C", "Buy", "Buy", "CONFIRMED", "bullish", "bullish"),
        ("C", "Overweight", "Buy", "CONFIRMED", "bullish", "bullish"),
        ("C", "Sell", "Sell", "INVALIDATED", "bullish", "bearish"),
        ("C", "Underweight", "Sell", "INVALIDATED", "bullish", "bearish"),
        ("C", "Hold", "Hold", "NEUTRAL", "bullish", None),
        ("P", "Sell", "Sell", "CONFIRMED", "bearish", "bearish"),
        ("P", "Underweight", "Sell", "CONFIRMED", "bearish", "bearish"),
        ("P", "Buy", "Buy", "INVALIDATED", "bearish", "bullish"),
        ("P", "Overweight", "Buy", "INVALIDATED", "bearish", "bullish"),
        ("P", "Hold", "Hold", "NEUTRAL", "bearish", None),
    ],
)
def test_refresh_long_option_thesis_distinguishes_confirmation_invalidation_and_neutral(
    right, portfolio, action, status, option_direction, current_direction
):
    result = refresh_long_option_thesis(right, portfolio, f"**Action**: {action}")

    assert result.status == status
    assert result.option_direction == option_direction
    assert result.current_direction == current_direction
    assert result.portfolio_rating == portfolio
    assert result.trader_action == action
    if status == "NEUTRAL":
        assert "not an opposite thesis" in result.reason


@pytest.mark.parametrize("right", ["", "X", None, 1])
def test_refresh_long_option_thesis_rejects_invalid_option_right(right):
    with pytest.raises(ValueError, match="option_right must be C or P"):
        refresh_long_option_thesis(right, "Buy", "**Action**: Buy")


def test_refresh_long_option_thesis_result_is_immutable():
    result = refresh_long_option_thesis("C", "Buy", "**Action**: Buy")

    with pytest.raises(AttributeError):
        result.status = "INVALIDATED"
