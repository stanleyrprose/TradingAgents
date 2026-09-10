"""The market analyst is bound (and prompt-instructed) to call
get_verified_market_snapshot; if the executor ToolNode doesn't register it, the
call fails and the model reports the tool "unavailable" and skips verification.

Regression guard for that wiring gap (snapshot bound to the LLM but missing from
the market ToolNode).
"""
import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.unit
def test_market_toolnode_can_execute_verified_snapshot():
    # _create_tool_nodes does not use self -> call unbound (avoids building LLMs).
    nodes = TradingAgentsGraph._create_tool_nodes(None)
    market_tools = set(nodes["market"].tools_by_name)
    assert "get_verified_market_snapshot" in market_tools, (
        "get_verified_market_snapshot is bound to the market analyst but not "
        "registered in the market ToolNode, so the model's call fails."
    )
    # the other core market tools must remain too
    assert {
        "get_stock_data",
        "get_indicators",
        "get_equity_option_context",
    } <= market_tools


@pytest.mark.unit
def test_news_toolnode_registers_cross_asset_and_core_news_tools():
    nodes = TradingAgentsGraph._create_tool_nodes(None)
    news_tools = set(nodes["news"].tools_by_name)

    assert {
        "get_cross_asset_context",
        "get_macro_indicators",
        "get_news",
        "get_global_news",
    } <= news_tools
