import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import tradingagents.agents.analysts.market_analyst as ma


class _CapturingToolLLM:
    def __init__(self):
        self.bound_tool_names = []
        self.system_prompt = ""

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]

        def invoke(prompt_value):
            self.system_prompt = prompt_value.to_messages()[0].content
            return AIMessage(content="report", tool_calls=[])

        return RunnableLambda(invoke)


def _invoke_market_analyst(ticker, analysis_symbol=None):
    llm = _CapturingToolLLM()
    state = {
        "trade_date": "2026-09-10",
        "company_of_interest": ticker,
        "asset_type": "stock",
        "messages": [],
    }
    if analysis_symbol is not None:
        state["analysis_symbol"] = analysis_symbol
    ma.create_market_analyst(llm)(state)
    return llm


@pytest.mark.unit
def test_occ_market_analyst_binds_option_context_and_routes_underlying_tools():
    ticker = "AAPL260918C00200000"
    llm = _invoke_market_analyst(ticker, analysis_symbol="AAPL")

    assert "get_equity_option_context" in llm.bound_tool_names
    assert ticker in llm.system_prompt
    assert "underlying analysis ticker `AAPL`" in llm.system_prompt
    assert "get_equity_option_context(ticker, curr_date)" in llm.system_prompt
    assert f"exact requested OCC ticker `{ticker}`" in llm.system_prompt
    assert "get_stock_data, get_indicators, and get_verified_market_snapshot" in llm.system_prompt
    assert "{market_ticker_reference}" not in llm.system_prompt
    assert (
        "call get_verified_market_snapshot for the underlying analysis ticker `AAPL` "
        "and the current date"
    ) in llm.system_prompt
    assert "underlying BUY or SELL view does not guarantee the option payoff" in llm.system_prompt
    assert f"final conclusion must concern the requested option contract `{ticker}`" in llm.system_prompt


@pytest.mark.unit
def test_normal_equity_market_analyst_does_not_bind_option_context():
    llm = _invoke_market_analyst("AAPL")

    assert "get_equity_option_context" not in llm.bound_tool_names
    assert "This is an equity option analysis" not in llm.system_prompt
    assert (
        "call get_verified_market_snapshot for this ticker and the current date"
        in llm.system_prompt
    )
    assert "{market_ticker_reference}" not in llm.system_prompt
