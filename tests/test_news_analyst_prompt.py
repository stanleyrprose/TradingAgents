"""Guard the news analyst prompt against tool-signature drift (#1116).

The prompt used to advertise ``get_news(query, ...)`` while the tool takes a
``ticker``, tricking the LLM into hallucinating free-text query calls.
"""
import inspect

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import tradingagents.agents.analysts.news_analyst as na
from tradingagents.agents.utils.news_data_tools import get_news


@pytest.mark.unit
def test_get_news_takes_ticker_not_query():
    arg_names = set(get_news.args.keys())
    assert "ticker" in arg_names
    assert "query" not in arg_names


@pytest.mark.unit
def test_news_prompt_matches_get_news_signature():
    src = inspect.getsource(na)
    assert "get_news(ticker, start_date, end_date)" in src
    assert "get_news(query" not in src


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


def _invoke_news_analyst(ticker):
    llm = _CapturingToolLLM()
    na.create_news_analyst(llm)(
        {
            "trade_date": "2026-09-10",
            "company_of_interest": ticker,
            "asset_type": "stock",
            "messages": [],
        }
    )
    return llm


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ticker", "profile_wording"),
    [("EURUSD", "asset_class=forex"), ("GC=F", "asset_class=commodity")],
)
def test_cross_asset_news_analyst_binds_tool_and_renders_instruction(
    ticker, profile_wording
):
    llm = _invoke_news_analyst(ticker)

    assert "get_cross_asset_context" in llm.bound_tool_names
    assert "get_cross_asset_context(ticker, current_date, 180)" in llm.system_prompt
    assert f"exact ticker `{ticker}`" in llm.system_prompt
    assert profile_wording in llm.system_prompt
    assert "market instrument" in llm.system_prompt


@pytest.mark.unit
def test_equity_news_analyst_does_not_bind_cross_asset_context():
    llm = _invoke_news_analyst("AAPL")

    assert "get_cross_asset_context" not in llm.bound_tool_names
    assert "get_cross_asset_context(ticker, current_date, 180)" not in llm.system_prompt
    assert "company-specific news" in llm.system_prompt
