from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    opponent_argument_or_opening,
)
from tradingagents.instrument_router import classify_instrument


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = opponent_argument_or_opening(
            investment_debate_state.get("current_response", ""), "bear analyst"
        )
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)
        profile = classify_instrument(state["company_of_interest"])
        is_ordinary_equity = (
            profile.primary_type == "stock"
            and profile.asset_class == "equity"
            and profile.instrument_kind == "stock"
        )

        if is_ordinary_equity:
            prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.
"""
        else:
            prompt = f"""You are a Bull Analyst advocating for a long position in the market instrument. Build a strong, evidence-based case without assuming it is a company.

Key points to focus on:
- Macro and Flow Drivers: Explain the macro regime, cross-asset relationships, liquidity, and flows that support the bullish case.
- Market Structure and Positioning: Discuss positioning, flows, or market structure only where the reports provide evidence; do not invent unavailable data.
- Price Behavior: Evaluate trend, momentum, and volatility, including how they affect upside potential and risk.
- Catalysts and Invalidation: Identify concrete bullish catalysts, key levels, and conditions that would invalidate the thesis.
- Direct Rebuttal: Engage directly with the bear analyst's claims using specific evidence, or open with your own case when the bear has not spoken.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Contextual/fundamental report (may be unavailable or not applicable): {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to present the bullish thesis, its catalysts, its invalidation conditions, and a direct rebuttal of the bear case.
"""

        prompt += get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
