from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    opponent_argument_or_opening,
)
from tradingagents.instrument_router import classify_instrument


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = opponent_argument_or_opening(
            investment_debate_state.get("current_response", ""), "bull analyst"
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
            prompt = f"""You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock.
"""
        else:
            prompt = f"""You are a Bear Analyst advocating against a long position in the market instrument. Build a strong, evidence-based downside case without assuming it is a company.

Key points to focus on:
- Macro and Flow Drivers: Explain the macro regime, cross-asset relationships, liquidity, and flows that create downside risk.
- Market Structure and Positioning: Discuss positioning, flows, or market structure only where the reports provide evidence; do not invent unavailable data.
- Price Behavior: Evaluate trend, momentum, and volatility, including signs of deterioration or asymmetric risk.
- Catalysts and Invalidation: Identify concrete bearish catalysts, key levels, and conditions that would invalidate the thesis.
- Direct Rebuttal: Engage directly with the bull analyst's claims using specific evidence, or open with your own case when the bull has not spoken.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Contextual/fundamental report (may be unavailable or not applicable): {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Use this information to present the bearish thesis, its catalysts, its invalidation conditions, and a direct rebuttal of the bull case.
"""

        prompt += get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
