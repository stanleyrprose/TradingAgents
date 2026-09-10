"""Tools for current equity-option context."""

from langchain_core.tools import tool

from tradingagents.dataflows.equity_options import fetch_equity_option_context


@tool
def get_equity_option_context(ticker: str, curr_date: str) -> str:
    """Return current Cboe delayed context for an exact OCC option symbol.

    Args:
        ticker: Exact OCC symbol, for example AAPL260116C00150000.
        curr_date: Analysis date in YYYY-MM-DD format.
    """
    return fetch_equity_option_context(ticker, curr_date)
