from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.cftc_positioning import fetch_cftc_positioning
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.instrument_router import classify_instrument

_FOREX_RATE_PROXIES = {
    "USD": "FEDFUNDS",
    "EUR": "ECBMRRFR",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "CHF": "IR3TIB01CHM156N",
    "CAD": "IR3TIB01CAM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IRSTCI01NZM156N",
    "CNY": "IRSTCI01CNM156N",
    "CNH": "IRSTCI01CNM156N",
}

_COMMODITY_DRIVER_BASKETS = {
    "CL=F": ("DCOILWTICO", "DTWEXBGS", "DGS10", "INDPRO"),
    "BZ=F": ("DCOILBRENTEU", "DTWEXBGS", "DGS10", "INDPRO"),
    "NG=F": ("DHHNGSP", "DTWEXBGS", "INDPRO"),
    "GC=F": ("DTWEXBGS", "DGS10", "T10YIE"),
    "SI=F": ("DTWEXBGS", "DGS10", "T10YIE"),
    "PL=F": ("DTWEXBGS", "DGS10", "T10YIE"),
    "PA=F": ("DTWEXBGS", "DGS10", "T10YIE"),
    "HG=F": ("DTWEXBGS", "DGS10", "INDPRO"),
}
_DEFAULT_COMMODITY_DRIVERS = ("DTWEXBGS", "CPIAUCSL", "INDPRO")


def _fetch_series(series_id: str, curr_date: str, look_back_days: int | None) -> str:
    """Fetch one series without allowing optional context to abort the tool."""
    try:
        return route_to_vendor(
            "get_macro_indicators", series_id, curr_date, look_back_days
        )
    except Exception:
        return "DATA_UNAVAILABLE"


def _render_series(
    sources: list[tuple[str, str]], curr_date: str, look_back_days: int | None
) -> str:
    sections = []
    for label, series_id in sources:
        report = _fetch_series(series_id, curr_date, look_back_days)
        sections.append(f"### {label} ({series_id})\n{report}")
    return "\n\n".join(sections)


@tool
def get_macro_indicators(
    indicator: Annotated[
        str,
        "Macro indicator: a friendly alias such as 'cpi', 'core_pce', "
        "'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve', "
        "'real_gdp', 'vix', or a raw FRED series ID such as 'CPIAUCSL'.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format; the end of the window"],
    look_back_days: Annotated[
        int | None, "Trailing window length in days; omit for a 1-year window"
    ] = None,
) -> str:
    """
    Retrieve a macroeconomic indicator time series from FRED (Federal Reserve
    Economic Data): policy rates, Treasury yields, inflation, labor, and growth.
    Returns the series title, units, frequency, the latest value, the change
    over the window, and a recent observation table. Uses the configured
    macro_data vendor.

    Args:
        indicator (str): Friendly alias or raw FRED series ID
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Trailing window length; omit for a 1-year window

    Returns:
        str: A formatted markdown report of the macro series
    """
    return route_to_vendor("get_macro_indicators", indicator, curr_date, look_back_days)


@tool
def get_cross_asset_context(
    ticker: str, curr_date: str, look_back_days: int | None = 180
) -> str:
    """Return rate-proxy context for forex or macro drivers for commodities."""
    profile = classify_instrument(ticker)

    if profile.asset_class == "forex":
        pair = profile.canonical_symbol.removesuffix("=X")
        base, quote = pair[:3], pair[3:6]
        notes = []
        sources = []
        seen_series = set()

        for role, currency in (("Base", base), ("Quote", quote)):
            series_id = _FOREX_RATE_PROXIES.get(currency)
            if series_id is None:
                notes.append(f"{role} currency {currency}: no rate proxy is mapped.")
                continue
            if currency == "CNH":
                notes.append(
                    "CNH uses the CNY rate proxy IRSTCI01CNM156N; it is not an "
                    "offshore-CNH-specific rate series."
                )
            if series_id in seen_series:
                notes.append(
                    f"{role} currency {currency} shares the already listed "
                    f"{series_id} proxy."
                )
                continue
            seen_series.add(series_id)
            sources.append((f"{role} {currency} rate proxy", series_id))

        if "DTWEXBGS" not in seen_series:
            sources.append(("Trade-weighted US dollar", "DTWEXBGS"))

        header = (
            f"## Cross-asset forex context: {profile.canonical_symbol}\n"
            "These policy/money-market series are rate proxies, not exact forward "
            "carry. International series may lag; observation dates matter."
        )
        note_text = "" if not notes else "\n\n" + "\n".join(f"- {note}" for note in notes)
        fred_context = (
            f"{header}{note_text}\n\n"
            f"{_render_series(sources, curr_date, look_back_days)}"
        )
        return (
            f"{fred_context}\n\n## CFTC positioning\n"
            f"{fetch_cftc_positioning(ticker, curr_date)}"
        )

    if profile.asset_class == "commodity":
        symbol = profile.analysis_symbol
        series_ids = _COMMODITY_DRIVER_BASKETS.get(
            symbol, _DEFAULT_COMMODITY_DRIVERS
        )
        sources = [("Driver", series_id) for series_id in series_ids]
        header = (
            f"## Cross-asset commodity context: {symbol}\n"
            "Inventory, physical supply-demand, and term structure are NOT provided "
            "by this tool."
        )
        fred_context = f"{header}\n\n{_render_series(sources, curr_date, look_back_days)}"
        return (
            f"{fred_context}\n\n## CFTC positioning\n"
            f"{fetch_cftc_positioning(ticker, curr_date)}"
        )

    return (
        f"NOT_APPLICABLE: {profile.canonical_symbol} is not classified as forex "
        "or commodity."
    )
