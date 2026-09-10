"""Deterministic, network-free instrument classification."""

import re
from dataclasses import dataclass

from tradingagents.dataflows.symbol_utils import crypto_base, normalize_symbol

# ``bond`` remains an accepted legacy spelling, but profiles expose
# ``fixed_income`` as their primary type.
PRIMARY_TYPES = (
    "stock",
    "crypto",
    "forex",
    "commodity",
    "index",
    "fixed_income",
    "rate",
    "fund",
    "option",
    "future",
    "derivative",
    "volatility",
    "bond",
)

_OPTION = re.compile(r"^(?P<root>[A-Z]{1,6})\d{6}[CP]\d{8}$")
_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")
_COMMODITY_ALIASES = frozenset(
    {
        "XAUUSD",
        "XAU",
        "GOLD",
        "XAGUSD",
        "XAG",
        "SILVER",
        "XPTUSD",
        "XPDUSD",
        "WTICOUSD",
        "USOIL",
        "WTI",
        "BCOUSD",
        "UKOIL",
        "BRENT",
        "NATGAS",
        "XNGUSD",
        "COPPER",
        "XCUUSD",
    }
)
_COMMODITY_FUTURES = frozenset(
    {
        "GC=F",
        "SI=F",
        "PL=F",
        "PA=F",
        "CL=F",
        "BZ=F",
        "NG=F",
        "HG=F",
        "ZC=F",
        "ZW=F",
        "ZS=F",
        "KC=F",
        "SB=F",
        "CT=F",
        "CC=F",
        "OJ=F",
    }
)
_INDEX_CFDS = frozenset(
    {
        "SPX500",
        "US500",
        "SPX",
        "NAS100",
        "US100",
        "USTEC",
        "US30",
        "DJI30",
        "WS30",
        "GER40",
        "GER30",
        "DE40",
        "UK100",
        "JP225",
        "JPN225",
        "FRA40",
        "EU50",
        "HK50",
    }
)
_DIRECT_INDICES = frozenset(
    {"^GSPC", "^NDX", "^DJI", "^RUT", "^FTSE", "^N225", "^HSI", "^GDAXI", "^FCHI", "^STOXX50E"}
)
_YAHOO_RATES = frozenset({"^IRX", "^FVX", "^TNX", "^TYX"})
_FRED_RATES = frozenset({"DGS1", "DGS2", "DGS5", "DGS10", "DGS30", "FEDFUNDS", "SOFR"})
_VOLATILITY_INDICES = frozenset({"^VIX", "^VXN", "^VXD", "^RVX", "^MOVE"})
_INDEX_FUTURES = frozenset({"ES=F", "NQ=F", "YM=F", "RTY=F"})
_FIXED_INCOME_FUTURES = frozenset({"ZB=F", "ZN=F", "ZF=F", "ZT=F"})
_FOREX_FUTURES = frozenset({"6E=F", "6J=F", "6B=F", "6A=F", "6C=F", "6S=F"})

# Small and intentionally explicit: detecting arbitrary funds requires network metadata.
_FUNDS = {
    **dict.fromkeys(("SPY", "QQQ", "DIA", "IWM", "VOO", "VTI"), "equity"),
    **dict.fromkeys(("TLT", "IEF", "SHY", "HYG", "LQD", "BND", "AGG"), "fixed_income"),
    **dict.fromkeys(("GLD", "SLV", "USO", "UNG", "DBC"), "commodity"),
    **dict.fromkeys(("IBIT", "FBTC", "BITO", "ETHA", "ETHE"), "crypto"),
    "UUP": "forex",
}


@dataclass(frozen=True)
class InstrumentProfile:
    raw_symbol: str
    canonical_symbol: str
    analysis_symbol: str
    primary_type: str
    asset_class: str
    instrument_kind: str
    pipeline_asset_type: str
    capability: str
    analysts: tuple[str, ...]
    notes: str
    can_run: bool


def _infer(raw: str, canonical: str) -> tuple[str, str, str]:
    """Return primary type, asset class, and instrument kind."""
    if _OPTION.fullmatch(raw):
        return "option", "equity", "option"
    if crypto_base(raw):
        return "crypto", "crypto", "spot"
    if canonical.endswith("=X"):
        return "forex", "forex", "spot"
    if raw in _COMMODITY_ALIASES:
        return "commodity", "commodity", "derivative"
    if raw in _COMMODITY_FUTURES:
        return "commodity", "commodity", "future"
    if raw in _INDEX_CFDS:
        return "derivative", "index", "cfd"
    if raw in _DIRECT_INDICES:
        return "index", "index", "index"
    if raw in _YAHOO_RATES or raw in _FRED_RATES:
        return "rate", "rate", "yield"
    if raw in _VOLATILITY_INDICES:
        return "volatility", "volatility", "index"
    if _ISIN.fullmatch(raw):
        return "fixed_income", "fixed_income", "bond"
    if raw in _FUNDS:
        return "fund", _FUNDS[raw], "etf"
    if raw in _INDEX_FUTURES:
        return "future", "index", "future"
    if raw in _FIXED_INCOME_FUTURES:
        return "future", "fixed_income", "future"
    if raw in _FOREX_FUTURES:
        return "future", "forex", "future"
    if raw.endswith("=F"):
        return "future", "other", "future"
    return "stock", "equity", "stock"


def _override_kind(primary: str) -> str:
    return {
        "stock": "stock",
        "crypto": "spot",
        "forex": "spot",
        "commodity": "derivative",
        "index": "index",
        "fixed_income": "bond",
        "rate": "yield",
        "fund": "fund",
        "option": "option",
        "future": "future",
        "derivative": "derivative",
        "volatility": "index",
    }[primary]


def _analytics(primary: str, asset_class: str, kind: str) -> tuple[str, tuple[str, ...], str]:
    ordinary_stock = primary == "stock" and asset_class == "equity" and kind == "stock"
    crypto_spot = primary == "crypto" and asset_class == "crypto" and kind == "spot"
    if ordinary_stock:
        return "FULL", ("market", "social", "news", "fundamentals"), ""
    if crypto_spot:
        return "FULL", ("market", "social", "news"), ""

    analysts = ("market", "social", "news") if primary == "fund" and asset_class in {
        "equity",
        "crypto",
    } else ("market", "news")
    missing = []
    if primary == "fund":
        missing.append("ETF/fund holdings, flows, and premium-discount analytics")
    if primary == "index" or asset_class == "index":
        missing.append("index breadth and constituent internals")
    if primary == "forex" or asset_class == "forex":
        missing.append("dedicated forex forward points/realized carry analytics")
    if primary == "rate" or asset_class == "rate":
        missing.append("rate curve and term-premium analytics")
    if primary == "volatility" or asset_class == "volatility":
        missing.append("volatility term structure and skew analytics")
    if primary == "fixed_income" or asset_class == "fixed_income":
        missing.append("fixed-income duration and spread analytics")
    if kind == "option":
        missing.append("option Greeks, chain, and IV-surface analytics")
    if kind == "future":
        missing.append("futures curve, basis, and roll analytics")
    if primary == "commodity" or asset_class == "commodity":
        missing.append("commodity inventory, physical supply-demand, and term-structure analytics")
    if kind in {"cfd", "derivative"}:
        missing.append("derivative contract semantics")
    if not missing:
        missing.append("domain-specific analytics")
    return "PARTIAL", analysts, "Missing analytics: " + "; ".join(missing) + "."


def _refine_commodity_notes(notes: str, analysis_symbol: str) -> str:
    """Reflect the commodity curve and EIA coverage available to analysis."""
    curve_symbols = {"GC=F", "SI=F", "HG=F", "CL=F", "NG=F"}
    if analysis_symbol not in curve_symbols:
        return notes

    notes = notes.replace(
        "futures curve, basis, and roll analytics",
        "cash basis, contract-specific roll-cost, and fair-value analytics",
    )
    physical_gap = "comprehensive physical supply-demand analytics"
    if analysis_symbol not in {"CL=F", "NG=F"}:
        physical_gap = "commodity inventory and " + physical_gap
    return notes.replace(
        "commodity inventory, physical supply-demand, and term-structure analytics",
        physical_gap,
    )


def _refine_general_futures_notes(notes: str, analysis_symbol: str) -> str:
    """Reflect curve coverage while retaining contract-family analytics gaps."""
    supported = _INDEX_FUTURES | _FIXED_INCOME_FUTURES | _FOREX_FUTURES
    if analysis_symbol not in supported:
        return notes

    notes = notes.replace(
        "futures curve, basis, and roll analytics",
        "cash basis, contract-specific roll-cost, and fair-value analytics",
    )
    if analysis_symbol in _FIXED_INCOME_FUTURES:
        notes = notes.replace(
            "fixed-income duration and spread analytics",
            "fixed-income duration and spread analytics; delivery-basket and "
            "cheapest-to-deliver (CTD) analytics",
        )
    return notes


def classify_instrument(raw_symbol, override_type=None):
    """Classify *raw_symbol* without market-data or other network access."""
    raw_value = str(raw_symbol)
    raw = raw_value.strip().upper().rstrip("+")
    canonical = normalize_symbol(raw_symbol)
    inferred_primary, asset_class, inferred_kind = _infer(raw, canonical)

    if override_type is not None:
        override = str(override_type).lower()
        if override not in PRIMARY_TYPES:
            raise ValueError("override_type must be one of: " + ", ".join(PRIMARY_TYPES))
        primary = "fixed_income" if override == "bond" else override
        kind = _override_kind(primary)
        if primary == "fund":
            asset_class = _FUNDS.get(raw, "other")
        elif primary == "commodity" and inferred_primary == "commodity" and inferred_kind == "future":
            kind = inferred_kind
            asset_class = "commodity"
        elif primary in {"option", "future", "derivative"}:
            meaningful_classes = {
                "equity",
                "crypto",
                "forex",
                "commodity",
                "fixed_income",
                "index",
                "volatility",
            }
            if (inferred_primary == "stock" and primary != "option") or asset_class not in meaningful_classes:
                asset_class = "other"
        else:
            asset_class = {
                "stock": "equity",
                "crypto": "crypto",
                "forex": "forex",
                "commodity": "commodity",
                "index": "index",
                "fixed_income": "fixed_income",
                "rate": "rate",
                "volatility": "volatility",
            }.get(primary, asset_class)
    else:
        primary, kind = inferred_primary, inferred_kind

    option_match = _OPTION.fullmatch(raw)
    analysis_symbol = option_match.group("root") if option_match else canonical
    use_crypto_proxy = asset_class == "crypto" and primary in {"option", "derivative"}
    crypto_spot = primary == "crypto" and asset_class == "crypto" and kind == "spot"
    pipeline_asset_type = "crypto" if crypto_spot or use_crypto_proxy else "stock"
    capability, analysts, notes = _analytics(primary, asset_class, kind)
    is_equity_occ_option = (
        option_match is not None
        and primary == "option"
        and asset_class == "equity"
        and kind == "option"
    )
    if is_equity_occ_option:
        notes = notes.replace(
            "option Greeks, chain, and IV-surface analytics",
            "historical option-chain/IV-surface analytics; model-independent "
            "option fair-value/scenario analytics",
        )
        notes += " Current Cboe delayed Greeks/IV/liquidity context is available."
    if asset_class == "commodity":
        notes = _refine_commodity_notes(notes, analysis_symbol)
    notes = _refine_general_futures_notes(notes, analysis_symbol)

    if option_match:
        notes += " Analysis uses the equity underlying as a proxy for the OCC contract."
    elif use_crypto_proxy:
        notes += " Analysis uses the crypto underlying as a proxy for the requested contract."

    can_run = raw not in _FRED_RATES and _ISIN.fullmatch(raw) is None
    if raw in _FRED_RATES:
        notes += " Cannot run yet: the current graph expects price bars, not a FRED rate series."
    elif _ISIN.fullmatch(raw):
        notes += " Cannot run yet: the current graph has no ISIN bond price-bar pipeline."

    return InstrumentProfile(
        raw_symbol=raw_value,
        canonical_symbol=canonical,
        analysis_symbol=analysis_symbol,
        primary_type=primary,
        asset_class=asset_class,
        instrument_kind=kind,
        pipeline_asset_type=pipeline_asset_type,
        capability=capability,
        analysts=analysts,
        notes=notes.strip(),
        can_run=can_run,
    )
