"""Deterministic, network-free instrument classification."""

import re
from dataclasses import dataclass

from tradingagents.dataflows.symbol_utils import crypto_base, normalize_symbol

PRIMARY_TYPES = ("stock", "crypto", "option", "future", "derivative", "bond", "commodity")
_OPTION = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")
_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")
_COMMODITY_ALIASES = frozenset({
    "XAUUSD", "XAU", "GOLD", "XAGUSD", "XAG", "SILVER", "XPTUSD", "XPDUSD", "WTICOUSD", "USOIL", "WTI",
    "BCOUSD", "UKOIL", "BRENT", "NATGAS", "XNGUSD", "COPPER", "XCUUSD",
})
_COMMODITY_FUTURES = frozenset({
    "GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "BZ=F", "NG=F", "HG=F",
    "ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F", "CT=F", "CC=F", "OJ=F",
})
_INDEX_CFDS = frozenset({
    "SPX500", "US500", "NAS100", "US100", "USTEC", "US30", "DJI30", "WS30",
    "GER40", "GER30", "DE40", "UK100", "JP225", "JPN225", "FRA40", "EU50", "HK50",
})


@dataclass(frozen=True)
class InstrumentProfile:
    raw_symbol: str
    canonical_symbol: str
    primary_type: str
    asset_class: str
    instrument_kind: str
    pipeline_asset_type: str
    capability: str
    analysts: tuple[str, ...]
    notes: str
    can_run: bool


def classify_instrument(raw_symbol, override_type=None):
    canonical = normalize_symbol(raw_symbol)
    raw = str(raw_symbol).strip().upper().rstrip("+")
    if override_type is not None:
        if override_type not in PRIMARY_TYPES:
            raise ValueError("override_type must be one of: " + ", ".join(PRIMARY_TYPES))
        primary = override_type
    elif _OPTION.fullmatch(raw):
        primary = "option"
    elif crypto_base(raw):
        primary = "crypto"
    elif raw in _COMMODITY_ALIASES or raw in _COMMODITY_FUTURES:
        primary = "commodity"
    elif raw in _INDEX_CFDS:
        primary = "derivative"
    elif raw in {"^TNX", "^FVX", "^TYX"} or _ISIN.fullmatch(raw):
        primary = "bond"
    elif raw.endswith("=F"):
        primary = "future"
    elif canonical.endswith("=X"):
        primary = "derivative"
    else:
        primary = "stock"

    is_isin = bool(_ISIN.fullmatch(raw))
    if primary == "stock":
        asset_class, kind, capability, analysts, notes = "stock", "cash", "FULL", ("market", "social", "news", "fundamentals"), ""
    elif primary == "crypto":
        asset_class, kind, capability, analysts, notes = "crypto", "cash", "FULL", ("market", "social", "news"), ""
    elif primary == "bond":
        asset_class, kind, capability, analysts, notes = "bond", "bond", "PARTIAL", ("market", "news"), "Missing analytics: bond duration/yield-spread."
    elif primary == "commodity":
        asset_class, kind = "commodity", ("future" if raw in _COMMODITY_FUTURES else "derivative")
        capability, analysts, notes = "PARTIAL", ("market", "news"), "Missing analytics: commodity inventory/term structure."
    elif primary == "future":
        asset_class, kind, capability, analysts, notes = "other", "future", "PARTIAL", ("market", "news"), "Missing analytics: futures curve/roll."
    elif primary == "option":
        asset_class, kind, capability, analysts, notes = "stock", "option", "PARTIAL", ("market", "news"), "Missing analytics: option Greeks/chain."
    else:
        asset_class, kind, capability, analysts, notes = "other", "derivative", "PARTIAL", ("market", "news"), "Missing analytics: generic derivative contract semantics."
    return InstrumentProfile(str(raw_symbol), canonical, primary, asset_class, kind, "crypto" if primary == "crypto" else "stock", capability, analysts, notes, not (primary == "bond" and is_isin))
