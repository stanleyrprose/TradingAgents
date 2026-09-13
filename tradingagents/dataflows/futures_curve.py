"""Narrow, point-in-time-safe futures curves from Yahoo Finance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

import yfinance as yf

from tradingagents.instrument_router import classify_instrument

logger = logging.getLogger(__name__)

_ALL_MONTHS = "FGHJKMNQUVXZ"
_CONTRACT_SPECS = {
    "GC=F": ("GC", ".CMX", "GJMQVZ"),
    "SI=F": ("SI", ".CMX", "HKNUZ"),
    "HG=F": ("HG", ".CMX", "HKNUZ"),
    "CL=F": ("CL", ".NYM", _ALL_MONTHS),
    "NG=F": ("NG", ".NYM", _ALL_MONTHS),
    "ES=F": ("ES", ".CME", "HMUZ"),
    "NQ=F": ("NQ", ".CME", "HMUZ"),
    "YM=F": ("YM", ".CBT", "HMUZ"),
    "RTY=F": ("RTY", ".CME", "HMUZ"),
    "ZB=F": ("ZB", ".CBT", "HMUZ"),
    "ZN=F": ("ZN", ".CBT", "HMUZ"),
    "ZF=F": ("ZF", ".CBT", "HMUZ"),
    "ZT=F": ("ZT", ".CBT", "HMUZ"),
    "6E=F": ("6E", ".CME", "HMUZ"),
    "6J=F": ("6J", ".CME", "HMUZ"),
    "6B=F": ("6B", ".CME", "HMUZ"),
    "6A=F": ("6A", ".CME", "HMUZ"),
    "6C=F": ("6C", ".CME", "HMUZ"),
    "6S=F": ("6S", ".CME", "HMUZ"),
}
_MONTH_CODES = {code: month for month, code in enumerate(_ALL_MONTHS, start=1)}

_COMMODITY_ROOTS = frozenset({"GC=F", "SI=F", "HG=F", "CL=F", "NG=F"})
_INDEX_ROOTS = frozenset({"ES=F", "NQ=F", "YM=F", "RTY=F"})
_TREASURY_ROOTS = frozenset({"ZB=F", "ZN=F", "ZF=F", "ZT=F"})
_FX_ROOTS = frozenset({"6E=F", "6J=F", "6B=F", "6A=F", "6C=F", "6S=F"})

_FAMILY_DETAILS = {
    "commodity": ("Commodity futures curve", None),
    "index": (
        "Index futures curve",
        "Index futures curve is not spot-index fair-value basis; dividends and financing matter.",
    ),
    "treasury": (
        "U.S. Treasury futures curve",
        "Treasury delivery basket, cheapest-to-deliver, and delivery option matter; this is not "
        "cash-bond basis/fair value.",
    ),
    "fx": (
        "FX futures curve",
        "FX futures curve is not exact OTC forward points or realized carry.",
    ),
}


@dataclass(frozen=True)
class _Contract:
    symbol: str
    delivery: date


@dataclass(frozen=True)
class _Observation:
    contract: _Contract
    close: float
    last_bar: date
    volume_5bar: float


def _unavailable(reason: str) -> str:
    return f"<futures curve unavailable: {reason}>"


@lru_cache(maxsize=128)
def _fetch_contract_history(contract: str, end_date: date):
    """Fetch a short history window; exceptions and empty results are not cached."""
    history = yf.Ticker(contract).history(
        start=(end_date - timedelta(days=10)).isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        auto_adjust=False,
    )
    if history is None or history.empty:
        raise ValueError("Yahoo returned no contract history")
    return history


def _listed_contracts(symbol: str, start: date) -> list[_Contract]:
    """Return at most six listed deliveries from the start month through +18 months."""
    root, suffix, listed_codes = _CONTRACT_SPECS[symbol]
    codes_by_month = {month: code for code, month in _MONTH_CODES.items()}
    contracts = []
    for offset in range(19):
        month_index = start.year * 12 + start.month - 1 + offset
        year, zero_based_month = divmod(month_index, 12)
        month = zero_based_month + 1
        code = codes_by_month[month]
        if code not in listed_codes:
            continue
        contracts.append(
            _Contract(
                symbol=f"{root}{code}{year % 100:02d}{suffix}",
                delivery=date(year, month, 1),
            )
        )
        if len(contracts) == 6:
            break
    return contracts


def _bar_date(value) -> date:
    """Normalize a yfinance index value to a calendar date."""
    if hasattr(value, "date"):
        result = value.date()
        if isinstance(result, date):
            return result
    return date.fromisoformat(str(value)[:10])


def _contract_observation(contract: _Contract, end_date: date) -> _Observation | None:
    history = _fetch_contract_history(contract.symbol, end_date)
    if "Close" not in history.columns or "Volume" not in history.columns:
        return None

    eligible = []
    for index, row in history.iterrows():
        try:
            bar_date = _bar_date(index)
            close = float(row["Close"])
            volume = float(row["Volume"])
        except (TypeError, ValueError, OverflowError):
            continue
        if bar_date <= end_date and close == close:
            eligible.append((bar_date, close, volume))

    if not eligible:
        return None
    eligible.sort(key=lambda item: item[0])
    last_bar, close, _volume = eligible[-1]
    if close <= 0 or (end_date - last_bar).days > 7:
        return None
    volume_5bar = sum(
        volume for _day, _close, volume in eligible[-5:] if volume == volume and volume > 0
    )
    return _Observation(contract, close, last_bar, volume_5bar)


def _number(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _volume(value: float) -> str:
    return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"


def _classification(spread: float) -> str:
    if spread > 0.10:
        return "Contango"
    if spread < -0.10:
        return "Backwardation"
    return "Flat"


def _spread(front: _Observation, later: _Observation) -> float:
    return (later.close / front.close - 1.0) * 100.0


def _annualized(spread: float, front: _Observation, later: _Observation) -> float:
    distance = (later.contract.delivery - front.contract.delivery).days
    return spread * 365.0 / distance


def _family(symbol: str) -> str:
    if symbol in _COMMODITY_ROOTS:
        return "commodity"
    if symbol in _INDEX_ROOTS:
        return "index"
    if symbol in _TREASURY_ROOTS:
        return "treasury"
    if symbol in _FX_ROOTS:
        return "fx"
    raise ValueError(f"unknown futures curve family for {symbol}")


def _supported_roots() -> str:
    return ", ".join(_CONTRACT_SPECS)


def fetch_futures_curve(ticker: str, end_date: str) -> str:
    """Return a narrow Yahoo futures curve without aborting the graph."""
    try:
        end = date.fromisoformat(end_date)
    except (TypeError, ValueError):
        return _unavailable("invalid end_date; expected YYYY-MM-DD")

    try:
        profile = classify_instrument(ticker)
        symbol = profile.analysis_symbol
        if symbol == "BZ=F":
            return _unavailable(
                "BZ=F/Brent is explicitly unsupported; the WTI curve is not used as a proxy"
            )
        supported_commodity_proxy = (
            profile.asset_class == "commodity" and symbol in _COMMODITY_ROOTS
        )
        if symbol not in _CONTRACT_SPECS or not (
            profile.instrument_kind == "future" or supported_commodity_proxy
        ):
            return _unavailable(
                f"not available/applicable for {profile.canonical_symbol}; "
                f"supported Yahoo roots are {_supported_roots()}"
            )

        valid = []
        for contract in _listed_contracts(symbol, end):
            try:
                observation = _contract_observation(contract, end)
            except Exception as exc:
                logger.debug(
                    "Futures contract history unavailable for %s: %s",
                    contract.symbol,
                    type(exc).__name__,
                )
                continue
            if observation is not None:
                valid.append(observation)

        family = _family(symbol)
        minimum_contracts = 2 if family == "treasury" else 3
        if len(valid) < minimum_contracts:
            required = "two" if family == "treasury" else "three"
            return _unavailable(
                f"fewer than {required} fresh, positive-price contracts were available for {symbol}"
            )

        anchor = max(valid[:4], key=lambda observation: observation.volume_5bar)
        later = [item for item in valid if item.contract.delivery > anchor.contract.delivery]
        limited_depth = family == "treasury" and len(later) == 1
        later_required = 1 if limited_depth else 2
        if len(later) < later_required:
            required = "one" if limited_depth else "two"
            return _unavailable(
                f"fewer than {required} later valid contracts were available after the active "
                f"{symbol} anchor"
            )
        selected = [anchor, *later[:later_required]]
        next_spread = _spread(selected[0], selected[1])

        family_header, family_caution = _FAMILY_DETAILS[family]
        lines = [f"{family_header} — {symbol} (as of {end.isoformat()})"]
        if profile.instrument_kind != "future":
            lines.append(
                "PROXY WARNING: The requested alias/derivative "
                f"{profile.raw_symbol.strip()} is analyzed using the {symbol} futures curve proxy."
            )
        lines.extend(
            [
                (
                    "Active/front anchor is a liquidity approximation: highest cumulative "
                    "5-bar volume among the first four valid delivery contracts."
                ),
            ]
        )
        if limited_depth:
            lines.append(
                "LIMITED CURVE DEPTH: only one later fresh contract after the active anchor "
                "was available; next2 metrics are unavailable."
            )
        lines.extend(
            [
                "Contract | Delivery | Close | Last bar | 5-bar volume",
                "--- | --- | ---: | --- | ---:",
            ]
        )
        lines.extend(
            f"{item.contract.symbol} | {item.contract.delivery:%Y-%m} | "
            f"{_number(item.close)} | {item.last_bar.isoformat()} | "
            f"{_volume(item.volume_5bar)}"
            for item in selected
        )
        lines.append(f"Next vs front spread: {next_spread:+.2f}%")
        if not limited_depth:
            next2_spread = _spread(selected[0], selected[2])
            lines.append(f"Next2 vs front spread: {next2_spread:+.2f}%")
        lines.extend(
            [
                f"Front-to-next curve classification: {_classification(next_spread)}",
                (
                    "Annualized slope proxy (next vs front): "
                    f"{_annualized(next_spread, selected[0], selected[1]):+.2f}%"
                ),
            ]
        )
        if not limited_depth:
            lines.append(
                "Annualized slope proxy (next2 vs front): "
                f"{_annualized(next2_spread, selected[0], selected[2]):+.2f}%"
            )
        if family_caution is not None:
            lines.append(family_caution)
        lines.append(
            "These are curve slope/roll proxies, NOT cash basis, cost-of-carry fair "
            "value, or guaranteed roll return."
        )
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Futures curve fetch failed: %s", type(exc).__name__)
        detail = str(exc).strip()
        reason = type(exc).__name__ if not detail else f"{type(exc).__name__}: {detail}"
        return _unavailable(reason)
