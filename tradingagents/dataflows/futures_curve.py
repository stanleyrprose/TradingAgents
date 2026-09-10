"""Narrow, point-in-time-safe commodity futures curves from Yahoo Finance."""

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
}
_MONTH_CODES = {code: month for month, code in enumerate(_ALL_MONTHS, start=1)}


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


def fetch_futures_curve(ticker: str, end_date: str) -> str:
    """Return a narrow Yahoo commodity futures curve without aborting the graph."""
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
        if profile.asset_class != "commodity" or symbol not in _CONTRACT_SPECS:
            return _unavailable(
                f"not available/applicable for {profile.canonical_symbol}; "
                "supported Yahoo roots are GC=F, SI=F, HG=F, CL=F, and NG=F"
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

        if len(valid) < 3:
            return _unavailable(
                f"fewer than three fresh, positive-price contracts were available for {symbol}"
            )

        anchor = max(valid[:4], key=lambda observation: observation.volume_5bar)
        later = [item for item in valid if item.contract.delivery > anchor.contract.delivery]
        if len(later) < 2:
            return _unavailable(
                f"fewer than two later valid contracts were available after the active {symbol} anchor"
            )
        selected = [anchor, *later[:2]]
        next_spread = _spread(selected[0], selected[1])
        next2_spread = _spread(selected[0], selected[2])

        lines = [f"Commodity futures curve — {symbol} (as of {end.isoformat()})"]
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
        lines.extend(
            [
                f"Next vs front spread: {next_spread:+.2f}%",
                f"Next2 vs front spread: {next2_spread:+.2f}%",
                f"Front-to-next curve classification: {_classification(next_spread)}",
                (
                    "Annualized slope proxy (next vs front): "
                    f"{_annualized(next_spread, selected[0], selected[1]):+.2f}%"
                ),
                (
                    "Annualized slope proxy (next2 vs front): "
                    f"{_annualized(next2_spread, selected[0], selected[2]):+.2f}%"
                ),
                (
                    "These are curve slope/roll proxies, NOT cash basis, cost-of-carry fair "
                    "value, or guaranteed roll return."
                ),
            ]
        )
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Futures curve fetch failed: %s", type(exc).__name__)
        detail = str(exc).strip()
        reason = type(exc).__name__ if not detail else f"{type(exc).__name__}: {detail}"
        return _unavailable(reason)
