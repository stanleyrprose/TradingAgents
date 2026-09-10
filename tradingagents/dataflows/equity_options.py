"""Current equity-option context from Cboe's delayed public chain."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime

import requests

logger = logging.getLogger(__name__)

_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{underlying}.json"
_UA = "tradingagents/0.4 (+https://github.com/TauricResearch/TradingAgents)"
_OCC_RE = re.compile(
    r"(?P<root>[A-Z]{1,6})(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})"
)
_NA = "DATA_UNAVAILABLE"


@dataclass(frozen=True)
class _OCCContract:
    symbol: str
    root: str
    expiry: date
    right: str
    strike: float


def _today() -> date:
    """Return the local calendar date; kept separate for deterministic tests."""
    return date.today()


def _unavailable(reason: str) -> str:
    return f"<equity option context unavailable: {reason}>"


def _parse_occ(symbol: object) -> _OCCContract | None:
    if not isinstance(symbol, str):
        return None
    match = _OCC_RE.fullmatch(symbol)
    if match is None:
        return None
    try:
        expiry = datetime.strptime(match["expiry"], "%y%m%d").date()
    except ValueError:
        return None
    return _OCCContract(
        symbol=symbol,
        root=match["root"],
        expiry=expiry,
        right=match["right"],
        strike=int(match["strike"]) / 1000.0,
    )


def _finite(value: object, *, positive: bool = False) -> float | None:
    try:
        number = float(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
    if number is None or not math.isfinite(number):
        return None
    if positive and number <= 0:
        return None
    return number


def _display(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return _NA
    rendered = f"{value:,.{decimals}f}"
    if decimals:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered if rendered not in {"", "-0"} else "0"


def _percent(value: float | None, decimals: int = 2) -> str:
    return _NA if value is None else f"{value:.{decimals}f}%"


def _row_number(row: dict, field: str, *, positive: bool = False) -> float | None:
    return _finite(row.get(field), positive=positive)


def _ratio(put_value: float, call_value: float) -> str:
    if call_value <= 0:
        return _NA
    return _display(put_value / call_value, 3)


def _chain_metrics(
    options: list,
    selected: _OCCContract,
    current_price: float,
) -> tuple[
    dict[str, float],
    float | None,
    tuple[float, _OCCContract, float, float] | None,
    tuple[float, _OCCContract, float, float] | None,
]:
    totals = {"call_oi": 0.0, "put_oi": 0.0, "call_volume": 0.0, "put_volume": 0.0}
    nearest_iv: dict[str, tuple[float, float] | None] = {"C": None, "P": None}
    skew_choice: dict[str, tuple[float, _OCCContract, float, float] | None] = {
        "C": None,
        "P": None,
    }

    for row in options:
        if not isinstance(row, dict):
            continue
        contract = _parse_occ(row.get("option"))
        if contract is None or contract.expiry != selected.expiry:
            continue

        side = "call" if contract.right == "C" else "put"
        totals[f"{side}_oi"] += _row_number(row, "open_interest") or 0.0
        totals[f"{side}_volume"] += _row_number(row, "volume") or 0.0

        iv = _row_number(row, "iv", positive=True)
        if iv is None:
            continue
        strike_distance = abs(contract.strike - current_price)
        current_nearest = nearest_iv[contract.right]
        if current_nearest is None or strike_distance < current_nearest[0]:
            nearest_iv[contract.right] = (strike_distance, iv)

        delta = _row_number(row, "delta")
        if delta is None:
            continue
        target = 0.25 if contract.right == "C" else -0.25
        delta_distance = abs(delta - target)
        current_skew = skew_choice[contract.right]
        if current_skew is None or delta_distance < current_skew[0]:
            skew_choice[contract.right] = (delta_distance, contract, delta, iv)

    atm_values = [item[1] for item in nearest_iv.values() if item is not None]
    atm_iv = sum(atm_values) / len(atm_values) if atm_values else None
    return totals, atm_iv, skew_choice["P"], skew_choice["C"]


def _render_report(
    contract: _OCCContract,
    row: dict,
    data: dict,
    payload: dict,
    current_price: float,
    today: date,
    options: list,
) -> str:
    bid = _row_number(row, "bid", positive=True)
    ask = _row_number(row, "ask", positive=True)
    midpoint = (bid + ask) / 2 if bid is not None and ask is not None else None
    spread_pct = (ask - bid) / midpoint * 100 if midpoint is not None else None
    last = _row_number(row, "last_trade_price")
    intrinsic = (
        max(current_price - contract.strike, 0.0)
        if contract.right == "C"
        else max(contract.strike - current_price, 0.0)
    )
    price_proxy = midpoint if midpoint is not None else last
    time_value = max(price_proxy - intrinsic, 0.0) if price_proxy is not None else None
    totals, atm_iv, skew_put, skew_call = _chain_metrics(
        options, contract, current_price
    )

    if skew_put is not None and skew_call is not None:
        put_contract, put_delta, put_iv = skew_put[1:]
        call_contract, call_delta, call_iv = skew_call[1:]
        skew = (put_iv - call_iv) * 100
        skew_text = (
            f"{_display(skew, 2)} percentage points "
            f"(put {put_contract.symbol}, delta {_display(put_delta, 3)}; "
            f"call {call_contract.symbol}, delta {_display(call_delta, 3)})"
        )
    else:
        skew_text = _NA

    timestamp = data.get("timestamp", payload.get("timestamp"))
    timestamp_text = str(timestamp).strip() if timestamp is not None else ""
    if not timestamp_text:
        timestamp_text = _NA
    iv30 = _finite(data.get("iv30"), positive=True)
    option_type = "call" if contract.right == "C" else "put"
    dte = (contract.expiry - today).days

    lines = [
        f"Cboe delayed equity option context — {contract.symbol}",
        f"Contract: {contract.symbol}",
        f"Underlying: {contract.root}",
        f"Expiry: {contract.expiry.isoformat()}",
        f"Type: {option_type}",
        f"Strike: {_display(contract.strike, 3)}",
        f"DTE: {dte}",
        f"Source timestamp: {timestamp_text}",
        f"Underlying price: {_display(current_price, 4)}",
        f"Bid: {_display(bid)} | Ask: {_display(ask)}",
        f"Midpoint: {_display(midpoint)}",
        f"Bid-ask spread as % of midpoint: {_percent(spread_pct)}",
        f"Last: {_display(last)}",
        f"Last trade time: {row.get('last_trade_time') or _NA}",
        f"Open interest: {_display(_row_number(row, 'open_interest'), 0)}",
        f"Volume: {_display(_row_number(row, 'volume'), 0)}",
        f"IV: {_percent((_row_number(row, 'iv') or 0) * 100) if _row_number(row, 'iv') is not None else _NA}",
        "Greeks: "
        f"delta {_display(_row_number(row, 'delta'))}; "
        f"gamma {_display(_row_number(row, 'gamma'))}; "
        f"vega {_display(_row_number(row, 'vega'))}; "
        f"theta {_display(_row_number(row, 'theta'))}; "
        f"rho {_display(_row_number(row, 'rho'))}",
        f"Theoretical value: {_display(_row_number(row, 'theo'))}",
        f"Moneyness (S/K - 1): {_percent((current_price / contract.strike - 1) * 100)}",
        f"Intrinsic value: {_display(intrinsic)}",
        f"Time-value proxy: {_display(time_value)}",
        "Same-expiry chain totals: "
        f"call OI {_display(totals['call_oi'], 0)}; put OI {_display(totals['put_oi'], 0)}; "
        f"call volume {_display(totals['call_volume'], 0)}; "
        f"put volume {_display(totals['put_volume'], 0)}",
        f"Same-expiry P/C OI ratio: {_ratio(totals['put_oi'], totals['call_oi'])}",
        f"Same-expiry P/C volume ratio: {_ratio(totals['put_volume'], totals['call_volume'])}",
        f"ATM IV proxy: {_percent(atm_iv * 100) if atm_iv is not None else _NA}",
        f"25-delta put-minus-call IV skew proxy: {skew_text}",
        f"Cboe IV30: {_percent(iv30) if iv30 is not None else _NA}",
        "Cautions: this is a delayed snapshot, not realtime; Greeks and IV are "
        "vendor-calculated; positioning ratios are descriptive, not directional; "
        "stale last trades and wide or zero markets reduce reliability.",
    ]
    return "\n".join(lines)


def fetch_equity_option_context(
    ticker: str,
    end_date: str,
    timeout: float = 15.0,
) -> str:
    """Return current Cboe context for one exact OCC equity-option contract."""
    contract = _parse_occ(ticker)
    if contract is None:
        return _unavailable("invalid OCC option symbol")

    try:
        requested_date = date.fromisoformat(end_date)
    except (TypeError, ValueError):
        return _unavailable("invalid end_date; expected YYYY-MM-DD")

    today = _today()
    if requested_date != today:
        return _unavailable(
            "current Cboe delayed chain cannot be used for historical/future "
            "point-in-time analysis"
        )

    try:
        response = requests.get(
            _URL.format(underlying=contract.root),
            headers={"User-Agent": _UA},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("response payload is not an object")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("response data is not an object")
        options = data.get("options")
        if not isinstance(options, list):
            raise ValueError("response options is not a list")
        current_price = _finite(data.get("current_price"), positive=True)
        if current_price is None:
            raise ValueError("current_price is missing or non-positive")
        selected_row = next(
            (
                row
                for row in options
                if isinstance(row, dict) and row.get("option") == contract.symbol
            ),
            None,
        )
        if selected_row is None:
            return _unavailable(f"exact contract {contract.symbol} not found")
        return _render_report(
            contract, selected_row, data, payload, current_price, today, options
        )
    except (requests.RequestException, TypeError, ValueError, OverflowError) as exc:
        logger.warning("Cboe equity option fetch failed: %s", type(exc).__name__)
        return _unavailable(type(exc).__name__)
