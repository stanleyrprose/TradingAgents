"""Current equity-option context from Cboe's delayed public chain."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import requests

from .option_scenarios import build_option_scenario_report

logger = logging.getLogger(__name__)

_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{underlying}.json"
_UA = "tradingagents/0.4 (+https://github.com/TauricResearch/TradingAgents)"
_OCC_RE = re.compile(
    r"(?P<root>[A-Z]{1,6})(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})"
)
_NA = "DATA_UNAVAILABLE"
_US_OPTION_MARKET_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class _OCCContract:
    symbol: str
    root: str
    expiry: date
    right: str
    strike: float


@dataclass(frozen=True)
class EquityOptionSnapshot:
    """Structured current delayed quote for one exact OCC equity option."""

    symbol: str
    underlying: str
    expiry: date
    right: str
    strike: float
    as_of: date
    dte: int
    source_timestamp: str | None
    underlying_spot: float
    bid: float | None
    ask: float | None
    midpoint: float | None
    spread_pct: float | None
    last: float | None
    iv: float | None
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    open_interest: float | None
    volume: float | None


@dataclass(frozen=True)
class EquityOptionSnapshotResult:
    """Structured exact-contract snapshot with fail-soft source semantics."""

    snapshot: EquityOptionSnapshot | None
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.snapshot is not None and self.unavailable_reason is None


def current_us_option_market_date(now: datetime | None = None) -> date:
    """Return the New York calendar date used by US equity-option sources."""
    if now is None:
        return datetime.now(_US_OPTION_MARKET_TZ).date()
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(_US_OPTION_MARKET_TZ).date()


def _today() -> date:
    """Return the current US option-market date; kept separate for deterministic tests."""
    return current_us_option_market_date()


def _source_market_date(data: dict, payload: dict) -> date | None:
    """Extract the source market date from Cboe's timestamp without assuming host timezone."""
    timestamp = data.get("timestamp", payload.get("timestamp"))
    if timestamp is None:
        return None
    text = str(timestamp).strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


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


def _nonnegative_number(row: dict, field: str) -> float | None:
    value = _row_number(row, field)
    return value if value is not None and value >= 0 else None


def _snapshot_from_row(
    contract: _OCCContract,
    row: dict,
    data: dict,
    payload: dict,
    spot: float,
    today: date,
) -> EquityOptionSnapshot:
    bid = _nonnegative_number(row, "bid")
    ask = _nonnegative_number(row, "ask")
    midpoint = None
    spread_pct = None
    if bid is not None and ask is not None and ask >= bid and bid + ask > 0:
        midpoint = (bid + ask) / 2
        if midpoint > 0:
            spread_pct = (ask - bid) / midpoint * 100

    timestamp = data.get("timestamp", payload.get("timestamp"))
    timestamp_text = str(timestamp).strip() if timestamp is not None else ""
    return EquityOptionSnapshot(
        symbol=contract.symbol,
        underlying=contract.root,
        expiry=contract.expiry,
        right=contract.right,
        strike=contract.strike,
        as_of=today,
        dte=(contract.expiry - today).days,
        source_timestamp=timestamp_text or None,
        underlying_spot=spot,
        bid=bid,
        ask=ask,
        midpoint=midpoint,
        spread_pct=spread_pct,
        last=_row_number(row, "last_trade_price"),
        iv=_row_number(row, "iv", positive=True),
        delta=_row_number(row, "delta"),
        gamma=_row_number(row, "gamma"),
        vega=_row_number(row, "vega"),
        theta=_row_number(row, "theta"),
        open_interest=_nonnegative_number(row, "open_interest"),
        volume=_nonnegative_number(row, "volume"),
    )


def fetch_equity_option_snapshots(
    tickers: tuple[str, ...] | list[str],
    end_date: str,
    timeout: float = 15.0,
) -> dict[str, EquityOptionSnapshotResult]:
    """Fetch exact OCC snapshots using at most one Cboe chain request per underlying."""

    requested = tuple(dict.fromkeys(tickers))
    results: dict[str, EquityOptionSnapshotResult] = {}
    contracts_by_root: dict[str, list[_OCCContract]] = {}
    for ticker in requested:
        contract = _parse_occ(ticker)
        if contract is None:
            results[ticker] = EquityOptionSnapshotResult(None, "invalid OCC option symbol")
            continue
        contracts_by_root.setdefault(contract.root, []).append(contract)

    try:
        requested_date = date.fromisoformat(end_date)
    except (TypeError, ValueError):
        reason = "invalid end_date; expected YYYY-MM-DD"
        for contracts in contracts_by_root.values():
            for contract in contracts:
                results[contract.symbol] = EquityOptionSnapshotResult(None, reason)
        return results

    today = _today()
    if requested_date != today:
        reason = (
            "current Cboe delayed chain cannot be used for historical/future point-in-time analysis"
        )
        for contracts in contracts_by_root.values():
            for contract in contracts:
                results[contract.symbol] = EquityOptionSnapshotResult(None, reason)
        return results

    for root, contracts in contracts_by_root.items():
        try:
            response = requests.get(
                _URL.format(underlying=root),
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
            source_date = _source_market_date(data, payload)
            if source_date is None:
                reason = "Cboe source timestamp is missing/unparseable; market date cannot be verified"
                for contract in contracts:
                    results[contract.symbol] = EquityOptionSnapshotResult(None, reason)
                continue
            if source_date != requested_date:
                reason = (
                    f"Cboe source market date {source_date.isoformat()} does not match requested "
                    f"US market date {requested_date.isoformat()}"
                )
                for contract in contracts:
                    results[contract.symbol] = EquityOptionSnapshotResult(None, reason)
                continue
            options = data.get("options")
            if not isinstance(options, list):
                raise ValueError("response options is not a list")
            spot = _finite(data.get("current_price"), positive=True)
            if spot is None:
                raise ValueError("current_price is missing or non-positive")
            rows = {
                item.get("option"): item
                for item in options
                if isinstance(item, dict) and isinstance(item.get("option"), str)
            }
            for contract in contracts:
                row = rows.get(contract.symbol)
                if row is None:
                    results[contract.symbol] = EquityOptionSnapshotResult(
                        None, f"exact contract {contract.symbol} not found"
                    )
                    continue
                results[contract.symbol] = EquityOptionSnapshotResult(
                    _snapshot_from_row(contract, row, data, payload, spot, today)
                )
        except (requests.RequestException, TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "Cboe equity option batch snapshot fetch failed for %s: %s",
                root,
                type(exc).__name__,
            )
            for contract in contracts:
                results[contract.symbol] = EquityOptionSnapshotResult(
                    None, type(exc).__name__
                )
    return results


def fetch_equity_option_snapshot(
    ticker: str,
    end_date: str,
    timeout: float = 15.0,
) -> EquityOptionSnapshotResult:
    """Return one structured current Cboe snapshot for an exact OCC contract.

    The function intentionally does not substitute midpoint/last for the bid. A
    long-position manager can therefore treat ``bid`` as a conservative delayed
    liquidation proxy without parsing the human-readable option context report.
    """

    return fetch_equity_option_snapshots([ticker], end_date, timeout=timeout)[ticker]


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
    positive_last = _row_number(row, "last_trade_price", positive=True)
    price_proxy = midpoint if midpoint is not None else positive_last
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
    premium = midpoint if midpoint is not None else positive_last
    iv = _row_number(row, "iv", positive=True)
    if premium is not None and iv is not None:
        scenario = build_option_scenario_report(
            symbol=contract.symbol,
            right=contract.right,
            strike=contract.strike,
            expiry=contract.expiry,
            as_of=today,
            spot=current_price,
            market_premium=premium,
            iv_decimal=iv,
            vendor_theo=_row_number(row, "theo"),
            vendor_delta=_row_number(row, "delta"),
            vendor_gamma=_row_number(row, "gamma"),
            vendor_theta=_row_number(row, "theta"),
        )
    else:
        missing = []
        if premium is None:
            missing.append("current premium reference")
        if iv is None:
            missing.append("positive IV")
        scenario = (
            "<deterministic option scenario unavailable: missing "
            + " and ".join(missing)
            + ">"
        )
    return "\n".join(lines) + "\n\n" + scenario


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
        source_date = _source_market_date(data, payload)
        if source_date is None:
            return _unavailable(
                "Cboe source timestamp is missing/unparseable; market date cannot be verified"
            )
        if source_date != requested_date:
            return _unavailable(
                f"Cboe source market date {source_date.isoformat()} does not match requested "
                f"US market date {requested_date.isoformat()}"
            )
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
