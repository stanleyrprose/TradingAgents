"""Deribit public options positioning summary for crypto sentiment."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .symbol_utils import crypto_base

logger = logging.getLogger(__name__)

_API = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
_UA = "tradingagents/0.4 (+https://github.com/TauricResearch/TradingAgents)"
_SUPPORTED = {"BTC", "ETH"}


def _side(name: str) -> str | None:
    suffix = name.rsplit("-", 1)[-1].upper()
    if suffix in {"C", "P"}:
        return suffix
    return None


def _number(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fetch_deribit_options_sentiment(
    ticker: str,
    end_date: str | None = None,
    timeout: float = 10.0,
) -> str:
    """Return current Deribit options positioning for BTC/ETH without auth.

    The summary is a positioning proxy, not a directional forecast. It
    aggregates put/call open interest and 24h volume across active options.
    Because the public endpoint is current-state, historical runs do not consume
    today's options positioning.
    """
    base = crypto_base(ticker)
    if not base or base not in _SUPPORTED:
        return (
            f"<deribit options unavailable: unsupported symbol "
            f"${ticker.upper()}>"
        )

    try:
        requested_date = date.fromisoformat(end_date) if end_date else None
    except (TypeError, ValueError):
        return "<deribit options unavailable: invalid end_date>"

    today = datetime.now(timezone.utc).date()
    if requested_date and requested_date < today:
        return (
            f"<deribit options unavailable for ${ticker.upper()} on {end_date}: "
            "current-state market data would leak into a historical run>"
        )

    query = urlencode({"currency": base, "kind": "option"})
    req = Request(f"{_API}?{query}", headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (OSError, ValueError) as exc:
        logger.warning("Deribit options fetch failed: %s", type(exc).__name__)
        return f"<deribit options unavailable: {type(exc).__name__}>"

    rows = payload.get("result", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        return "<deribit options unavailable: no options data>"

    stats = {
        "C": {"oi": 0.0, "volume": 0.0},
        "P": {"oi": 0.0, "volume": 0.0},
    }
    iv_values: list[float] = []
    underlying: list[float] = []
    count = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        side = _side(str(row.get("instrument_name", "")))
        if side is None:
            continue
        count += 1
        stats[side]["oi"] += _number(row.get("open_interest")) or 0.0
        stats[side]["volume"] += _number(row.get("volume")) or 0.0

        mark_iv = _number(row.get("mark_iv"))
        if mark_iv is not None:
            iv_values.append(mark_iv)

        underlying_price = _number(row.get("underlying_price"))
        if underlying_price is not None:
            underlying.append(underlying_price)

    def _ratio(numerator: float, denominator: float) -> str:
        if denominator <= 0:
            return "NA"
        return f"{numerator / denominator:.2f}"

    put_call_oi = _ratio(stats["P"]["oi"], stats["C"]["oi"])
    put_call_volume = _ratio(stats["P"]["volume"], stats["C"]["volume"])
    avg_iv = sum(iv_values) / len(iv_values) if iv_values else None
    avg_underlying = sum(underlying) / len(underlying) if underlying else None

    lines = [
        f"Deribit {base} options — {base}-SPECIFIC DERIVATIVES POSITIONING",
        f"Active option instruments: {count}",
        f"Put/Call open-interest ratio: {put_call_oi}",
        f"Put/Call 24h volume ratio: {put_call_volume}",
        (
            f"Call OI: {stats['C']['oi']:.2f} {base} | "
            f"Put OI: {stats['P']['oi']:.2f} {base}"
        ),
        (
            f"Call 24h volume: {stats['C']['volume']:.2f} {base} | "
            f"Put 24h volume: {stats['P']['volume']:.2f} {base}"
        ),
    ]
    if avg_iv is not None:
        lines.append(f"Mean mark IV across listed options: {avg_iv:.2f}%")
    if avg_underlying is not None:
        lines.append(f"Mean underlying reference price: ${avg_underlying:,.2f}")
    lines.append(
        "Interpretation caution: put/call ratios are positioning proxies, "
        "not standalone bullish/bearish signals."
    )
    return "\n".join(lines)
