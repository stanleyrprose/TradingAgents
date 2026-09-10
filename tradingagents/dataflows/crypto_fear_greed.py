"""Crypto Fear & Greed Index provider (Alternative.me)."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_API = "https://api.alternative.me/fng/?limit=0"
_UA = "tradingagents/0.4 (+https://github.com/TauricResearch/TradingAgents)"


def _to_date(ts: str | int) -> date:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).date()


def _unavailable(reason: str) -> str:
    return f"<crypto fear-greed unavailable: {reason}>"


def fetch_crypto_fear_greed(
    end_date: str | None = None,
    timeout: float = 10.0,
) -> str:
    """Return a point-in-time-safe Crypto Fear & Greed regime summary.

    The index is a broad crypto-market sentiment indicator, not an ETH-specific
    signal. For historical analysis, the latest observation on or before
    ``end_date`` is used so future sentiment cannot leak into backtests.
    """
    req = Request(_API, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (OSError, ValueError) as exc:
        logger.warning("Fear & Greed fetch failed: %s", type(exc).__name__)
        return _unavailable(type(exc).__name__)

    try:
        cutoff = date.fromisoformat(end_date) if end_date else None
    except (TypeError, ValueError):
        return _unavailable("invalid end_date")

    rows = payload.get("data", []) if isinstance(payload, dict) else []
    parsed: list[tuple[date, int, str]] = []
    for row in rows:
        try:
            point_date = _to_date(row["timestamp"])
            value = int(row["value"])
            label = str(row["value_classification"])
        except (KeyError, TypeError, ValueError):
            continue
        if cutoff and point_date > cutoff:
            continue
        parsed.append((point_date, value, label))

    if not parsed:
        return _unavailable("no point-in-time data")

    parsed.sort(key=lambda x: x[0], reverse=True)
    latest = parsed[0]

    def _at_or_before(days_back: int) -> tuple[str, int, str] | None:
        target = latest[0] - timedelta(days=days_back)
        for point_date, value, label in parsed[1:]:
            if point_date <= target:
                return point_date, value, label
        return None

    lines = [
        "Crypto Fear & Greed (Alternative.me) — BROAD MARKET REGIME, not ETH-specific",
        f"Latest at/before {latest[0].isoformat()}: {latest[1]}/100 — {latest[2]}",
    ]
    for label, days in (("7d prior", 7), ("30d prior", 30)):
        point = _at_or_before(days)
        if point:
            lines.append(
                f"{label} ({point[0].isoformat()}): {point[1]}/100 — {point[2]}"
            )
    return "\n".join(lines)
