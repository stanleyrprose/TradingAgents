"""Official StockTwits sentiment provider.

TradingAgents uses StockTwits' authenticated Firestream Sentiment V2 endpoint
when credentials are configured. The former unauthenticated per-symbol stream is
not used as an automated fallback: it now commonly returns HTTP 403 and current
StockTwits terms require automated access to use an approved API.

Credentials are read from ``STOCKTWITS_USERNAME`` and
``STOCKTWITS_PASSWORD``. They are used only to build the HTTP Basic
Authorization header and are never logged.

The public function keeps its historical ``fetch_stocktwits_messages`` name for
call-site compatibility, but the preferred official source returns aggregate
sentiment and message-volume metrics rather than individual post bodies.
"""

from __future__ import annotations

import base64
import http.client
import json
import logging
import os
from datetime import datetime, timezone
from urllib.request import Request, urlopen

from .symbol_utils import crypto_base

logger = logging.getLogger(__name__)

_SENTIMENT_API = (
    "https://api-gw-prd.stocktwits.com/api-middleware/external/"
    "sentiment/v2/{ticker}/detail"
)
_UA = "tradingagents/0.4 (+https://github.com/TauricResearch/TradingAgents)"


def _stocktwits_symbol(ticker: str) -> str:
    """Map a crypto pair to StockTwits' ``<BASE>.X`` convention."""
    base = crypto_base(ticker)
    return f"{base}.X" if base else ticker.strip().upper()


def _firestream_credentials() -> tuple[str, str] | None:
    """Return configured Firestream credentials without logging them."""
    username = os.getenv("STOCKTWITS_USERNAME")
    password = os.getenv("STOCKTWITS_PASSWORD")
    if username and password:
        return username, password
    return None


def _fetch_authenticated_sentiment(ticker: str, timeout: float) -> str | None:
    """Fetch and format official StockTwits aggregate sentiment."""
    credentials = _firestream_credentials()
    if credentials is None:
        return None

    username, password = credentials
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    symbol = _stocktwits_symbol(ticker)
    url = _SENTIMENT_API.format(ticker=symbol)
    req = Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "application/json",
            "Authorization": f"Basic {token}",
        },
    )

    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning(
            "Authenticated StockTwits sentiment failed for %s: %s",
            ticker,
            type(exc).__name__,
        )
        return None

    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    sentiment = data.get("sentiment", {}) if isinstance(data, dict) else {}
    volume = data.get("messageVolume", {}) if isinstance(data, dict) else {}
    timeframes = data.get("timeframes", {}) if isinstance(data, dict) else {}
    if not sentiment and not timeframes:
        return None

    lines = [
        f"Official StockTwits sentiment for ${symbol} "
        "(authenticated Firestream)"
    ]
    for period in ("now", "15m", "24h"):
        sent = sentiment.get(period, {}) if isinstance(sentiment, dict) else {}
        vol = volume.get(period, {}) if isinstance(volume, dict) else {}
        if sent or vol:
            lines.append(
                f"{period}: sentiment={sent.get('labelNormalized', 'NA')} "
                f"({sent.get('valueNormalized', 'NA')}), "
                f"message_volume={vol.get('labelNormalized', 'NA')} "
                f"({vol.get('valueNormalized', 'NA')})"
            )

    for period in ("1D", "1W", "1M"):
        frame = timeframes.get(period, {}) if isinstance(timeframes, dict) else {}
        sent = frame.get("sentiment", {}) if isinstance(frame, dict) else {}
        if sent:
            lines.append(
                f"{period}: sentiment={sent.get('labelNormalized', 'NA')} "
                f"({sent.get('valueNormalized', 'NA')})"
            )

    return "\n".join(lines)


def fetch_stocktwits_messages(
    ticker: str,
    limit: int = 30,
    timeout: float = 10.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Return current StockTwits aggregate sentiment for ``ticker``.

    The official Sentiment V2 detail endpoint is current-state data. Historical
    windows therefore return a placeholder instead of leaking today's sentiment
    into a backtest. ``limit`` is retained for API compatibility with existing
    TradingAgents call sites.
    """
    del limit

    if start_date and end_date:
        today = datetime.now(timezone.utc).date().isoformat()
        if end_date < today:
            return (
                f"<no StockTwits sentiment for ${ticker.upper()} within "
                f"{start_date}..{end_date} "
                "(official sentiment API is current-state only)>"
            )

    if _firestream_credentials() is None:
        return "<stocktwits unavailable: official API credentials not configured>"

    result = _fetch_authenticated_sentiment(ticker, timeout)
    if result is None:
        return "<stocktwits unavailable: authenticated API request failed>"
    return result
