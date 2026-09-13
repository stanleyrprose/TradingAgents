"""Historical social sentiment must not leak current data into a backtest (#1220).

StockTwits and Reddit fetchers pull only recent items, so for a historical run
they must be trimmed to the analysis window (and yield a clear placeholder when
nothing qualifies) rather than showing today's chatter as if it were from the
as-of date. All three sources share dataflows.date_window.in_window.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tradingagents.dataflows import reddit, stocktwits
from tradingagents.dataflows.date_window import in_window


class _JsonResp:
    """Minimal urlopen() context-manager stub returning a JSON body."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


# --- shared window helper ---------------------------------------------------

@pytest.mark.unit
def test_in_window_bounds_and_exclusive_upper():
    start = datetime(2026, 5, 1)
    end = datetime(2026, 5, 9)
    assert in_window(datetime(2026, 5, 5, tzinfo=timezone.utc), start, end) is True
    assert in_window(datetime(2026, 5, 9, 23, 59, tzinfo=timezone.utc), start, end) is True
    # exactly midnight after end -> excluded (no leak)
    assert in_window(datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc), start, end) is False
    # offset-aware converted, not truncated: 05-10T01:00+05:00 == 05-09T20:00Z
    assert in_window(datetime.fromisoformat("2026-05-10T01:00:00+05:00"), start, end) is True


@pytest.mark.unit
def test_in_window_undated_excluded_in_backtest_kept_live():
    old = datetime(2026, 5, 9)
    assert in_window(None, datetime(2026, 5, 1), old) is False       # historical
    now = datetime.now(timezone.utc)
    assert in_window(None, now, now) is True                          # live


# --- StockTwits -------------------------------------------------------------

@pytest.mark.unit
def test_stocktwits_historical_window_never_uses_current_sentiment(monkeypatch):
    monkeypatch.setenv("STOCKTWITS_USERNAME", "user")
    monkeypatch.setenv("STOCKTWITS_PASSWORD", "pass")
    with monkeypatch.context() as mp:
        mp.setattr(
            stocktwits,
            "urlopen",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("historical StockTwits request should not hit network")
            ),
        )
        out = stocktwits.fetch_stocktwits_messages(
            "AAPL",
            start_date="2026-05-01",
            end_date="2026-05-08",
        )
    assert "no StockTwits sentiment" in out
    assert "2026-05-01..2026-05-08" in out
    assert "current-state only" in out


@pytest.mark.unit
def test_stocktwits_live_call_requires_official_credentials(monkeypatch):
    monkeypatch.delenv("STOCKTWITS_USERNAME", raising=False)
    monkeypatch.delenv("STOCKTWITS_PASSWORD", raising=False)
    out = stocktwits.fetch_stocktwits_messages("AAPL")
    assert "official API credentials not configured" in out


# --- Reddit -----------------------------------------------------------------

def _epoch(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


@pytest.mark.unit
def test_reddit_historical_window_excludes_recent(monkeypatch):
    posts = [{"title": "NOW", "created_utc": _epoch("2026-08-30"), "source": "rss"}]
    monkeypatch.setattr(reddit, "_fetch_subreddit", lambda *a, **k: posts)
    out = reddit.fetch_reddit_posts(
        "AAPL", subreddits=("stocks",), inter_request_delay=0,
        start_date="2026-05-01", end_date="2026-05-08",
    )
    assert "NOW" not in out
    assert "no posts" in out.lower() or "no reddit posts" in out.lower()


@pytest.mark.unit
def test_reddit_live_window_keeps_in_range(monkeypatch):
    posts = [{"title": "INRANGE", "created_utc": _epoch("2026-05-05"), "source": "rss"}]
    monkeypatch.setattr(reddit, "_fetch_subreddit", lambda *a, **k: posts)
    out = reddit.fetch_reddit_posts(
        "AAPL", subreddits=("stocks",), inter_request_delay=0,
        start_date="2026-05-01", end_date="2026-05-08",
    )
    assert "INRANGE" in out
