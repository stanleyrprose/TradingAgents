"""StockTwits official authenticated sentiment provider tests."""

from __future__ import annotations

import http.client
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import stocktwits


def _raise(exc):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            raise exc

    return _Resp()


@pytest.mark.unit
class TestStockTwitsResilience:
    @pytest.mark.parametrize(
        "exc",
        [
            http.client.IncompleteRead(b""),
            HTTPError("url", 503, "down", {}, None),
            TimeoutError("slow"),
        ],
    )
    def test_transport_errors_return_placeholder(self, monkeypatch, exc):
        monkeypatch.setenv("STOCKTWITS_USERNAME", "user")
        monkeypatch.setenv("STOCKTWITS_PASSWORD", "pass")
        with patch.object(stocktwits, "urlopen", return_value=_raise(exc)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out == "<stocktwits unavailable: authenticated API request failed>"

    def test_missing_credentials_do_not_attempt_network(self, monkeypatch):
        monkeypatch.delenv("STOCKTWITS_USERNAME", raising=False)
        monkeypatch.delenv("STOCKTWITS_PASSWORD", raising=False)
        with patch.object(stocktwits, "urlopen") as mocked:
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        mocked.assert_not_called()
        assert out == "<stocktwits unavailable: official API credentials not configured>"


@pytest.mark.unit
class TestStockTwitsCryptoSymbols:
    @pytest.mark.parametrize(
        ("ticker", "expected"),
        [
            ("BTC-USD", "BTC.X"),
            ("eth-usd", "ETH.X"),
            ("SOL-USD", "SOL.X"),
            ("BTCUSD", "BTC.X"),
            ("BTC-USDT", "BTC.X"),
            ("AMD", "AMD"),
            ("BRK-B", "BRK-B"),
            ("GOLD", "GOLD"),
            ("XYZ-USD", "XYZ-USD"),
        ],
    )
    def test_symbol_mapping(self, ticker, expected):
        assert stocktwits._stocktwits_symbol(ticker) == expected


@pytest.mark.unit
def test_authenticated_firestream_sentiment_is_preferred(monkeypatch):
    monkeypatch.setenv("STOCKTWITS_USERNAME", "user")
    monkeypatch.setenv("STOCKTWITS_PASSWORD", "pass")
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return (
                b'{"data":{"sentiment":{"24h":{"labelNormalized":"BULLISH",'
                b'"valueNormalized":68.9}},"messageVolume":{"24h":'
                b'{"labelNormalized":"HIGH","valueNormalized":78.4}},'
                b'"timeframes":{"1D":{"sentiment":{"labelNormalized":"BULLISH",'
                b'"valueNormalized":68.9}}}}}'
            )

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["authorization"] = req.get_header("Authorization")
        return Resp()

    with patch.object(stocktwits, "urlopen", side_effect=fake_urlopen):
        out = stocktwits.fetch_stocktwits_messages("ETH-USD")

    assert "external/sentiment/v2/ETH.X/detail" in seen["url"]
    assert seen["authorization"].startswith("Basic ")
    assert "Official StockTwits sentiment for $ETH.X" in out
    assert "24h: sentiment=BULLISH (68.9)" in out
    assert "message_volume=HIGH (78.4)" in out


@pytest.mark.unit
def test_historical_window_skips_current_state_api(monkeypatch):
    monkeypatch.setenv("STOCKTWITS_USERNAME", "user")
    monkeypatch.setenv("STOCKTWITS_PASSWORD", "pass")
    with patch.object(stocktwits, "urlopen") as mocked:
        out = stocktwits.fetch_stocktwits_messages(
            "ETH-USD",
            start_date="2026-01-01",
            end_date="2026-01-07",
        )
    mocked.assert_not_called()
    assert "official sentiment API is current-state only" in out
