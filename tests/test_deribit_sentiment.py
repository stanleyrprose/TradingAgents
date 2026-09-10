import json
from unittest.mock import patch

from tradingagents.dataflows.deribit_sentiment import fetch_deribit_options_sentiment


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_aggregates_option_positioning():
    rows = [
        {"instrument_name": "ETH-30JAN26-3000-C", "open_interest": 10, "volume": 4, "mark_iv": 50, "underlying_price": 2800},
        {"instrument_name": "ETH-30JAN26-2500-P", "open_interest": 20, "volume": 2, "mark_iv": 70, "underlying_price": 2800},
    ]
    with patch("tradingagents.dataflows.deribit_sentiment.urlopen", return_value=_Response({"result": rows})):
        result = fetch_deribit_options_sentiment("ETH-USD")
    assert "Active option instruments: 2" in result
    assert "Put/Call open-interest ratio: 2.00" in result
    assert "Put/Call 24h volume ratio: 0.50" in result
    assert "Mean mark IV across listed options: 60.00%" in result


def test_rejects_unsupported_and_historical_requests_without_network():
    with patch("tradingagents.dataflows.deribit_sentiment.urlopen") as urlopen:
        assert "unsupported symbol" in fetch_deribit_options_sentiment("SOL")
        assert "historical run" in fetch_deribit_options_sentiment("ETH-USD", "2020-01-01")
    urlopen.assert_not_called()
