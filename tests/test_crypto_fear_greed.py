import json
from datetime import datetime, timezone
from unittest.mock import patch

from tradingagents.dataflows.crypto_fear_greed import fetch_crypto_fear_greed


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _ts(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def test_uses_latest_observations_at_or_before_cutoff():
    rows = [
        {"timestamp": str(_ts("2026-01-15")), "value": "80", "value_classification": "Greed"},
        {"timestamp": str(_ts("2026-01-08")), "value": "60", "value_classification": "Greed"},
        {"timestamp": str(_ts("2025-12-16")), "value": "25", "value_classification": "Fear"},
        {"timestamp": str(_ts("2026-01-20")), "value": "90", "value_classification": "Greed"},
    ]
    with patch("tradingagents.dataflows.crypto_fear_greed.urlopen", return_value=_Response({"data": rows})):
        result = fetch_crypto_fear_greed("2026-01-15")
    assert "Latest at/before 2026-01-15: 80/100" in result
    assert "7d prior (2026-01-08): 60/100" in result
    assert "30d prior (2025-12-16): 25/100" in result


def test_network_failure_returns_placeholder():
    with patch("tradingagents.dataflows.crypto_fear_greed.urlopen", side_effect=OSError):
        assert fetch_crypto_fear_greed() == "<crypto fear-greed unavailable: OSError>"
