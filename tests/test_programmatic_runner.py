import runpy
from unittest.mock import patch


def test_detect_only_never_constructs_graph(capsys):
    with patch("sys.argv", ["run_tradingagents.py", "BTCUSD", "--detect-only"]), patch(
        "tradingagents.graph.trading_graph.TradingAgentsGraph"
    ) as graph:
        namespace = runpy.run_path("scripts/run_tradingagents.py", run_name="runner_test")
        assert namespace["main"]() == 0
    assert graph.call_count == 0
    output = capsys.readouterr().out
    assert "primary_type: crypto" in output
    assert "pipeline_asset_type: crypto" in output
