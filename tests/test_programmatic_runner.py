import runpy
from unittest.mock import MagicMock, patch


def _load_runner():
    return runpy.run_path("scripts/run_tradingagents.py", run_name="runner_test")


def test_detect_only_never_constructs_graph(capsys):
    with patch("sys.argv", ["run_tradingagents.py", "BTCUSD", "--detect-only"]), patch(
        "tradingagents.graph.trading_graph.TradingAgentsGraph"
    ) as graph:
        assert _load_runner()["main"]() == 0
    assert graph.call_count == 0
    output = capsys.readouterr().out
    assert "canonical_symbol: BTC-USD" in output
    assert "analysis_symbol: BTC-USD" in output
    assert "primary_type: crypto" in output


def test_non_runnable_symbol_exits_before_graph_construction(capsys):
    with patch("sys.argv", ["run_tradingagents.py", "DGS10"]), patch(
        "tradingagents.graph.trading_graph.TradingAgentsGraph"
    ) as graph:
        assert _load_runner()["main"]() == 2
    assert graph.call_count == 0
    assert "requires vendor price bars" in capsys.readouterr().out


def test_runner_uses_occ_underlying_for_propagate_and_reports():
    graph = MagicMock()
    graph.propagate.return_value = ({"state": "complete"}, "HOLD")
    graph.save_reports.return_value = "/tmp/report"
    with patch("sys.argv", ["run_tradingagents.py", "AAPL260918C00200000"]), patch(
        "tradingagents.graph.trading_graph.TradingAgentsGraph", return_value=graph
    ):
        assert _load_runner()["main"]() == 0

    graph.propagate.assert_called_once()
    assert graph.propagate.call_args.args[:2] == ("AAPL", graph.propagate.call_args.args[1])
    assert graph.propagate.call_args.kwargs == {"asset_type": "stock"}
    graph.save_reports.assert_called_once_with({"state": "complete"}, "AAPL")


def test_runner_accepts_legacy_bond_type(capsys):
    with patch("sys.argv", ["run_tradingagents.py", "AAPL", "--type", "bond", "--detect-only"]):
        assert _load_runner()["main"]() == 0
    assert "primary_type: fixed_income" in capsys.readouterr().out
