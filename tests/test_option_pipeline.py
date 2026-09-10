import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from tradingagents.dataflows.option_selector import OptionCandidate, OptionSelectionResult

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_option_pipeline.py"


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("option_pipeline_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(symbol: str, score: float = 88.5) -> OptionCandidate:
    return OptionCandidate(
        symbol=symbol,
        expiry=date(2026, 9, 18),
        right="C" if "C" in symbol[-9:-7] else "P",
        strike=200.0,
        dte=8,
        bid=4.8,
        ask=5.2,
        midpoint=5.0,
        spread_pct=8.0,
        delta=0.55,
        abs_delta=0.55,
        iv=0.3,
        oi=500,
        volume=100,
        theta=-0.05,
        theta_burden=0.01,
        breakeven=205.0,
        required_move=0.025,
        components=(),
        score=score,
    )


def _install_graph_module(graph_factory):
    module = ModuleType("tradingagents.graph.trading_graph")
    module.TradingAgentsGraph = graph_factory
    return patch.dict(sys.modules, {"tradingagents.graph.trading_graph": module})


def _profile(symbol: str):
    return SimpleNamespace(
        canonical_symbol=symbol,
        analysis_symbol="AAPL",
        pipeline_asset_type="stock",
        analysts=("market", "news"),
    )


def test_unavailable_prints_report_and_never_imports_graph(capsys):
    module = _load_pipeline()
    result = OptionSelectionResult((), "<unavailable>", "no contracts")
    sys.modules.pop("tradingagents.graph.trading_graph", None)
    with patch.object(module, "rank_equity_option_contracts", return_value=result):
        assert module.main(["AAPL", "--direction", "bullish"]) == 2

    assert capsys.readouterr().out == "<unavailable>\n"
    assert "tradingagents.graph.trading_graph" not in sys.modules


def test_analyze_top_one_forwards_selection_identity_and_prints_compact_output(capsys):
    module = _load_pipeline()
    candidate = _candidate("AAPL260918C00200000")
    selection = OptionSelectionResult((candidate,), "SELECTION")
    graph = MagicMock()
    graph.propagate.return_value = ({"done": True}, "SELL")
    graph.save_reports.return_value = "/tmp/report-one"
    graph_factory = MagicMock(return_value=graph)
    with (
        patch.object(module, "rank_equity_option_contracts", return_value=selection) as selector,
        patch.object(module, "classify_instrument", side_effect=_profile) as classify,
        _install_graph_module(graph_factory),
    ):
        rc = module.main(
            ["aapl", "--direction", "bullish", "--date", "2026-09-10", "--min-dte", "10", "--max-dte", "30", "--target-delta", "0.6", "--top", "3"]
        )

    assert rc == 0
    selector.assert_called_once_with("aapl", "bullish", "2026-09-10", min_dte=10, max_dte=30, target_delta=0.6, top_n=3)
    classify.assert_called_once_with(candidate.symbol)
    graph_factory.assert_called_once()
    graph.propagate.assert_called_once_with(candidate.symbol, "2026-09-10", asset_type="stock", analysis_symbol="AAPL")
    graph.save_reports.assert_called_once_with({"done": True}, candidate.symbol)
    output = capsys.readouterr().out
    assert "rank: 1" in output and candidate.symbol in output
    assert "selector score: 88.50" in output and "decision: SELL" in output
    assert "report: /tmp/report-one" in output


def test_analyze_top_three_uses_fresh_graphs_and_saves_each(capsys):
    module = _load_pipeline()
    candidates = tuple(_candidate(f"AAPL260918C{strike:08d}", 90 - index) for index, strike in enumerate((200000, 205000, 210000)))
    graphs = [MagicMock() for _ in candidates]
    for index, graph in enumerate(graphs):
        graph.propagate.return_value = ({"index": index}, "HOLD")
        graph.save_reports.return_value = f"/tmp/report-{index}"
    factory = MagicMock(side_effect=graphs)
    with (
        patch.object(module, "rank_equity_option_contracts", return_value=OptionSelectionResult(candidates, "REPORT")),
        patch.object(module, "classify_instrument", side_effect=_profile) as classify,
        _install_graph_module(factory),
    ):
        assert module.main(["AAPL", "--direction", "bullish", "--date", "2026-09-10", "--analyze-top", "3"]) == 0

    assert factory.call_count == 3
    assert [call.args[0] for call in classify.call_args_list] == [item.symbol for item in candidates]
    for index, (candidate, graph) in enumerate(zip(candidates, graphs, strict=True)):
        graph.propagate.assert_called_once_with(candidate.symbol, "2026-09-10", asset_type="stock", analysis_symbol="AAPL")
        graph.save_reports.assert_called_once_with({"index": index}, candidate.symbol)


def test_failure_continues_and_later_candidate_saves_report(capsys):
    module = _load_pipeline()
    candidates = tuple(_candidate(f"AAPL260918C{strike:08d}") for strike in (200000, 205000, 210000))
    graphs = [MagicMock() for _ in candidates]
    graphs[0].propagate.return_value = ({}, "HOLD")
    graphs[0].save_reports.return_value = "/tmp/first"
    graphs[1].propagate.side_effect = RuntimeError("boom")
    graphs[2].propagate.return_value = ({"later": True}, "BUY")
    graphs[2].save_reports.return_value = "/tmp/later"
    with (
        patch.object(module, "rank_equity_option_contracts", return_value=OptionSelectionResult(candidates, "REPORT")),
        patch.object(module, "classify_instrument", side_effect=_profile),
        _install_graph_module(MagicMock(side_effect=graphs)),
    ):
        assert module.main(["AAPL", "--direction", "bullish", "--analyze-top", "3"]) == 1

    graphs[2].save_reports.assert_called_once_with({"later": True}, candidates[2].symbol)
    assert "failed: boom" in capsys.readouterr().out


def test_graph_decision_never_rewrites_selected_direction():
    module = _load_pipeline()
    cases = [
        ("bullish", _candidate("AAPL260918C00200000"), "SELL"),
        ("bearish", _candidate("AAPL260918P00200000"), "BUY"),
    ]
    for direction, candidate, decision in cases:
        graph = MagicMock()
        graph.propagate.return_value = ({}, decision)
        graph.save_reports.return_value = "/tmp/report"
        classify = MagicMock(side_effect=_profile)
        with (
            patch.object(module, "rank_equity_option_contracts", return_value=OptionSelectionResult((candidate,), "REPORT")),
            patch.object(module, "classify_instrument", classify),
            _install_graph_module(MagicMock(return_value=graph)),
        ):
            assert module.main(["AAPL", "--direction", direction]) == 0
        classify.assert_called_once_with(candidate.symbol)
        graph.propagate.assert_called_once()
        assert graph.propagate.call_args.args[0] == candidate.symbol
