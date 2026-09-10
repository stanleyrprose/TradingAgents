import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest

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


def _underlying_profile(symbol: str = "AAPL"):
    return SimpleNamespace(
        primary_type="stock",
        asset_class="equity",
        instrument_kind="stock",
        can_run=True,
        canonical_symbol=symbol,
        analysis_symbol=symbol,
        pipeline_asset_type="stock",
        analysts=("market", "news"),
    )


def _option_profile(symbol: str):
    return SimpleNamespace(
        primary_type="option",
        asset_class="option",
        instrument_kind="option",
        can_run=True,
        canonical_symbol=symbol,
        analysis_symbol="AAPL",
        pipeline_asset_type="stock",
        analysts=("market", "news"),
    )


def _auto_classify(symbol: str):
    return _underlying_profile() if symbol.upper() == "AAPL" else _option_profile(symbol)


def _graph(final_state, decision, report_path):
    graph = MagicMock()
    graph.propagate.return_value = (final_state, decision)
    graph.save_reports.return_value = report_path
    return graph


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


def test_auto_bullish_analyzes_exact_call_with_fresh_graph_and_saves_both_reports():
    module = _load_pipeline()
    candidate = _candidate("AAPL260918C00200000")
    underlying_state = {
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: BUY",
        "underlying": True,
    }
    underlying_graph = _graph(underlying_state, "Overweight", "/tmp/underlying")
    option_graph = _graph({"option": True}, "SELL", "/tmp/option")
    graph_factory = MagicMock(side_effect=(underlying_graph, option_graph))
    config_factory = MagicMock(return_value={"config": True})

    with (
        patch.object(module, "_graph_dependencies", return_value=(graph_factory, config_factory)),
        patch.object(module, "classify_instrument", side_effect=_auto_classify),
        patch.object(
            module,
            "rank_equity_option_contracts",
            return_value=OptionSelectionResult((candidate,), "CALL SELECTION"),
        ) as selector,
    ):
        rc = module.main(
            [
                "AAPL",
                "--auto-direction",
                "--date",
                "2026-09-10",
                "--min-dte",
                "10",
                "--max-dte",
                "30",
                "--target-delta",
                "0.6",
                "--top",
                "3",
            ]
        )

    assert rc == 0
    selector.assert_called_once_with(
        "AAPL", "bullish", "2026-09-10", min_dte=10, max_dte=30, target_delta=0.6, top_n=3
    )
    assert graph_factory.call_count == 2
    underlying_graph.propagate.assert_called_once_with(
        "AAPL", "2026-09-10", asset_type="stock", analysis_symbol="AAPL"
    )
    underlying_graph.save_reports.assert_called_once_with(underlying_state, "AAPL")
    option_graph.propagate.assert_called_once_with(
        candidate.symbol, "2026-09-10", asset_type="stock", analysis_symbol="AAPL"
    )
    option_graph.save_reports.assert_called_once_with({"option": True}, candidate.symbol)


def test_auto_bearish_analyzes_exact_put_without_rewriting_direction():
    module = _load_pipeline()
    candidate = _candidate("AAPL260918P00200000")
    underlying_state = {"trader_investment_plan": "FINAL TRANSACTION PROPOSAL: SELL"}
    underlying_graph = _graph(underlying_state, "Underweight", "/tmp/underlying")
    option_graph = _graph({"put": True}, "BUY", "/tmp/put")
    graph_factory = MagicMock(side_effect=(underlying_graph, option_graph))

    with (
        patch.object(module, "_graph_dependencies", return_value=(graph_factory, MagicMock())),
        patch.object(module, "classify_instrument", side_effect=_auto_classify),
        patch.object(
            module,
            "rank_equity_option_contracts",
            return_value=OptionSelectionResult((candidate,), "PUT SELECTION"),
        ) as selector,
    ):
        assert module.main(["AAPL", "--auto-direction"]) == 0

    selector.assert_called_once()
    assert selector.call_args.args[1] == "bearish"
    assert graph_factory.call_count == 2
    option_graph.propagate.assert_called_once()
    assert option_graph.propagate.call_args.args[0] == candidate.symbol


def test_auto_neutral_hold_saves_underlying_report_without_selecting_options():
    module = _load_pipeline()
    state = {"trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD"}
    underlying_graph = _graph(state, "Hold", "/tmp/neutral")
    graph_factory = MagicMock(return_value=underlying_graph)

    with (
        patch.object(module, "_graph_dependencies", return_value=(graph_factory, MagicMock())),
        patch.object(module, "classify_instrument", return_value=_underlying_profile()),
        patch.object(module, "rank_equity_option_contracts") as selector,
    ):
        assert module.main(["AAPL", "--auto-direction"]) == 0

    underlying_graph.save_reports.assert_called_once_with(state, "AAPL")
    selector.assert_not_called()
    graph_factory.assert_called_once()


@pytest.mark.parametrize(
    ("decision", "trader_plan"),
    [
        ("Underweight", "FINAL TRANSACTION PROPOSAL: HOLD"),
        ("Overweight", "FINAL TRANSACTION PROPOSAL: HOLD"),
        ("Buy", "FINAL TRANSACTION PROPOSAL: SELL"),
        ("Sell", "FINAL TRANSACTION PROPOSAL: BUY"),
        ("Buy", "The trader might buy after further review."),
        ("REVIEW", "FINAL TRANSACTION PROPOSAL: BUY"),
    ],
)
def test_auto_conservative_no_trade_never_selects_or_constructs_option_graph(decision, trader_plan):
    module = _load_pipeline()
    state = {"trader_investment_plan": trader_plan}
    graph_factory = MagicMock(return_value=_graph(state, decision, "/tmp/no-trade"))

    with (
        patch.object(module, "_graph_dependencies", return_value=(graph_factory, MagicMock())),
        patch.object(module, "classify_instrument", return_value=_underlying_profile()),
        patch.object(module, "rank_equity_option_contracts") as selector,
    ):
        assert module.main(["AAPL", "--auto-direction"]) == 0

    selector.assert_not_called()
    graph_factory.assert_called_once()


@pytest.mark.parametrize("symbol", ["BTCUSD", "GC=F"])
def test_auto_invalid_non_equity_returns_before_graph_dependencies_and_selector(symbol, capsys):
    module = _load_pipeline()
    invalid_profile = SimpleNamespace(
        primary_type="crypto" if symbol == "BTCUSD" else "future",
        asset_class="crypto" if symbol == "BTCUSD" else "commodity",
        instrument_kind="crypto" if symbol == "BTCUSD" else "future",
        can_run=True,
        canonical_symbol=symbol,
    )
    with (
        patch.object(module, "classify_instrument", return_value=invalid_profile),
        patch.object(module, "_graph_dependencies") as dependencies,
        patch.object(module, "rank_equity_option_contracts") as selector,
    ):
        assert module.main([symbol, "--auto-direction"]) == 2

    dependencies.assert_not_called()
    selector.assert_not_called()
    assert capsys.readouterr().out == (
        "<underlying thesis unavailable: auto-direction requires a runnable US "
        "equity stock or equity fund with a 1-6 letter Cboe symbol>\n"
    )


def test_historical_auto_direction_passes_date_then_stops_on_unavailable_chain():
    module = _load_pipeline()
    state = {"trader_investment_plan": "FINAL TRANSACTION PROPOSAL: BUY"}
    underlying_graph = _graph(state, "Buy", "/tmp/historical")
    graph_factory = MagicMock(return_value=underlying_graph)
    unavailable = OptionSelectionResult((), "<unavailable>", "current chain unavailable")

    with (
        patch.object(module, "_graph_dependencies", return_value=(graph_factory, MagicMock())),
        patch.object(module, "classify_instrument", return_value=_underlying_profile()),
        patch.object(module, "rank_equity_option_contracts", return_value=unavailable) as selector,
    ):
        assert module.main(["AAPL", "--auto-direction", "--date", "2026-09-09"]) == 2

    selector.assert_called_once_with(
        "AAPL", "bullish", "2026-09-09", min_dte=7, max_dte=45, target_delta=0.55, top_n=5
    )
    underlying_graph.propagate.assert_called_once_with(
        "AAPL", "2026-09-09", asset_type="stock", analysis_symbol="AAPL"
    )
    graph_factory.assert_called_once()


def test_historical_neutral_auto_direction_never_calls_selector():
    module = _load_pipeline()
    graph_factory = MagicMock(
        return_value=_graph(
            {"trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD"},
            "Hold",
            "/tmp/historical-neutral",
        )
    )
    with (
        patch.object(module, "_graph_dependencies", return_value=(graph_factory, MagicMock())),
        patch.object(module, "classify_instrument", return_value=_underlying_profile()),
        patch.object(module, "rank_equity_option_contracts") as selector,
    ):
        assert module.main(["AAPL", "--auto-direction", "--date", "2026-09-09"]) == 0

    selector.assert_not_called()
    graph_factory.assert_called_once()


def test_manual_direction_remains_selector_first_with_one_option_graph():
    module = _load_pipeline()
    candidate = _candidate("AAPL260918C00200000")
    events = []
    option_graph = _graph({"manual": True}, "BUY", "/tmp/manual")
    graph_factory = MagicMock(return_value=option_graph)

    def select(*args, **kwargs):
        events.append("selector")
        return OptionSelectionResult((candidate,), "MANUAL")

    def dependencies():
        events.append("dependencies")
        return graph_factory, MagicMock()

    with (
        patch.object(module, "rank_equity_option_contracts", side_effect=select) as selector,
        patch.object(module, "_graph_dependencies", side_effect=dependencies),
        patch.object(module, "classify_instrument", side_effect=_option_profile) as classify,
    ):
        assert module.main(["AAPL", "--direction", "bullish"]) == 0

    assert events == ["selector", "dependencies"]
    selector.assert_called_once()
    classify.assert_called_once_with(candidate.symbol)
    graph_factory.assert_called_once()
    option_graph.propagate.assert_called_once_with(
        candidate.symbol, ANY, asset_type="stock", analysis_symbol="AAPL"
    )


def test_direction_arguments_are_mutually_exclusive_and_required():
    parser = _load_pipeline()._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["AAPL"])
    with pytest.raises(SystemExit):
        parser.parse_args(["AAPL", "--direction", "bullish", "--auto-direction"])
