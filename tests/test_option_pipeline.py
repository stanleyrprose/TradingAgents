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


def test_sizing_disabled_preserves_output_and_never_calls_allocator(capsys):
    module = _load_pipeline()
    candidate = _candidate("AAPL260918C00200000")
    graph = _graph({"trader_investment_plan": "Action: BUY"}, "Buy", "/tmp/legacy")
    with (
        patch.object(
            module,
            "rank_equity_option_contracts",
            return_value=OptionSelectionResult((candidate,), "LEGACY SELECTION"),
        ),
        patch.object(module, "classify_instrument", side_effect=_profile),
        patch.object(module, "allocate_long_option_premium_risk") as allocator,
        _install_graph_module(MagicMock(return_value=graph)),
    ):
        assert module.main(["AAPL", "--direction", "bullish"]) == 0

    allocator.assert_not_called()
    assert capsys.readouterr().out == (
        "LEGACY SELECTION\n"
        f"rank: 1 | symbol: {candidate.symbol} | selector score: 88.50 | "
        "decision: Buy | report: /tmp/legacy\n"
    )


@pytest.mark.parametrize(
    "sizing_args",
    [
        ["--premium-budget", "0"],
        ["--premium-budget", "nan"],
        ["--account-equity", "10000"],
        ["--max-risk-pct", "2"],
        ["--account-equity", "10000", "--max-risk-pct", "101"],
    ],
)
def test_invalid_sizing_is_rejected_before_selector_or_any_graph(sizing_args, capsys):
    module = _load_pipeline()
    with (
        patch.object(module, "rank_equity_option_contracts") as selector,
        patch.object(module, "_graph_dependencies") as dependencies,
    ):
        assert (
            module.main(["AAPL", "--auto-direction", *sizing_args]) == 2
        )

    selector.assert_not_called()
    dependencies.assert_not_called()
    output = capsys.readouterr().out
    assert output.startswith("option sizing error: ")
    assert len(output.splitlines()) == 1


@pytest.mark.parametrize(
    ("symbol", "decision", "trader_plan", "approved"),
    [
        ("AAPL260918C00200000", "Rating: Buy", "Action: BUY", True),
        (
            "AAPL260918C00200000",
            "Rating: Overweight",
            "FINAL TRANSACTION PROPOSAL: BUY",
            True,
        ),
        ("AAPL260918P00200000", "Buy", "Action: Buy", True),
        ("AAPL260918C00200000", "Underweight", "Action: BUY", False),
        ("AAPL260918C00200000", "Buy", "Action: HOLD", False),
        ("AAPL260918C00200000", "REVIEW", "Action: BUY", False),
        ("AAPL260918C00200000", "not parseable", "we should buy", False),
    ],
)
def test_post_analysis_long_contract_approval_is_strict(
    symbol, decision, trader_plan, approved, capsys
):
    module = _load_pipeline()
    candidate = _candidate(symbol)
    graph = _graph({"trader_investment_plan": trader_plan}, decision, "/tmp/option")
    allocation = SimpleNamespace(report="ALLOCATION", available=True, has_position=True)
    with (
        patch.object(
            module,
            "rank_equity_option_contracts",
            return_value=OptionSelectionResult((candidate,), "SELECTION"),
        ),
        patch.object(module, "classify_instrument", side_effect=_profile),
        patch.object(
            module, "allocate_long_option_premium_risk", return_value=allocation
        ) as allocator,
        _install_graph_module(MagicMock(return_value=graph)),
    ):
        assert module.main(
            ["AAPL", "--direction", "bearish", "--premium-budget", "1000"]
        ) == 0

    if approved:
        allocator.assert_called_once_with(
            [candidate],
            account_equity=None,
            max_risk_pct=None,
            premium_budget=1000.0,
        )
    else:
        allocator.assert_not_called()
    output = capsys.readouterr().out
    expected_gate = "APPROVED" if approved else "REJECTED"
    assert f"allocation gate: {expected_gate}" in output
    assert ("ALLOCATION" in output) is approved


def test_top_three_real_allocation_uses_equal_buckets_only_for_approved(capsys):
    module = _load_pipeline()
    candidates = (
        _candidate("AAPL260918C00200000", score=99),
        _candidate("AAPL260918C00205000", score=98),
        _candidate("AAPL260918C00210000", score=1),
    )
    graphs = (
        _graph({"trader_investment_plan": "Action: BUY"}, "Buy", "/tmp/one"),
        _graph({"trader_investment_plan": "Action: HOLD"}, "Buy", "/tmp/two"),
        _graph({"trader_investment_plan": "Action: BUY"}, "Overweight", "/tmp/three"),
    )
    with (
        patch.object(
            module,
            "rank_equity_option_contracts",
            return_value=OptionSelectionResult(candidates, "SELECTION"),
        ),
        patch.object(module, "classify_instrument", side_effect=_profile),
        _install_graph_module(MagicMock(side_effect=graphs)),
    ):
        assert module.main(
            [
                "AAPL",
                "--direction",
                "bullish",
                "--analyze-top",
                "3",
                "--premium-budget",
                "1200",
            ]
        ) == 0

    output = capsys.readouterr().out
    assert f"| 1 | {candidates[0].symbol} | $5.20 | $600.00 |" in output
    assert f"| 2 | {candidates[2].symbol} | $5.20 | $600.00 |" in output
    assert f"| 2 | {candidates[1].symbol}" not in output
    assert "Selector score is intentionally not a sizing weight" in output


def test_graph_failure_is_excluded_while_later_approval_is_allocated():
    module = _load_pipeline()
    failed_candidate = _candidate("AAPL260918C00200000")
    approved_candidate = _candidate("AAPL260918C00205000")
    failed_graph = MagicMock()
    failed_graph.propagate.side_effect = RuntimeError("analysis failed")
    approved_graph = _graph(
        {"trader_investment_plan": "Action: BUY"}, "Buy", "/tmp/approved"
    )
    allocation = SimpleNamespace(report="ALLOCATION", available=True, has_position=True)
    with (
        patch.object(
            module,
            "rank_equity_option_contracts",
            return_value=OptionSelectionResult(
                (failed_candidate, approved_candidate), "SELECTION"
            ),
        ),
        patch.object(module, "classify_instrument", side_effect=_profile),
        patch.object(
            module, "allocate_long_option_premium_risk", return_value=allocation
        ) as allocator,
        _install_graph_module(MagicMock(side_effect=(failed_graph, approved_graph))),
    ):
        assert module.main(
            [
                "AAPL",
                "--direction",
                "bullish",
                "--analyze-top",
                "3",
                "--premium-budget",
                "1000",
            ]
        ) == 1

    allocator.assert_called_once_with(
        [approved_candidate],
        account_equity=None,
        max_risk_pct=None,
        premium_budget=1000.0,
    )


@pytest.mark.parametrize(
    ("sizing_args", "expected_kwargs"),
    [
        (
            ["--premium-budget", "700"],
            {"account_equity": None, "max_risk_pct": None, "premium_budget": 700.0},
        ),
        (
            ["--account-equity", "50000", "--max-risk-pct", "2"],
            {
                "account_equity": 50000.0,
                "max_risk_pct": 2.0,
                "premium_budget": None,
            },
        ),
        (
            [
                "--account-equity",
                "50000",
                "--max-risk-pct",
                "2",
                "--premium-budget",
                "600",
            ],
            {
                "account_equity": 50000.0,
                "max_risk_pct": 2.0,
                "premium_budget": 600.0,
            },
        ),
    ],
)
def test_valid_budget_forms_reach_allocator(sizing_args, expected_kwargs):
    module = _load_pipeline()
    candidate = _candidate("AAPL260918C00200000")
    graph = _graph({"trader_investment_plan": "Action: BUY"}, "Buy", "/tmp/report")
    allocation = SimpleNamespace(report="ALLOCATION", available=True, has_position=True)
    with (
        patch.object(
            module,
            "rank_equity_option_contracts",
            return_value=OptionSelectionResult((candidate,), "SELECTION"),
        ),
        patch.object(module, "classify_instrument", side_effect=_profile),
        patch.object(
            module, "allocate_long_option_premium_risk", return_value=allocation
        ) as allocator,
        _install_graph_module(MagicMock(return_value=graph)),
    ):
        assert module.main(
            ["AAPL", "--direction", "bullish", *sizing_args]
        ) == 0

    allocator.assert_called_once_with([candidate], **expected_kwargs)


def test_both_caps_use_smaller_budget_with_real_allocator(capsys):
    module = _load_pipeline()
    candidate = _candidate("AAPL260918C00200000")
    graph = _graph({"trader_investment_plan": "Action: BUY"}, "Buy", "/tmp/report")
    with (
        patch.object(
            module,
            "rank_equity_option_contracts",
            return_value=OptionSelectionResult((candidate,), "SELECTION"),
        ),
        patch.object(module, "classify_instrument", side_effect=_profile),
        _install_graph_module(MagicMock(return_value=graph)),
    ):
        assert module.main(
            [
                "AAPL",
                "--direction",
                "bullish",
                "--account-equity",
                "50000",
                "--max-risk-pct",
                "2",
                "--premium-budget",
                "600",
            ]
        ) == 0

    assert "Risk budget: $600.00" in capsys.readouterr().out


def test_unaffordable_approved_contract_is_valid_zero_position(capsys):
    module = _load_pipeline()
    candidate = _candidate("AAPL260918C00200000")
    graph = _graph({"trader_investment_plan": "Action: BUY"}, "Buy", "/tmp/report")
    with (
        patch.object(
            module,
            "rank_equity_option_contracts",
            return_value=OptionSelectionResult((candidate,), "SELECTION"),
        ),
        patch.object(module, "classify_instrument", side_effect=_profile),
        _install_graph_module(MagicMock(return_value=graph)),
    ):
        assert module.main(
            ["AAPL", "--direction", "bullish", "--premium-budget", "100"]
        ) == 0

    output = capsys.readouterr().out
    assert f"| 1 | {candidate.symbol} | $5.20 | $100.00 | 0 |" in output
    assert "NO OPTION POSITION: 0 whole contracts fit" in output


def test_sized_auto_no_option_trade_stops_before_selector_and_allocator():
    module = _load_pipeline()
    state = {"trader_investment_plan": "Action: HOLD"}
    graph_factory = MagicMock(return_value=_graph(state, "Hold", "/tmp/no-trade"))
    with (
        patch.object(module, "_graph_dependencies", return_value=(graph_factory, MagicMock())),
        patch.object(module, "classify_instrument", return_value=_underlying_profile()),
        patch.object(module, "rank_equity_option_contracts") as selector,
        patch.object(module, "allocate_long_option_premium_risk") as allocator,
    ):
        assert module.main(
            ["AAPL", "--auto-direction", "--premium-budget", "1000"]
        ) == 0

    selector.assert_not_called()
    allocator.assert_not_called()
