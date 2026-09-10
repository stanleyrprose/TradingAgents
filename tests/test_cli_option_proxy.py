"""Interactive CLI coverage for requested option identity and analysis proxy."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cli.main as cli_main
from cli.models import AssetType

OCC_SYMBOL = "AAPL260918C00200000"
ANALYSIS_DATE = "2026-09-10"


def test_user_selections_add_occ_identity_and_underlying_proxy():
    env = {
        "TRADINGAGENTS_LLM_PROVIDER": "openai",
        "TRADINGAGENTS_QUICK_THINK_LLM": "quick-model",
        "TRADINGAGENTS_DEEP_THINK_LLM": "deep-model",
        "TRADINGAGENTS_OUTPUT_LANGUAGE": "English",
        "TRADINGAGENTS_MAX_DEBATE_ROUNDS": "1",
        "TRADINGAGENTS_MAX_RISK_ROUNDS": "1",
    }
    config = dict(
        cli_main.DEFAULT_CONFIG,
        llm_provider="openai",
        quick_think_llm="quick-model",
        deep_think_llm="deep-model",
        output_language="English",
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
    )

    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(cli_main, "DEFAULT_CONFIG", config),
        patch.object(cli_main, "console"),
        patch.object(cli_main, "fetch_announcements", return_value=None),
        patch.object(cli_main, "display_announcements"),
        patch.object(cli_main, "get_ticker", return_value=OCC_SYMBOL),
        patch.object(cli_main, "detect_asset_type", return_value=AssetType.STOCK) as detect,
        patch.object(cli_main, "get_analysis_date", return_value=ANALYSIS_DATE),
        patch.object(cli_main, "select_analysts", return_value=[]) as select_analysts,
        patch.object(cli_main, "ensure_api_key"),
    ):
        selections = cli_main.get_user_selections()

    assert selections["ticker"] == OCC_SYMBOL
    assert selections["requested_ticker"] == OCC_SYMBOL
    assert selections["analysis_symbol"] == "AAPL"
    detect.assert_called_once_with(OCC_SYMBOL)
    select_analysts.assert_called_once_with(AssetType.STOCK)


@pytest.mark.parametrize(
    ("selection_identity", "expected_requested", "expected_analysis"),
    [
        (
            {"requested_ticker": OCC_SYMBOL, "analysis_symbol": "AAPL"},
            OCC_SYMBOL,
            "AAPL",
        ),
        ({}, "AAPL", "AAPL"),
    ],
)
def test_run_analysis_routes_requested_identity_and_analysis_proxy(
    tmp_path, selection_identity, expected_requested, expected_analysis
):
    ticker = OCC_SYMBOL if selection_identity else "AAPL"
    selections = {
        "ticker": ticker,
        "asset_type": "stock",
        "analysis_date": ANALYSIS_DATE,
        "analysts": [SimpleNamespace(value="market")],
        **selection_identity,
    }
    graph = MagicMock()
    instrument_context = object()
    initial_state = {"company_of_interest": expected_requested}
    graph.resolve_instrument_context.return_value = instrument_context
    graph.propagator.create_initial_state.return_value = initial_state
    graph.propagator.get_graph_args.return_value = {}
    graph.begin_checkpoint.return_value = "checkpoint-thread"
    graph.checkpoint_input.return_value = initial_state
    graph.graph.stream.return_value = []
    tracker = MagicMock()
    tracker.format_summary.return_value = "summary"
    report_path = tmp_path / "saved" / "report.md"

    with (
        patch.object(cli_main, "get_user_selections", return_value=selections),
        patch.object(
            cli_main, "_build_run_config", return_value={"results_dir": str(tmp_path)}
        ),
        patch.object(cli_main, "TradingAgentsGraph", return_value=graph),
        patch.object(cli_main, "message_buffer", cli_main.MessageBuffer()),
        patch.object(cli_main, "build_analyst_execution_plan", return_value=object()),
        patch.object(cli_main, "AnalystWallTimeTracker", return_value=tracker),
        patch.object(cli_main, "get_initial_analyst_node", return_value="Market Analyst"),
        patch.object(cli_main, "create_layout", return_value=MagicMock()),
        patch.object(cli_main, "update_display"),
        patch.object(cli_main, "Live") as live,
        patch.object(
            cli_main.typer,
            "prompt",
            side_effect=["Y", str(tmp_path / "saved"), "N"],
        ),
        patch.object(cli_main, "save_report_to_disk", return_value=report_path) as save_report,
    ):
        live.return_value.__enter__.return_value = None
        cli_main.run_analysis()

    graph.resolve_instrument_context.assert_called_once_with(
        expected_requested, "stock", analysis_symbol=expected_analysis
    )
    graph.propagator.create_initial_state.assert_called_once_with(
        expected_requested,
        ANALYSIS_DATE,
        asset_type="stock",
        instrument_context=instrument_context,
        analysis_symbol=expected_analysis,
    )
    graph.begin_checkpoint.assert_called_once_with(
        expected_requested, ANALYSIS_DATE, "stock"
    )
    graph.clear_checkpoint_on_success.assert_called_once_with(
        expected_requested, ANALYSIS_DATE, "stock"
    )
    save_report.assert_called_once_with({}, expected_requested, tmp_path / "saved")
    assert (tmp_path / expected_requested / ANALYSIS_DATE / "message_tool.log").exists()
