import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.dataflows.equity_options import (
    EquityOptionSnapshot,
    EquityOptionSnapshotResult,
)

SCRIPT = Path(__file__).parents[1] / "scripts" / "manage_option_position.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("option_position_script_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot():
    return EquityOptionSnapshot(
        symbol="AAPL260918C00300000",
        underlying="AAPL",
        expiry=date(2026, 9, 18),
        right="C",
        strike=300.0,
        as_of=date(2026, 9, 10),
        dte=8,
        source_timestamp="2026-09-10 15:45:00",
        underlying_spot=315.0,
        bid=16.0,
        ask=18.0,
        midpoint=17.0,
        spread_pct=11.76,
        last=16.5,
        iv=0.30,
        delta=0.75,
        gamma=0.0123,
        vega=0.2345,
        theta=-0.1234,
        open_interest=100.0,
        volume=10.0,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--entry-premium", "0", "--contracts", "1"],
        ["--entry-premium", "5", "--contracts", "0"],
        ["--entry-premium", "5", "--contracts", "1", "--stop-loss-pct", "101"],
        ["--entry-premium", "5", "--contracts", "1", "--exit-at-dte", "-1"],
    ],
)
def test_invalid_position_or_policy_stops_before_snapshot_fetch(args, capsys):
    module = _load_script()
    with patch.object(module, "fetch_equity_option_snapshot") as fetch:
        assert module.main(["AAPL260918C00300000", *args]) == 2

    fetch.assert_not_called()
    assert capsys.readouterr().out.startswith("option position policy error: ")


def test_unavailable_snapshot_is_fail_closed_without_evaluator(capsys):
    module = _load_script()
    unavailable = EquityOptionSnapshotResult(None, "exact contract not found")
    with (
        patch.object(module, "fetch_equity_option_snapshot", return_value=unavailable) as fetch,
        patch.object(module, "evaluate_long_option_position") as evaluate,
    ):
        assert module.main(
            [
                "AAPL260918C00300000",
                "--entry-premium",
                "5",
                "--contracts",
                "1",
                "--date",
                "2026-09-10",
            ]
        ) == 2

    fetch.assert_called_once_with("AAPL260918C00300000", "2026-09-10")
    evaluate.assert_not_called()
    assert capsys.readouterr().out == (
        "<option position management unavailable: exact contract not found>\n"
    )


def test_report_only_forwards_exact_snapshot_and_no_hidden_policy(capsys):
    module = _load_script()
    snapshot = _snapshot()
    result = SimpleNamespace(status="REPORT_ONLY", report="POSITION REPORT_ONLY")
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ) as fetch,
        patch.object(
            module, "evaluate_long_option_position", return_value=result
        ) as evaluate,
    ):
        assert module.main(
            [
                snapshot.symbol,
                "--entry-premium",
                "5.25",
                "--contracts",
                "2",
                "--date",
                "2026-09-10",
            ]
        ) == 0

    fetch.assert_called_once_with(snapshot.symbol, "2026-09-10")
    evaluate.assert_called_once_with(
        snapshot,
        entry_premium=5.25,
        contracts=2,
        take_profit_pct=None,
        stop_loss_pct=None,
        exit_at_dte=None,
        max_theta_burn_pct_per_day=None,
    )
    assert capsys.readouterr().out == "POSITION REPORT_ONLY\n"


def test_explicit_policy_is_forwarded_and_exit_is_normal_result(capsys):
    module = _load_script()
    snapshot = _snapshot()
    result = SimpleNamespace(status="EXIT", report="POSITION EXIT")
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ) as fetch,
        patch.object(
            module, "evaluate_long_option_position", return_value=result
        ) as evaluate,
    ):
        rc = module.main(
            [
                snapshot.symbol,
                "--entry-premium",
                "5",
                "--contracts",
                "1",
                "--date",
                "2026-09-10",
                "--take-profit-pct",
                "50",
                "--stop-loss-pct",
                "40",
                "--exit-at-dte",
                "5",
                "--max-theta-burn-pct-per-day",
                "3",
            ]
        )

    assert rc == 0
    fetch.assert_called_once()
    evaluate.assert_called_once_with(
        snapshot,
        entry_premium=5.0,
        contracts=1,
        take_profit_pct=50.0,
        stop_loss_pct=40.0,
        exit_at_dte=5,
        max_theta_burn_pct_per_day=3.0,
    )
    assert capsys.readouterr().out == "POSITION EXIT\n"


@pytest.mark.parametrize("status", ["HOLD", "EXIT", "REPORT_ONLY"])
def test_normal_decision_statuses_return_zero(status):
    module = _load_script()
    snapshot = _snapshot()
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ),
        patch.object(
            module,
            "evaluate_long_option_position",
            return_value=SimpleNamespace(status=status, report=status),
        ),
    ):
        assert module.main(
            [snapshot.symbol, "--entry-premium", "5", "--contracts", "1"]
        ) == 0


def test_review_returns_two_for_automation_fail_closed(capsys):
    module = _load_script()
    snapshot = _snapshot()
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ),
        patch.object(
            module,
            "evaluate_long_option_position",
            return_value=SimpleNamespace(status="REVIEW", report="POSITION REVIEW"),
        ),
    ):
        assert module.main(
            [snapshot.symbol, "--entry-premium", "5", "--contracts", "1"]
        ) == 2

    assert capsys.readouterr().out == "POSITION REVIEW\n"


def _refresh_result(status, *, option_direction="bullish", current_direction=None):
    return SimpleNamespace(
        status=status,
        option_direction=option_direction,
        current_direction=current_direction,
        portfolio_rating="Hold" if current_direction is None else ("Buy" if current_direction == "bullish" else "Sell"),
        trader_action="Hold" if current_direction is None else ("Buy" if current_direction == "bullish" else "Sell"),
        reason=f"refresh {status}",
    )


def test_refresh_only_invalidated_is_informational_not_forced_exit(capsys):
    module = _load_script()
    snapshot = _snapshot()
    refresh = _refresh_result("INVALIDATED", current_direction="bearish")
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ),
        patch.object(
            module,
            "evaluate_long_option_position",
            return_value=SimpleNamespace(status="HOLD", report="POSITION HOLD"),
        ),
        patch.object(
            module,
            "_refresh_underlying_thesis",
            return_value=(refresh, "/tmp/AAPL-report.md"),
        ) as refresh_call,
    ):
        rc = module.main(
            [
                snapshot.symbol,
                "--entry-premium",
                "5",
                "--contracts",
                "1",
                "--date",
                "2026-09-10",
                "--refresh-thesis",
            ]
        )

    assert rc == 0
    refresh_call.assert_called_once_with(snapshot, "2026-09-10")
    output = capsys.readouterr().out
    assert "Status: **INVALIDATED**" in output
    assert "informational only" in output
    assert "FINAL LIFECYCLE STATUS: EXIT" not in output


def test_exit_on_invalidation_implies_refresh_and_overrides_hold(capsys):
    module = _load_script()
    snapshot = _snapshot()
    refresh = _refresh_result("INVALIDATED", current_direction="bearish")
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ),
        patch.object(
            module,
            "evaluate_long_option_position",
            return_value=SimpleNamespace(status="HOLD", report="POSITION HOLD"),
        ),
        patch.object(
            module,
            "_refresh_underlying_thesis",
            return_value=(refresh, "/tmp/AAPL-report.md"),
        ) as refresh_call,
    ):
        rc = module.main(
            [
                snapshot.symbol,
                "--entry-premium",
                "5",
                "--contracts",
                "1",
                "--date",
                "2026-09-10",
                "--exit-on-thesis-invalidation",
            ]
        )

    assert rc == 0
    refresh_call.assert_called_once_with(snapshot, "2026-09-10")
    output = capsys.readouterr().out
    assert "Policy: EXIT on explicit opposite consensus" in output
    assert "FINAL LIFECYCLE STATUS: EXIT" in output


def test_exit_on_invalidation_neutral_does_not_force_exit(capsys):
    module = _load_script()
    snapshot = _snapshot()
    refresh = _refresh_result("NEUTRAL", current_direction=None)
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ),
        patch.object(
            module,
            "evaluate_long_option_position",
            return_value=SimpleNamespace(status="HOLD", report="POSITION HOLD"),
        ),
        patch.object(
            module,
            "_refresh_underlying_thesis",
            return_value=(refresh, "/tmp/AAPL-report.md"),
        ),
    ):
        rc = module.main(
            [
                snapshot.symbol,
                "--entry-premium",
                "5",
                "--contracts",
                "1",
                "--exit-on-thesis-invalidation",
            ]
        )

    assert rc == 0
    output = capsys.readouterr().out
    assert "Status: **NEUTRAL**" in output
    assert "FINAL LIFECYCLE STATUS: EXIT" not in output


def test_explicit_refresh_failure_is_fail_closed(capsys):
    module = _load_script()
    snapshot = _snapshot()
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ),
        patch.object(
            module,
            "evaluate_long_option_position",
            return_value=SimpleNamespace(status="HOLD", report="POSITION HOLD"),
        ),
        patch.object(
            module,
            "_refresh_underlying_thesis",
            side_effect=RuntimeError("graph failed"),
        ),
    ):
        rc = module.main(
            [
                snapshot.symbol,
                "--entry-premium",
                "5",
                "--contracts",
                "1",
                "--refresh-thesis",
            ]
        )

    assert rc == 2
    assert "<underlying thesis refresh unavailable: graph failed>" in capsys.readouterr().out


def test_refresh_helper_analyzes_underlying_not_occ_symbol():
    module = _load_script()
    snapshot = _snapshot()
    profile = SimpleNamespace(
        can_run=True,
        asset_class="equity",
        analysts=("market", "news"),
        canonical_symbol="AAPL",
        pipeline_asset_type="stock",
        analysis_symbol="AAPL",
    )
    graph = SimpleNamespace()
    graph.propagate = MagicMock(
        return_value=({"trader_investment_plan": "**Action**: Buy"}, "Overweight")
    )
    graph.save_reports = MagicMock(return_value="/tmp/AAPL-report.md")
    graph_factory = MagicMock(return_value=graph)
    config_factory = MagicMock(return_value={"provider": "codex_cli"})
    refresh = _refresh_result("CONFIRMED", current_direction="bullish")

    with (
        patch.object(module, "classify_instrument", return_value=profile) as classify,
        patch.object(
            module, "_graph_dependencies", return_value=(graph_factory, config_factory)
        ),
        patch.object(
            module, "refresh_long_option_thesis", return_value=refresh
        ) as refresh_gate,
    ):
        result, report = module._refresh_underlying_thesis(snapshot, "2026-09-10")

    assert result is refresh
    assert report == "/tmp/AAPL-report.md"
    classify.assert_called_once_with("AAPL")
    graph_factory.assert_called_once_with(
        selected_analysts=["market", "news"],
        config={"provider": "codex_cli"},
        debug=False,
    )
    graph.propagate.assert_called_once_with(
        "AAPL",
        "2026-09-10",
        asset_type="stock",
        analysis_symbol="AAPL",
    )
    graph.save_reports.assert_called_once_with(
        {"trader_investment_plan": "**Action**: Buy"}, "AAPL"
    )
    refresh_gate.assert_called_once_with("C", "Overweight", "**Action**: Buy")


@pytest.mark.parametrize(
    ("base", "refresh", "exit_policy", "expected"),
    [
        ("REPORT_ONLY", "INVALIDATED", False, "REPORT_ONLY"),
        ("HOLD", "INVALIDATED", False, "HOLD"),
        ("HOLD", "INVALIDATED", True, "EXIT"),
        ("REVIEW", "INVALIDATED", True, "EXIT"),
        ("EXIT", "CONFIRMED", False, "EXIT"),
        ("HOLD", "NEUTRAL", True, "HOLD"),
        ("REVIEW", "NEUTRAL", True, "REVIEW"),
    ],
)
def test_final_status_only_uses_explicit_opposite_thesis_policy(
    base, refresh, exit_policy, expected
):
    assert _load_script()._final_status(
        base, refresh, exit_on_invalidation=exit_policy
    ) == expected


def test_roll_tuning_args_require_plan_roll_before_snapshot_fetch(capsys):
    module = _load_script()
    with patch.object(module, "fetch_equity_option_snapshot") as fetch:
        rc = module.main(
            [
                "AAPL260918C00300000",
                "--entry-premium",
                "5",
                "--contracts",
                "1",
                "--roll-target-delta",
                "0.55",
            ]
        )

    assert rc == 2
    fetch.assert_not_called()
    assert "roll tuning arguments require --plan-roll" in capsys.readouterr().out


def test_plan_roll_neutral_refresh_never_calls_selector(capsys):
    module = _load_script()
    snapshot = _snapshot()
    refresh = _refresh_result("NEUTRAL", current_direction=None)
    no_roll = SimpleNamespace(status="NO_ROLL", report="ROLL NO_ROLL")
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ),
        patch.object(
            module,
            "evaluate_long_option_position",
            return_value=SimpleNamespace(status="HOLD", report="POSITION HOLD"),
        ),
        patch.object(
            module,
            "_refresh_underlying_thesis",
            return_value=(refresh, "/tmp/AAPL-report.md"),
        ) as refresh_call,
        patch.object(module, "rank_equity_option_contracts") as selector,
        patch.object(module, "plan_long_option_roll", return_value=no_roll) as planner,
        patch.object(module, "compare_hold_vs_roll") as hold_roll,
    ):
        rc = module.main(
            [
                snapshot.symbol,
                "--entry-premium",
                "5",
                "--contracts",
                "1",
                "--date",
                "2026-09-10",
                "--plan-roll",
            ]
        )

    assert rc == 0
    refresh_call.assert_called_once_with(snapshot, "2026-09-10")
    selector.assert_not_called()
    planner.assert_called_once_with(
        snapshot,
        refresh,
        contracts=1,
        selection=None,
        top_n=3,
    )
    hold_roll.assert_not_called()
    assert "ROLL NO_ROLL" in capsys.readouterr().out


def test_plan_roll_confirmed_refresh_calls_selector_with_current_delta_and_later_dte(capsys):
    module = _load_script()
    snapshot = _snapshot()
    refresh = _refresh_result("CONFIRMED", current_direction="bullish")
    selection = SimpleNamespace(available=True)
    compare = SimpleNamespace(status="COMPARE", report="ROLL COMPARE")
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ),
        patch.object(
            module,
            "evaluate_long_option_position",
            return_value=SimpleNamespace(status="HOLD", report="POSITION HOLD"),
        ),
        patch.object(
            module,
            "_refresh_underlying_thesis",
            return_value=(refresh, "/tmp/AAPL-report.md"),
        ),
        patch.object(
            module, "rank_equity_option_contracts", return_value=selection
        ) as selector,
        patch.object(module, "plan_long_option_roll", return_value=compare) as planner,
        patch.object(
            module,
            "compare_hold_vs_roll",
            return_value=SimpleNamespace(report="HOLD VS ROLL"),
        ) as hold_roll,
    ):
        rc = module.main(
            [
                snapshot.symbol,
                "--entry-premium",
                "5",
                "--contracts",
                "2",
                "--date",
                "2026-09-10",
                "--plan-roll",
            ]
        )

    assert rc == 0
    selector.assert_called_once_with(
        "AAPL",
        "bullish",
        "2026-09-10",
        min_dte=snapshot.dte + 1,
        max_dte=365,
        target_delta=abs(snapshot.delta),
        top_n=3,
    )
    planner.assert_called_once_with(
        snapshot,
        refresh,
        contracts=2,
        selection=selection,
        top_n=3,
    )
    hold_roll.assert_called_once_with(
        snapshot,
        compare,
        entry_premium=5.0,
        contracts=2,
    )
    output = capsys.readouterr().out
    assert "ROLL COMPARE" in output
    assert "HOLD VS ROLL" in output


def test_plan_roll_explicit_tuning_overrides_default_selector_targets():
    module = _load_script()
    snapshot = _snapshot()
    refresh = _refresh_result("CONFIRMED", current_direction="bullish")
    args = module._parser().parse_args(
        [
            snapshot.symbol,
            "--entry-premium",
            "5",
            "--contracts",
            "1",
            "--date",
            "2026-09-10",
            "--plan-roll",
            "--roll-min-dte",
            "30",
            "--roll-max-dte",
            "90",
            "--roll-target-delta",
            "0.50",
            "--roll-top",
            "5",
        ]
    )
    with patch.object(module, "rank_equity_option_contracts") as selector:
        module._rank_roll_candidates(snapshot, refresh, args)

    selector.assert_called_once_with(
        "AAPL",
        "bullish",
        "2026-09-10",
        min_dte=30,
        max_dte=90,
        target_delta=0.50,
        top_n=5,
    )


def test_plan_roll_invalidated_refresh_never_calls_selector():
    module = _load_script()
    snapshot = _snapshot()
    refresh = _refresh_result("INVALIDATED", current_direction="bearish")
    args = module._parser().parse_args(
        [
            snapshot.symbol,
            "--entry-premium",
            "5",
            "--contracts",
            "1",
            "--plan-roll",
        ]
    )
    with patch.object(module, "rank_equity_option_contracts") as selector:
        assert module._rank_roll_candidates(snapshot, refresh, args) is None

    selector.assert_not_called()


def test_plan_roll_missing_current_delta_requires_explicit_target_before_selector():
    module = _load_script()
    snapshot = _snapshot()
    snapshot = snapshot.__class__(**{**snapshot.__dict__, "delta": None})
    refresh = _refresh_result("CONFIRMED", current_direction="bullish")
    args = module._parser().parse_args(
        [
            snapshot.symbol,
            "--entry-premium",
            "5",
            "--contracts",
            "1",
            "--plan-roll",
        ]
    )
    with (
        patch.object(module, "rank_equity_option_contracts") as selector,
        pytest.raises(ValueError, match="provide --roll-target-delta"),
    ):
        module._rank_roll_candidates(snapshot, refresh, args)

    selector.assert_not_called()


def test_roll_review_returns_two_for_fail_closed_automation():
    module = _load_script()
    snapshot = _snapshot()
    refresh = _refresh_result("CONFIRMED", current_direction="bullish")
    selection = SimpleNamespace(available=True)
    review = SimpleNamespace(status="REVIEW", report="ROLL REVIEW")
    with (
        patch.object(
            module,
            "fetch_equity_option_snapshot",
            return_value=EquityOptionSnapshotResult(snapshot),
        ),
        patch.object(
            module,
            "evaluate_long_option_position",
            return_value=SimpleNamespace(status="HOLD", report="POSITION HOLD"),
        ),
        patch.object(
            module,
            "_refresh_underlying_thesis",
            return_value=(refresh, "/tmp/AAPL-report.md"),
        ),
        patch.object(module, "rank_equity_option_contracts", return_value=selection),
        patch.object(module, "plan_long_option_roll", return_value=review),
    ):
        rc = module.main(
            [
                snapshot.symbol,
                "--entry-premium",
                "5",
                "--contracts",
                "1",
                "--plan-roll",
            ]
        )

    assert rc == 2
