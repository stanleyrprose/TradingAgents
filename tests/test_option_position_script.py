import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
