import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tradingagents.option_position_registry import OptionPositionRegistry

CALL1 = "AAPL260925C00320000"
CALL2 = "AAPL261016C00320000"
PUT1 = "AAPL260925P00320000"


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "manage_options.py"
    spec = importlib.util.spec_from_file_location("manage_options_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry(tmp_path):
    return OptionPositionRegistry(tmp_path / "options.sqlite3")


def _open(registry, *, position_id="opt_call", symbol=CALL1):
    return registry.open_position(
        symbol,
        entry_premium=8.25,
        contracts=2,
        entry_date="2026-09-11",
        exit_policy={
            "take_profit_pct": 50.0,
            "stop_loss_pct": 40.0,
            "exit_at_dte": 5,
            "max_theta_burn_pct_per_day": 5.0,
        },
        greek_limits={"max_abs_delta_shares": 150.0},
        initial_thesis={"direction": "bullish" if symbol == CALL1 else "bearish"},
        position_id=position_id,
    )


def test_unique_underlying_resolves_and_passes_stored_basis_and_exit_policy(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry)
    manager_main = MagicMock(return_value=0)

    with patch.object(
        module,
        "_load_position_manager",
        return_value=SimpleNamespace(main=manager_main),
    ):
        rc = module.main(
            [
                "AAPL",
                "--db",
                str(registry.path),
                "--date",
                "2026-09-11",
            ]
        )

    assert rc == 0
    manager_main.assert_called_once_with(
        [
            CALL1,
            "--entry-premium",
            "8.25",
            "--lifecycle-net-premium",
            "8.25",
            "--contracts",
            "2",
            "--date",
            "2026-09-11",
            "--take-profit-pct",
            "50.0",
            "--stop-loss-pct",
            "40.0",
            "--exit-at-dte",
            "5",
            "--max-theta-burn-pct-per-day",
            "5.0",
        ]
    )
    output = capsys.readouterr().out
    assert "Position ID: opt_call" in output
    assert "Lifecycle net premium basis: $8.2500/share" in output
    assert "Greek-policy note" in output


def test_after_roll_passes_current_leg_cost_and_cumulative_lifecycle_basis_separately(tmp_path):
    module = _load_script()
    registry = _registry(tmp_path)
    position = _open(registry)
    registry.record_roll(
        position.position_id,
        CALL2,
        close_credit=11.0,
        new_entry_premium=15.0,
        roll_date="2026-09-20",
    )
    manager_main = MagicMock(return_value=0)

    with patch.object(
        module,
        "_load_position_manager",
        return_value=SimpleNamespace(main=manager_main),
    ):
        rc = module.main(
            [
                "AAPL",
                "--db",
                str(registry.path),
                "--date",
                "2026-09-20",
            ]
        )

    assert rc == 0
    argv = manager_main.call_args.args[0]
    assert argv[0] == CALL2
    assert argv[argv.index("--entry-premium") + 1] == "15.0"
    assert argv[argv.index("--lifecycle-net-premium") + 1] == "12.25"
    assert argv[argv.index("--contracts") + 1] == "2"


def test_refresh_and_roll_tuning_flags_are_forwarded_without_journal_mutation(tmp_path):
    module = _load_script()
    registry = _registry(tmp_path)
    position = _open(registry)
    before_events = registry.list_events(position.position_id)
    manager_main = MagicMock(return_value=0)

    with patch.object(
        module,
        "_load_position_manager",
        return_value=SimpleNamespace(main=manager_main),
    ):
        rc = module.main(
            [
                position.position_id,
                "--db",
                str(registry.path),
                "--date",
                "2026-09-11",
                "--refresh-thesis",
                "--exit-on-thesis-invalidation",
                "--plan-roll",
                "--roll-top",
                "5",
                "--roll-min-dte",
                "30",
                "--roll-max-dte",
                "90",
                "--roll-target-delta",
                "0.55",
            ]
        )

    assert rc == 0
    argv = manager_main.call_args.args[0]
    for token in (
        "--refresh-thesis",
        "--exit-on-thesis-invalidation",
        "--plan-roll",
        "--roll-top",
        "--roll-min-dte",
        "--roll-max-dte",
        "--roll-target-delta",
    ):
        assert token in argv
    assert registry.list_events(position.position_id) == before_events


def test_resolve_only_never_loads_position_manager_or_external_tools(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    position = _open(registry)

    with patch.object(module, "_load_position_manager") as loader:
        rc = module.main(
            [
                position.position_id,
                "--db",
                str(registry.path),
                "--date",
                "2026-09-11",
                "--plan-roll",
                "--resolve-only",
            ]
        )

    assert rc == 0
    loader.assert_not_called()
    output = capsys.readouterr().out
    assert "# Resolved manager inputs" in output
    assert CALL1 in output
    assert "--plan-roll" in output


def test_multiple_open_positions_for_underlying_fail_before_manager_load(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry, position_id="opt_call")
    _open(registry, position_id="opt_put", symbol=PUT1)

    with patch.object(module, "_load_position_manager") as loader:
        rc = module.main(["AAPL", "--db", str(registry.path)])

    assert rc == 2
    loader.assert_not_called()
    output = capsys.readouterr().out
    assert "multiple open option positions" in output
    assert "opt_call" in output and "opt_put" in output


def test_closed_position_is_not_resolved_by_underlying_or_current_symbol(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    position = _open(registry)
    registry.close_position(
        position.position_id,
        close_premium=9.0,
        close_date="2026-09-12",
    )

    with patch.object(module, "_load_position_manager") as loader:
        assert module.main(["AAPL", "--db", str(registry.path)]) == 2
        assert module.main([CALL1, "--db", str(registry.path)]) == 2

    loader.assert_not_called()
    assert "no open option position" in capsys.readouterr().out


def test_current_symbol_after_roll_can_resolve_position(tmp_path):
    module = _load_script()
    registry = _registry(tmp_path)
    position = _open(registry)
    registry.record_roll(
        position.position_id,
        CALL2,
        close_credit=11.0,
        new_entry_premium=15.0,
        roll_date="2026-09-20",
    )
    manager_main = MagicMock(return_value=0)

    with patch.object(
        module,
        "_load_position_manager",
        return_value=SimpleNamespace(main=manager_main),
    ):
        assert module.main([CALL2, "--db", str(registry.path), "--date", "2026-09-20"]) == 0

    assert manager_main.call_args.args[0][0] == CALL2


def test_manager_return_code_is_propagated(tmp_path):
    module = _load_script()
    registry = _registry(tmp_path)
    position = _open(registry)
    manager_main = MagicMock(return_value=2)

    with patch.object(
        module,
        "_load_position_manager",
        return_value=SimpleNamespace(main=manager_main),
    ):
        rc = module.main([position.position_id, "--db", str(registry.path)])

    assert rc == 2


def test_stored_greek_limits_are_displayed_but_not_forwarded_as_hidden_enforcement(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    position = _open(registry)
    manager_main = MagicMock(return_value=0)

    with patch.object(
        module,
        "_load_position_manager",
        return_value=SimpleNamespace(main=manager_main),
    ):
        assert module.main([position.position_id, "--db", str(registry.path)]) == 0

    argv = manager_main.call_args.args[0]
    assert "--max-abs-delta-shares" not in argv
    output = capsys.readouterr().out
    assert "Stored Greek limits" in output
    assert "not automatically re-enforced" in output


def test_unknown_query_fails_closed_before_manager_load(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)

    with patch.object(module, "_load_position_manager") as loader:
        rc = module.main(["AAPL", "--db", str(registry.path)])

    assert rc == 2
    loader.assert_not_called()
    assert "no open option position" in capsys.readouterr().out
