import importlib.util
import json
from pathlib import Path

import pytest

from tradingagents.option_position_registry import OptionPositionRegistry

CALL1 = "AAPL260925C00320000"
CALL2 = "AAPL261016C00320000"
PUT1 = "AAPL260925P00320000"


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "options_registry.py"
    spec = importlib.util.spec_from_file_location("options_registry_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _db(tmp_path):
    return tmp_path / "options.sqlite3"


def _open(module, db, *, position_id="opt_cli", symbol=CALL1):
    return module.main(
        [
            "--db",
            str(db),
            "open",
            symbol,
            "--entry-premium",
            "8.25",
            "--contracts",
            "2",
            "--entry-date",
            "2026-09-11",
            "--position-id",
            position_id,
            "--take-profit-pct",
            "50",
            "--stop-loss-pct",
            "40",
            "--max-abs-delta-shares",
            "150",
            "--thesis-direction",
            "bullish" if "C" in symbol[-9:-8] else "bearish",
            "--thesis-note",
            "initial view",
        ]
    )


def test_open_persists_policies_and_initial_thesis(tmp_path, capsys):
    module = _load_script()
    db = _db(tmp_path)

    assert _open(module, db) == 0
    output = capsys.readouterr().out
    assert "Position ID: opt_cli" in output
    assert CALL1 in output

    position = OptionPositionRegistry(db).get_position("opt_cli")
    assert position.exit_policy == {"stop_loss_pct": 40.0, "take_profit_pct": 50.0}
    assert position.greek_limits == {"max_abs_delta_shares": 150.0}
    assert position.initial_thesis == {
        "direction": "bullish",
        "note": "initial view",
    }


def test_list_and_show_json_are_machine_readable(tmp_path, capsys):
    module = _load_script()
    db = _db(tmp_path)
    assert _open(module, db) == 0
    capsys.readouterr()

    assert module.main(["--db", str(db), "--json", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) == 1
    assert listed[0]["position_id"] == "opt_cli"

    assert module.main(["--db", str(db), "--json", "show", "AAPL"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["current_symbol"] == CALL1
    assert shown["lifecycle_net_premium_per_share"] == pytest.approx(8.25)


def test_events_show_append_only_lifecycle_order(tmp_path, capsys):
    module = _load_script()
    db = _db(tmp_path)
    assert _open(module, db) == 0
    capsys.readouterr()

    assert module.main(
        [
            "--db",
            str(db),
            "thesis",
            "AAPL",
            "--date",
            "2026-09-12",
            "--status",
            "CONFIRMED",
            "--current-direction",
            "bullish",
            "--portfolio-rating",
            "Overweight",
            "--trader-action",
            "Buy",
        ]
    ) == 0
    capsys.readouterr()
    assert module.main(
        [
            "--db",
            str(db),
            "roll",
            "AAPL",
            "--new-symbol",
            CALL2,
            "--close-credit",
            "11",
            "--new-entry-premium",
            "15",
            "--date",
            "2026-09-20",
        ]
    ) == 0
    capsys.readouterr()

    assert module.main(["--db", str(db), "--json", "events", "opt_cli"]) == 0
    events = json.loads(capsys.readouterr().out)
    assert [event["event_type"] for event in events] == [
        "OPEN",
        "THESIS_REFRESH",
        "ROLL",
    ]


def test_policy_replaces_only_supplied_group_and_can_clear(tmp_path, capsys):
    module = _load_script()
    db = _db(tmp_path)
    assert _open(module, db) == 0
    capsys.readouterr()

    assert module.main(
        [
            "--db",
            str(db),
            "policy",
            "AAPL",
            "--take-profit-pct",
            "60",
            "--date",
            "2026-09-12",
        ]
    ) == 0
    capsys.readouterr()
    position = OptionPositionRegistry(db).get_position("opt_cli")
    assert position.exit_policy == {"take_profit_pct": 60.0}
    assert position.greek_limits == {"max_abs_delta_shares": 150.0}

    assert module.main(
        [
            "--db",
            str(db),
            "policy",
            "AAPL",
            "--clear-greek-limits",
            "--date",
            "2026-09-13",
        ]
    ) == 0
    capsys.readouterr()
    position = OptionPositionRegistry(db).get_position("opt_cli")
    assert position.greek_limits == {}
    assert position.exit_policy == {"take_profit_pct": 60.0}


def test_policy_without_replacement_fails_without_mutation(tmp_path, capsys):
    module = _load_script()
    db = _db(tmp_path)
    assert _open(module, db) == 0
    capsys.readouterr()
    registry = OptionPositionRegistry(db)
    before = registry.get_position("opt_cli")
    event_count = len(registry.list_events("opt_cli"))

    assert module.main(["--db", str(db), "policy", "AAPL"]) == 2
    assert "requires an exit/Greek policy replacement" in capsys.readouterr().out
    after = registry.get_position("opt_cli")
    assert after.exit_policy == before.exit_policy
    assert len(registry.list_events("opt_cli")) == event_count


def test_roll_command_updates_current_leg_and_lifecycle_basis(tmp_path, capsys):
    module = _load_script()
    db = _db(tmp_path)
    assert _open(module, db) == 0
    capsys.readouterr()

    assert module.main(
        [
            "--db",
            str(db),
            "roll",
            "AAPL",
            "--new-symbol",
            CALL2,
            "--close-credit",
            "11",
            "--new-entry-premium",
            "15",
            "--date",
            "2026-09-20",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert CALL2 in output
    assert "Lifecycle net premium/share: $12.2500" in output

    position = OptionPositionRegistry(db).get_position("opt_cli")
    assert position.current_leg_entry_premium == pytest.approx(15.0)
    assert position.lifecycle_net_premium_per_share == pytest.approx(12.25)
    assert position.roll_count == 1


def test_close_command_moves_position_out_of_default_open_list(tmp_path, capsys):
    module = _load_script()
    db = _db(tmp_path)
    assert _open(module, db) == 0
    capsys.readouterr()

    assert module.main(
        [
            "--db",
            str(db),
            "close",
            "AAPL",
            "--close-premium",
            "10",
            "--date",
            "2026-09-12",
            "--reason",
            "manual test close",
        ]
    ) == 0
    capsys.readouterr()

    assert module.main(["--db", str(db), "--json", "list"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert module.main(["--db", str(db), "--json", "list", "--closed"]) == 0
    closed = json.loads(capsys.readouterr().out)
    assert closed[0]["status"] == "CLOSED"

    assert module.main(["--db", str(db), "--json", "show", "opt_cli"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["close_premium"] == pytest.approx(10.0)


def test_ambiguous_underlying_fails_closed_and_requires_position_id(tmp_path, capsys):
    module = _load_script()
    db = _db(tmp_path)
    assert _open(module, db, position_id="opt_call") == 0
    capsys.readouterr()
    assert _open(module, db, position_id="opt_put", symbol=PUT1) == 0
    capsys.readouterr()

    assert module.main(["--db", str(db), "show", "AAPL"]) == 2
    output = capsys.readouterr().out
    assert "multiple open option positions" in output
    assert "opt_call" in output and "opt_put" in output

    assert module.main(["--db", str(db), "show", "opt_put"]) == 0
    assert PUT1 in capsys.readouterr().out


def test_invalid_open_policy_does_not_create_position(tmp_path, capsys):
    module = _load_script()
    db = _db(tmp_path)

    rc = module.main(
        [
            "--db",
            str(db),
            "open",
            CALL1,
            "--entry-premium",
            "8.25",
            "--contracts",
            "2",
            "--entry-date",
            "2026-09-11",
            "--stop-loss-pct",
            "101",
        ]
    )

    assert rc == 2
    assert "stop_loss_pct" in capsys.readouterr().out
    assert OptionPositionRegistry(db).list_positions() == ()


def test_thesis_command_persists_optional_fields_without_none_values(tmp_path, capsys):
    module = _load_script()
    db = _db(tmp_path)
    assert _open(module, db) == 0
    capsys.readouterr()

    assert module.main(
        [
            "--db",
            str(db),
            "thesis",
            "AAPL",
            "--status",
            "NEUTRAL",
            "--current-direction",
            "none",
            "--reason",
            "no consensus",
            "--date",
            "2026-09-12",
        ]
    ) == 0
    capsys.readouterr()

    position = OptionPositionRegistry(db).get_position("opt_cli")
    assert position.latest_thesis == {"reason": "no consensus", "status": "NEUTRAL"}
