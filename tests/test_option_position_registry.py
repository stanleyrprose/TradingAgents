import json
import os
import sqlite3
from pathlib import Path

import pytest

from tradingagents.option_position_registry import (
    OptionPositionRegistry,
    default_registry_path,
)

CALL1 = "AAPL260925C00320000"
CALL2 = "AAPL261016C00320000"
CALL3 = "AAPL261120C00320000"
PUT1 = "AAPL260925P00320000"
MSFT_CALL = "MSFT261016C00400000"


def _registry(tmp_path):
    return OptionPositionRegistry(tmp_path / "options.sqlite3")


def _open(registry, **overrides):
    values = {
        "symbol": CALL1,
        "entry_premium": 8.25,
        "contracts": 2,
        "entry_date": "2026-09-11",
        "exit_policy": {"take_profit_pct": 50.0, "stop_loss_pct": 40.0},
        "greek_limits": {"max_abs_delta_shares": 150.0},
        "initial_thesis": {"direction": "bullish", "source": "manual"},
        "position_id": "opt_test",
    }
    values.update(overrides)
    return registry.open_position(**values)


def test_default_registry_path_can_be_overridden(monkeypatch, tmp_path):
    target = tmp_path / "custom.sqlite3"
    monkeypatch.setenv("TRADINGAGENTS_OPTION_REGISTRY_DB", str(target))
    assert default_registry_path() == target


def test_open_position_creates_projection_and_open_event(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)

    assert position.position_id == "opt_test"
    assert position.underlying == "AAPL"
    assert position.option_right == "C"
    assert position.status == "OPEN"
    assert position.original_symbol == CALL1
    assert position.current_symbol == CALL1
    assert position.original_entry_premium == pytest.approx(8.25)
    assert position.current_leg_entry_premium == pytest.approx(8.25)
    assert position.lifecycle_net_premium_per_share == pytest.approx(8.25)
    assert position.contracts == 2
    assert position.opened_at == "2026-09-11"
    assert position.current_leg_opened_at == "2026-09-11"
    assert position.exit_policy["take_profit_pct"] == 50.0
    assert position.greek_limits["max_abs_delta_shares"] == 150.0
    assert position.initial_thesis == {"direction": "bullish", "source": "manual"}
    assert position.latest_thesis == position.initial_thesis
    assert position.roll_count == 0

    events = registry.list_events(position.position_id)
    assert [event.event_type for event in events] == ["OPEN"]
    assert events[0].payload["symbol"] == CALL1
    assert events[0].payload["entry_premium"] == 8.25


def test_database_uses_wal_and_schema_version_one(tmp_path):
    registry = _registry(tmp_path)
    with sqlite3.connect(registry.path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_registry_file_is_owner_only_when_chmod_supported(tmp_path):
    registry = _registry(tmp_path)
    if os.name == "posix":
        assert registry.path.stat().st_mode & 0o777 == 0o600


def test_unknown_schema_version_fails_closed(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=99")
    with pytest.raises(RuntimeError, match="unsupported option registry schema version 99"):
        OptionPositionRegistry(path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"symbol": "AAPL"}, "exact OCC"),
        ({"entry_premium": 0}, "entry_premium"),
        ({"entry_premium": float("nan")}, "entry_premium"),
        ({"contracts": 0}, "contracts"),
        ({"contracts": True}, "contracts"),
        ({"entry_date": "2026-09-26"}, "after option expiry"),
        ({"entry_date": "bad"}, "YYYY-MM-DD"),
    ],
)
def test_open_position_rejects_invalid_trade_facts(tmp_path, kwargs, match):
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match=match):
        _open(registry, **kwargs)


def test_json_payload_rejects_nan_and_nonserializable_values(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="JSON-serializable finite values"):
        _open(registry, greek_limits={"bad": float("nan")})
    with pytest.raises(ValueError, match="JSON-serializable finite values"):
        _open(registry, initial_thesis={"bad": Path("x")})


def test_list_and_resolve_open_position_by_id_symbol_or_underlying(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)

    assert registry.list_positions() == (position,)
    assert registry.list_positions(underlying="aapl") == (position,)
    assert registry.resolve_open_position(position.position_id) == position
    assert registry.resolve_open_position(CALL1.lower()) == position
    assert registry.resolve_open_position("aapl") == position


def test_underlying_resolution_fails_closed_when_multiple_open_positions_match(tmp_path):
    registry = _registry(tmp_path)
    first = _open(registry, position_id="opt_one")
    second = _open(
        registry,
        symbol=PUT1,
        position_id="opt_two",
        initial_thesis={"direction": "bearish"},
    )

    with pytest.raises(ValueError, match="multiple open option positions") as exc:
        registry.resolve_open_position("AAPL")
    assert first.position_id in str(exc.value)
    assert second.position_id in str(exc.value)
    assert registry.resolve_open_position(first.position_id) == first


def test_unknown_open_position_query_raises_key_error(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(KeyError, match="no open option position"):
        registry.resolve_open_position("AAPL")


def test_set_policies_updates_projection_and_appends_event(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)

    updated = registry.set_policies(
        position.position_id,
        exit_policy={"take_profit_pct": 60.0},
        occurred_at="2026-09-12",
    )

    assert updated.exit_policy == {"take_profit_pct": 60.0}
    assert updated.greek_limits == position.greek_limits
    events = registry.list_events(position.position_id)
    assert [event.event_type for event in events] == ["OPEN", "POLICY_UPDATE"]
    assert events[-1].occurred_at == "2026-09-12"
    assert events[-1].payload["greek_limits"] == position.greek_limits


def test_record_thesis_updates_latest_without_rewriting_initial(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)
    refreshed = {
        "status": "CONFIRMED",
        "current_direction": "bullish",
        "portfolio_rating": "Overweight",
        "trader_action": "Buy",
        "report_path": "/tmp/report.md",
    }

    updated = registry.record_thesis(
        position.position_id,
        refreshed,
        occurred_at="2026-09-12",
    )

    assert updated.initial_thesis == position.initial_thesis
    assert updated.latest_thesis == refreshed
    assert registry.list_events(position.position_id)[-1].event_type == "THESIS_REFRESH"


def test_roll_preserves_original_basis_and_updates_current_and_lifecycle_basis(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)

    rolled = registry.record_roll(
        position.position_id,
        CALL2,
        close_credit=11.0,
        new_entry_premium=15.0,
        roll_date="2026-09-20",
    )

    assert rolled.original_symbol == CALL1
    assert rolled.original_entry_premium == pytest.approx(8.25)
    assert rolled.current_symbol == CALL2
    assert rolled.current_leg_entry_premium == pytest.approx(15.0)
    assert rolled.lifecycle_net_premium_per_share == pytest.approx(12.25)
    assert rolled.current_leg_opened_at == "2026-09-20"
    assert rolled.contracts == 2
    assert rolled.roll_count == 1

    event = registry.list_events(position.position_id)[-1]
    assert event.event_type == "ROLL"
    assert event.payload["from_symbol"] == CALL1
    assert event.payload["to_symbol"] == CALL2
    assert event.payload["previous_lifecycle_net_premium_per_share"] == pytest.approx(8.25)
    assert event.payload["new_lifecycle_net_premium_per_share"] == pytest.approx(12.25)


def test_second_roll_accumulates_lifecycle_net_premium(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)
    registry.record_roll(
        position.position_id,
        CALL2,
        close_credit=11.0,
        new_entry_premium=15.0,
        roll_date="2026-09-20",
    )
    rolled = registry.record_roll(
        position.position_id,
        CALL3,
        close_credit=16.0,
        new_entry_premium=18.0,
        roll_date="2026-10-10",
    )

    assert rolled.current_leg_entry_premium == pytest.approx(18.0)
    assert rolled.lifecycle_net_premium_per_share == pytest.approx(14.25)
    assert rolled.roll_count == 2
    assert [event.event_type for event in registry.list_events(position.position_id)] == [
        "OPEN",
        "ROLL",
        "ROLL",
    ]


@pytest.mark.parametrize(
    ("symbol", "roll_date", "match"),
    [
        (PUT1, "2026-09-20", "same underlying and call/put direction"),
        (MSFT_CALL, "2026-09-20", "same underlying and call/put direction"),
        (CALL1, "2026-09-20", "strictly later"),
        (CALL2, "2026-09-10", "cannot precede"),
        (CALL2, "2026-10-17", "after replacement expiry"),
    ],
)
def test_roll_rejects_invalid_replacement_or_time(tmp_path, symbol, roll_date, match):
    registry = _registry(tmp_path)
    position = _open(registry)
    with pytest.raises(ValueError, match=match):
        registry.record_roll(
            position.position_id,
            symbol,
            close_credit=11.0,
            new_entry_premium=15.0,
            roll_date=roll_date,
        )


def test_close_position_records_final_lifecycle_pnl_and_removes_from_open_resolution(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)
    registry.record_roll(
        position.position_id,
        CALL2,
        close_credit=11.0,
        new_entry_premium=15.0,
        roll_date="2026-09-20",
    )

    closed = registry.close_position(
        position.position_id,
        close_premium=17.0,
        close_date="2026-10-10",
        reason="manual exit",
    )

    assert closed.status == "CLOSED"
    assert closed.closed_at == "2026-10-10"
    assert closed.close_premium == pytest.approx(17.0)
    assert registry.list_positions() == ()
    assert registry.list_positions(status="CLOSED") == (closed,)
    with pytest.raises(KeyError):
        registry.resolve_open_position(position.position_id)

    event = registry.list_events(position.position_id)[-1]
    assert event.event_type == "CLOSE"
    assert event.payload["final_lifecycle_net_pnl_per_share"] == pytest.approx(4.75)
    assert event.payload["reason"] == "manual exit"


def test_close_date_cannot_precede_current_leg_open(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)
    with pytest.raises(ValueError, match="cannot precede"):
        registry.close_position(
            position.position_id,
            close_premium=9.0,
            close_date="2026-09-10",
        )


def test_closed_position_rejects_mutations(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)
    registry.close_position(
        position.position_id,
        close_premium=9.0,
        close_date="2026-09-12",
    )

    with pytest.raises(ValueError, match="closed"):
        registry.set_policies(position.position_id, exit_policy={})
    with pytest.raises(ValueError, match="closed"):
        registry.record_thesis(position.position_id, {"status": "NEUTRAL"})
    with pytest.raises(ValueError, match="closed"):
        registry.record_roll(
            position.position_id,
            CALL2,
            close_credit=9.0,
            new_entry_premium=12.0,
            roll_date="2026-09-12",
        )
    with pytest.raises(ValueError, match="already closed"):
        registry.close_position(
            position.position_id,
            close_premium=9.0,
            close_date="2026-09-12",
        )


def test_event_journal_is_append_only_at_sqlite_layer(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)
    event = registry.list_events(position.position_id)[0]

    with sqlite3.connect(registry.path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE option_position_events SET event_type='CHANGED' WHERE event_id=?",
                (event.event_id,),
            )
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "DELETE FROM option_position_events WHERE event_id=?",
                (event.event_id,),
            )


def test_full_event_order_is_auditable(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry)
    registry.set_policies(
        position.position_id,
        exit_policy={"take_profit_pct": 60},
        occurred_at="2026-09-12",
    )
    registry.record_thesis(
        position.position_id,
        {"status": "CONFIRMED", "current_direction": "bullish"},
        occurred_at="2026-09-13",
    )
    registry.record_roll(
        position.position_id,
        CALL2,
        close_credit=11,
        new_entry_premium=15,
        roll_date="2026-09-20",
    )
    registry.close_position(
        position.position_id,
        close_premium=17,
        close_date="2026-10-10",
    )

    events = registry.list_events(position.position_id)
    assert [event.event_type for event in events] == [
        "OPEN",
        "POLICY_UPDATE",
        "THESIS_REFRESH",
        "ROLL",
        "CLOSE",
    ]
    assert [event.occurred_at for event in events] == [
        "2026-09-11",
        "2026-09-12",
        "2026-09-13",
        "2026-09-20",
        "2026-10-10",
    ]
    json.dumps([event.payload for event in events], allow_nan=False)
