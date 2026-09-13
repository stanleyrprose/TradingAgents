from datetime import date

import pytest

from tradingagents.dataflows.equity_options import (
    EquityOptionSnapshot,
    EquityOptionSnapshotResult,
)
from tradingagents.option_portfolio_dashboard import build_option_portfolio_dashboard
from tradingagents.option_position_registry import OptionPositionRegistry

TODAY = date(2026, 9, 11)
CALL1 = "AAPL260925C00320000"
CALL2 = "AAPL261016C00320000"
PUT1 = "AAPL260925P00320000"
MSFT_CALL = "MSFT261016C00400000"


def _registry(tmp_path):
    return OptionPositionRegistry(tmp_path / "options.sqlite3")


def _open(
    registry,
    symbol,
    position_id,
    *,
    entry=8.25,
    contracts=1,
    exit_policy=None,
    thesis=None,
):
    return registry.open_position(
        symbol,
        entry_premium=entry,
        contracts=contracts,
        entry_date="2026-09-11",
        exit_policy=exit_policy,
        initial_thesis=thesis or {},
        position_id=position_id,
    )


def _snapshot(
    symbol,
    *,
    underlying="AAPL",
    strike=320.0,
    dte=14,
    spot=325.0,
    bid=10.0,
    ask=10.5,
    delta=0.60,
    gamma=0.02,
    vega=0.30,
    theta=-0.20,
):
    right = "P" if "P" in symbol[-9:-8] else "C"
    expiry = date(2026, 9, 25) if dte == 14 else date(2026, 10, 16)
    midpoint = None if bid is None or ask is None else (bid + ask) / 2
    spread = None if midpoint in (None, 0) else (ask - bid) / midpoint * 100
    return EquityOptionSnapshotResult(
        EquityOptionSnapshot(
            symbol=symbol,
            underlying=underlying,
            expiry=expiry,
            right=right,
            strike=strike,
            as_of=TODAY,
            dte=dte,
            source_timestamp="2026-09-11 10:00:00",
            underlying_spot=spot,
            bid=bid,
            ask=ask,
            midpoint=midpoint,
            spread_pct=spread,
            last=bid,
            iv=0.25,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            open_interest=100,
            volume=10,
        )
    )


def test_dashboard_triage_uses_only_explicit_policy_semantics_and_fixed_order(tmp_path):
    registry = _registry(tmp_path)
    exit_pos = _open(
        registry,
        CALL1,
        "exit",
        exit_policy={"take_profit_pct": 10.0},
    )
    review_pos = _open(registry, PUT1, "review", exit_policy={"stop_loss_pct": 20.0})
    hold_pos = _open(
        registry,
        MSFT_CALL,
        "hold",
        entry=12.0,
        exit_policy={"take_profit_pct": 50.0, "stop_loss_pct": 40.0},
    )
    report_pos = _open(registry, CALL2, "report", entry=15.0)
    positions = (report_pos, hold_pos, review_pos, exit_pos)
    snapshots = {
        CALL1: _snapshot(CALL1, bid=10.0),
        PUT1: EquityOptionSnapshotResult(None, "Timeout"),
        MSFT_CALL: _snapshot(
            MSFT_CALL,
            underlying="MSFT",
            dte=35,
            spot=410,
            bid=12.5,
            ask=13.0,
            delta=0.55,
        ),
        CALL2: _snapshot(CALL2, dte=35, bid=16.0, ask=16.5),
    }

    result = build_option_portfolio_dashboard(positions, snapshots, as_of=TODAY)

    assert [row.position_id for row in result.rows] == ["exit", "review", "hold", "report"]
    assert [row.status for row in result.rows] == ["EXIT", "REVIEW", "HOLD", "REPORT_ONLY"]
    assert result.exit_count == 1
    assert result.review_count == 1
    assert result.hold_count == 1
    assert result.report_only_count == 1
    assert "hidden weighted score" in result.report
    assert "stored thesis" in result.report.lower()


def test_rolled_position_separates_current_leg_and_lifecycle_pnl(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry, CALL1, "rolled", contracts=2)
    rolled = registry.record_roll(
        position.position_id,
        CALL2,
        close_credit=11.0,
        new_entry_premium=15.0,
        roll_date="2026-09-11",
    )
    snapshots = {CALL2: _snapshot(CALL2, dte=35, bid=17.0, ask=17.5)}

    result = build_option_portfolio_dashboard((rolled,), snapshots, as_of=TODAY)
    row = result.rows[0]

    assert row.current_leg_entry_premium == pytest.approx(15.0)
    assert row.lifecycle_net_premium_per_share == pytest.approx(12.25)
    assert row.current_leg_pnl_dollars == pytest.approx(400.0)
    assert row.lifecycle_pnl_dollars == pytest.approx(950.0)
    assert result.total_current_leg_pnl_dollars == pytest.approx(400.0)
    assert result.total_lifecycle_pnl_dollars == pytest.approx(950.0)


def test_underlying_summary_nets_only_within_same_underlying(tmp_path):
    registry = _registry(tmp_path)
    call = _open(registry, CALL1, "call")
    put = _open(registry, PUT1, "put")
    msft = _open(registry, MSFT_CALL, "msft", entry=12.0)
    snapshots = {
        CALL1: _snapshot(CALL1, delta=0.60, gamma=0.02, theta=-0.20),
        PUT1: _snapshot(PUT1, delta=-0.40, gamma=0.03, theta=-0.25),
        MSFT_CALL: _snapshot(
            MSFT_CALL,
            underlying="MSFT",
            dte=35,
            spot=410,
            bid=12.5,
            ask=13,
            delta=0.50,
            gamma=0.01,
            theta=-0.10,
        ),
    }

    result = build_option_portfolio_dashboard((call, put, msft), snapshots, as_of=TODAY)
    by_underlying = {item.underlying: item for item in result.underlyings}

    assert by_underlying["AAPL"].net_delta_shares == pytest.approx(20.0)
    assert by_underlying["AAPL"].gross_delta_shares == pytest.approx(100.0)
    assert by_underlying["AAPL"].net_gamma_delta_shares_per_dollar == pytest.approx(5.0)
    assert by_underlying["MSFT"].net_delta_shares == pytest.approx(50.0)
    assert "Cross-underlying share deltas are not netted together" in result.report


def test_unavailable_snapshot_makes_strict_book_totals_unavailable(tmp_path):
    registry = _registry(tmp_path)
    first = _open(registry, CALL1, "first")
    second = _open(registry, PUT1, "second")
    snapshots = {
        CALL1: _snapshot(CALL1),
        PUT1: EquityOptionSnapshotResult(None, "Timeout"),
    }

    result = build_option_portfolio_dashboard((first, second), snapshots, as_of=TODAY)

    assert result.total_liquidation_value is None
    assert result.total_current_leg_pnl_dollars is None
    assert result.total_lifecycle_pnl_dollars is None
    review = next(row for row in result.rows if row.position_id == "second")
    assert review.status == "REVIEW"
    assert review.unavailable_reason == "Timeout"
    assert "market data unavailable: Timeout" in review.attention_reason


def test_missing_batch_key_is_review_not_exception(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry, CALL1, "missing")

    result = build_option_portfolio_dashboard((position,), {}, as_of=TODAY)

    assert result.rows[0].status == "REVIEW"
    assert "snapshot missing from batch result" in result.rows[0].attention_reason


def test_same_status_uses_shorter_dte_then_stable_identity(tmp_path):
    registry = _registry(tmp_path)
    near = _open(registry, CALL1, "z_near")
    far = _open(registry, CALL2, "a_far", entry=15.0)
    snapshots = {
        CALL1: _snapshot(CALL1, dte=14),
        CALL2: _snapshot(CALL2, dte=35),
    }

    result = build_option_portfolio_dashboard((far, near), snapshots, as_of=TODAY)
    assert [row.position_id for row in result.rows] == ["z_near", "a_far"]


def test_closed_positions_are_rejected(tmp_path):
    registry = _registry(tmp_path)
    position = _open(registry, CALL1, "closed")
    closed = registry.close_position(
        position.position_id,
        close_premium=9.0,
        close_date="2026-09-11",
    )

    with pytest.raises(ValueError, match="OPEN positions only"):
        build_option_portfolio_dashboard((closed,), {}, as_of=TODAY)


def test_empty_book_is_valid_and_has_zero_totals():
    result = build_option_portfolio_dashboard((), {}, as_of=TODAY)

    assert result.rows == ()
    assert result.underlyings == ()
    assert result.total_liquidation_value == 0
    assert result.total_current_leg_pnl_dollars == 0
    assert result.total_lifecycle_pnl_dollars == 0
    assert "Open positions: 0" in result.report
