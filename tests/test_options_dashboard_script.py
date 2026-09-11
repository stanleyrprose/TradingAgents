import importlib.util
import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

from tradingagents.dataflows.equity_options import (
    EquityOptionSnapshot,
    EquityOptionSnapshotResult,
)
from tradingagents.option_position_registry import OptionPositionRegistry

TODAY = date(2026, 9, 11)
CALL1 = "AAPL260925C00320000"
PUT1 = "AAPL260925P00320000"
MSFT_CALL = "MSFT261016C00400000"


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "options_dashboard.py"
    spec = importlib.util.spec_from_file_location("options_dashboard_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry(tmp_path):
    return OptionPositionRegistry(tmp_path / "options.sqlite3")


def _open(registry, symbol, position_id, *, entry=8.25, policy=None):
    return registry.open_position(
        symbol,
        entry_premium=entry,
        contracts=1,
        entry_date=TODAY.isoformat(),
        exit_policy=policy,
        position_id=position_id,
    )


def _snapshot(symbol, *, underlying="AAPL", bid=10.0, delta=0.6, dte=14):
    expiry = date(2026, 9, 25) if dte == 14 else date(2026, 10, 16)
    right = "P" if "P" in symbol[-9:-8] else "C"
    return EquityOptionSnapshotResult(
        EquityOptionSnapshot(
            symbol=symbol,
            underlying=underlying,
            expiry=expiry,
            right=right,
            strike=320.0,
            as_of=TODAY,
            dte=dte,
            source_timestamp="2026-09-11 10:00:00",
            underlying_spot=325.0 if underlying == "AAPL" else 410.0,
            bid=bid,
            ask=bid + 0.5,
            midpoint=bid + 0.25,
            spread_pct=0.5 / (bid + 0.25) * 100,
            last=bid,
            iv=0.25,
            delta=delta,
            gamma=0.02,
            vega=0.30,
            theta=-0.20,
            open_interest=100,
            volume=10,
        )
    )


def test_dashboard_fetches_all_open_symbols_in_one_batch_api_call(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry, CALL1, "aapl_call", policy={"take_profit_pct": 10.0})
    _open(registry, PUT1, "aapl_put")
    _open(registry, MSFT_CALL, "msft", entry=12.0, policy={"take_profit_pct": 50.0})
    snapshots = {
        CALL1: _snapshot(CALL1, bid=10.0),
        PUT1: _snapshot(PUT1, bid=8.0, delta=-0.4),
        MSFT_CALL: _snapshot(MSFT_CALL, underlying="MSFT", bid=12.5, delta=0.5, dte=35),
    }

    with patch.object(module, "fetch_equity_option_snapshots", return_value=snapshots) as fetch:
        rc = module.main(["--db", str(registry.path), "--date", TODAY.isoformat()])

    assert rc == 0
    fetch.assert_called_once_with([CALL1, PUT1, MSFT_CALL], TODAY.isoformat())
    output = capsys.readouterr().out
    assert "# Option portfolio daily dashboard" in output
    assert "EXIT 1" in output
    assert "REPORT_ONLY 1" in output
    assert "AAPL" in output and "MSFT" in output


def test_underlying_filter_limits_registry_positions_before_fetch(tmp_path):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry, CALL1, "aapl")
    _open(registry, MSFT_CALL, "msft", entry=12.0)

    with patch.object(
        module,
        "fetch_equity_option_snapshots",
        return_value={CALL1: _snapshot(CALL1)},
    ) as fetch:
        rc = module.main(
            [
                "--db",
                str(registry.path),
                "--date",
                TODAY.isoformat(),
                "--underlying",
                "aapl",
            ]
        )

    assert rc == 0
    fetch.assert_called_once_with([CALL1], TODAY.isoformat())


def test_empty_book_skips_network_and_prints_valid_dashboard(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)

    with patch.object(module, "fetch_equity_option_snapshots") as fetch:
        rc = module.main(["--db", str(registry.path), "--date", TODAY.isoformat()])

    assert rc == 0
    fetch.assert_not_called()
    assert "Open positions: 0" in capsys.readouterr().out


def test_json_output_is_machine_readable_and_excludes_markdown_report(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry, CALL1, "aapl", policy={"take_profit_pct": 50.0})

    with patch.object(
        module,
        "fetch_equity_option_snapshots",
        return_value={CALL1: _snapshot(CALL1)},
    ):
        rc = module.main(
            ["--db", str(registry.path), "--date", TODAY.isoformat(), "--json"]
        )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["as_of"] == TODAY.isoformat()
    assert payload["rows"][0]["position_id"] == "aapl"
    assert payload["counts"]["HOLD"] == 1
    assert "report" not in payload


def test_historical_batch_guard_surfaces_as_review_without_lookahead(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry, CALL1, "aapl")
    guarded = {
        CALL1: EquityOptionSnapshotResult(
            None,
            "current Cboe delayed chain cannot be used for historical/future point-in-time analysis",
        )
    }

    with patch.object(module, "fetch_equity_option_snapshots", return_value=guarded) as fetch:
        rc = module.main(["--db", str(registry.path), "--date", "2026-09-10"])

    assert rc == 0
    fetch.assert_called_once_with([CALL1], "2026-09-10")
    output = capsys.readouterr().out
    assert "REVIEW 1" in output
    assert "historical/future point-in-time analysis" in output


def test_invalid_date_fails_before_registry_network_refresh(tmp_path, capsys):
    module = _load_script()
    registry = _registry(tmp_path)
    _open(registry, CALL1, "aapl")

    with patch.object(module, "fetch_equity_option_snapshots") as fetch:
        rc = module.main(["--db", str(registry.path), "--date", "bad-date"])

    assert rc == 2
    fetch.assert_not_called()
    assert "date must be YYYY-MM-DD" in capsys.readouterr().out
