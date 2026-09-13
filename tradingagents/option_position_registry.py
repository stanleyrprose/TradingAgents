"""Persistent SQLite registry and append-only journal for long equity-option positions."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.dataflows.equity_options import _parse_occ

_SCHEMA_VERSION = 1
_DEFAULT_DB = Path.home() / ".tradingagents" / "positions" / "options.sqlite3"
_ENV_DB = "TRADINGAGENTS_OPTION_REGISTRY_DB"


@dataclass(frozen=True)
class OptionPositionRecord:
    """Current projection of one option trade lifecycle."""

    position_id: str
    underlying: str
    option_right: str
    status: str
    original_symbol: str
    current_symbol: str
    original_entry_premium: float
    current_leg_entry_premium: float
    lifecycle_net_premium_per_share: float
    contracts: int
    opened_at: str
    current_leg_opened_at: str
    closed_at: str | None
    close_premium: float | None
    exit_policy: dict[str, Any]
    greek_limits: dict[str, Any]
    initial_thesis: dict[str, Any]
    latest_thesis: dict[str, Any]
    roll_count: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class OptionPositionEvent:
    """One immutable journal event."""

    event_id: int
    position_id: str
    event_type: str
    occurred_at: str
    payload: dict[str, Any]
    created_at: str


def default_registry_path() -> Path:
    override = os.getenv(_ENV_DB)
    return Path(override).expanduser() if override else _DEFAULT_DB


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _date_text(value: object, name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError(f"{name} must be YYYY-MM-DD") from exc
    raise ValueError(f"{name} must be YYYY-MM-DD")


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number greater than zero")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return number


def _nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _contracts(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("contracts must be a positive whole number")
    return value


def _json_object(value: Mapping[str, Any] | None, name: str) -> tuple[dict[str, Any], str]:
    obj = {} if value is None else dict(value)
    try:
        encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-serializable finite values") from exc
    return obj, encoded


def _json_load(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    return value if isinstance(value, dict) else {}


def _position_id(value: str | None = None) -> str:
    if value is None:
        return f"opt_{uuid.uuid4().hex}"
    text = value.strip()
    if not text or len(text) > 128:
        raise ValueError("position_id must be a nonempty string up to 128 characters")
    return text


def _contract(symbol: object):
    contract = _parse_occ(symbol)
    if contract is None:
        raise ValueError("symbol must be an exact OCC US equity-option symbol")
    return contract


def _row_to_position(row: sqlite3.Row) -> OptionPositionRecord:
    return OptionPositionRecord(
        position_id=row["position_id"],
        underlying=row["underlying"],
        option_right=row["option_right"],
        status=row["status"],
        original_symbol=row["original_symbol"],
        current_symbol=row["current_symbol"],
        original_entry_premium=float(row["original_entry_premium"]),
        current_leg_entry_premium=float(row["current_leg_entry_premium"]),
        lifecycle_net_premium_per_share=float(row["lifecycle_net_premium_per_share"]),
        contracts=int(row["contracts"]),
        opened_at=row["opened_at"],
        current_leg_opened_at=row["current_leg_opened_at"],
        closed_at=row["closed_at"],
        close_premium=None if row["close_premium"] is None else float(row["close_premium"]),
        exit_policy=_json_load(row["exit_policy_json"]),
        greek_limits=_json_load(row["greek_limits_json"]),
        initial_thesis=_json_load(row["initial_thesis_json"]),
        latest_thesis=_json_load(row["latest_thesis_json"]),
        roll_count=int(row["roll_count"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_event(row: sqlite3.Row) -> OptionPositionEvent:
    return OptionPositionEvent(
        event_id=int(row["event_id"]),
        position_id=row["position_id"],
        event_type=row["event_type"],
        occurred_at=row["occurred_at"],
        payload=_json_load(row["payload_json"]),
        created_at=row["created_at"],
    )


class OptionPositionRegistry:
    """Transactional current projection plus append-only lifecycle journal."""

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(db_path).expanduser() if db_path is not None else default_registry_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, _SCHEMA_VERSION}:
                raise RuntimeError(
                    f"unsupported option registry schema version {version}; expected {_SCHEMA_VERSION}"
                )
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS option_positions (
                    position_id TEXT PRIMARY KEY,
                    underlying TEXT NOT NULL,
                    option_right TEXT NOT NULL CHECK (option_right IN ('C','P')),
                    status TEXT NOT NULL CHECK (status IN ('OPEN','CLOSED')),
                    original_symbol TEXT NOT NULL,
                    current_symbol TEXT NOT NULL,
                    original_entry_premium REAL NOT NULL CHECK (original_entry_premium > 0),
                    current_leg_entry_premium REAL NOT NULL CHECK (current_leg_entry_premium > 0),
                    lifecycle_net_premium_per_share REAL NOT NULL,
                    contracts INTEGER NOT NULL CHECK (contracts > 0),
                    opened_at TEXT NOT NULL,
                    current_leg_opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    close_premium REAL CHECK (close_premium IS NULL OR close_premium >= 0),
                    exit_policy_json TEXT NOT NULL DEFAULT '{}',
                    greek_limits_json TEXT NOT NULL DEFAULT '{}',
                    initial_thesis_json TEXT NOT NULL DEFAULT '{}',
                    latest_thesis_json TEXT NOT NULL DEFAULT '{}',
                    roll_count INTEGER NOT NULL DEFAULT 0 CHECK (roll_count >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS option_position_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (position_id) REFERENCES option_positions(position_id)
                );

                CREATE INDEX IF NOT EXISTS idx_option_positions_open_underlying
                    ON option_positions(status, underlying);
                CREATE INDEX IF NOT EXISTS idx_option_positions_open_symbol
                    ON option_positions(status, current_symbol);
                CREATE INDEX IF NOT EXISTS idx_option_events_position
                    ON option_position_events(position_id, event_id);

                CREATE TRIGGER IF NOT EXISTS option_events_no_update
                BEFORE UPDATE ON option_position_events
                BEGIN
                    SELECT RAISE(ABORT, 'option position journal is append-only');
                END;

                CREATE TRIGGER IF NOT EXISTS option_events_no_delete
                BEFORE DELETE ON option_position_events
                BEGIN
                    SELECT RAISE(ABORT, 'option position journal is append-only');
                END;
                """
            )
            if version == 0:
                conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        with suppress(OSError):
            os.chmod(self.path, 0o600)

    @staticmethod
    def _append_event(
        conn: sqlite3.Connection,
        position_id: str,
        event_type: str,
        occurred_at: str,
        payload: Mapping[str, Any],
    ) -> None:
        _, payload_json = _json_object(payload, "event payload")
        conn.execute(
            """
            INSERT INTO option_position_events
                (position_id, event_type, occurred_at, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (position_id, event_type, occurred_at, payload_json, _utc_now()),
        )

    def open_position(
        self,
        symbol: str,
        *,
        entry_premium: object,
        contracts: object,
        entry_date: object,
        exit_policy: Mapping[str, Any] | None = None,
        greek_limits: Mapping[str, Any] | None = None,
        initial_thesis: Mapping[str, Any] | None = None,
        position_id: str | None = None,
    ) -> OptionPositionRecord:
        contract = _contract(symbol)
        premium = _positive_number(entry_premium, "entry_premium")
        count = _contracts(contracts)
        opened = _date_text(entry_date, "entry_date")
        if date.fromisoformat(opened) > contract.expiry:
            raise ValueError("entry_date cannot be after option expiry")
        pid = _position_id(position_id)
        exit_obj, exit_json = _json_object(exit_policy, "exit_policy")
        greek_obj, greek_json = _json_object(greek_limits, "greek_limits")
        thesis_obj, thesis_json = _json_object(initial_thesis, "initial_thesis")
        canonical = str(symbol).upper()
        now = _utc_now()
        payload = {
            "symbol": canonical,
            "underlying": contract.root,
            "right": contract.right,
            "entry_premium": premium,
            "contracts": count,
            "exit_policy": exit_obj,
            "greek_limits": greek_obj,
            "initial_thesis": thesis_obj,
        }
        with self._immediate() as conn:
            conn.execute(
                """
                INSERT INTO option_positions (
                    position_id, underlying, option_right, status,
                    original_symbol, current_symbol,
                    original_entry_premium, current_leg_entry_premium,
                    lifecycle_net_premium_per_share, contracts,
                    opened_at, current_leg_opened_at,
                    exit_policy_json, greek_limits_json,
                    initial_thesis_json, latest_thesis_json,
                    roll_count, created_at, updated_at
                ) VALUES (?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    pid,
                    contract.root,
                    contract.right,
                    canonical,
                    canonical,
                    premium,
                    premium,
                    premium,
                    count,
                    opened,
                    opened,
                    exit_json,
                    greek_json,
                    thesis_json,
                    thesis_json,
                    now,
                    now,
                ),
            )
            self._append_event(conn, pid, "OPEN", opened, payload)
        return self.get_position(pid)

    def get_position(self, position_id: str) -> OptionPositionRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM option_positions WHERE position_id=?", (position_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown option position: {position_id}")
        return _row_to_position(row)

    def list_positions(
        self,
        *,
        underlying: str | None = None,
        status: str | None = "OPEN",
    ) -> tuple[OptionPositionRecord, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if underlying is not None:
            clauses.append("underlying=?")
            params.append(underlying.upper())
        if status is not None:
            normalized = status.upper()
            if normalized not in {"OPEN", "CLOSED"}:
                raise ValueError("status must be OPEN, CLOSED, or None")
            clauses.append("status=?")
            params.append(normalized)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM option_positions{where} ORDER BY opened_at, position_id",
                params,
            ).fetchall()
        return tuple(_row_to_position(row) for row in rows)

    def resolve_open_position(self, query: str) -> OptionPositionRecord:
        text = query.strip()
        if not text:
            raise ValueError("position query must not be empty")
        upper = text.upper()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM option_positions
                WHERE status='OPEN' AND (
                    position_id=? OR current_symbol=? OR underlying=?
                )
                ORDER BY opened_at, position_id
                """,
                (text, upper, upper),
            ).fetchall()
        unique = {row["position_id"]: row for row in rows}
        if not unique:
            raise KeyError(f"no open option position matches: {query}")
        if len(unique) > 1:
            matches = ", ".join(sorted(unique))
            raise ValueError(
                f"multiple open option positions match {query}: {matches}; use position_id"
            )
        return _row_to_position(next(iter(unique.values())))

    def set_policies(
        self,
        position_id: str,
        *,
        exit_policy: Mapping[str, Any] | None = None,
        greek_limits: Mapping[str, Any] | None = None,
        occurred_at: object | None = None,
    ) -> OptionPositionRecord:
        current = self.get_position(position_id)
        if current.status != "OPEN":
            raise ValueError("cannot update policies on a closed position")
        exit_obj = current.exit_policy if exit_policy is None else dict(exit_policy)
        greek_obj = current.greek_limits if greek_limits is None else dict(greek_limits)
        exit_obj, exit_json = _json_object(exit_obj, "exit_policy")
        greek_obj, greek_json = _json_object(greek_obj, "greek_limits")
        day = _date_text(occurred_at or date.today(), "occurred_at")
        with self._immediate() as conn:
            conn.execute(
                """
                UPDATE option_positions
                SET exit_policy_json=?, greek_limits_json=?, updated_at=?
                WHERE position_id=? AND status='OPEN'
                """,
                (exit_json, greek_json, _utc_now(), position_id),
            )
            self._append_event(
                conn,
                position_id,
                "POLICY_UPDATE",
                day,
                {"exit_policy": exit_obj, "greek_limits": greek_obj},
            )
        return self.get_position(position_id)

    def record_thesis(
        self,
        position_id: str,
        thesis: Mapping[str, Any],
        *,
        occurred_at: object | None = None,
    ) -> OptionPositionRecord:
        current = self.get_position(position_id)
        if current.status != "OPEN":
            raise ValueError("cannot record a live thesis on a closed position")
        thesis_obj, thesis_json = _json_object(thesis, "thesis")
        day = _date_text(occurred_at or date.today(), "occurred_at")
        with self._immediate() as conn:
            conn.execute(
                "UPDATE option_positions SET latest_thesis_json=?, updated_at=? WHERE position_id=?",
                (thesis_json, _utc_now(), position_id),
            )
            self._append_event(conn, position_id, "THESIS_REFRESH", day, thesis_obj)
        return self.get_position(position_id)

    def record_roll(
        self,
        position_id: str,
        new_symbol: str,
        *,
        close_credit: object,
        new_entry_premium: object,
        roll_date: object,
    ) -> OptionPositionRecord:
        current = self.get_position(position_id)
        if current.status != "OPEN":
            raise ValueError("cannot roll a closed position")
        old_contract = _contract(current.current_symbol)
        new_contract = _contract(new_symbol)
        if new_contract.root != old_contract.root or new_contract.right != old_contract.right:
            raise ValueError("roll replacement must keep the same underlying and call/put direction")
        if new_contract.expiry <= old_contract.expiry:
            raise ValueError("roll replacement expiry must be strictly later than the current expiry")
        credit = _nonnegative_number(close_credit, "close_credit")
        debit = _positive_number(new_entry_premium, "new_entry_premium")
        day = _date_text(roll_date, "roll_date")
        roll_day = date.fromisoformat(day)
        if roll_day < date.fromisoformat(current.current_leg_opened_at):
            raise ValueError("roll_date cannot precede the current leg open date")
        if roll_day > new_contract.expiry:
            raise ValueError("roll_date cannot be after replacement expiry")
        canonical = new_symbol.upper()
        lifecycle = current.lifecycle_net_premium_per_share - credit + debit
        payload = {
            "from_symbol": current.current_symbol,
            "to_symbol": canonical,
            "contracts": current.contracts,
            "close_credit_per_share": credit,
            "new_entry_premium_per_share": debit,
            "previous_lifecycle_net_premium_per_share": current.lifecycle_net_premium_per_share,
            "new_lifecycle_net_premium_per_share": lifecycle,
        }
        with self._immediate() as conn:
            conn.execute(
                """
                UPDATE option_positions
                SET current_symbol=?, current_leg_entry_premium=?,
                    lifecycle_net_premium_per_share=?, current_leg_opened_at=?,
                    roll_count=roll_count+1, updated_at=?
                WHERE position_id=? AND status='OPEN'
                """,
                (canonical, debit, lifecycle, day, _utc_now(), position_id),
            )
            self._append_event(conn, position_id, "ROLL", day, payload)
        return self.get_position(position_id)

    def close_position(
        self,
        position_id: str,
        *,
        close_premium: object,
        close_date: object,
        reason: str | None = None,
    ) -> OptionPositionRecord:
        current = self.get_position(position_id)
        if current.status != "OPEN":
            raise ValueError("position is already closed")
        premium = _nonnegative_number(close_premium, "close_premium")
        day = _date_text(close_date, "close_date")
        if date.fromisoformat(day) < date.fromisoformat(current.current_leg_opened_at):
            raise ValueError("close_date cannot precede the current leg open date")
        payload = {
            "symbol": current.current_symbol,
            "contracts": current.contracts,
            "close_premium_per_share": premium,
            "reason": reason or "",
            "final_lifecycle_net_pnl_per_share": premium - current.lifecycle_net_premium_per_share,
        }
        with self._immediate() as conn:
            conn.execute(
                """
                UPDATE option_positions
                SET status='CLOSED', closed_at=?, close_premium=?, updated_at=?
                WHERE position_id=? AND status='OPEN'
                """,
                (day, premium, _utc_now(), position_id),
            )
            self._append_event(conn, position_id, "CLOSE", day, payload)
        return self.get_position(position_id)

    def list_events(self, position_id: str) -> tuple[OptionPositionEvent, ...]:
        self.get_position(position_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM option_position_events
                WHERE position_id=? ORDER BY event_id
                """,
                (position_id,),
            ).fetchall()
        return tuple(_row_to_event(row) for row in rows)
