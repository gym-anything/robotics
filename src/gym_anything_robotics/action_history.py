"""Bounded, durable privileged diagnostics; never used to choose robot actions."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Mapping
from uuid import uuid4
import zlib

from .errors import ObservationUnavailableAfterCommitError, RoboticsEnvironmentError

DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_RECORDS = 10_000


class HistoryUnavailableError(RoboticsEnvironmentError):
    """History could not be persisted before an operation; no operation began."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def diagnostic_json(value: Any, paths: list[str], path: str = "", *, preserve_arrays: bool = False) -> Any:
    """Preserve invalid numerical evidence as strings, with explicit JSON pointers."""
    if isinstance(value, float) and not math.isfinite(value):
        paths.append(path)
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): diagnostic_json(v, paths, path + "/" + str(k).replace("~", "~0").replace("/", "~1"), preserve_arrays=preserve_arrays)
                for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [diagnostic_json(v, paths, f"{path}/{i}", preserve_arrays=preserve_arrays) for i, v in enumerate(value)]
    if preserve_arrays:
        import numpy as np
        if isinstance(value, np.ndarray) and value.dtype.kind in 'biuf' and np.isfinite(value).all():
            # Freeze the before-state independently of subsequent mutation.
            # Nonfinite arrays follow the existing diagnostic path below.
            return value.copy()
    if hasattr(value, "tolist"):
        return diagnostic_json(value.tolist(), paths, path)
    if hasattr(value, "item"):
        return diagnostic_json(value.item(), paths, path)
    raise TypeError(f"Cannot record {type(value).__name__}")


class ActionHistory:
    """SQLite journal with compressed JSON records and explicit oldest-first retention.

    Compressed payload bytes are bounded; SQLite indexes, reusable free pages and
    its transient transaction journal are additional storage, not image archives.
    A started row is committed before motion. An interrupted process therefore
    leaves an incomplete row instead of inventing an after-state.
    """

    def __init__(self, path: Path | None = None, *, max_bytes: int = DEFAULT_MAX_BYTES,
                 max_records: int = DEFAULT_MAX_RECORDS, binary_arrays: bool = False) -> None:
        if max_bytes < 1024 or max_records < 1:
            raise ValueError("history limits must be >= 1024 bytes and >= 1 record")
        self.path = None if path is None else Path(path).resolve()
        self.binary_arrays = bool(binary_arrays)
        if self.binary_arrays:
            import zstandard  # Check optional storage dependency before actions.
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.max_records = max_bytes, max_records
        self._lock = threading.RLock()
        self._db = sqlite3.connect(":memory:" if self.path is None else str(self.path),
                                   check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                request_id TEXT NOT NULL, session_id TEXT NOT NULL,
                kind TEXT NOT NULL, role TEXT NOT NULL, action_index INTEGER,
                started_at TEXT NOT NULL, outcome TEXT NOT NULL,
                payload BLOB NOT NULL, size INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS request_records ON records(request_id, seq);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            INSERT OR IGNORE INTO metadata VALUES ('evicted_records', 0);
        """)
        with self._db:
            self._prune(0, 0)

    def _prune(self, extra_bytes: int, extra_count: int, keep_id: str = "") -> None:
        size, count = self._db.execute("SELECT COALESCE(SUM(size),0), COUNT(*) FROM records").fetchone()
        while size + extra_bytes > self.max_bytes or count + extra_count > self.max_records:
            row = self._db.execute("SELECT id,size FROM records WHERE id != ? ORDER BY seq LIMIT 1",
                                   (keep_id,)).fetchone()
            if row is None:
                raise HistoryUnavailableError("History record exceeds configured retention budget")
            self._db.execute("DELETE FROM records WHERE id=?", (row["id"],))
            self._db.execute("UPDATE metadata SET value=value+1 WHERE key='evicted_records'")
            size -= row["size"]
            count -= 1

    def write(self, entry: dict[str, Any], *, update: bool = False) -> None:
        if self.binary_arrays:
            from .physics_trace import pack
            payload = pack(entry, compression='zstd')
        else:
            payload = zlib.compress(json.dumps(entry, allow_nan=False, separators=(",", ":")).encode(), 1)
        if len(payload) > self.max_bytes:
            raise HistoryUnavailableError("History record exceeds configured retention budget")
        try:
            with self._lock, self._db:
                old = self._db.execute("SELECT size FROM records WHERE id=?", (entry["id"],)).fetchone()
                if update and old is None:
                    raise HistoryUnavailableError("Started history record is missing")
                self._prune(len(payload) - (old[0] if old else 0), 0 if old else 1, entry["id"])
                if update:
                    self._db.execute("UPDATE records SET outcome=?,payload=?,size=? WHERE id=?",
                                     (entry["outcome"], payload, len(payload), entry["id"]))
                else:
                    self._db.execute("""INSERT INTO records
                        (id,request_id,session_id,kind,role,action_index,started_at,outcome,payload,size)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""", (
                        entry["id"], entry["request_id"], entry["session_id"], entry["kind"],
                        entry["role"], entry["action_index"], entry["started_at"], entry["outcome"],
                        payload, len(payload)))
        except sqlite3.Error as error:
            raise HistoryUnavailableError(f"History persistence failed: {error}") from error

    def status(self) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("""SELECT COUNT(*) AS retained_records, COALESCE(SUM(size),0) AS payload_bytes,
                MIN(seq) AS first_seq, MAX(seq) AS last_seq,
                COALESCE(SUM(outcome='incomplete'),0) AS incomplete_records FROM records""").fetchone()
            return {**dict(row), "durable": self.path is not None,
                    "max_payload_bytes": self.max_bytes, "max_records": self.max_records,
                    "evicted_records": self._db.execute("SELECT value FROM metadata WHERE key='evicted_records'").fetchone()[0],
                    "database_bytes": 0 if self.path is None else self.path.stat().st_size}

    def list(self, *, after_seq: int = 0, limit: int = 100,
             request_id: str | None = None) -> dict[str, Any]:
        if after_seq < 0 or not 1 <= limit <= 200:
            raise ValueError("history requires after_seq >= 0 and limit in 1..200")
        if request_id is not None and not 1 <= len(request_id) <= 200:
            raise ValueError("invalid request_id")
        with self._lock:
            rows = self._db.execute("""SELECT seq,id,request_id,session_id,kind,role,action_index,started_at,outcome
                FROM records WHERE seq > ? AND (? IS NULL OR request_id=?) ORDER BY seq LIMIT ?""",
                (after_seq, request_id, request_id, limit + 1)).fetchall()
            return {"records": [dict(row) for row in rows[:limit]], "has_more": len(rows) > limit,
                    "next_after_seq": rows[min(len(rows), limit)-1]["seq"] if rows else after_seq,
                    "retention": self.status()}

    def get(self, record_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT seq,payload FROM records WHERE id=?", (record_id,)).fetchone()
            if row is None:
                raise KeyError("History record is unknown or was evicted")
            if row['payload'].startswith(b'PTZ1'):
                from .physics_trace import unpack
                entry = unpack(row['payload'])['data']
            else:
                entry = json.loads(zlib.decompress(row['payload']))
            return {**entry, "seq": row["seq"]}

    def close(self) -> None:
        with self._lock:
            self._db.close()


class HistoryRecorder:
    def __init__(self, store: ActionHistory, get_state: Callable[[], dict[str, Any]]) -> None:
        self.store, self.get_state = store, get_state
        self.session_id = str(uuid4())
        self.started_at = utc_now()
        self.recording_fault: str | None = None

    def status(self) -> dict[str, Any]:
        return {**self.store.status(), "session_id": self.session_id,
                "session_started_at": self.started_at, "recording_fault": self.recording_fault,
                "granularity": "before_and_after_each_decoded_action",
                "physics_substeps": "backend_dependent_see_each_record_trace"}

    def snapshot(self) -> dict[str, Any]:
        began = utc_now()
        paths: list[str] = []
        try:
            state = diagnostic_json(self.get_state(), paths, preserve_arrays=self.store.binary_arrays)
            return {"capture_started_at": began, "captured_at": utc_now(), "state": state,
                    "nonfinite_paths": paths, "error": None}
        except Exception as error:
            return {"capture_started_at": began, "captured_at": utc_now(), "state": None,
                    "nonfinite_paths": [], "error": {"type": type(error).__name__, "message": str(error)}}

    @contextmanager
    def span(self, *, kind: str, payload: Mapping[str, Any], request_id: str,
             role: str, action_index: int | None = None, idempotency_key: str | None = None,
             allow_missing_before: bool = False):
        if self.recording_fault is not None:
            raise HistoryUnavailableError("History recording failed previously; inspect privileged history status")
        before = self.snapshot()
        invalid_before = before["error"] is not None or bool(before["nonfinite_paths"])
        entry = {"id": str(uuid4()), "request_id": request_id, "session_id": self.session_id,
                 "kind": kind, "role": role, "action_index": action_index,
                 "idempotency_key": idempotency_key, "command": dict(payload),
                 "started_at": utc_now(), "finished_at": None, "wall_duration_s": None,
                 "outcome": "incomplete", "before": before, "after": None, "error": None}
        if invalid_before and not allow_missing_before:
            entry["outcome"] = "not_executed"
            entry["error"] = {"type": "BeforeStateUnavailable", "message": "No finite pre-command state; command not executed"}
            self.store.write(entry)
            raise HistoryUnavailableError(entry["error"]["message"])
        self.store.write(entry)  # Must commit before physical execution.
        began = time.monotonic()
        try:
            yield entry
        except BaseException as error:
            entry["outcome"] = ("committed_without_observation" if isinstance(error, ObservationUnavailableAfterCommitError)
                                else "failed")
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        else:
            entry["outcome"] = "completed"
        finally:
            entry["wall_duration_s"] = time.monotonic() - began
            entry["after"] = self.snapshot()
            entry["finished_at"] = utc_now()
            try:
                self.store.write(entry, update=True)
            except Exception as error:
                # Never misreport executed motion as unexecuted because logging failed.
                # The committed before-record remains incomplete; block further motion.
                self.recording_fault = f"{type(error).__name__}: {error}"
