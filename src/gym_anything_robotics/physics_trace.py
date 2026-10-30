"""Privileged, paginated physics-step evidence. No simulator dependencies."""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import struct
import threading
from typing import Any
import zlib

import numpy as np

from .action_history import diagnostic_json, utc_now

DEFAULT_TRACE_BYTES = 1024 * 1024 * 1024
DEFAULT_TRACE_ITEMS = 250_000


def install_retention_indexes(db: sqlite3.Connection) -> None:
    """Upgrade storage only; never rewrite or discard recorded samples.

    Size columns follow large BLOBs in the original tables. An unindexed SUM
    reads those payload pages on *every* physics step. Covering indexes keep
    retention accounting in small index pages, including on existing journals.
    Safe to apply to an idle live controller without replacing its Python state.
    """
    with db:
        db.execute("CREATE INDEX IF NOT EXISTS item_retention ON item(trace_id,size,kind)")
        db.execute("CREATE INDEX IF NOT EXISTS trace_retention ON trace(metadata_size)")

def install_retention_accounting(db: sqlite3.Connection) -> None:
    """Transactional counts, including old-writer inserts and rollbacks.

    A covering SUM still rescans every earlier contact event on every step.
    Migrate once under a write lock, preserving all payloads and eviction rules.
    Triggers keep accounting atomic with each row and work across connections.
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("""CREATE TABLE IF NOT EXISTS retention_totals (
            id INTEGER PRIMARY KEY CHECK(id=1), data_bytes INTEGER NOT NULL,
            items INTEGER NOT NULL, metadata_bytes INTEGER NOT NULL, headers INTEGER NOT NULL)""")
        # Recompute on writer open, also auditing an older journal's counters.
        data, items=db.execute("SELECT COALESCE(SUM(size),0),COUNT(*) FROM item").fetchone()
        meta, headers=db.execute("SELECT COALESCE(SUM(metadata_size),0),COUNT(*) FROM trace").fetchone()
        db.execute("INSERT OR REPLACE INTO retention_totals VALUES(1,?,?,?,?)",(data,items,meta,headers))
        for table,size,bytecol,countcol in (
            ('item','size','data_bytes','items'),('trace','metadata_size','metadata_bytes','headers')):
            for operation,expression,count in (
                ('INSERT',f'NEW.{size}','+1'),('DELETE',f'-OLD.{size}','-1')):
                db.execute(f"""CREATE TRIGGER IF NOT EXISTS retention_{table}_{operation.lower()}
                    AFTER {operation} ON {table} BEGIN
                    UPDATE retention_totals SET {bytecol}={bytecol}+({expression}),
                        {countcol}={countcol}{count} WHERE id=1; END""")
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS retention_{table}_resize
                AFTER UPDATE OF {size} ON {table} BEGIN
                UPDATE retention_totals SET {bytecol}={bytecol}+NEW.{size}-OLD.{size}
                WHERE id=1; END""")
        db.commit()
    except BaseException:
        db.rollback()
        raise


class ValidatedJSON:
    """Internal fast path for an already validated, producer-owned JSON tree.

    The producer must supply only finite JSON primitives/containers, no NumPy
    arrays or custom objects, and must not mutate it during pack(). Final JSON
    encoding still rejects non-finite/unsupported values. This is not a wire
    type: decoded data and the encoded format are unchanged.
    """
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value


def pack(value: Any, *, compression: str = 'zlib') -> bytes:
    # Avoid converting hundreds of thousands of floats to Python objects in the
    # stepping loop. Store exact numeric array bytes; expand only on HTTP reads.
    arrays, buffers, paths = [], [], []
    offset = 0

    def encode(item, location):
        nonlocal offset
        # These values need neither recursive conversion nor an error pointer.
        # Construct escaped JSON paths only when diagnostic_json may use one.
        # In a contact-rich sample this avoids tens of thousands of identical
        # finite-value path constructions while preserving the exact wire data.
        if item is None or isinstance(item, (str, int, bool)):
            return item
        if isinstance(item, float) and math.isfinite(item):
            return item
        if isinstance(item, np.ndarray) and item.dtype.kind in "biuf":
            array = item if item.flags.c_contiguous else np.ascontiguousarray(item)
            raw = array.tobytes()
            arrays.append({"path": location, "dtype": array.dtype.str, "shape": array.shape,
                           "offset": offset, "bytes": len(raw)})
            buffers.append(raw)
            offset += len(raw)
            return None
        if isinstance(item, dict):
            return {str(k): encode(v, location + [str(k)]) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [encode(v, location + [i]) for i, v in enumerate(item)]
        if isinstance(item, ValidatedJSON):
            return item.value
        pointer = "".join("/" + str(k).replace("~", "~0").replace("/", "~1") for k in location)
        return diagnostic_json(item, paths, pointer)

    header = json.dumps({"data": encode(value, []), "arrays": arrays, "nonfinite_paths": paths},
                        separators=(",", ":"), allow_nan=False).encode()
    raw = b"PT1\0" + struct.pack("<Q", len(header)) + header + b"".join(buffers)
    if compression == 'zstd':
        import zstandard
        return b'PTZ1'+zstandard.ZstdCompressor(level=1).compress(raw)
    if compression != 'zlib':
        raise ValueError('Unknown trace compression')
    return zlib.compress(raw, 1)


def unpack(value: bytes, *, arrays: bool = False) -> dict:
    if value.startswith(b'PTZ1'):
        import zstandard
        raw = zstandard.ZstdDecompressor().decompress(value[4:])
    else:
        raw = zlib.decompress(value)
    if not raw.startswith(b"PT1\0"):
        return json.loads(raw)  # Initial JSON-only trace format.
    header_size = struct.unpack("<Q", raw[4:12])[0]
    header = json.loads(raw[12:12+header_size])
    data = memoryview(raw)[12+header_size:]
    result = header["data"]
    paths = header["nonfinite_paths"]
    for description in header["arrays"]:
        dtype = np.dtype(description["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Invalid trace array dtype")
        start, size = description["offset"], description["bytes"]
        if start < 0 or start + size > len(data):
            raise ValueError("Invalid trace array offset")
        array = np.frombuffer(data[start:start+size], dtype=dtype).reshape(description["shape"])
        location = description["path"]
        pointer = "".join("/" + str(k).replace("~", "~0").replace("/", "~1") for k in location)
        values = array.copy() if arrays else array.tolist()
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            values = diagnostic_json(array.tolist(), paths, pointer)
        if not location:
            result = values
        else:
            target = result
            for component in location[:-1]:
                target = target[component]
            target[location[-1]] = values
    return {"data": result, "nonfinite_paths": paths}


def missing_ranges(start: int, end: int, ticks: list[int]) -> list[list[int]]:
    """Inclusive missing end-of-step ticks, without expanding a huge tick range."""
    cursor = start + 1
    result = []
    for tick in ticks:
        if tick < cursor:
            continue
        if tick > end:
            break
        if cursor < tick:
            result.append([cursor, tick - 1])
        cursor = tick + 1
    if cursor <= end:
        result.append([cursor, end])
    return result


class PhysicsTraceStore:
    """Rolling compressed journal; readers can inspect a running command.

    Retention evicts items (not sampling). Trace summaries retain eviction counts.
    At most 10,000 trace headers survive; their metadata counts toward byte limits.
    SQLite indexes/free pages/WAL use additional storage and are reported separately.
    """

    def __init__(self, path: Path, *, readonly: bool = False,
                 max_bytes: int = DEFAULT_TRACE_BYTES, max_items: int = DEFAULT_TRACE_ITEMS,
                 compression: str = 'zlib', synchronous: str = 'FULL'):
        # synchronous='NORMAL' (opt-in, WAL mode) still commits every append
        # and survives an application crash; only an OS crash or power loss
        # can roll back the latest commits (never corrupt). It avoids an
        # fsync per physics tick (measured 9.7 ms of an 11.7 ms/tick trace).
        if synchronous not in ('FULL', 'NORMAL'):
            raise ValueError('synchronous must be FULL or NORMAL')
        if max_bytes < 1024 or max_items < 1:
            raise ValueError("Trace limits must be >=1024 bytes and >=1 item")
        self.path = Path(path).resolve()
        if compression not in ('zlib', 'zstd'):
            raise ValueError('Unknown trace compression')
        if compression == 'zstd':
            import zstandard  # Fail before recording/physics if unavailable.
        self.compression = compression
        self.readonly = readonly
        self._lock = threading.RLock()
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc"),
                                  uri=True, check_same_thread=False, timeout=10)
        self.db.row_factory = sqlite3.Row
        if readonly:
            return
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(f"PRAGMA synchronous={synchronous}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS trace (
                id TEXT PRIMARY KEY, request_id TEXT NOT NULL, start_tick INTEGER NOT NULL,
                last_tick INTEGER NOT NULL, end_tick INTEGER, dt REAL NOT NULL,
                started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
                captured INTEGER NOT NULL DEFAULT 0, capture_errors INTEGER NOT NULL DEFAULT 0,
                evicted_samples INTEGER NOT NULL DEFAULT 0, evicted_events INTEGER NOT NULL DEFAULT 0,
                error TEXT, metadata BLOB NOT NULL, metadata_size INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS item (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL,
                kind TEXT NOT NULL, tick INTEGER NOT NULL, event_type TEXT,
                payload BLOB NOT NULL, size INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS trace_items ON item(trace_id,kind,tick,seq);
            CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), max_bytes INTEGER, max_items INTEGER);
        """)
        install_retention_indexes(self.db)
        install_retention_accounting(self.db)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO config VALUES (1,?,?)", (max_bytes, max_items))
            self._prune()

    def _prune(self, keep_id: str = ""):
        max_bytes, max_items = self.db.execute("SELECT max_bytes,max_items FROM config WHERE id=1").fetchone()
        data_bytes,items,meta_bytes,headers=self.db.execute(
            "SELECT data_bytes,items,metadata_bytes,headers FROM retention_totals WHERE id=1").fetchone()
        # Remove old headers (and their items) only for metadata/header limits.
        while headers > 10_000 or meta_bytes > max_bytes:
            row = self.db.execute("SELECT id,metadata_size FROM trace WHERE id != ? ORDER BY started_at LIMIT 1", (keep_id,)).fetchone()
            if row is None:
                raise OSError("Trace metadata exceeds storage budget")
            size, count = self.db.execute("SELECT COALESCE(SUM(size),0),COUNT(*) FROM item WHERE trace_id=?", (row["id"],)).fetchone()
            self.db.execute("DELETE FROM item WHERE trace_id=?", (row["id"],))
            self.db.execute("DELETE FROM trace WHERE id=?", (row["id"],))
            meta_bytes -= row["metadata_size"]
            data_bytes -= size
            items -= count
            headers -= 1
        while meta_bytes + data_bytes > max_bytes or items > max_items:
            row = self.db.execute("SELECT seq,trace_id,kind,size FROM item ORDER BY seq LIMIT 1").fetchone()
            if row is None:
                raise OSError("Trace retention cannot satisfy storage budget")
            self.db.execute("DELETE FROM item WHERE seq=?", (row["seq"],))
            column = "evicted_samples" if row["kind"] == "sample" else "evicted_events"
            self.db.execute(f"UPDATE trace SET {column}={column}+1 WHERE id=?", (row["trace_id"],))
            data_bytes -= row["size"]
            items -= 1

    @contextmanager
    def _read_snapshot(self):
        with self._lock:
            own = not self.db.in_transaction
            if own:
                self.db.execute("BEGIN")
            try:
                yield
            finally:
                if own:
                    self.db.rollback()

    def begin(self, trace_id: str, request_id: str, tick: int, dt_s: float, metadata: dict) -> None:
        encoded = pack(metadata, compression=self.compression)
        with self._lock, self.db:
            old = self.db.execute("SELECT request_id FROM trace WHERE id=?", (trace_id,)).fetchone()
            if old:
                if old[0] != request_id:
                    raise ValueError("Trace ID belongs to a different request")
                self.db.execute("UPDATE trace SET status='recording',end_tick=NULL,finished_at=NULL WHERE id=?", (trace_id,))
                return
            self.db.execute("""INSERT INTO trace
                (id,request_id,start_tick,last_tick,dt,started_at,status,metadata,metadata_size)
                VALUES (?,?,?,?,?,?,'recording',?,?)""",
                (trace_id, request_id, tick, tick, dt_s, utc_now(), encoded, len(encoded)))
            self._prune(trace_id)

    def append(self, trace_id: str, tick: int, sample: dict | None, events: list[dict]) -> None:
        values = ([] if sample is None else [("sample", tick, None, pack(sample, compression=self.compression))])
        values += [("event", int(event["tick"]), event["type"], pack(event, compression=self.compression)) for event in events]
        with self._lock, self.db:
            if self.db.execute("SELECT id FROM trace WHERE id=?", (trace_id,)).fetchone() is None:
                raise KeyError("Trace header unavailable")
            self.db.executemany("INSERT INTO item(trace_id,kind,tick,event_type,payload,size) VALUES(?,?,?,?,?,?)",
                                [(trace_id, kind, at, event_type, payload, len(payload)) for kind, at, event_type, payload in values])
            self.db.execute("UPDATE trace SET last_tick=MAX(last_tick,?),captured=captured+? WHERE id=?",
                            (tick, int(sample is not None), trace_id))
            self._prune(trace_id)

    def append_batch(self, trace_id: str, steps: list) -> None:
        """Several physics steps' (tick, sample or None, events) in one transaction: the same rows in the same order,
        accounting and retention result as calling append() for each in turn (retention evicts oldest items first, so
        one pass after the batch keeps the same items), with one commit instead of one per step."""
        values, captured, last = [], 0, None
        for tick, sample, events in steps:
            if sample is not None:
                values.append(("sample", tick, None, pack(sample, compression=self.compression)))
                captured += 1
            values += [("event", int(event["tick"]), event["type"], pack(event, compression=self.compression))
                       for event in events]
            last = tick if last is None else max(last, tick)
        if last is None:
            return
        with self._lock, self.db:
            if self.db.execute("SELECT id FROM trace WHERE id=?", (trace_id,)).fetchone() is None:
                raise KeyError("Trace header unavailable")
            self.db.executemany("INSERT INTO item(trace_id,kind,tick,event_type,payload,size) VALUES(?,?,?,?,?,?)",
                                [(trace_id, kind, at, event_type, payload, len(payload)) for kind, at, event_type, payload in values])
            self.db.execute("UPDATE trace SET last_tick=MAX(last_tick,?),captured=captured+? WHERE id=?",
                            (last, captured, trace_id))
            self._prune(trace_id)

    def finish(self, trace_id: str, tick: int, *, status: str = "completed",
               capture_errors: int = 0, error: str | None = None):
        with self._lock, self.db:
            self.db.execute("""UPDATE trace SET last_tick=MAX(last_tick,?),end_tick=?,finished_at=?,
                status=?,capture_errors=capture_errors+?,error=COALESCE(?,error) WHERE id=?""",
                (tick, tick, utc_now(), status, capture_errors, error, trace_id))

    def summary(self, trace_id: str) -> dict:
        with self._read_snapshot():
            row = self.db.execute("SELECT * FROM trace WHERE id=?", (trace_id,)).fetchone()
            if row is None:
                raise KeyError("Trace unavailable: not recorded or header evicted")
            result = dict(row)
            result["metadata"] = unpack(result.pop("metadata"))
            ticks = [r[0] for r in self.db.execute("SELECT tick FROM item WHERE trace_id=? AND kind='sample' ORDER BY tick", (trace_id,))]
            end = result["end_tick"] if result["end_tick"] is not None else result["last_tick"]
            gaps = missing_ranges(result["start_tick"], end, ticks)
            stats = self.db.execute("SELECT kind,COUNT(*) AS n,MIN(tick) AS first_tick,MAX(tick) AS last_tick FROM item WHERE trace_id=? GROUP BY kind", (trace_id,)).fetchall()
            event_stats = self.db.execute("SELECT event_type,COUNT(*) AS n,MIN(tick) AS first_tick,MAX(tick) AS last_tick FROM item WHERE trace_id=? AND kind='event' GROUP BY event_type", (trace_id,)).fetchall()
            max_bytes, max_items = self.db.execute("SELECT max_bytes,max_items FROM config WHERE id=1").fetchone()
            return {**result, "stride_physics_steps": 1, "sample_rate_hz": 1.0 / result["dt"],
                    "end_tick_known": result["end_tick"] is not None,
                    "expected_samples_through_last_tick": end - result["start_tick"],
                    "retained_samples": len(ticks), "missing_tick_ranges": gaps,
                    "complete": (result["status"] == "completed" and not gaps
                                 and result["capture_errors"] == 0 and result["evicted_events"] == 0),
                    "retained_items": [dict(r) for r in stats],
                    "event_summary": [dict(r) for r in event_stats],
                    "retention": {"max_payload_bytes": max_bytes, "max_items": max_items,
                                  "max_trace_headers": 10_000, "database_bytes": self.path.stat().st_size,
                                  "wal_bytes": Path(str(self.path) + "-wal").stat().st_size if Path(str(self.path) + "-wal").exists() else 0}}

    def page(self, trace_id: str, *, kind: str = "sample", after_seq: int = 0,
             start_tick: int | None = None, end_tick: int | None = None, limit: int = 16) -> dict:
        if kind not in {"sample", "event"} or after_seq < 0 or not 1 <= limit <= 128:
            raise ValueError("Trace requires kind sample/event, after_seq>=0, limit in 1..128")
        if (start_tick is not None and start_tick < 0) or (end_tick is not None and end_tick < 0):
            raise ValueError("Trace ticks must be nonnegative")
        if start_tick is not None and end_tick is not None and end_tick < start_tick:
            raise ValueError("end_tick must be >= start_tick")
        with self._read_snapshot():
            summary = self.summary(trace_id)
            rows = self.db.execute("""SELECT seq,tick,payload FROM item WHERE trace_id=? AND kind=? AND seq>?
                AND (? IS NULL OR tick>=?) AND (? IS NULL OR tick<=?) ORDER BY seq LIMIT ?""",
                (trace_id, kind, after_seq, start_tick, start_tick, end_tick, end_tick, limit+1)).fetchall()
            return {"summary": summary, "kind": kind,
                    "items": [{"seq": r["seq"], "tick": r["tick"], **unpack(r["payload"])} for r in rows[:limit]],
                    "has_more": len(rows) > limit,
                    "next_after_seq": rows[min(limit,len(rows))-1]["seq"] if rows else after_seq}

    def close(self):
        self.db.close()
