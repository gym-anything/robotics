"""Bounded SQLite evidence store. No pickle, action retries or automatic restoration."""

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import time

from .trace_codec import encode, decode


class RecordingUnavailable(RuntimeError):
    pass


class Store:
    def __init__(self, directory, *, budget_bytes=1024**3, reserve_bytes=128 * 1024**2):
        if budget_bytes < 1024**2 or reserve_bytes < 0:
            raise ValueError("Storage budget must be >=1 MiB and reserve nonnegative")
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget, self.reserve = budget_bytes, reserve_bytes
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "evidence.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                key TEXT NOT NULL, created_ns INTEGER NOT NULL,
                payload BLOB NOT NULL, digest TEXT NOT NULL, UNIQUE(kind,key));
            CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
            INSERT OR IGNORE INTO counters VALUES ('evicted_trace',0),('evicted_event',0),
                ('trace_sequence',0),('failed_trace',0);
        """)
        if "phase" not in {row[1] for row in self.db.execute("PRAGMA table_info(records)")}:
            self.db.execute("ALTER TABLE records ADD COLUMN phase TEXT")
        for key, payload in self.db.execute(
            "SELECT key,payload FROM records WHERE kind='command' AND phase IS NULL"
        ).fetchall():
            self.db.execute(
                "UPDATE records SET phase=? WHERE kind='command' AND key=?",
                (decode(payload).get("phase"), key),
            )
        self.db.commit()
        self.bytes = self.db.execute(
            "SELECT COALESCE(SUM(length(payload)),0) FROM records"
        ).fetchone()[0]

    def preflight(self):
        if shutil.disk_usage(self.root).free < self.reserve:
            raise RecordingUnavailable("Recording disk reserve reached; physics was not advanced")

    def put(self, kind, key, value, *, encoded=False):
        data = value if encoded else encode(value)
        if len(data) > self.budget:
            raise RecordingUnavailable("One record exceeds the storage budget")
        with self.lock:
            self.preflight()
            try:
                with self.db:
                    old_size = self.db.execute(
                        "SELECT length(payload) FROM records WHERE kind=? AND key=?",
                        (kind, str(key)),
                    ).fetchone()
                    self.db.execute(
                        "INSERT INTO records(kind,key,created_ns,payload,digest,phase) VALUES(?,?,?,?,?,?) "
                        "ON CONFLICT(kind,key) DO UPDATE SET payload=excluded.payload,digest=excluded.digest,phase=excluded.phase",
                        (
                            kind,
                            str(key),
                            time.time_ns(),
                            data,
                            hashlib.sha256(data).hexdigest(),
                            (decode(data) if encoded else value).get("phase")
                            if kind == "command"
                            else None,
                        ),
                    )
                    if kind == "trace":
                        self.db.execute(
                            "UPDATE counters SET value=max(value,?) WHERE name='trace_sequence'",
                            (int(key),),
                        )
                    total = self.bytes + len(data) - (old_size[0] if old_size else 0)
                    while total > self.budget:
                        old = self.db.execute(
                            "SELECT seq,kind,length(payload) FROM records WHERE kind IN ('trace','event') ORDER BY seq LIMIT 1"
                        ).fetchone()
                        if old is None:
                            raise RecordingUnavailable(
                                "Non-evictable action/checkpoint evidence fills the budget"
                            )
                        self.db.execute("DELETE FROM records WHERE seq=?", (old[0],))
                        self.db.execute(
                            "UPDATE counters SET value=value+1 WHERE name=?", ("evicted_" + old[1],)
                        )
                        total -= old[2]
                self.bytes = total
                # Incremental vacuum reuses released pages; WAL auto-checkpoints at 1000 pages.
                self.db.execute("PRAGMA incremental_vacuum(32)")
            except (sqlite3.Error, OSError) as exc:
                raise RecordingUnavailable(str(exc)) from exc

    def get(self, kind, key, *, arrays=False, raw=False):
        with self.lock:
            row = self.db.execute(
                "SELECT payload,digest FROM records WHERE kind=? AND key=?", (kind, str(key))
            ).fetchone()
            if row is None:
                raise KeyError(key)
            if hashlib.sha256(row[0]).hexdigest() != row[1]:
                raise RecordingUnavailable("Stored record failed content-integrity verification")
            return row[0] if raw else decode(row[0], arrays=arrays)

    def record_trace_failure(self, sequence):
        with self.lock, self.db:
            self.db.execute("UPDATE counters SET value=value+1 WHERE name='failed_trace'")
            self.db.execute(
                "UPDATE counters SET value=max(value,?) WHERE name='trace_sequence'", (sequence,)
            )

    def keys(self, kind):
        with self.lock:
            return [
                row[0]
                for row in self.db.execute(
                    "SELECT key FROM records WHERE kind=? ORDER BY seq", (kind,)
                )
            ]

    def retain(self, kind, count):
        with self.lock, self.db:
            deleted = self.db.execute(
                "DELETE FROM records WHERE kind=? AND seq NOT IN (SELECT seq FROM records WHERE kind=? ORDER BY seq DESC LIMIT ?)",
                (kind, kind, count),
            ).rowcount
            if deleted:
                self.db.execute(
                    "INSERT INTO counters(name,value) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value",
                    ("evicted_" + kind, deleted),
                )
            self.bytes = self.db.execute(
                "SELECT COALESCE(SUM(length(payload)),0) FROM records"
            ).fetchone()[0]

    def trace_page(self, after, limit):
        with self.lock:
            keys = self.db.execute(
                "SELECT key FROM records WHERE kind='trace' AND CAST(key AS INTEGER)>? ORDER BY CAST(key AS INTEGER) LIMIT ?",
                (after, limit),
            ).fetchall()
            return [self.get("trace", key[0]) for key in keys]

    def metadata(self):
        with self.lock:
            rows = self.db.execute(
                "SELECT kind,count(*),sum(length(payload)) FROM records GROUP BY kind"
            ).fetchall()
            interval = self.db.execute(
                "SELECT min(CAST(key AS INTEGER)),max(CAST(key AS INTEGER)) FROM records WHERE kind='trace'"
            ).fetchone()
            counters = dict(self.db.execute("SELECT name,value FROM counters"))
            return {
                "durable": True,
                "budget_bytes": self.budget,
                "reserve_bytes": self.reserve,
                "trace_oldest_sequence": interval[0],
                "trace_latest_sequence": interval[1],
                "records": {k: {"count": n, "bytes": b} for k, n, b in rows},
                "eviction": {
                    key: value for key, value in counters.items() if key.startswith("evicted_")
                },
                "trace_sequence": max(counters["trace_sequence"], interval[1] or 0),
                "failed_trace": counters["failed_trace"],
                "unfinished_commands": [
                    row[0]
                    for row in self.db.execute(
                        "SELECT key FROM records WHERE kind='command' AND phase IN ('started','interrupted') ORDER BY seq"
                    )
                ],
                "commit": "SQLite FULL synchronous transaction per stored physics tick",
            }

    def close(self):
        with self.lock:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.close()


def snapshot_bytes(env):
    """Schema-limited data, never executable Python objects."""
    from dataclasses import asdict

    snapshot = env.snapshot()
    ink = snapshot.pop("ink")
    snapshot["ink_data"] = {
        "tiles": [[list(k), *v] for k, v in ink.tiles.items()],
        "dirty": [list(k) for k in ink.dirty],
        **{
            name: getattr(ink, name)
            for name in (
                "reservoir",
                "eraser_load",
                "total_deposited_ul",
                "total_removed_ul",
                "revision",
            )
        },
    }
    return encode(
        {
            "schema": "whiteboard.checkpoint.v2",
            "config": asdict(env.config),
            "mechanics_fingerprint": mechanics_fingerprint(env),
            "task": asdict(env.task),
            "body_labels": list(env.model.body_label),
            "snapshot": snapshot,
        }
    )


def restore_bytes(env, payload):
    from dataclasses import asdict
    from .ink import InkSurface

    data = decode(payload, arrays=True)

    # Canonical JSON removes tuple/list representation differences only.
    def canonical(value):
        return json.dumps(value, sort_keys=True)

    if (
        data["schema"] != "whiteboard.checkpoint.v2"
        or data.get("mechanics_fingerprint") != mechanics_fingerprint(env)
        or canonical(data["config"]) != canonical(asdict(env.config))
        or canonical(data["task"]) != canonical(asdict(env.task))
        or data["body_labels"] != list(env.model.body_label)
    ):
        raise ValueError(
            "Checkpoint schema/configuration/task/body layout mismatch; no state changed"
        )
    state = data["snapshot"]
    value = state.pop("ink_data")
    ink = InkSurface(
        env.config.board, env.config.texture_width, env.config.texture_height, env.config.materials
    )
    ink.tiles = {tuple(k): (a, b) for k, a, b in value.pop("tiles")}
    ink.dirty = {tuple(k) for k in value.pop("dirty")}
    for key, item in value.items():
        setattr(ink, key, item)
    state["ink"] = ink
    return env.restore(state)


def mechanics_fingerprint(env):
    """Do not apply a numeric snapshot to changed mass/contact/mesh code."""
    if hasattr(env, "_mechanics_fingerprint"):
        return env._mechanics_fingerprint
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in (
        "env.py",
        "geometry.py",
        "robot.py",
        "rigid_contact.py",
        "cpu_collision.py",
        "contact.py",
        "deformable_contact.py",
        "elasticity.py",
        "friction.py",
        "ink.py",
        "gpu_runtime.py",
        "solver.py",
        "energy.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    digest.update(str(env.fixture_mode).encode())
    digest.update(b"newton1.5.0/warp1.16.0/mujoco3.11.0")
    return digest.hexdigest()
