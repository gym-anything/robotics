"""Bounded, teacher-only physics trace. Recording never advances or retries physics."""

from collections import deque
import time
import numpy as np
from .trace_codec import encode, decode


class Trace:
    def __init__(self, budget_bytes=64 * 1024 * 1024):
        self.budget_bytes = budget_bytes
        self.frames = deque()
        self.events = deque(maxlen=2048)
        self.bytes = self.sequence = self.evicted = self.failed = 0
        self.command_id = None
        self._pairs = set()
        self.store = None

    def attach(self, store):
        self.store = store
        metadata = store.metadata()
        self.sequence = metadata["trace_sequence"]
        self.failed = metadata["failed_trace"]
        self.frames.clear()
        self.bytes = self.evicted = 0
        self._pairs.clear()
        self.event("process_or_environment_start", 0, continuity=False)

    def event(self, kind, tick, **detail):
        event = {
            "kind": kind,
            "tick": tick,
            "sequence": self.sequence,
            "command_id": self.command_id,
            "wall_time_ns": time.time_ns(),
            **detail,
        }
        self.events.append(event)
        if self.store is not None:
            self.store.put("event", str(event["wall_time_ns"]), event)

    def record(self, env):
        self.sequence += 1
        try:
            import mujoco

            m, d = env.solver.mj_model, env.solver.mj_data
            active = np.flatnonzero(np.asarray(d.contact.efc_address) >= 0)
            geom_ids = np.asarray(d.contact.geom)[active]
            unique = np.unique(np.sort(geom_ids, axis=1), axis=0)
            pairs = set(map(tuple, unique.tolist()))
            forces = np.zeros((len(active), 6))
            for row, index in enumerate(active):
                mujoco.mj_contactForce(m, d, int(index), forces[row])
            contacts = {
                "geoms": geom_ids,
                "point_m": np.asarray(d.contact.pos)[active],
                "frame_rows": np.asarray(d.contact.frame)[active].reshape(-1, 3, 3),
                "distance_m": np.asarray(d.contact.dist)[active],
                "wrench_contact_frame": forces,
            }
            if pairs != self._pairs:
                self.event(
                    "contact_change",
                    env.tick,
                    entered=[list(p) for p in sorted(pairs - self._pairs)],
                    exited=[list(p) for p in sorted(self._pairs - pairs)],
                )
                self._pairs = pairs
            row = {
                "sequence": self.sequence,
                "tick": env.tick,
                "time_s": env.tick * env.config.physics_dt,
                "wall_time_ns": time.time_ns(),
                "command_id": self.command_id,
                "body_q": env.state_0.body_q.numpy(),
                "body_qd": env.state_0.body_qd.numpy(),
                "joint_q": env.state_0.joint_q.numpy(),
                "joint_qd": env.state_0.joint_qd.numpy(),
                "joint_target_q": env.control.joint_target_q.numpy(),
                "actuator_force_nm_or_n": d.qfrc_actuator,
                "contacts": contacts,
                "tool_contacts": env.last_contacts,
                "deformations": env.deformations,
                "physics_audit": env.last_audit,
                "ink": env.ink.metadata(),
            }
            payload = encode(row)
            if self.store is not None:
                self.store.put("trace", self.sequence, payload, encoded=True)
            if len(payload) > self.budget_bytes:
                raise MemoryError("One trace frame exceeds the retention budget")
            self.frames.append((self.sequence, payload))
            self.bytes += len(payload)
            while self.bytes > self.budget_bytes:
                _, old = self.frames.popleft()
                self.bytes -= len(old)
                self.evicted += 1
        except Exception as exc:
            self.failed += 1
            if self.store is not None:
                try:
                    self.store.record_trace_failure(self.sequence)
                except Exception:
                    pass  # The started command journal also exposes an interrupted write.
            # Preserve the original failure if the disk itself cannot store an event.
            try:
                self.event("recording_failure", env.tick, error=type(exc).__name__)
            except Exception:
                pass
            from .storage import RecordingUnavailable

            raise RecordingUnavailable(
                f"Trace failed after executed tick {env.tick}: {exc}"
            ) from exc

    def page(self, after=0, limit=100):
        if (
            any(isinstance(v, bool) or not isinstance(v, int) for v in (after, limit))
            or after < 0
            or not 1 <= limit <= 200
        ):
            raise ValueError("after must be nonnegative; limit must be in [1,200]")
        rows = []
        for seq, payload in self.frames:
            if seq > after:
                rows.append(decode(payload))
                if len(rows) == limit:
                    break
        durable = None
        if self.store is not None:
            rows = self.store.trace_page(after, limit)
            durable = self.store.metadata()
        return {
            "frames": rows,
            "next_after": rows[-1]["sequence"] if rows else after,
            "sampling": "Every executed physics tick; post-step poses, solver contact wrench from that integration step",
            "oldest_sequence": durable["trace_oldest_sequence"]
            if durable
            else self.frames[0][0]
            if self.frames
            else None,
            "latest_sequence": self.sequence,
            "evicted_frames": durable["eviction"]["evicted_trace"] if durable else self.evicted,
            "failed_frames": self.failed,
            "retained_bytes": durable["records"].get("trace", {}).get("bytes", 0)
            if durable
            else self.bytes,
            "budget_bytes": durable["budget_bytes"] if durable else self.budget_bytes,
            "memory_cache": {
                "evicted_frames": self.evicted,
                "retained_bytes": self.bytes,
                "budget_bytes": self.budget_bytes,
            },
            "events": list(self.events),
            "event_retention": 2048,
            "complete_since_start": self.failed == 0
            and (
                durable["eviction"]["evicted_trace"] == 0 and not durable["unfinished_commands"]
                if durable
                else self.evicted == 0
            ),
            "disk": durable,
        }
