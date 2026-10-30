"""Bounded privileged per-command physics trace in the shared format (gym_anything_robotics.physics_trace), so the
HTTP service's /v1/history/{id}/trace routes and `gar export --include-traces` read it. Solver quantities, not sensors.

One sample per physics step: every body's pose (body_q) and the joint positions, the scanner's trigger travel and
switch, the seat sensor, keys down, grip forces, the fastest body and the contact count. Events: switch, seat and key
changes, protective stops, scanner frames and reads.
"""

import shutil

import numpy as np

LIMIT_BYTES = 256*1024*1024


class Trace:
    def __init__(self, env, path, max_bytes=LIMIT_BYTES):
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        self.env = env
        self.store = PhysicsTraceStore(path, max_bytes=max_bytes, max_items=max(100_000, max_bytes//256), synchronous="NORMAL")
        self.current, self._last = None, None
        env.trace_callback = self._on_tick

    def _state(self):
        w = self.env.world
        return dict(trigger=bool(w.trigger.state), seated=bool(w.seated), keys=sorted(k for k, s in w.key_switch.items() if s.state),
                    reflexes=len(w.reflexes), frames=len(self.env.frames), reads=len(self.env.firmware.reads))

    def begin(self, context):
        if self.current:
            raise RuntimeError("Action trace already active")
        if shutil.disk_usage(self.store.path.parent).free < 512*1024**2:
            raise OSError("Trace disk reserve unavailable; no action executed")
        w = self.env.world
        key = str(context["action_id"])
        self.store.begin(key, str(context["request_id"]), int(w.tick), float(w.config.rigid_dt), dict(
            source="barcode-scanning/newton1.5", body_labels=list(w.model.body_label), joint_labels=list(w.model.joint_label),
            sampling="Every completed physics step; no additional physics steps.",
            fields="body_q (x, y, z, qx, qy, qz, qw per body), joint_q, trigger_travel_mm, trigger_switch, seated, keys, "
                   "grip_force_n per arm (per finger), max_body_speed_m_s, contacts",
            events="trigger_switch, seat, key, protective_stop, scanner_frame, read",
            missing="No measured force sensors; contact forces are not recorded."))
        self.current, self._last = key, self._state()
        return dict(available=True, id=key, store_path=str(self.store.path),
                    note="Privileged solver trace (body poses, trigger, seat, keys, grip), not sensor data")

    def _on_tick(self, env):
        if self.current is None:
            return
        w = env.world
        qd = w.state.body_qd.numpy()
        sample = dict(tick=int(w.tick), body_q=w.state.body_q.numpy().astype(np.float32),
                      joint_q=w.state.joint_q.numpy().astype(np.float32), trigger_travel_mm=float(w.trigger_travel_m()*1e3),
                      trigger_switch=bool(w.trigger.state), seated=bool(w.seated),
                      keys=sorted(k for k, s in w.key_switch.items() if s.state),
                      grip_force_n={k: float(v) for k, v in w.grip_force.items()},
                      max_body_speed_m_s=float(np.linalg.norm(qd[:, :3], axis=1).max()), contacts=int(w.solver.mj_data.ncon))
        self.store.append(self.current, int(w.tick), sample, self._events(int(w.tick)))

    def _events(self, t):
        """Device events since the last call (the scanner's frames are rendered and decoded after the action's physics,
        so their frame and read events reach the trace when the command finishes)."""
        env, w = self.env, self.env.world
        now, was, events = self._state(), self._last, []
        if now["trigger"] != was["trigger"]:
            events.append(dict(type="trigger_switch", tick=t, closed=now["trigger"]))
        if now["seated"] != was["seated"]:
            events.append(dict(type="seat", tick=t, seated=now["seated"]))
        for k in sorted(set(now["keys"]) ^ set(was["keys"])):
            events.append(dict(type="key", tick=t, key=k, down=k in now["keys"]))
        for r in w.reflexes[was["reflexes"]:now["reflexes"]]:
            events.append(dict(type="protective_stop", tick=t, record={k: v for k, v in dict(r).items() if isinstance(v, (int, float, str, bool))}))
        for f in env.frames[was["frames"]:now["frames"]]:
            events.append(dict(type="scanner_frame", tick=int(f["tick"]), found=[x["text"] for x in f.get("found", [])],
                               reported=bool(f.get("reported"))))
        for r in env.firmware.reads[was["reads"]:now["reads"]]:
            events.append(dict(type="read", tick=t, read={k: v for k, v in dict(r).items() if isinstance(v, (int, float, str, bool))}))
        self._last = now
        return events

    def finish(self, error=None):
        if self.current is None:
            return None
        key = self.current
        tail = self._events(int(self.env.world.tick))
        if tail:
            self.store.append(key, int(self.env.world.tick), None, tail)
        self.current = None
        fault = error or self.env.world.fault
        self.store.finish(key, int(self.env.world.tick), status="failed" if fault else "completed", error=fault)
        return self.store.summary(key)

    def close(self):
        self.finish()
        self.store.close()
