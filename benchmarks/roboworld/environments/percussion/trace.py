"""Privileged per-command physics trace in GAR's shared PhysicsTraceStore format (wine_pouring/trace.py pattern), so
`GET /v1/history/{id}/trace`, `/trace/events` and `gar export --include-traces` read it. One sample every world tick
(1 ms); solver quantities, not sensors. Recording only: no extra physics.

A sample holds every body pose and velocity, the joint state and targets, grip commands, whether the tick was an
impact window, and the vibration energy of every instrument. Events: protective stops and every closed vibro-contact
segment (striker, surface, start/end sample, peak force, impulse, approach speed, struck point)."""

import shutil
import numpy as np

from gym_anything_robotics.physics_trace import PhysicsTraceStore

LIMIT_BYTES = 512*1024*1024


class Trace:
    def __init__(self, env, path, max_bytes=LIMIT_BYTES):
        self.env = env
        self.store = PhysicsTraceStore(path, max_bytes=max_bytes, max_items=max(100000, max_bytes//64),
                                       compression="zlib", synchronous="NORMAL")
        self.current = None
        self._reflexes = 0
        self._segments = 0
        self._windows = 0
        self._prev = env.trace_callback
        env.trace_callback = self._on_tick

    def begin(self, context):
        if self.current:
            raise RuntimeError("Action trace already active")
        if shutil.disk_usage(self.store.path.parent).free < 512*1024**2:
            raise OSError("Trace disk reserve unavailable; no action executed")
        e, w = self.env, self.env.world
        key = str(context["action_id"])
        self.store.begin(key, str(context["request_id"]), int(w.tick), float(e.config.rigid_dt), dict(
            source="percussion/newton1.5+modal-vibro-contact", config=dict(e.config.__dict__),
            body_labels=list(w.model.body_label), items={n: int(it["body"]) for n, it in w.scene.items.items()
                                                         if "body" in it},
            sampling="every completed 1 ms world tick; no additional physics steps (impact windows run 48 MuJoCo steps "
                     "of 20.8 us inside one tick)",
            instruments="vibration energy (J) of each modal structure at the end of the tick; vibro-contact segments as "
                        "events: solver quantities, not sensors",
            missing="per-sample contact forces and modal states are not traced (contact segments carry peak force and "
                    "impulse); microphone signals are in the state/observation routes"))
        self.current = key
        self._reflexes = len(w.reflexes)
        self._segments = len(w.contact_log)
        self._windows = int(w.timing["window_ticks"])
        return dict(available=True, id=key, store_path=str(self.store.path))

    def _on_tick(self, env):
        if self._prev:
            self._prev(env)
        if self.current is None:
            return
        w = env.world
        events = [{**{k: v for k, v in r.items() if isinstance(v, (int, float, str))}, "type": "protective_stop",
                   "tick": int(w.tick)} for r in w.reflexes[self._reflexes:]]
        self._reflexes = len(w.reflexes)
        for seg in w.contact_log[self._segments:]:
            events.append(dict(type="vibro_contact", tick=int(w.tick), striker=seg["striker"], surface=seg["surface"],
                               start_sample=int(seg["start"]), end_sample=int(seg["end"]), peak_n=float(seg["peak_N"]),
                               impulse_ns=float(seg["impulse_Ns"]), approach_m_s=float(seg["v_in"]),
                               point_m=[float(v) for v in seg["local"]]))
        self._segments = len(w.contact_log)
        win = int(w.timing["window_ticks"])
        sample = dict(tick=int(w.tick), body_q=w.state.body_q.numpy(), body_qd=w.state.body_qd.numpy(),
                      joint_q=w.state.joint_q.numpy(), joint_qd=w.state.joint_qd.numpy(),
                      joint_target_q=w.control.joint_target_q.numpy(),
                      grip_force_n={k: float(v) for k, v in w.grip_force.items()}, impact_window=win > self._windows,
                      vibration_energy_j={k: float(v.energy()) for k, v in w.structs.items()})
        self._windows = win
        self.store.append(self.current, int(w.tick), sample, events)

    def finish(self):
        if self.current is None:
            return None
        key = self.current
        self.store.finish(key, int(self.env.world.tick))
        self.current = None
        s = self.store.summary(key)
        return dict(id=key, store_path=str(self.store.path), complete=s["complete"], retained_samples=s["retained_samples"])

    def close(self):
        self.store.close()
