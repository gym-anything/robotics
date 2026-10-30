"""Privileged per-command physics trace in GAR's shared PhysicsTraceStore format (gym_anything_robotics.physics_trace;
wine_pouring/trace.py pattern), so `GET /v1/history/{id}/trace`, `/trace/events` and `gar export --include-traces`
read it. One sample every physics tick (1 ms); solver quantities, not sensors. Recording only: no extra physics.

A sample holds every body pose and velocity, the joint state and targets, grip and wrench commands, each bit's
thrust, torque, feed and temperature, and each tool's state. Events: protective stops, hole events (seated, cap
collapse, tear-out onset, through, rebar contact, cleaning passes), tool events (clutch trips, punch strikes) and
broken or burnt bits."""

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
        self._marks = (0, 0, 0)
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
            source="drilling/newton1.5+mujoco3.11", body_labels=list(w.model.body_label),
            parts={n: int(p["body"]) for n, p in w.scene.parts.items()},
            sampling="every completed 1 ms physics tick; no additional physics steps",
            bits="per bit: solved thrust on the hole floor, applied cutting torque, feed per revolution, mode, depth, tip "
                 "temperature: solver and model quantities, not sensors",
            missing="no force sensors; hole records are in the state route"))
        self.current = key
        self._marks = (len(w.reflexes), len(w.events), len(w.hm.events))
        return dict(available=True, id=key, store_path=str(self.store.path))

    def _on_tick(self, env):
        if self._prev:
            self._prev(env)
        if self.current is None:
            return
        w = env.world
        r0, e0, h0 = self._marks
        events = [{**{k: v for k, v in r.items() if isinstance(v, (int, float, str))}, "type": "protective_stop"}
                  for r in w.reflexes[r0:]]
        events += [{k: v for k, v in x.items() if isinstance(v, (int, float, str))} for x in w.events[e0:]]
        events += [{k: v for k, v in x.items() if isinstance(v, (int, float, str))} for x in w.hm.events[h0:]]
        self._marks = (len(w.reflexes), len(w.events), len(w.hm.events))
        bits = {n: {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in d.items()
                    if k in ("thrust_n", "torque_nm", "feed_mm_rev", "mode", "depth_m", "w_rel_rad_s", "lateral_n")}
                for n, d in w.hm.diag.items()}
        for n, it in w.hm.items.items():
            if it.kind == "bit":
                bits.setdefault(n, {})["temp_c"] = float(it.state["temp_c"])
        sample = dict(tick=int(w.tick), body_q=w.bq, body_qd=w.bqd, joint_q=w.jq, joint_qd=w.jqd,
                      joint_target_q=w.tq, grip_force_n={k: float(v) for k, v in getattr(w, "grip_force", {}).items()},
                      grip_applied_n={k: float(w.grip_applied_n(k)) for k in getattr(w, "grip_force", {})},
                      wrench_cmd={k: [float(x) for x in v[0]] for k, v in getattr(w, "wrench_cmd", {}).items()},
                      bits=bits)
        if w.tick % 50 == 0:
            sample["tools"] = {n: t.state() for n, t in w.scene.tools.items()}
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
