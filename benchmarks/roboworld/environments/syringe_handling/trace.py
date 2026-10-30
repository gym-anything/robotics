"""Privileged per-command physics trace in GAR's shared PhysicsTraceStore format (gym_anything_robotics.physics_trace;
the wine_pouring / drilling / ingredient_mixing pattern), so `GET /v1/history/{id}/trace`, `/trace/events` and `gar
export --include-traces` read it. One sample every physics tick (1 ms); solver quantities, not sensors. Recording only.

A sample holds every body pose and velocity, the joint state and targets, grip commands and applied grip forces, and
every 20 ticks the equipment's state (each syringe's plunger, contents, pressure and Luer joint; each needle's track;
vial, lock and line; liquid by place). Events: protective stops and world events (touches, punctures, fits)."""

import shutil

from gym_anything_robotics.physics_trace import PhysicsTraceStore

LIMIT_BYTES = 512*1024*1024


class Trace:
    def __init__(self, env, path, max_bytes=LIMIT_BYTES):
        self.env = env
        self.store = PhysicsTraceStore(path, max_bytes=max_bytes, max_items=max(100000, max_bytes//64),
                                       compression="zlib", synchronous="NORMAL")
        self.current = None
        self._marks = (0, 0)
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
            source="syringe_handling/newton1.5+mujoco+liquid-network", body_labels=list(w.model.body_label),
            objects={n: int(b) for n, b in w.scene.bodies.items()},
            sampling="every completed 1 ms physics tick; no additional physics steps",
            equipment="every 20 ticks: the equipment's state as env.equipment_state() gives it (solver quantities, not "
                      "sensors)"))
        self.current = key
        self._marks = (len(w.reflexes), len(w.events))
        return dict(available=True, id=key, store_path=str(self.store.path))

    def _on_tick(self, env):
        if self._prev:
            self._prev(env)
        if self.current is None:
            return
        w = env.world
        r0, e0 = self._marks
        events = [{**{k: v for k, v in r.items() if isinstance(v, (int, float, str))}, "type": "protective_stop"}
                  for r in w.reflexes[r0:]]
        # A world event is logged as dict(t=sim seconds, event=name, ...) (devices.py); the store files every event under
        # its "tick" and "type" (physics_trace.py append). Without them the first event of a command would raise
        # KeyError here, after physics has advanced, and stop the command.
        events += [{**{k: v for k, v in x.items() if isinstance(v, (int, float, str))},
                    "tick": int(w.tick), "type": str(x.get("event", "world_event"))} for x in w.events[e0:]]
        self._marks = (len(w.reflexes), len(w.events))
        sample = dict(tick=int(w.tick), body_q=w.bq, body_qd=w.bqd, joint_q=w.jq, joint_qd=w.jqd, joint_target_q=w.tq,
                      grip_force_n={k: float(v) for k, v in getattr(w, "grip_force", {}).items()},
                      grip_applied_n={k: float(w.grip_applied_n(k)) for k in getattr(w, "grip_force", {})},
                      )
        if w.tick % 20 == 0:
            sample["equipment"] = env.equipment_state()
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
