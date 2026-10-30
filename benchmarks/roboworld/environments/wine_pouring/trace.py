"""Privileged per-command physics trace in GAR's shared PhysicsTraceStore format (gym_anything_robotics.physics_trace;
deposit_cans/trace.py pattern), so `GET /v1/history/{id}/trace`, `/trace/events` and `gar export --include-traces`
read it. One sample every physics tick (1 ms); solver quantities, not sensors. Recording only: no extra physics.

A sample holds every body pose and velocity, the joint state and targets, grip commands, the liquid step's substeps,
iterations, guard and limiter counts, and the liquid's wrench on each glassware body. Events: protective stops and
liquid-limiter activity (speed cap engaged, a particle repaired out of a wall)."""

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
            source="wine-pouring/newton1.5+dfsph", config=e.config.__dict__ if hasattr(e.config, "__dict__") else {},
            body_labels=list(w.model.body_label), items={n: int(it["body"]) for n, it in w.scene.items.items()},
            sampling="every completed 1 ms physics tick; no additional physics steps",
            liquid="DFSPH step info (substeps, iterations, guard hits incl. robot links, speed-cap and wall-repair counts) "
                   "and the liquid wrench on each glassware body: solver quantities, not sensors",
            missing="particle positions are not traced (use the state/checkpoint routes); no force sensors"))
        self.current = key
        self._reflexes = len(w.reflexes)
        return dict(available=True, id=key, store_path=str(self.store.path))

    def _on_tick(self, env):
        if self._prev:
            self._prev(env)
        if self.current is None:
            return
        w = env.world
        L = w.liquid
        info = dict(L.last) if L is not None else {}
        events = [{**{k: v for k, v in r.items() if isinstance(v, (int, float, str))}, "type": "protective_stop",
                   "tick": int(w.tick)} for r in w.reflexes[self._reflexes:]]
        self._reflexes = len(w.reflexes)
        if info.get("capped"):
            events.append(dict(type="liquid_speed_cap", tick=int(w.tick), particles=int(info["capped"])))
        if info.get("wall_repairs"):
            events.append(dict(type="liquid_wall_repair", tick=int(w.tick), particles=int(info["wall_repairs"])))
        sample = dict(tick=int(w.tick), body_q=w.state.body_q.numpy(), body_qd=w.state.body_qd.numpy(),
                      joint_q=w.state.joint_q.numpy(), joint_qd=w.state.joint_qd.numpy(),
                      joint_target_q=w.control.joint_target_q.numpy(),
                      grip_force_n={k: float(v) for k, v in w.grip_force.items()},
                      liquid_step={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                                   for k, v in info.items()},
                      liquid_wrench_on_items={n: np.asarray(w.liquid_wrench[it["body"]], float).tolist()
                                              for n, it in w.scene.items.items()} if L is not None else {})
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
