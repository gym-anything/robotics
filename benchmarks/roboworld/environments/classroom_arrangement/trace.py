"""Privileged per-command physics trace in GAR's shared PhysicsTraceStore format (gym_anything_robotics.physics_trace;
the deposit_cans / wine_pouring pattern), so `GET /v1/history/{id}/trace`, `/trace/events` and
`gar export --include-traces` read it. One sample every physics tick (0.5 ms: GAR's store counts a trace complete
only without missing ticks) with the joint state (all furniture free joints included) and targets; every 20th tick
(10 ms) adds the full body poses and velocities, base, wrench and grip records. Solver quantities, not sensors.
Recording only: no extra physics.

Events: protective stops."""

import shutil
import numpy as np

from gym_anything_robotics.physics_trace import PhysicsTraceStore

LIMIT_BYTES = 256*1024*1024
EVERY = 20


class Trace:
    def __init__(self, world, path, max_bytes=LIMIT_BYTES, every=EVERY):
        self.world, self.every = world, every
        self.store = PhysicsTraceStore(path, max_bytes=max_bytes, max_items=max(100000, max_bytes//64),
                                       compression="zlib", synchronous="NORMAL")
        self.current = None
        self._reflexes = 0
        world.recorders.append(self._on_tick)

    def begin(self, context):
        if self.current:
            raise RuntimeError("Action trace already active")
        if shutil.disk_usage(self.store.path.parent).free < 512*1024**2:
            raise OSError("Trace disk reserve unavailable; no action executed")
        w = self.world
        key = str(context.get("action_id") or context.get("command") or f"tick-{w.tick}")
        self.store.begin(key, str(context.get("request_id", key)), int(w.tick), float(0.0005), dict(
            source="classroom-arrangement/newton1.5+mujoco",
            body_labels=list(w.model.body_label), items={n: int(it["body"]) for n, it in w.scene.items.items()},
            arms=list(w.arms), sampling=f"every physics tick (joint state, targets); every {self.every} ticks also body "
                                        "poses/velocities, base, wrench and grip; no additional physics steps",
            missing="contact forces are not traced per contact (use the reflex events and hand-wrench estimates); "
                    "no force sensors"))
        self.current = key
        self._reflexes = len(w.reflexes)
        return dict(available=True, id=key, store_path=str(self.store.path))

    def _on_tick(self, w):
        if self.current is None:
            return
        events = [{**{k: v for k, v in r.items() if isinstance(v, (int, float, str))}, "type": "protective_stop"}
                  for r in w.reflexes[self._reflexes:]]
        self._reflexes = len(w.reflexes)
        sample = dict(tick=int(w.tick), time_s=float(w.sim_time), joint_q=w.state.joint_q.numpy(),
                      joint_qd=w.state.joint_qd.numpy(), joint_target_q=w.control.joint_target_q.numpy())
        if w.tick % self.every == 0 or events:
            sample.update(body_q=w.state.body_q.numpy(), body_qd=w.state.body_qd.numpy(),
                          base_pose=[float(v) for v in w.base_pose()], base_cmd=[float(v) for v in w.base.cmd],
                          external_torque_nm={a: np.asarray(w._tau_f[a], float).tolist() for a in w.arms},
                          hand_wrench={a: np.asarray(w._wrench_f[a], float).tolist() for a in w.arms},
                          grip_force_n={k: float(v) for k, v in w.grip_force.items()})
        self.store.append(self.current, int(w.tick), sample, events)

    def finish(self):
        if self.current is None:
            return None
        key = self.current
        self.store.finish(key, int(self.world.tick))
        self.current = None
        s = self.store.summary(key)
        return dict(id=key, store_path=str(self.store.path), complete=s["complete"], retained_samples=s["retained_samples"])

    def close(self):
        if self._on_tick in self.world.recorders:
            self.world.recorders.remove(self._on_tick)
        self.store.close()
