"""Privileged per-command physics trace in GAR's shared PhysicsTraceStore format (classroom_arrangement.trace's
pattern): one sample every 0.5 ms tick (joint state and targets); every 20 ticks also body poses/velocities, base,
hand wrench, grip, and the paint summary (applicator loads, pool volumes, film volumes per chart, airborne parcels).
Events: protective stops, lap marks, drops landing. Recording only: no extra physics."""

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
        self._marks = {}
        world.recorders.append(self._on_tick)

    def begin(self, context):
        if self.current:
            raise RuntimeError("Action trace already active")
        if shutil.disk_usage(self.store.path.parent).free < 512*1024**2:
            raise OSError("Trace disk reserve unavailable; no action executed")
        w = self.world
        key = str(context.get("action_id") or context.get("command") or f"tick-{w.tick}")
        self.store.begin(key, str(context.get("request_id", key)), int(w.tick), float(0.0005), dict(
            source="painting/newton1.5+mujoco+film",
            body_labels=list(w.model.body_label), items={n: int(it["body"]) for n, it in w.scene.items.items()},
            arms=list(w.arms), sampling=f"every physics tick (joint state, targets); every {self.every} ticks also body "
                                        "poses/velocities, base, wrench, grip and the paint summary; no additional physics",
            missing="film fields are not traced per cell (checkpoints carry them); no force sensors"))
        self.current = key
        self._reflexes = len(w.reflexes)
        self._marks = {n: len(c.events) for n, c in (w.paint.charts.items() if w.paint else [])}
        return dict(available=True, id=key, store_path=str(self.store.path))

    def _paint(self, w):
        p = w.paint
        if p is None:
            return {}
        out = dict(applicators={n: {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                                    for k, v in a.state().items() if not isinstance(v, (list, dict))}
                                for n, a in p.applicators.items()},
                   pools_mL={n: r.volume*1e6 for n, r in p.reservoirs.items()})
        out["film_mL"] = {n: float((c.s.sum() + c.w.sum())*c.area*1e6) for n, c in p.charts.items() if c.bbox is not None}
        return out

    def _on_tick(self, w):
        if self.current is None:
            return
        events = [{**{k: v for k, v in r.items() if isinstance(v, (int, float, str))}, "type": "protective_stop"}
                  for r in w.reflexes[self._reflexes:]]
        self._reflexes = len(w.reflexes)
        if w.paint is not None:
            for n, c in w.paint.charts.items():
                k0 = self._marks.get(n, 0)
                for e in c.events[k0:]:
                    # GAR's trace store files every event under its tick: film events (drops landing, laps) carry none
                    # of their own, and a drop landing during a recorded HTTP command would stop the backend with
                    # KeyError 'tick'
                    events.append({"type": e.get("kind", "film"), "chart": n,
                                   **{k: v for k, v in e.items() if isinstance(v, (int, float)) and k != "kind"},
                                   "tick": int(w.tick)})
                self._marks[n] = len(c.events)
        sample = dict(tick=int(w.tick), time_s=float(w.sim_time), joint_q=w.state.joint_q.numpy(),
                      joint_qd=w.state.joint_qd.numpy(), joint_target_q=w.control.joint_target_q.numpy())
        if w.tick % self.every == 0 or events:
            sample.update(body_q=w.state.body_q.numpy(), body_qd=w.state.body_qd.numpy(),
                          base_pose=[float(v) for v in w.base_pose()], base_cmd=[float(v) for v in w.base.cmd],
                          external_torque_nm={a: np.asarray(w._tau_f[a], float).tolist() for a in w.arms},
                          hand_wrench={a: np.asarray(w._wrench_f[a], float).tolist() for a in w.arms},
                          grip_force_n={k: float(v) for k, v in w.grip_force.items()}, paint=self._paint(w))
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
