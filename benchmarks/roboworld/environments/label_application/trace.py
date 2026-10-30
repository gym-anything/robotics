"""Privileged per-command physics trace in GAR's shared PhysicsTraceStore format (gym_anything_robotics.physics_trace;
the deposit_cans / wine_pouring pattern), so `GET /v1/history/{id}/trace`, `/trace/events` and
`gar export --include-traces` read it. Solver quantities, not sensors. Recording only: no extra physics.

One sample every environment tick (one 2 ms label step, the public tick; GAR's store counts a trace complete only
without missing ticks): the rigid tick, joint state and targets, bonded vertices by collider, fingertip-pad loads,
pockets (without vertex lists), the label's largest node speed and energy release rate, moving peel-front nodes and
the label's reactions on each collider. Events: velocity reflexes, the tear that ends the episode, and a collider's
first / last bonded vertex in the command. The reference a command returns carries `store_path`, which GAR's
history routes read."""

import shutil
import numpy as np

from gym_anything_robotics.physics_trace import PhysicsTraceStore

LIMIT_BYTES = 256*1024*1024


class Trace:
    def __init__(self, env, path, max_bytes=LIMIT_BYTES):
        self.env = env
        self.store = PhysicsTraceStore(path, max_bytes=max_bytes, max_items=max(100000, max_bytes//64),
                                       compression="zlib", synchronous="NORMAL")
        self.current = None
        self._reflexes = 0
        self._bonded = set()
        self._frozen = None
        self._prev = env.trace_callback
        env.trace_callback = self._on_tick

    @property
    def path(self):
        return self.store.path

    def _bonded_by_collider(self, lab):
        ids = np.unique(lab.bond[lab.bond >= 0])
        return {lab.colliders[int(c)].name: int((lab.bond == c).sum()) for c in ids}

    def begin(self, context):
        if self.current:
            raise RuntimeError("Action trace already active")
        if shutil.disk_usage(self.store.path.parent).free < 512*1024**2:
            raise OSError("Trace disk reserve unavailable; no action executed")
        e, w = self.env, self.env.world
        key = str(context["action_id"])
        lab = w.label
        bonded = self._bonded_by_collider(lab)
        self.store.begin(key, str(context["request_id"]), int(w.steps), float(e.config.label_dt), dict(
            source="label-application/newton1.5+label-shell",
            tick="environment tick = one label step (label_dt) = label_substeps rigid ticks; rigid_tick is the "
                 "MuJoCo tick count",
            label_dt_s=float(e.config.label_dt), label_substeps=int(e.config.label_substeps),
            rigid_dt_s=float(e.config.rigid_dt), generation=int(e.generation), start_rigid_tick=int(w.tick),
            body_labels=list(w.model.body_label), items={n: int(it["body"]) for n, it in w.scene.items.items()},
            colliders=[dict(name=c.name, role=c.role, body=int(c.body), surface=c.surface.name) for c in lab.colliders],
            joints={n: dict(q=int(s["q"]), dof=int(s["dof"])) for n, s in w.joints.items()},
            initial_bonded_vertices=bonded, initial_termination=w.label_frozen,
            sampling="every completed environment tick (label step); no additional physics steps",
            note="Privileged solver trace (label bond/contact/pocket state, pad loads), not sensor data",
            missing="label node positions are not traced (GET /v1/geometry/label, checkpoints); no force sensors"))
        self.current = key
        self._reflexes = len(w.reflexes)
        self._bonded = set(bonded)
        self._frozen = w.label_frozen
        return dict(available=True, id=key, store_path=str(self.store.path),
                    note="Privileged solver trace (label bond/contact/pocket state, pad loads), not sensor data")

    def _on_tick(self, env):
        if self._prev:
            self._prev(env)
        if self.current is None:
            return
        w = env.world
        lab = w.label
        tick = int(w.steps)
        events = [{**{k: v for k, v in r.items() if isinstance(v, (int, float, str))}, "type": "velocity_reflex",
                   "rigid_tick": int(r["tick"]), "tick": tick} for r in w.reflexes[self._reflexes:]]
        self._reflexes = len(w.reflexes)
        bonded = self._bonded_by_collider(lab)
        for name in sorted(set(bonded) - self._bonded):
            events.append(dict(type="bond_onset", tick=tick, collider=name, vertices=bonded[name]))
        for name in sorted(self._bonded - set(bonded)):
            events.append(dict(type="bond_released", tick=tick, collider=name))
        self._bonded = set(bonded)
        if w.label_frozen is not None and self._frozen is None:
            events.append(dict(type="label_torn", **{k: v for k, v in w.label_frozen.items()
                                                    if k != "tick"}, tick=int(w.label_frozen["tick"])))
        self._frozen = w.label_frozen
        sample = dict(
            tick=tick, rigid_tick=int(w.tick), time_s=float(tick*env.config.label_dt),
            joint_q=w.state.joint_q.numpy(), joint_qd=w.state.joint_qd.numpy(), joint_target_q=np.asarray(w.targets),
            bonded=bonded,
            pad_loads_n={w.model.body_label[b]: float(f) for b, f in getattr(w, "last_pad_loads", {}).items()},
            pockets=[{k: v for k, v in p.items() if k != "vertices"} for p in getattr(lab, "pockets", [])],
            max_label_speed=float(np.linalg.norm(lab.v, axis=1).max()),
            max_G=float(lab.G.max()), front_nodes_moving=int((lab.front_speed > 0).sum()),
            reactions={k: [float(x) for x in f] for k, (f, t, b) in lab.reactions.items() if np.any(f)},
            torn=bool(lab.torn), terminated=w.label_frozen is not None)
        self.store.append(self.current, tick, sample, events)

    def finish(self):
        if self.current is None:
            return None
        key = self.current
        self.store.finish(key, int(self.env.world.steps))
        self.current = None
        s = self.store.summary(key)
        return dict(id=key, store_path=str(self.store.path), complete=s["complete"],
                    retained_samples=s["retained_samples"])

    def close(self):
        if self.env.trace_callback == self._on_tick:
            self.env.trace_callback = self._prev
        self.store.close()
