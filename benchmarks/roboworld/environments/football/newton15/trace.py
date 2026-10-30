"""Privileged every-tick state/contact record using GAR's bounded journal."""

import shutil
import numpy as np
from gym_anything_robotics.physics_trace import PhysicsTraceStore


class Trace:
    def __init__(self, world, path, *, max_bytes=512 * 1024**2):
        self.world = world
        self.store = PhysicsTraceStore(
            path, max_bytes=max_bytes, max_items=100_000, compression="zstd"
        )
        self.current, self.previous = None, set()

    def begin(self, context):
        if self.current:
            raise RuntimeError("Trace already active")
        if shutil.disk_usage(self.store.path.parent).free < 256 * 1024**2:
            raise OSError("Insufficient trace disk reserve; action not started")
        w = self.world
        self.current = str(context["action_id"])
        self.store.begin(
            self.current,
            str(context["request_id"]),
            w.tick,
            w.config.dt,
            {
                "source": "football-newton15",
                "generation": w.generation,
                "sampling": "Every native 500 Hz physics tick; constitutive microsteps summarized by impulse/peak force",
                "contacts": "Native force witnesses at step start, body state at step end; columnar arrays; geom order retained; active means efc_address>=0",
                "microsteps": "20 kHz shoe/pitch and 24 kHz net internals are not individually serialized",
            },
        )
        self.previous = set()
        w.trace_callback = self.append
        return {
            "available": True,
            "id": self.current,
            "store_path": str(self.store.path),
        }

    def append(self, w):
        import mujoco

        m, d = w.solver.mj_model, w.solver.mj_data
        if (
            w.tick % 500 == 0
            and shutil.disk_usage(self.store.path.parent).free < 256 * 1024**2
        ):
            raise OSError("Trace disk reserve exhausted; physics stopped")
        count = int(d.ncon)
        # MuJoCo also emits inactive proximity witnesses. Keep all witnesses,
        # but do not call entering its broad margin a physical contact event.
        contacts = {
            "geom_ids": np.empty((count, 2), np.int32),
            "position_world_m": np.empty((count, 3)),
            "frame": np.empty((count, 3, 3)),
            "distance_m": np.empty(count),
            "efc_address": np.empty(count, np.int32),
            "wrench_contact_frame": np.zeros((count, 6)),
            "force_world_on_geom2_n": np.empty((count, 3)),
        }
        pairs = set()
        for i in range(count):
            c = d.contact[i]
            contacts["geom_ids"][i] = (c.geom1, c.geom2)
            contacts["position_world_m"][i] = c.pos
            contacts["frame"][i] = c.frame.reshape(3, 3)
            contacts["distance_m"][i] = c.dist
            contacts["efc_address"][i] = c.efc_address
            mujoco.mj_contactForce(m, d, i, contacts["wrench_contact_frame"][i])
            contacts["force_world_on_geom2_n"][i] = (
                contacts["frame"][i].T @ contacts["wrench_contact_frame"][i, :3]
            )
            if c.efc_address >= 0:
                pairs.add(tuple(sorted((int(c.geom1), int(c.geom2)))))
        events = [
            {"type": kind, "tick": w.tick, "geom_ids": pair}
            for kind, changed in (
                ("contact_begin", pairs - self.previous),
                ("contact_end", self.previous - pairs),
            )
            for pair in sorted(changed)
        ]
        self.previous = pairs
        self.store.append(
            self.current,
            w.tick,
            {
                "tick": w.tick,
                "generation": w.generation,
                "body_q": w.poses,
                "body_qd": w.velocities,
                "joint_q": w.q,
                "joint_qd": w.qd,
                "targets": w.targets,
                "native_contacts": contacts,
                "custom_contacts": w.last,
                "custom_body_wrenches_world": w.wrenches,
                "nets": {
                    name: {"position": net.position, "velocity": net.velocity}
                    for name, net in w.nets.items()
                },
            },
            events,
        )

    def finish(self):
        if self.current is None:
            return None
        self.world.trace_callback = None
        key, self.current = self.current, None
        self.store.finish(
            key,
            self.world.tick,
            status="failed" if self.world.fault else "completed",
            error=self.world.fault,
        )
        return self.store.summary(key)

    def close(self):
        self.finish()
        self.store.close()
