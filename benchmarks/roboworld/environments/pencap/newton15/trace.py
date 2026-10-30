"""Bounded privileged per-step records, readable by the shared HTTP history API."""

import shutil
from gym_anything_robotics.physics_trace import PhysicsTraceStore


class Trace:
    def __init__(self, env, path, *, source="pencap-newton15",
                 contact_description="Shape-pair transitions recomputed from the entire buffer, including nub adapter"):
        self.env = env
        self.source, self.contact_description = source, contact_description
        self.store = PhysicsTraceStore(path, max_bytes=256*1024**2, max_items=100_000, compression="zstd")
        self.current, self.previous = None, None
        self.solved_contacts = env.pipeline.contacts()

    def begin(self, context):
        if self.current:
            raise RuntimeError("Trace already active")
        if shutil.disk_usage(self.store.path.parent).free < 256*1024**2:
            self.env.fault = "Insufficient trace disk reserve; no action executed"
            raise OSError(self.env.fault)
        command = str(context["action_id"])
        try:
            self.store.begin(command, str(context["request_id"]), self.env.tick, self.env.config.dt,
                {"source": self.source, "generation": self.env.generation,
                 "sampling": "Every physics tick, post-integration state; contact witnesses refer to step start",
                 "force_impulses": "Native solver contact force on shape0 in world axes; force times dt is step impulse; moments are contact torques, not COM moments",
                 "events": self.contact_description})
        except Exception as exc:
            self.env.fault = f"Trace could not begin before tick {self.env.tick}: {exc}"
            raise
        self.current, self.previous = command, None
        self.env.trace_callback = self.append
        return {"available": True, "id": command, "store_path": str(self.store.path)}

    def append(self, env):
        c = env.contacts
        n = int(c.rigid_contact_count.numpy()[0])
        contacts = {k: getattr(c, "rigid_contact_"+k).numpy()[:n] for k in
            ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1")}
        # Extract the just-completed native solve into a DIFFERENT buffer.
        # Diagnostics must not overwrite collision inputs or advance physics.
        env.solver.update_contacts(self.solved_contacts)
        solved = self.solved_contacts
        count = int(solved.rigid_contact_count.numpy()[0])
        force = solved.force.numpy()[:count]
        solved_contacts = {k: getattr(solved, "rigid_contact_"+k).numpy()[:count]
            for k in ("shape0", "shape1", "point0", "point1")}
        solved_contacts.update(force_world_on_shape0_n=force[:, :3],
            contact_torque_world_on_shape0_nm=force[:, 3:],
            impulse_world_on_shape0_n_s=force[:, :3]*env.config.dt)
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(contacts["shape0"], contacts["shape1"])}
        events = []
        if self.previous is not None:
            for kind, changed in (("contact_begin", pairs-self.previous), ("contact_end", self.previous-pairs)):
                events.extend({"type": kind, "tick": env.tick, "shape_ids": list(p)} for p in sorted(changed))
        self.previous = pairs
        if shutil.disk_usage(self.store.path.parent).free < 256*1024**2:
            raise OSError("Trace disk reserve reached after step")
        sample = {"tick": env.tick, "generation": env.generation, "contacts_step_start_tick": env.tick-1,
            **{k: getattr(env.state, k).numpy() for k in ("body_q", "body_qd", "joint_q", "joint_qd")},
            "actuator_setpoints": env.control.joint_target_q.numpy(), "requested_targets": env.targets,
            "contacts": contacts, "solved_contacts": solved_contacts, "evaluation": env.evaluation}
        self.store.append(self.current, env.tick, sample, events)

    def finish(self):
        if self.current is None:
            return None
        command, self.current = self.current, None
        self.env.trace_callback = None
        self.store.finish(command, self.env.tick, status="failed" if self.env.fault else "completed", error=self.env.fault)
        return self.store.summary(command)

    def close(self):
        self.finish()
        self.store.close()
