"""Shared HTTP-compatible, bounded every-substep privileged trace adapter."""

from pathlib import Path
import shutil
from gym_anything_robotics.physics_trace import PhysicsTraceStore


class SessionTrace:
    def __init__(self, session, path, *, max_bytes=256*1024**2, max_items=250_000):
        self.session = session
        self.store = PhysicsTraceStore(Path(path), max_bytes=max_bytes, max_items=max_items, compression='zstd')
        self.current = None
        self.previous_contacts = None

    def begin(self, context):
        if self.current is not None:
            raise RuntimeError("An action trace is already active")
        if shutil.disk_usage(self.store.path.parent).free < 128*1024**2:
            raise OSError("Trace disk reserve reached before action")
        command = str(context["action_id"])
        self.store.begin(command, str(context["request_id"]), self.session.tick, self.session.dt_s,
            {"source": "titration-session", "generation": self.session.generation,
             "fields": "Newton body/joint poses and velocities, targets, contacts/wrenches, magnetic loads",
             "sampling": "Every physics substep, after integration; contact data is the solve at step start",
             "liquid_sampling": ({"pressure_and_rigid_load_stride_ticks": 1,
                 "positions_velocities_species_stride_ticks": self.session.trace_liquid_stride,
                 "interpolation": "none", "before_after_full_state": "provided by shared HTTP action history"}
                 if self.session.liquid is not None else None),
             "contact_events": "First sample is a baseline, not a claim that its contacts began during this action"})
        # Do not install callbacks or claim an active trace before its header
        # has been durably committed. A failed begin has advanced no physics.
        self.current = command
        self.session.command_id = self.current
        self.session.trace_callback = self.append
        self.previous_contacts = None
        return {"available": True, "id": self.current, "store_path": str(self.store.path)}

    def append(self, sample):
        contacts = sample["contacts"]
        pairs = {tuple(sorted(map(int, pair))) for pair, active in
            zip(contacts["geometry_ids"], contacts["constraint_address"]) if active >= 0}
        if self.previous_contacts is None:
            events = [{"type": "contact_baseline", "tick": sample["tick"],
                "geometry_ids": [list(pair) for pair in sorted(pairs)],
                "sample_step_start_tick": sample["contact_sample_step_start_tick"]}]
        else:
            events = [{"type": "contact_begin" if added else "contact_end", "tick": sample["tick"],
                "sample_step_start_tick": sample["contact_sample_step_start_tick"],
                "geometry_ids": list(pair)} for added, values in
                ((True, pairs-self.previous_contacts), (False, self.previous_contacts-pairs)) for pair in sorted(values)]
        if shutil.disk_usage(self.store.path.parent).free < 128*1024**2:
            raise OSError("Trace disk reserve reached after physics step")
        self.store.append(self.current, sample["tick"], sample, events)
        self.previous_contacts = pairs

    def finish(self):
        if self.current is None:
            return None
        command, self.current = self.current, None
        self.session.trace_callback = None
        self.session.command_id = None
        fault = self.session.world.fault
        self.store.finish(command, self.session.tick, status="failed" if fault else "completed", error=fault)
        return self.store.summary(command)

    def close(self):
        self.finish()
        self.store.close()
