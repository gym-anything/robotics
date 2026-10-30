"""Bounded teacher-only physics-tick history, including deformable state."""
import shutil
import numpy as np
from pencap.newton15.trace import Trace as RigidTrace


class Trace(RigidTrace):
    def __init__(self, env, path):
        super().__init__(env, path, source="laundry-newton15",
            contact_description="Rigid and cloth particle/shape begin/end transitions; lagged reaction force timing explicit")
        from .readback import Readback
        self.readback = Readback()

    def append(self, env):
        if shutil.disk_usage(self.store.path.parent).free < 256*1024**2:
            raise OSError("Trace reserve reached after physics step")
        c = env.contacts
        env.solver.update_contacts(self.solved_contacts)
        keys = ("shape0", "shape1", "point0", "point1", "normal")
        softkeys = ("particle", "shape", "body_pos", "normal")
        arrays = {"rigid_count":c.rigid_contact_count, "soft_count":c.soft_contact_count,
            "actuator_effort":env.state.mujoco.qfrc_actuator,
            "solved_count":self.solved_contacts.rigid_contact_count,
            "setpoints":env.control.joint_target_q, "solved_force":self.solved_contacts.force,
            **{"rigid_"+k:getattr(c, "rigid_contact_"+k) for k in keys},
            **{"soft_"+k:getattr(c, "soft_contact_"+k) for k in softkeys},
            **{"solved_"+k:getattr(self.solved_contacts, "rigid_contact_"+k) for k in keys[:-1]}}
        if env.config.coupling == "lagged":
            arrays.update(soft_force=env.sim.solver.collision.forces,
                applied=env.sim.applied_wrenches, next_force=env.sim.solver.collision.wrenches)
        data = self.readback.read(arrays)
        n = min(int(data["rigid_count"][0]), c.rigid_contact_max)
        m = min(int(data["soft_count"][0]), c.soft_contact_max)
        count = min(int(data["solved_count"][0]), self.solved_contacts.rigid_contact_max)
        rigid = {k:data["rigid_"+k][:n] for k in keys}
        soft = {k:data["soft_"+k][:m] for k in softkeys}
        if env.config.coupling == "lagged":
            soft["force_world_on_particle_n"] = data["soft_force"][:m]
            coupling = dict(method="native_style3d_contact_lagged_reaction", delay_ticks=1,
                applied_body_wrenches=data["applied"], next_tick_body_wrenches=data["next_force"])
        else:
            coupling = env.sim.coupling_snapshot()
        pairs = {("rigid", int(a), int(b)) for a, b in zip(rigid["shape0"], rigid["shape1"])}
        pairs |= {("cloth", int(a), int(b)) for a, b in zip(soft["particle"], soft["shape"])}
        events = []
        if self.previous is not None:
            for kind, changed in (("contact_begin", pairs-self.previous), ("contact_end", self.previous-pairs)):
                events.extend(dict(type=kind, tick=env.tick, pair=list(p)) for p in sorted(changed))
        self.previous = pairs
        sample = dict(tick=env.tick, generation=env.generation,
            **env.step_arrays,
            actuator_setpoints=data["setpoints"], requested_targets=env.targets,
            actuator_effort_dof=data["actuator_effort"],
            actuator_effort_units="Newton joint DOF order; Nm for revolute, N for prismatic; includes robot gravity compensation",
            rigid_contacts=rigid, cloth_contacts=soft,
            solved_rigid_contacts={k:data["solved_"+k][:count] for k in keys[:-1]},
            solved_rigid_force=data["solved_force"][:count],
            coupling=coupling,
            timing="Geometry witnesses are step-start; coupling method and force delay are explicit in coupling",
            table_ccd=dict(env.sim.table_plane_ccd_summary()), ipc=dict(env.sim.ipc_self_contact_summary()))
        ctrl=env.sim._ipc_self_contact_controller
        if ctrl.config.local_contact_projection:
            sample["local_self_contact_projection_impulse_kg_m_s"]=ctrl.last_projection_impulse
            sample["local_self_contact_projection_timing"]="Candidate cloth/table normal correction before final CCD; table involvement and residual global clipping are reported in ipc"
        self.store.append(self.current, env.tick, sample, events)
