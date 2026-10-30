"""Native contact generation and MuJoCo Warp integration, shared by the environment and its mechanism."""

import numpy as np
from .runtime import require_engine


class Dynamics:
    def __init__(self, builder, c, *, diagnostic_mujoco_contacts=False):
        self.newton, self.wp = require_engine()
        self.config, self.builder = c, builder
        with self.wp.ScopedDevice(c.device):
            self.model = builder.finalize(device=c.device)
            self.model.request_contact_attributes("force")
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            # The baseline uses rigid contact constraints, not measured linear
            # material springs. Use native, damped contact dynamics.
            # This avoids force-space gains whose refsafe clamp erased small
            # pin clearances, without disabling that guard or changing dt.
            self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            contact_ref = self.model.mujoco.solref.numpy()
            contact_ref[:] = (c.contact_time_constant, c.contact_damping_ratio)
            if c.scaled_scene_contacts:
                large = np.array([not s.startswith(("plug_", "driver_", "keypin_", "key_blade_", "wrong_blade_"))
                                  and s not in ("shell", "lock_casing")
                                  and not (c.shackle_swivel and s.startswith("shackle_"))
                                  for s in builder.shape_label])
                contact_ref[large] = (.01, 1.)
                priorities = self.model.mujoco.geom_priority.numpy()
                priorities[large] = 1
                self.model.mujoco.geom_priority.assign(priorities)
            if c.shackle_swivel:
                shackle = np.array([s.startswith("shackle_") for s in builder.shape_label])
                contact_ref[shackle] = (c.shackle_contact_time_constant, c.shackle_contact_damping_ratio)
            self.model.mujoco.solref.assign(contact_ref)
            impedance = self.model.mujoco.geom_solimp.numpy()
            impedance[:, :2] = c.contact_impedance
            if c.scaled_scene_contacts:
                impedance[large, :2] = .95
            self.model.mujoco.geom_solimp.assign(impedance)
            # Revolute guide stops use authored native time-constant dynamics.
            # Copying the old force-space limit kd=500 into a micrometre-scale
            # body's rotational inertia made the timestep clamp erase the stop.
            self.model.mujoco.solreflimit_mode.fill_(SOLREF_MODE_RAW)
            refs = self.model.mujoco.solreflimit.numpy()
            refs[:] = (c.limit_time_constant, 1.)
            self.model.mujoco.solreflimit.assign(refs)
            limits = self.model.mujoco.solimplimit.numpy()
            limits[:, :2] = .9999
            self.model.mujoco.solimplimit.assign(limits)
            self.pipeline = self.newton.CollisionPipeline(self.model,
                rigid_contact_max=c.contact_capacity, broad_phase="sap", reduce_contacts=True,
                contact_matching="latest", contact_report=True)
            self.solver = self.newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=diagnostic_mujoco_contacts,
                integrator="implicitfast", cone="elliptic", solver=c.solver_algorithm,
                iterations=c.solver_iterations, ls_iterations=30,
                nconmax=c.contact_capacity, njmax=c.contact_capacity*3)
            self.state, self.next_state = self.model.state(), self.model.state()
            self.newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
            self.next_state.assign(self.state)
            self.control, self.contacts = self.model.control(), self.pipeline.contacts()
        self.tick, self.fault = 0, None

    def step(self, joint_forces=None, *, body_forces=None):
        if self.fault:
            raise RuntimeError(self.fault)
        self.state.clear_forces()
        if body_forces is not None:
            self.state.body_f.assign(np.asarray(body_forces, np.float32))
        self.control.joint_f.zero_()
        if joint_forces is not None:
            self.control.joint_f.assign(np.asarray(joint_forces, np.float32))
        try:
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next_state, self.control, self.contacts, self.config.dt)
            for attr in ("body_q", "body_qd", "joint_q", "joint_qd"):
                if not np.isfinite(getattr(self.next_state, attr).numpy()).all():
                    raise FloatingPointError(f"Non-finite {attr} at attempted tick {self.tick+1}")
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise
