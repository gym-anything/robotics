"""Newton 1.5 laboratory articulation/contact subsystem, independent of optics."""

import math
import numpy as np

from .laboratory import build_laboratory
from .rigid_contact import create_solver
from .runtime import require_newton
from .actuator_targets import ActuatorTargets


class LaboratoryMechanics:
    def __init__(self, config=None):
        newton, self.wp = require_newton()
        self.lab, self.model = build_laboratory(config)
        self.state, self.next_state = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.control = self.model.control()
        self.actuator_targets = ActuatorTargets(self.lab.robot_joints,
                                               self.control.joint_target_q.numpy())
        self.solver = create_solver(self.model)
        self.tick, self.time_s = 0, 0.
        self.fault = None
        self._validate_initial_contacts()

    def _validate_initial_contacts(self):
        """Reject assembled scene overlaps before the first physics step."""
        import mujoco

        m, d = self.solver.mj_model, self.solver.mj_data
        self.solver._update_mjc_data(d, self.model, self.state)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        overlaps = [{"geoms": [m.geom(int(g)).name for g in c.geom], "depth_m": -float(c.dist)}
                    for c in d.contact if c.dist < -.0001]
        if overlaps:
            raise ValueError(f"Initial rigid shapes overlap by more than 0.1 mm: {overlaps[:8]}")

    def command_joints(self, targets):
        """Only actual Panda actuator targets; never set joint/body positions."""
        self.actuator_targets.command(targets)

    def joint(self, name):
        joint = self.lab.apparatus.joints[name]
        q = self.lab.builder.joint_q_start[joint]
        v = self.lab.builder.joint_qd_start[joint]
        return float(self.state.joint_q.numpy()[q]), float(self.state.joint_qd.numpy()[v])

    def step(self, dt_s, *, assisted_joint_torques=None, body_wrenches=None, force_models=()):
        """One real Newton step; optional wrenches are labelled fixture access."""
        if self.fault:
            raise RuntimeError(f"Mechanics faulted: {self.fault}")
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Positive finite physics dt required")
        self.state.clear_forces()
        torques = np.zeros(self.model.joint_dof_count, np.float32)
        for name, value in (assisted_joint_torques or {}).items():
            if name not in self.lab.apparatus.joints or not math.isfinite(value):
                raise ValueError("Invalid assisted joint torque")
            j = self.lab.apparatus.joints[name]
            torques[self.lab.builder.joint_qd_start[j]] = value
        self.control.joint_f.assign(torques)
        if body_wrenches is not None:
            forces = np.asarray(body_wrenches, np.float32)
            if forces.shape != (self.model.body_count, 6) or not np.isfinite(forces).all():
                raise ValueError("Finite body-count x 6 wrench array required")
            self.state.body_f.assign(forces)
        try:
            self.control.joint_target_q.assign(self.actuator_targets.advance(dt_s))
            for force_model in force_models:
                force_model.apply(self.state, dt_s)
            self.solver.step(self.state, self.next_state, self.control, None, dt_s)
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                if not np.isfinite(getattr(self.next_state, name).numpy()).all():
                    raise FloatingPointError(f"Nonfinite {name} after Newton step")
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
            self.time_s += dt_s
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def proprioception(self):
        q, v = self.state.joint_q.numpy(), self.state.joint_qd.numpy()
        targets = self.actuator_targets.requested
        setpoints = self.control.joint_target_q.numpy()
        return {name: {"position": float(q[s["q"]]), "velocity": float(v[s["dof"]]),
                       "target": float(targets[s["q"]]), "servo_setpoint": float(setpoints[s["q"]])}
                for name, s in self.lab.robot_joints.items()}
