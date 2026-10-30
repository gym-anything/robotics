"""Published Unitree G1 actuator limits on top of the football PD profile.

The profile's PD gains and peak torques are unchanged.  A plain PD with only a
torque clamp delivers that peak torque at any speed and has no rotor inertia,
so legal step commands spun joints 2.4-6x past the pinned URDF velocity
limits.  Two hardware properties are added:

* Reflected rotor and gear inertia of each joint's motor (MuJoCo armature).
* The motor torque-speed envelope.  Back-EMF against the supply voltage makes
  the available torque fall linearly with speed, to zero at the no-load speed;
  the current limit caps it at the profile peak torque.  Torque that opposes
  the motion (braking) stays available up to that cap.

Body motors: Unitree's published actuator tables (unitreerobotics/unitree_rl_lab,
``assets/robots/unitree_actuators.py``) give each motor's T-N knee point
(X1 rad/s, Y1 N m), no-load speed X2 and reflected inertia; ``unitree.py``
assigns them to G1 29-DoF joints, with the ankle and waist roll/pitch driven
by two N5020-16 through a parallel linkage.  The envelope keeps that motor's
back-EMF slope Y1 / (X2 - X1) and reaches zero torque at the pinned URDF
``<limit velocity>`` of the joint, so a self-driven joint cannot exceed it.

Dex3-1 finger drives have no published T-N data or rotor inertia.  They are
ideal DC motors whose stall torque is the profile effort and whose no-load
speed is the URDF velocity: ``clip(PD, -E, E) - (E / w0) w``.  The back-EMF
term is joint damping so MuJoCo integrates it implicitly; a finger link's
mechanical time constant is far below the 2 ms step, where an explicitly
lagged limit would chatter.
"""

import re

import numpy as np

# knee-point speed (rad/s), knee-point torque (N m), no-load speed (rad/s),
# reflected inertia (kg m^2)
UNITREE_MOTORS = {
    "N7520-14.3": (22.63, 71.0, 35.52, 0.01017752),
    "N7520-22.5": (14.5, 111.0, 22.7, 0.025101925),
    "N5020-16": (30.86, 24.8, 40.13, 0.003609725),
    "W4010-25": (15.3, 4.8, 24.76, 0.00425),
}
# joint-name pattern, motor, motors driving the joint
G1_JOINT_MOTORS = (
    (r".*_hip_(pitch|yaw)_joint|waist_yaw_joint", "N7520-14.3", 1),
    (r".*_hip_roll_joint|.*_knee_joint", "N7520-22.5", 1),
    (r".*_ankle_(pitch|roll)_joint|waist_(roll|pitch)_joint", "N5020-16", 2),
    (r".*_shoulder_(pitch|roll|yaw)_joint|.*_elbow_joint|.*_wrist_roll_joint", "N5020-16", 1),
    (r".*_wrist_(pitch|yaw)_joint", "W4010-25", 1),
)


def motor_of(joint):
    for pattern, motor, count in G1_JOINT_MOTORS:
        if re.fullmatch(pattern, joint):
            return motor, count
    return None, 0


class G1Actuators:
    """Adds armature/back-EMF to the builder, then bounds each body joint's
    actuator torque to the envelope at the start of every physics tick."""

    def __init__(self, builder, joints, velocity_limits):
        self.joints = {}
        for name, a in joints.items():
            limit = float(velocity_limits[name])
            effort = float(a["effort_nm"])
            motor, count = motor_of(name)
            if motor is None:
                if "_hand_" not in name:
                    raise ValueError(f"No actuator model for G1 joint {name}")
                damping = effort / limit
                builder.joint_damping[a["dof"]] = damping
                self.joints[name] = dict(
                    dof=a["dof"], drive="Dex3-1 ideal DC motor (no published T-N data)",
                    velocity_limit_rad_s=limit, effort_nm=effort, back_emf_damping_nm_s=damping,
                    armature_kg_m2=0.0,
                )
                continue
            knee_speed, knee_torque, no_load_speed, armature = UNITREE_MOTORS[motor]
            slope = count * knee_torque / (no_load_speed - knee_speed)
            builder.joint_armature[a["dof"]] = count * armature
            self.joints[name] = dict(
                dof=a["dof"], drive=f"{count} x Unitree {motor}", velocity_limit_rad_s=limit,
                effort_nm=effort, back_emf_slope_nm_s=slope, stall_torque_nm=slope * limit,
                armature_kg_m2=count * armature,
            )
        self.enveloped = [n for n, j in self.joints.items() if "back_emf_slope_nm_s" in j]

    def bind(self, world):
        """Map to native joints; effective inertia for the implicit envelope."""
        import mujoco

        m = world.solver.mj_model
        jnt_dof = world.solver.mjc_jnt_to_newton_dof.numpy().reshape(-1)
        mj_dof = world.solver.mjc_dof_to_newton_dof.numpy().reshape(-1)
        rows = [self.joints[n] for n in self.enveloped]
        self.dofs = np.array([j["dof"] for j in rows])
        self.mj_joints = np.array([int(np.flatnonzero(jnt_dof == d)[0]) for d in self.dofs])
        native_dofs = [int(np.flatnonzero(mj_dof == d)[0]) for d in self.dofs]
        self.effort = m.jnt_actfrcrange[self.mj_joints, 1].copy()
        if not np.allclose(self.effort, [j["effort_nm"] for j in rows], rtol=1e-6):
            raise RuntimeError("Native actuator force range differs from the profile effort")
        self.slope = np.array([j["back_emf_slope_nm_s"] for j in rows])
        self.stall = np.array([j["stall_torque_nm"] for j in rows])
        # Smallest (unconstrained, free-floating) joint-space inertia at the
        # reset pose, armature included: 1 / (M^-1)_ii.  Contacts only add
        # inertia, so the implicit factor below never under-damps.
        d = mujoco.MjData(m)
        q = world.state_0.joint_q.numpy()
        for name, a in world.joints.items():
            joint = int(np.flatnonzero(jnt_dof == a["dof"])[0])
            d.qpos[m.jnt_qposadr[joint]] = q[a["q"]]
        mujoco.mj_forward(m, d)
        unit = np.zeros((len(native_dofs), m.nv))
        unit[np.arange(len(native_dofs)), native_dofs] = 1.0
        response = np.zeros_like(unit)
        mujoco.mj_solveM(m, d, response, unit)
        self.inertia = 1.0 / response[np.arange(len(native_dofs)), native_dofs]
        self.implicit = world.config.dt * self.slope / self.inertia
        for j, inertia, x in zip(rows, self.inertia, self.implicit):
            j.update(effective_inertia_kg_m2=float(inertia), implicit_factor=float(x))
        if np.any(self.stall / (1.0 + self.implicit) < self.effort):
            raise RuntimeError("Implicit envelope would reduce stall torque")
        self.model = m

    def apply(self, world):
        # Torque available over the next step, with the back-EMF term taken at
        # the end-of-step speed of the joint accelerating under it (linearly
        # implicit; identical to the explicit line for heavy links).
        omega = world.qd[self.dofs].astype(np.float64)
        scale = 1.0 + self.implicit
        upper = np.clip((self.stall - self.slope * omega) / scale, -self.effort, self.effort)
        lower = np.clip((-self.stall - self.slope * omega) / scale, -self.effort, self.effort)
        self.model.jnt_actfrcrange[self.mj_joints, 0] = lower
        self.model.jnt_actfrcrange[self.mj_joints, 1] = upper

    def report(self):
        return {
            "model": "published G1 motor torque-speed envelope and reflected inertia",
            "sources": [
                "unitreerobotics/unitree_rl_lab assets/robots/unitree_actuators.py (T-N X1/Y1/X2, armature)",
                "unitreerobotics/unitree_rl_lab assets/robots/unitree.py (G1 29-DoF joint-to-motor map)",
                "pinned g1_29dof_with_hand_rev_1_0.urdf <limit velocity>",
            ],
            "joints": self.joints,
        }
