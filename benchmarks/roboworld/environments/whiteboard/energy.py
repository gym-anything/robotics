"""Read-only SI energy/momentum ledger for every executed integration step.

Constraint and actuator powers use native generalized force/velocity. Custom
contact/apparatus powers use COM wrenches. An independently computed global
impulse balance includes wall/floor contact impulses and gravity. No mj_forward,
extra step, trajectory modification or force replacement is performed.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class Ledger:
    def __init__(self, env):
        self.mass = env.model.body_mass.numpy().astype(float)
        self.inertia = env.model.body_inertia.numpy().astype(float)
        self.com = env.com
        self.map = env.solver.mjc_body_to_newton.numpy()[0]
        self.gravity = np.asarray(env.solver.mj_model.opt.gravity).copy()
        self.geom_body = self.map[env.solver.mj_model.geom_bodyid]

    def mechanical(self, env):
        q = env.state_0.body_q.numpy().astype(float)
        v = env.state_0.body_qd.numpy().astype(float)
        R = Rotation.from_quat(q[:, 3:]).as_matrix()
        world_inertia = R @ self.inertia @ R.transpose(0, 2, 1)
        x = q[:, :3] + np.einsum("bij,bj->bi", R, self.com)
        p = self.mass[:, None] * v[:, :3]
        spin = np.einsum("bij,bj->bi", world_inertia, v[:, 3:])
        kinetic = float(0.5 * np.sum(p * v[:, :3]) + 0.5 * np.sum(spin * v[:, 3:]))
        potential = float(-np.sum(self.mass * (x @ self.gravity)))
        native = env.solver.mj_data.qvel.copy()
        # Joint armature is geared motor rotor inertia: its kinetic energy is
        # not carried by any rigid body, so it is booked separately.
        rotor = float(0.5 * np.sum(env.solver.mj_model.dof_armature * native * native))
        return {
            "kinetic_j": kinetic,
            "rotor_kinetic_j": rotor,
            "potential_j": potential,
            "energy_j": kinetic + rotor + potential,
            "momentum": p.sum(axis=0),
            "angular_momentum": (spin + np.cross(x, p)).sum(axis=0),
            "velocity": v,
            "com_world": x,
            "native_velocity": native,
            "elastic_j": sum(r["energy_j"] for r in env.deformations.values()),
        }

    def finish(self, env, before, custom_forces, synchronized):
        import mujoco

        after = self.mechanical(env)
        m, d = env.solver.mj_model, env.solver.mj_data
        dt = env.config.physics_dt
        mid = 0.5 * (before["native_velocity"] + d.qvel)
        work = {
            name: float(np.dot(getattr(d, "qfrc_" + name), mid) * dt)
            for name in ("actuator", "constraint", "passive")
        }
        work["custom"] = float(
            np.sum(custom_forces * 0.5 * (before["velocity"] + after["velocity"])) * dt
        )
        force = self.mass.sum() * self.gravity + custom_forces[:, :3].sum(axis=0)
        torque = np.cross(before["com_world"], self.mass[:, None] * self.gravity).sum(axis=0)
        torque += (custom_forces[:, 3:] + np.cross(before["com_world"], custom_forces[:, :3])).sum(
            axis=0
        )
        # The same native forces and every contact are retained. Batch the
        # read-only vector algebra instead of allocating hundreds of tiny
        # arrays/cross-products each physics tick. Internal contacts cancel.
        c = d.contact
        bodies = self.geom_body[np.asarray(c.geom)]
        signs = (bodies[:, 1] >= 0).astype(int) - (bodies[:, 0] >= 0).astype(int)
        indices = np.flatnonzero((np.asarray(c.efc_address) >= 0) & (signs != 0))
        wrenches = np.zeros((len(indices), 6))
        for row, index in enumerate(indices):
            mujoco.mj_contactForce(m, d, int(index), wrenches[row])
        axes = np.asarray(c.frame)[indices].reshape(-1, 3, 3)
        f = np.einsum("nji,nj->ni", axes, wrenches[:, :3]) * signs[indices, None]
        t = np.einsum("nji,nj->ni", axes, wrenches[:, 3:]) * signs[indices, None]
        force += f.sum(axis=0)
        torque += (t + np.cross(np.asarray(c.pos)[indices], f)).sum(axis=0)
        contacts = len(indices)
        delta = after["energy_j"] - before["energy_j"]
        return {
            "rigid_energy_j": after["energy_j"],
            "kinetic_j": after["kinetic_j"],
            "rotor_kinetic_j": after["rotor_kinetic_j"],
            "potential_j": after["potential_j"],
            "elastic_energy_j": after["elastic_j"],
            "elastic_energy_change_j": after["elastic_j"] - before["elastic_j"],
            "rigid_energy_change_j": delta,
            "work_j": work,
            "energy_residual_j": delta - sum(work.values()),
            "momentum_residual_ns": (after["momentum"] - before["momentum"] - force * dt).tolist(),
            "angular_momentum_residual_nms": (
                after["angular_momentum"] - before["angular_momentum"] - torque * dt
            ).tolist(),
            "boundary_contacts": contacts,
            "valid_native_work_interval": not synchronized,
            "scope": "Discrete trapezoidal work audit; residual includes integrator/contact regularization error, not a claimed exact conservation identity",
        }
