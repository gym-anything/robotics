"""Pinned free-floating G1 and live operational mass; no task policy."""

import numpy as np

from football.operational_mass import point_inverse_mass_response
from football.robot_adapter import RobotObservation


def add_robot(builder, wp, config):
    """Legacy joint addresses and each joint's pinned URDF velocity limit."""
    from robots.backends.newton import add_robot as build, mount
    runtime = build(builder,'g1','football',instance='robot',prefix='',
        placement=mount(config.robot_position,config.robot_yaw))
    velocity = {n: a['velocity_limit'] for n, a in runtime.joints.items()}
    return runtime.metadata['legacy_addresses'], velocity


class Robot:
    def __init__(self, world):
        self.world = world
        self.dof_names = tuple(world.joints)
        self.dof_index = {name: i for i, name in enumerate(self.dof_names)}

    def observe(self, time_s=None):
        w = self.world
        names = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link")
        p, quat = w.views["pelvis"].get_world_pose()
        return RobotObservation(
            time_s=w.time if time_s is None else time_s,
            base_position_world_m=tuple(p),
            base_orientation_world_wxyz=tuple(quat),
            base_linear_velocity_world_m_s=tuple(
                w.views["pelvis"].get_linear_velocity()
            ),
            base_angular_velocity_world_rad_s=tuple(
                w.views["pelvis"].get_angular_velocity()
            ),
            joint_position_rad=tuple(w.q[a["q"]] for a in w.joints.values()),
            joint_velocity_rad_s=tuple(w.qd[a["dof"]] for a in w.joints.values()),
            link_positions_world_m={
                n: tuple(w.views[n].get_world_pose()[0]) for n in names
            },
            link_orientations_world_wxyz={
                n: tuple(w.views[n].get_world_pose()[1]) for n in names
            },
            link_linear_velocities_world_m_s={
                n: tuple(w.views[n].get_linear_velocity()) for n in names
            },
            link_angular_velocities_world_rad_s={
                n: tuple(w.views[n].get_angular_velocity()) for n in names
            },
        )

    def inverse_mass(self, point, normal, foot="right_ankle_roll_link"):
        """Query the native live mass matrix without mutating native solver data.

        Use the free articulated tensor. A planted-support ideal-constraint
        approximation is used only while that foot has an actual ground contact.
        """
        import mujoco

        w = self.world
        m, source = w.solver.mj_model, w.solver.mj_data
        d = w.mass_query_data
        d.qpos[:] = source.qpos
        d.qvel[:] = source.qvel
        mujoco.mj_forward(m, d)
        mass = np.empty((m.nv, m.nv))
        mujoco.mj_fullM(m, d, mass)
        body = w.mj_body_ids[w.body_ids[foot]]
        j = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, j, None, np.asarray(point, dtype=float), body)
        free = point_inverse_mass_response(mass, j)
        support_name = (
            "left_ankle_roll_link"
            if foot.startswith("right")
            else "right_ankle_roll_link"
        )
        support = w.mj_body_ids[w.body_ids[support_name]]
        grounded = any(
            {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} == {0, support}
            and c.dist <= 0.001
            for c in source.contact[: source.ncon]
        )
        response = free
        if grounded:
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, support)
            response = point_inverse_mass_response(
                mass, j, constraint_jacobian=np.vstack([jp, jr])
            )
        n = np.asarray(normal)
        return {
            "point_inverse_mass_world_per_kg": response.tolist(),
            "free_point_inverse_mass_world_per_kg": free.tolist(),
            "support_ground_contact": grounded,
            "effective_mass_kg": float(1.0 / (n @ response @ n)),
            "source": "Newton MuJoCo live M and point Jacobian; contact-conditioned support approximation",
        }
