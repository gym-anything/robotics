"""Adapter for the existing constitutive laws; never assigns outgoing poses.

Newton spatial velocities/forces are linear/force first, then angular/torque,
about the centre of mass in world axes. Legacy view quaternions are WXYZ.
"""

import numpy as np


class RigidView:
    def __init__(self, world, body):
        self.world, self.body = world, body
        self._rigid_prim_view = self

    def get_world_pose(self):
        q = self.world.poses[self.body]
        return q[:3].copy(), q[[6, 3, 4, 5]].copy()

    def get_linear_velocity(self):
        return self.world.velocities[self.body, :3].copy()

    def get_angular_velocity(self):
        return self.world.velocities[self.body, 3:].copy()

    def apply_forces_and_torques_at_pos(self, *, forces, torques, is_global):
        if not is_global:
            raise ValueError(
                "Constitutive adapter only accepts world-frame COM wrenches"
            )
        wrench = np.r_[np.asarray(forces).reshape(3), np.asarray(torques).reshape(3)]
        if not np.isfinite(wrench).all():
            raise FloatingPointError("Non-finite constitutive wrench")
        self.world.wrenches[self.body] += wrench
