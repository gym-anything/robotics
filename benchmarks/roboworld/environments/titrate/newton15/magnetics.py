"""Dipole-driven stir bar; actual Newton contact and fluid loads govern motion.

Ideal motor dipole is an electrically powered boundary. Dipole moments are
engineering reconstructions from N35-scale remanence and core dimensions,
not IKA factory magnetic-field measurements. No bar pose/velocity is assigned.
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def dipole_force_torque(displacement, bar_moment, drive_moment):
    r = np.asarray(displacement, dtype=float)
    distance = np.linalg.norm(r)
    if not np.isfinite(r).all() or distance < .008:
        raise ValueError("Dipole approximation invalid inside the drive-magnet envelope")
    n = r/distance
    b, d = np.asarray(bar_moment), np.asarray(drive_moment)
    field = 1e-7/distance**3*(3*np.dot(d, n)*n-d)
    force = 3e-7/distance**4*(np.dot(b, n)*d+np.dot(d, n)*b+np.dot(b, d)*n-5*np.dot(b, n)*np.dot(d, n)*n)
    torque = np.cross(b, field)
    return force, torque, -float(np.dot(b, field))


@dataclass
class MagneticDrive:
    body: int
    position: tuple
    rpm: float = 0.
    phase: float = 0.
    motor_moment_am2: float = .375
    bar_moment_am2: float = .0675
    support_body: int | None = None
    support_com: tuple = (0., 0., 0.)

    def __post_init__(self):
        if not np.isfinite([*self.position, self.rpm, self.phase, self.motor_moment_am2, self.bar_moment_am2]).all():
            raise ValueError("Finite magnetic drive parameters required")
        if not 0 <= self.rpm <= 1500 or self.motor_moment_am2 <= 0 or self.bar_moment_am2 <= 0:
            raise ValueError("Drive specification outside declared operating envelope")
        self.last = {}

    def apply(self, state, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite magnetic timestep required")
        if self.rpm == 0:
            # A stopped permanent magnet still attracts and orients the bar.
            omega = 0.
        else:
            omega = self.rpm*2*np.pi/60
        self.phase = float((self.phase+omega*dt) % (2*np.pi))
        poses = state.body_q.numpy()
        q = poses[self.body]
        bar = Rotation.from_quat(q[3:]).apply([self.bar_moment_am2, 0., 0.])
        motor = self.motor_moment_am2*np.array([np.cos(self.phase), np.sin(self.phase), 0.])
        position = np.asarray(self.position, float)
        axis = np.array([0., 0., 1.])
        if self.support_body is not None:
            support = poses[self.support_body]
            rotation = Rotation.from_quat(support[3:])
            position = support[:3]+rotation.apply(position)
            motor, axis = rotation.apply(motor), rotation.apply(axis)
            support_com = support[:3]+rotation.apply(self.support_com)
        displacement = q[:3]-position
        f, torque, potential = dipole_force_torque(displacement, bar, motor)
        external = state.body_f.numpy()
        external[self.body, :3] += f
        external[self.body, 3:] += torque
        reaction_torque = -torque-np.cross(displacement, f)
        if self.support_body is not None:
            external[self.support_body, :3] -= f
            external[self.support_body, 3:] += reaction_torque+np.cross(position-support_com, -f)
        state.body_f.assign(external)
        self.last = {"force_n": f.tolist(), "torque_nm": torque.tolist(), "potential_j": potential,
                      "drive_rpm": self.rpm, "phase_rad": self.phase,
                      "motor_position_m": position.tolist(), "reaction_torque_nm": reaction_torque.tolist(),
                      "ideal_motor_mechanical_power_w": -float(np.dot(reaction_torque, omega*axis))}
