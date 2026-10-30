"""Multi-rate two-way coupling between a PhysX ball and cable goal net.

The cable graph is integrated at its validated constitutive rate only when the
ball approaches the net.  Its accumulated ball impulse is applied to the live
PhysX rigid body as an average wrench; no pose, velocity, spin, or task result
is assigned.  Equal-and-opposite contact reactions remain inside the cable
graph and ultimately reach its declared anchors.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .football_math import dot, norm, sub
from .football_specs import BallSpec
from .net_model import CableNetPatch


def _vec(values) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)


class NetBallCoupler:
    def __init__(
        self,
        *,
        ball_view,
        net: CableNetPatch,
        ball: BallSpec = BallSpec(),
        contact_substep_hz: float = 24_000.0,
        activation_distance_m: float = 0.80,
        predictive_lookahead_s: float = 0.050,
        post_contact_settle_s: float = 0.35,
        gravity_m_s2: float = 9.80665,
        recurrent_contact: bool = False,
    ) -> None:
        if not math.isfinite(contact_substep_hz) or contact_substep_hz <= 0.0:
            raise ValueError("net contact substep rate must be positive and finite")
        if not math.isfinite(activation_distance_m) or activation_distance_m <= ball.radius_m:
            raise ValueError("net activation distance must exceed the ball radius")
        if predictive_lookahead_s <= 0.0 or post_contact_settle_s <= 0.0:
            raise ValueError("net lookahead and post-contact duration must be positive")
        self.ball_view = ball_view
        self.net = net
        self.ball = ball
        self.contact_substep_hz = float(contact_substep_hz)
        self.contact_substep_dt_s = 1.0 / self.contact_substep_hz
        self.activation_distance_m = activation_distance_m
        self.predictive_lookahead_s = predictive_lookahead_s
        self.post_contact_settle_s = post_contact_settle_s
        self.gravity_m_s2 = gravity_m_s2
        self.recurrent_contact = recurrent_contact
        self.reset(reset_net=False)

    def reset(self, *, reset_net: bool = True) -> None:
        if reset_net:
            self.net.reset()
        self.activated = False
        self.ever_activated = False
        self.contact_seen = False
        self.contact_started_s: float | None = None
        self.contact_ended_s: float | None = None
        self.outer_contact_steps = 0
        self.solver_substeps = 0
        self.contact_substeps = 0
        self.peak_contact_force_n = 0.0
        self.maximum_deflection_m = 0.0
        self.impulse_on_ball_n_s = np.zeros(3, dtype=np.float64)
        self.angular_impulse_on_ball_n_m_s = np.zeros(3, dtype=np.float64)

    def signed_plane_distance_m(self, center_world_m) -> float:
        reference = self.net.surface_corners_world_m[0]
        return dot(
            sub(_vec(center_world_m), reference),
            self.net.reference_normal_world,
        )

    def step(self, time_s: float, dt_s: float) -> dict[str, Any]:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("net coupler timestep must be positive and finite")
        position, _ = self.ball_view.get_world_pose()
        center = _vec(position)
        velocity = _vec(self.ball_view.get_linear_velocity())
        omega = _vec(self.ball_view.get_angular_velocity())
        plane_distance = self.signed_plane_distance_m(center)
        normal_speed = dot(velocity, self.net.reference_normal_world)
        if plane_distance < 0.0 and normal_speed > 1.0e-9:
            time_to_plane_s = -plane_distance / normal_speed
        elif plane_distance > 0.0 and normal_speed < -1.0e-9:
            time_to_plane_s = -plane_distance / normal_speed
        else:
            time_to_plane_s = float("inf")
        imminent = (
            abs(plane_distance) <= self.activation_distance_m
            and 0.0 <= time_to_plane_s <= self.predictive_lookahead_s
        )
        already_geometrically_near = (
            abs(plane_distance)
            <= self.ball.radius_m + 0.5 * self.net.prior.cord_diameter_m + 0.01
            and abs(normal_speed) > 0.02
        )
        post_contact = (
            self.contact_seen
            and self.contact_ended_s is not None
            and time_s - self.contact_ended_s <= self.post_contact_settle_s
        )
        if self.recurrent_contact:
            # Later shots must wake the graph again. Legacy single-shot mode
            # intentionally retains its original activation rule for comparison.
            self.activated = post_contact or imminent or already_geometrically_near
        else:
            self.activated = post_contact if self.contact_seen else imminent or already_geometrically_near
        self.ever_activated = self.ever_activated or self.activated
        if not self.activated:
            return {
                "time_s": time_s,
                "activated": False,
                "signed_plane_distance_m": plane_distance,
                "normal_speed_m_s": normal_speed,
                "predicted_time_to_plane_s": (
                    None if not math.isfinite(time_to_plane_s) else time_to_plane_s
                ),
                "contact_count": 0,
                "maximum_contact_force_n": 0.0,
                "maximum_deflection_m": self.maximum_deflection_m,
                "force_on_ball_world_n": (0.0, 0.0, 0.0),
                "torque_on_ball_world_n_m": (0.0, 0.0, 0.0),
                "solver_substeps": 0,
                "contact_substeps": 0,
            }

        substep_count = max(1, int(math.ceil(dt_s / self.contact_substep_dt_s)))
        h = dt_s / substep_count
        local_center = np.asarray(center, dtype=np.float64)
        local_velocity = np.asarray(velocity, dtype=np.float64)
        local_omega = np.asarray(omega, dtype=np.float64)
        force_impulse = np.zeros(3, dtype=np.float64)
        torque_impulse = np.zeros(3, dtype=np.float64)
        maximum_contact_force = 0.0
        maximum_deflection = 0.0
        maximum_contact_count = 0
        contact_substeps = 0
        for substep in range(substep_count):
            result = self.net.step(
                ball_center_world_m=_vec(local_center),
                ball_velocity_world_m_s=_vec(local_velocity),
                ball_angular_velocity_world_rad_s=_vec(local_omega),
                ball=self.ball,
                dt_s=h,
            )
            force = np.asarray(result.force_on_ball_world_n, dtype=np.float64)
            torque = np.asarray(result.torque_on_ball_world_n_m, dtype=np.float64)
            force_impulse += h * force
            torque_impulse += h * torque
            maximum_contact_force = max(
                maximum_contact_force, result.maximum_contact_force_n
            )
            maximum_deflection = max(maximum_deflection, result.maximum_deflection_m)
            maximum_contact_count = max(maximum_contact_count, result.contact_count)
            if result.contact_count > 0:
                contact_substeps += 1
                self.contact_substeps += 1
                if not self.contact_seen:
                    self.contact_seen = True
                    self.contact_started_s = time_s + substep * h
                self.contact_ended_s = time_s + (substep + 1) * h

            # This is only the local predictor used to evaluate the next
            # constitutive sample.  PhysX receives the integrated impulse and
            # remains the sole owner of the actual ball state.
            local_velocity += h * force / self.ball.mass_kg
            local_velocity[2] -= h * self.gravity_m_s2
            local_omega += h * torque / self.ball.thin_shell_inertia_kg_m2
            local_center += h * local_velocity
            self.solver_substeps += 1

        sync_nodes = getattr(self.net, "sync_nodes", None)
        if sync_nodes is not None:
            sync_nodes()

        average_force = force_impulse / dt_s
        average_torque = torque_impulse / dt_s
        if norm(_vec(force_impulse)) > 0.0 or norm(_vec(torque_impulse)) > 0.0:
            self.ball_view._rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=np.asarray([average_force], dtype=np.float32),
                torques=np.asarray([average_torque], dtype=np.float32),
                is_global=True,
            )
            self.outer_contact_steps += 1
            self.impulse_on_ball_n_s += force_impulse
            self.angular_impulse_on_ball_n_m_s += torque_impulse
        self.peak_contact_force_n = max(
            self.peak_contact_force_n, maximum_contact_force
        )
        self.maximum_deflection_m = max(
            self.maximum_deflection_m, maximum_deflection
        )
        return {
            "time_s": time_s,
            "activated": True,
            "signed_plane_distance_m": plane_distance,
            "normal_speed_m_s": normal_speed,
            "predicted_time_to_plane_s": (
                None if not math.isfinite(time_to_plane_s) else time_to_plane_s
            ),
            "contact_count": maximum_contact_count,
            "maximum_contact_force_n": maximum_contact_force,
            "maximum_deflection_m": maximum_deflection,
            "force_on_ball_world_n": _vec(average_force),
            "torque_on_ball_world_n_m": _vec(average_torque),
            "solver_substeps": substep_count,
            "contact_substeps": contact_substeps,
        }

    def report(self) -> dict[str, Any]:
        return {
            "model_status": self.net.prior.evidence_status,
            "contact_substep_hz": self.contact_substep_hz,
            "activation_distance_m": self.activation_distance_m,
            "predictive_lookahead_s": self.predictive_lookahead_s,
            "post_contact_settle_s": self.post_contact_settle_s,
            "recurrent_contact": self.recurrent_contact,
            "activated": self.activated,
            "ever_activated": self.ever_activated,
            "contact_seen": self.contact_seen,
            "contact_started_s": self.contact_started_s,
            "contact_ended_s": self.contact_ended_s,
            "outer_contact_steps": self.outer_contact_steps,
            "solver_substeps": self.solver_substeps,
            "contact_substeps": self.contact_substeps,
            "peak_contact_force_n": self.peak_contact_force_n,
            "maximum_deflection_m": self.maximum_deflection_m,
            "impulse_on_ball_n_s": self.impulse_on_ball_n_s.tolist(),
            "angular_impulse_on_ball_n_m_s": (
                self.angular_impulse_on_ball_n_m_s.tolist()
            ),
            "node_count": len(self.net.nodes),
            "edge_count": len(self.net.edges),
            "total_cord_mass_kg": self.net.total_mass_kg,
            "surface_corners_world_m": self.net.surface_corners_world_m,
        }
