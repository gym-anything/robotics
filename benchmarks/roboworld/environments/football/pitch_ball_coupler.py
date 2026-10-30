"""Multi-rate compliant pitch wrench for a live PhysX football.

The pitch constitutive law is integrated at 20 kHz while the humanoid scene
runs at the official G1 policy's 500 Hz physics rate.  Only the integrated
contact wrench is applied to PhysX.  Gravity and already-computed external
wrenches may enter the local predictor so simultaneous kick/aero loading is
seen by the pitch law, but those external forces are never applied twice.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import numpy as np

from .football_math import norm
from .football_specs import BallSpec
from .pitch_model import PitchContactModel, PitchContactState, PitchPrior


def _vec(values) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)


class PitchBallCoupler:
    def __init__(
        self,
        *,
        ball_view,
        ball: BallSpec = BallSpec(),
        prior: PitchPrior = PitchPrior(),
        contact_substep_hz: float = 20_000.0,
        gravity_m_s2: float = 9.80665,
    ) -> None:
        if not math.isfinite(contact_substep_hz) or contact_substep_hz <= 0.0:
            raise ValueError("pitch substep rate must be positive and finite")
        self.ball_view = ball_view
        self.ball = ball
        self.prior = prior
        self.model = PitchContactModel(ball, prior)
        self.contact_substep_hz = float(contact_substep_hz)
        self.contact_substep_dt_s = 1.0 / self.contact_substep_hz
        self.gravity_m_s2 = gravity_m_s2
        # The contact law itself supplies the impact/resting transition scale.
        # At delta_g its elastic force exactly supports the ball weight.  A
        # few times sqrt(g*delta_g) is the corresponding low-amplitude speed
        # scale; no episode trajectory or desired outcome enters this choice.
        self.static_equilibrium_compression_m = (
            self.ball.mass_kg
            * self.gravity_m_s2
            / self.prior.normal_stiffness_n_per_m_pow_n
        ) ** (1.0 / self.prior.normal_exponent)
        self.quasistatic_compression_limit_m = (
            4.0 * self.static_equilibrium_compression_m
        )
        self.quasistatic_normal_speed_limit_m_s = 3.0 * math.sqrt(
            self.gravity_m_s2 * self.static_equilibrium_compression_m
        )
        self.reset()

    def reset(self) -> None:
        self.state = PitchContactState()
        # Once a compliant impact starts, its 20 kHz trajectory must persist
        # across the 500 Hz PhysX boundaries.  Re-seeding the micro-integrator
        # from the coarse position every outer step changes the constitutive
        # trajectory and spuriously dissipates rebound energy.
        self._contact_active = False
        self._awaiting_geometric_release = False
        self._local_center: np.ndarray | None = None
        self._local_velocity: np.ndarray | None = None
        self._local_omega: np.ndarray | None = None
        self._local_dt_s: float | None = None
        self._unseen_force_n = np.zeros(3, dtype=np.float64)
        # Coarse-body indentation left over when the micro-integrator released
        # (ratcheted down while the ball rises out).  A later impact is
        # measured from it rather than converting it into stored energy.
        self._stale_compression_m: float | None = None
        self.first_contact_s: float | None = None
        self.last_contact_s: float | None = None
        self.outer_contact_steps = 0
        self.contact_substeps = 0
        self.peak_force_n = 0.0
        self.maximum_compression_m = 0.0
        self.impulse_on_ball_n_s = np.zeros(3, dtype=np.float64)
        self.angular_impulse_on_ball_n_m_s = np.zeros(3, dtype=np.float64)
        self.maximum_force_closure_error_n = 0.0
        self.maximum_origin_moment_closure_error_n_m = 0.0
        self.maximum_position_disagreement_m = 0.0
        self.maximum_velocity_disagreement_m_s = 0.0
        self.maximum_angular_velocity_disagreement_rad_s = 0.0
        self.unmodelled_impulse_on_ball_n_s = np.zeros(3, dtype=np.float64)
        self.maximum_unmodelled_velocity_change_m_s = 0.0
        self.maximum_unseen_force_n = 0.0
        self.maximum_stale_release_compression_m = 0.0
        self.dynamic_outer_contact_steps = 0
        self.quasistatic_outer_contact_steps = 0

    def step(
        self,
        time_s: float,
        dt_s: float,
        *,
        external_force_on_ball_world_n=(0.0, 0.0, 0.0),
        external_torque_on_ball_world_n_m=(0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("pitch coupler timestep must be positive and finite")
        position, _ = self.ball_view.get_world_pose()
        actual_center = np.asarray(position, dtype=np.float64)
        actual_velocity = np.asarray(
            self.ball_view.get_linear_velocity(), dtype=np.float64
        )
        actual_omega = np.asarray(
            self.ball_view.get_angular_velocity(), dtype=np.float64
        )
        external_force = np.asarray(
            external_force_on_ball_world_n, dtype=np.float64
        )
        external_torque = np.asarray(
            external_torque_on_ball_world_n_m, dtype=np.float64
        )

        # A nonlinear compliant contact can release while the outer
        # semi-implicit step still leaves the coarse sphere a fraction below
        # the geometric plane.  Do not immediately start a second, fictitious
        # impact.  PhysX already has the correct outgoing impulse and normally
        # clears this sub-millimetre positional lag on the following step.
        if self._awaiting_geometric_release:
            stale = max(0.0, self.ball.radius_m - float(actual_center[2]))
            self._stale_compression_m = (
                stale
                if self._stale_compression_m is None
                else min(self._stale_compression_m, stale)
            )
            if (
                actual_center[2] >= self.ball.radius_m - 1.0e-7
                and actual_velocity[2] >= 0.0
            ):
                self._awaiting_geometric_release = False
                self._stale_compression_m = None
            elif actual_velocity[2] <= 0.0:
                # A very small rebound can turn around before the coarse
                # centre clears the plane.  Re-enter through the quasi-static
                # branch instead of suppressing support indefinitely.
                self._awaiting_geometric_release = False
            else:
                return {
                    "time_s": time_s,
                    "compression_m": max(
                        0.0, self.ball.radius_m - float(actual_center[2])
                    ),
                    "normal_force_n": 0.0,
                    "force_on_ball_world_n": (0.0, 0.0, 0.0),
                    "torque_on_ball_world_n_m": (0.0, 0.0, 0.0),
                    "contact_substeps": 0,
                    "sliding": False,
                    "integration_regime": "released_substep_lag",
                    "predictor_external_force_world_n": _vec(external_force),
                    "predictor_external_torque_world_n_m": _vec(
                        external_torque
                    ),
                }

        if self._contact_active:
            assert self._local_center is not None
            assert self._local_velocity is not None
            assert self._local_omega is not None
            local_center = self._local_center.copy()
            local_velocity = self._local_velocity.copy()
            local_omega = self._local_omega.copy()
            self.maximum_position_disagreement_m = max(
                self.maximum_position_disagreement_m,
                float(np.linalg.norm(local_center - actual_center)),
            )
            self.maximum_velocity_disagreement_m_s = max(
                self.maximum_velocity_disagreement_m_s,
                float(np.linalg.norm(local_velocity - actual_velocity)),
            )
            self.maximum_angular_velocity_disagreement_rad_s = max(
                self.maximum_angular_velocity_disagreement_rad_s,
                float(np.linalg.norm(local_omega - actual_omega)),
            )
            # The predictor cannot see forces that act on the live ball during
            # the outer step: native contacts (robot links, goal frame) and
            # labelled fixtures.  Their impulse is the difference between the
            # live velocity and the shadow's prediction of it.  Re-anchor the
            # shadow to the live velocity and add the displacement that the
            # same step-constant acceleration gives in the micro-integrator,
            # dt * dv / 2, then hold the total unseen force over the next step
            # (a zero-order-hold interface predictor).  The coarse-step
            # position offset itself is kept, so without unseen forces (dv is
            # float32 round-off) the trajectory is unchanged.
            assert self._local_dt_s is not None
            velocity_change = actual_velocity - local_velocity
            local_center += 0.5 * self._local_dt_s * velocity_change
            local_velocity = actual_velocity.copy()
            local_omega = actual_omega.copy()
            self._unseen_force_n = (
                self._unseen_force_n
                + self.ball.mass_kg * velocity_change / self._local_dt_s
            )
            self.unmodelled_impulse_on_ball_n_s += (
                self._local_dt_s * self._unseen_force_n
            )
            self.maximum_unmodelled_velocity_change_m_s = max(
                self.maximum_unmodelled_velocity_change_m_s,
                float(np.linalg.norm(velocity_change)),
            )
            self.maximum_unseen_force_n = max(
                self.maximum_unseen_force_n,
                float(np.linalg.norm(self._unseen_force_n)),
            )
        else:
            local_center = actual_center.copy()
            local_velocity = actual_velocity.copy()
            local_omega = actual_omega.copy()
            self._unseen_force_n = np.zeros(3, dtype=np.float64)
        unseen_force = self._unseen_force_n

        actual_compression = max(
            0.0, self.ball.radius_m - float(actual_center[2])
        )
        quasistatic_interval = (
            not self._contact_active
            and actual_compression <= self.quasistatic_compression_limit_m
            and abs(float(actual_velocity[2]))
            <= self.quasistatic_normal_speed_limit_m_s
        )
        if self._stale_compression_m is not None and not self._contact_active:
            stale = min(self._stale_compression_m, actual_compression)
            if quasistatic_interval or stale <= 0.0:
                self._stale_compression_m = None
            else:
                # A dynamic contact that starts after a latched release is
                # measured from the coarse overlap left at release, which the
                # micro-integrator had already unloaded.
                self._stale_compression_m = stale
                local_center[2] += stale
                self.maximum_stale_release_compression_m = max(
                    self.maximum_stale_release_compression_m, stale
                )

        substep_count = max(1, int(math.ceil(dt_s / self.contact_substep_dt_s)))
        h = dt_s / substep_count
        force_impulse = np.zeros(3, dtype=np.float64)
        torque_impulse = np.zeros(3, dtype=np.float64)
        outer_peak_force = 0.0
        outer_maximum_compression = 0.0
        contact_substeps = 0
        sliding = False
        event_contact_seen = self._contact_active
        separated = False
        for substep in range(substep_count):
            compression = max(0.0, self.ball.radius_m - float(local_center[2]))
            contact_point = (
                float(local_center[0]),
                float(local_center[1]),
                0.0,
            )
            if separated:
                wrench = self.model.ball_wrench(
                    compression_m=0.0,
                    ball_center_world_m=_vec(local_center),
                    ball_velocity_world_m_s=_vec(local_velocity),
                    ball_angular_velocity_world_rad_s=_vec(local_omega),
                    contact_point_world_m=contact_point,
                    normal_pitch_to_ball_world=(0.0, 0.0, 1.0),
                    dt_s=h,
                    state=PitchContactState(),
                )
            else:
                wrench = self.model.ball_wrench(
                    compression_m=compression,
                    ball_center_world_m=_vec(local_center),
                    ball_velocity_world_m_s=_vec(local_velocity),
                    ball_angular_velocity_world_rad_s=_vec(local_omega),
                    contact_point_world_m=contact_point,
                    normal_pitch_to_ball_world=(0.0, 0.0, 1.0),
                    dt_s=h,
                    state=self.state,
                )
            force = np.asarray(wrench.force_on_ball_world_n, dtype=np.float64)
            torque = np.asarray(wrench.torque_on_ball_world_n_m, dtype=np.float64)
            force_impulse += h * force
            torque_impulse += h * torque
            force_magnitude = wrench.normal.total_n
            outer_peak_force = max(outer_peak_force, force_magnitude)
            outer_maximum_compression = max(
                outer_maximum_compression, compression
            )
            sliding = sliding or wrench.state.sliding
            if force_magnitude > 0.0:
                event_contact_seen = True
                contact_substeps += 1
                self.contact_substeps += 1
                substep_time = time_s + substep * h
                if self.first_contact_s is None:
                    self.first_contact_s = substep_time
                self.last_contact_s = substep_time + h

                pitch_force = np.asarray(
                    wrench.force_on_pitch_world_n, dtype=np.float64
                )
                force_error = np.linalg.norm(force + pitch_force)
                pitch_torque_origin = np.asarray(
                    wrench.torque_on_pitch_world_n_m, dtype=np.float64
                )
                moment_ball_origin = torque + np.cross(local_center, force)
                moment_error = np.linalg.norm(
                    moment_ball_origin + pitch_torque_origin
                )
                self.maximum_force_closure_error_n = max(
                    self.maximum_force_closure_error_n, float(force_error)
                )
                self.maximum_origin_moment_closure_error_n_m = max(
                    self.maximum_origin_moment_closure_error_n_m,
                    float(moment_error),
                )
            elif event_contact_seen:
                # The unilateral law has reached zero on unloading.  Keep it
                # released for the remainder of this outer interval rather
                # than allowing a numerical re-contact before clearance.
                separated = True
            self.state = wrench.state

            # Local coupled predictor only.  PhysX independently integrates
            # gravity and all external wrenches once the averaged pitch wrench
            # has been applied below.
            total_predictor_force = force + external_force + unseen_force
            total_predictor_force[2] -= self.ball.mass_kg * self.gravity_m_s2
            local_velocity += h * total_predictor_force / self.ball.mass_kg
            local_omega += (
                h
                * (torque + external_torque)
                / self.ball.thin_shell_inertia_kg_m2
            )
            local_center += h * local_velocity

        if event_contact_seen and not separated and not quasistatic_interval:
            self._contact_active = True
            self._local_center = local_center.copy()
            self._local_velocity = local_velocity.copy()
            self._local_omega = local_omega.copy()
            self._local_dt_s = dt_s
            self._stale_compression_m = None
        elif separated:
            self._contact_active = False
            self._local_center = None
            self._local_velocity = None
            self._local_omega = None
            self._awaiting_geometric_release = True
            self._stale_compression_m = None
            self.state = PitchContactState()
        elif quasistatic_interval:
            # Resting/rolling support is re-evaluated from the live PhysX
            # state at every outer boundary.  Persisting a shadow position in
            # this regime would integrate tiny interface truncation errors
            # forever and eventually let a supported ball drift underground.
            self._contact_active = False
            self._local_center = None
            self._local_velocity = None
            self._local_omega = None

        average_force = force_impulse / dt_s
        average_torque = torque_impulse / dt_s
        if norm(_vec(force_impulse)) > 0.0 or norm(_vec(torque_impulse)) > 0.0:
            self.ball_view._rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=np.asarray([average_force], dtype=np.float32),
                torques=np.asarray([average_torque], dtype=np.float32),
                is_global=True,
            )
            self.outer_contact_steps += 1
            if quasistatic_interval:
                self.quasistatic_outer_contact_steps += 1
            else:
                self.dynamic_outer_contact_steps += 1
            self.impulse_on_ball_n_s += force_impulse
            self.angular_impulse_on_ball_n_m_s += torque_impulse
        self.peak_force_n = max(self.peak_force_n, outer_peak_force)
        self.maximum_compression_m = max(
            self.maximum_compression_m, outer_maximum_compression
        )
        return {
            "time_s": time_s,
            "compression_m": outer_maximum_compression,
            "normal_force_n": outer_peak_force,
            "force_on_ball_world_n": _vec(average_force),
            "torque_on_ball_world_n_m": _vec(average_torque),
            "contact_substeps": contact_substeps,
            "sliding": sliding,
            "integration_regime": (
                "quasistatic_live_reseed"
                if quasistatic_interval
                else "persistent_dynamic_subcycle"
            ),
            "predictor_external_force_world_n": _vec(external_force),
            "predictor_external_torque_world_n_m": _vec(external_torque),
        }

    def report(self) -> dict[str, Any]:
        return {
            "model_status": self.prior.evidence_status,
            "prior": asdict(self.prior),
            "contact_substep_hz": self.contact_substep_hz,
            "static_equilibrium_compression_m": (
                self.static_equilibrium_compression_m
            ),
            "quasistatic_compression_limit_m": (
                self.quasistatic_compression_limit_m
            ),
            "quasistatic_normal_speed_limit_m_s": (
                self.quasistatic_normal_speed_limit_m_s
            ),
            "first_contact_s": self.first_contact_s,
            "last_contact_s": self.last_contact_s,
            "outer_contact_steps": self.outer_contact_steps,
            "dynamic_outer_contact_steps": self.dynamic_outer_contact_steps,
            "quasistatic_outer_contact_steps": (
                self.quasistatic_outer_contact_steps
            ),
            "contact_substeps": self.contact_substeps,
            "peak_force_n": self.peak_force_n,
            "maximum_compression_m": self.maximum_compression_m,
            "impulse_on_ball_n_s": self.impulse_on_ball_n_s.tolist(),
            "angular_impulse_on_ball_n_m_s": (
                self.angular_impulse_on_ball_n_m_s.tolist()
            ),
            "maximum_force_closure_error_n": self.maximum_force_closure_error_n,
            "maximum_origin_moment_closure_error_n_m": (
                self.maximum_origin_moment_closure_error_n_m
            ),
            "maximum_position_disagreement_m": (
                self.maximum_position_disagreement_m
            ),
            "maximum_velocity_disagreement_m_s": (
                self.maximum_velocity_disagreement_m_s
            ),
            "maximum_angular_velocity_disagreement_rad_s": (
                self.maximum_angular_velocity_disagreement_rad_s
            ),
            "unmodelled_impulse_on_ball_n_s": (
                self.unmodelled_impulse_on_ball_n_s.tolist()
            ),
            "maximum_unmodelled_velocity_change_m_s": (
                self.maximum_unmodelled_velocity_change_m_s
            ),
            "maximum_held_unseen_force_n": self.maximum_unseen_force_n,
            "maximum_stale_release_compression_m": (
                self.maximum_stale_release_compression_m
            ),
            "post_reset_ball_state_writes": 0,
        }
