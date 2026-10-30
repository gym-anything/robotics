"""Two-way reduced-order coupling between a pressurised ball and robot foot.

The coupler applies equal-and-opposite wrenches to live PhysX rigid bodies.  It
never assigns an outgoing velocity, pose, spin, or task result.

Foot--ball contact lasts only a few milliseconds, whereas the whole humanoid
does not need to be stepped at tens of kilohertz.  The constitutive law is
therefore integrated on an internal fixed-rate clock and its impulse is
returned to PhysX as the average wrench over the enclosing physics step.  This
is a multi-rate force coupling, not a prescribed post-impact velocity.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .foot_contact import (
    ConvexFootProxy,
    FootPatchSpec,
    SphereFootContactGeometry,
    load_convex_foot_proxy,
    sphere_convex_contact,
)
from .football_math import ZERO, add, cross, dot, norm, quat_rotate, scale, sub
from .football_specs import BallSpec
from .impact_model import ContactState, PressureShellContactModel
from .operational_mass import (
    effective_mass_from_point_response,
    rigid_body_point_inverse_mass_response,
)


def _vec(values) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)


FootInverseMassProvider = Callable[
    [tuple[float, float, float], tuple[float, float, float]],
    Mapping[str, Any],
]
FootContactGeometryProvider = Callable[..., SphereFootContactGeometry]


class FootBallCoupler:
    def __init__(
        self,
        *,
        ball_view,
        foot_view,
        foot_center_of_mass_local_m: tuple[float, float, float],
        ball: BallSpec = BallSpec(),
        patch: FootPatchSpec = FootPatchSpec(),
        contact_model: PressureShellContactModel | None = None,
        contact_substep_hz: float = 20_000.0,
        foot_inverse_mass_provider: FootInverseMassProvider | None = None,
        foot_contact_geometry_provider: FootContactGeometryProvider | None = None,
        foot_proxy: ConvexFootProxy | None = None,
    ) -> None:
        if not math.isfinite(contact_substep_hz) or contact_substep_hz <= 0.0:
            raise ValueError("contact substep rate must be positive and finite")
        self.ball_view = ball_view
        self.foot_view = foot_view
        self.foot_center_of_mass_local_m = foot_center_of_mass_local_m
        self.ball = ball
        self.patch = patch
        self.model = contact_model or PressureShellContactModel(ball)
        self.contact_substep_hz = float(contact_substep_hz)
        self.contact_substep_dt_s = 1.0 / self.contact_substep_hz
        self.foot_inverse_mass_provider = foot_inverse_mass_provider
        if foot_contact_geometry_provider is not None and foot_proxy is not None:
            raise ValueError(
                "provide either a foot contact geometry provider or a convex "
                "foot proxy, not both"
            )
        self.foot_contact_geometry_provider = foot_contact_geometry_provider
        if foot_contact_geometry_provider is None:
            proxy_path = Path(self.patch.proxy_relative_path)
            if not proxy_path.is_absolute():
                proxy_path = Path(__file__).resolve().parent / proxy_path
            self.foot_proxy = foot_proxy or load_convex_foot_proxy(
                proxy_path
            )
            # Validate the complete pinned-asset contract immediately rather
            # than discovering a bad proxy at first impact.
            sphere_convex_contact(
                ball_center_world_m=(1.0, 0.0, 0.0),
                ball_radius_m=self.ball.radius_m,
                foot_center_world_m=(0.0, 0.0, 0.0),
                foot_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
                patch=self.patch,
                proxy=self.foot_proxy,
            )
            self.contact_geometry_audit = self.foot_proxy.audit()
        else:
            self.foot_proxy = None
            self.contact_geometry_audit = {
                "model_status": "EXPLICITLY INJECTED CONTACT GEOMETRY PROVIDER",
                "provider": getattr(
                    foot_contact_geometry_provider,
                    "__qualname__",
                    repr(foot_contact_geometry_provider),
                ),
            }
        self.reset()

    def _contact_geometry(
        self,
        *,
        ball_center_world_m: tuple[float, float, float],
        foot_center_world_m: tuple[float, float, float],
        foot_orientation_world_wxyz: tuple[float, float, float, float],
    ) -> SphereFootContactGeometry:
        arguments = {
            "ball_center_world_m": ball_center_world_m,
            "ball_radius_m": self.ball.radius_m,
            "foot_center_world_m": foot_center_world_m,
            "foot_orientation_world_wxyz": foot_orientation_world_wxyz,
            "patch": self.patch,
        }
        if self.foot_contact_geometry_provider is not None:
            return self.foot_contact_geometry_provider(**arguments)
        return sphere_convex_contact(**arguments, proxy=self.foot_proxy)

    def reset(self) -> None:
        self.state = ContactState()
        self.engaged = False
        self.awaiting_geometric_separation = False
        # Live overlap left when the internal mode opened, ratcheted down as
        # the bodies separate.  Only overlap up to it is treated as stale.
        self.stale_overlap_m: float | None = None
        self.release_hysteresis_steps = 0
        self.maximum_release_hysteresis_overlap_m = 0.0
        self.reengagements_beyond_stale_overlap = 0
        self.compression_m = 0.0
        self._predicted_relative_velocity: tuple[float, float, float] | None = None
        self._predicted_dt_s: float | None = None
        self._unmodelled_relative_acceleration: tuple[float, float, float] = ZERO
        self.maximum_unmodelled_closing_speed_change_m_s = 0.0
        self.maximum_unmodelled_relative_acceleration_m_s2 = 0.0
        self.unmodelled_compression_correction_m = 0.0
        self.boundary_stored_energy_j = 0.0
        self.contact_started_s: float | None = None
        self.contact_ended_s: float | None = None
        self.peak_force_n = 0.0
        self.peak_compression_m = 0.0
        self.impulse_on_ball_n_s = np.zeros(3, dtype=np.float64)
        self.impulse_on_foot_n_s = np.zeros(3, dtype=np.float64)
        self.maximum_force_closure_error_n = 0.0
        self.maximum_origin_moment_closure_error_n_m = 0.0
        self.contact_steps = 0
        self.contact_substeps = 0
        self.operational_response_evaluations = 0
        self.foot_directional_effective_masses_kg: list[float] = []
        self.combined_directional_reduced_masses_kg: list[float] = []
        self.operational_response_sources: set[str] = set()
        self.subcycle_contact_work_extracted_j = 0.0
        self.subcycle_relative_kinetic_loss_j = 0.0
        self.subcycle_constitutive_dissipation_j = 0.0
        self.subcycle_numerical_energy_creation_j = 0.0
        self.contact_initial_stored_energy_j = 0.0
        self.minimum_incremental_constitutive_dissipation_j = float("inf")
        self.maximum_subcycle_work_closure_error_j = 0.0

    @staticmethod
    def _zero_sample(
        *,
        time_s: float,
        geometry,
        foot_com,
        relative_velocity,
        engaged: bool,
    ) -> dict[str, Any]:
        return {
            "time_s": time_s,
            "geometry": asdict(geometry),
            "foot_center_of_mass_world_m": foot_com,
            "relative_velocity_foot_minus_ball_world_m_s": relative_velocity,
            "compression_m": 0.0,
            "normal_force_n": 0.0,
            "force_on_ball_world_n": ZERO,
            "torque_on_ball_world_n_m": ZERO,
            "force_on_foot_world_n": ZERO,
            "torque_on_foot_world_n_m": ZERO,
            "sliding": False,
            "engaged": engaged,
            "contact_substeps": 0,
            "point_inverse_mass_response": None,
        }

    def _point_inverse_mass_response(
        self,
        *,
        contact_point: tuple[float, float, float],
        ball_arm: tuple[float, float, float],
        normal: tuple[float, float, float],
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Build the live foot + hollow-ball relative point response."""

        if self.foot_inverse_mass_provider is None:
            foot_response = (
                np.eye(3, dtype=np.float64)
                / self.patch.effective_mass_prior_kg
            )
            provider_audit: Mapping[str, Any] = {}
            source = "FootPatchSpec isotropic effective-mass fallback prior"
            evidence_status = (
                "FALLBACK PRIOR / NOT CONFIGURATION-DEPENDENT / "
                "NOT HARDWARE-CALIBRATED"
            )
        else:
            provider_audit = self.foot_inverse_mass_provider(
                contact_point, normal
            )
            if "point_inverse_mass_world_per_kg" not in provider_audit:
                raise ValueError(
                    "foot inverse-mass provider must return "
                    "point_inverse_mass_world_per_kg"
                )
            foot_response = np.asarray(
                provider_audit["point_inverse_mass_world_per_kg"],
                dtype=np.float64,
            )
            source = str(provider_audit.get("source", "runtime provider"))
            evidence_status = str(
                provider_audit.get("evidence_status", "UNLABELLED MODEL VALUE")
            )
        if foot_response.shape != (3, 3) or not np.all(np.isfinite(foot_response)):
            raise ValueError(
                "foot inverse-mass provider returned a non-finite or non-3x3 tensor"
            )
        foot_response = 0.5 * (foot_response + foot_response.T)
        minimum_eigenvalue = float(np.linalg.eigvalsh(foot_response)[0])
        if minimum_eigenvalue < -1.0e-9:
            raise ValueError(
                "foot inverse-mass tensor is not positive semidefinite "
                f"(minimum eigenvalue {minimum_eigenvalue})"
            )

        ball_inertia = np.eye(3, dtype=np.float64) * (
            self.ball.thin_shell_inertia_kg_m2
        )
        ball_response = rigid_body_point_inverse_mass_response(
            self.ball.mass_kg,
            ball_inertia,
            ball_arm,
        )
        combined_response = foot_response + ball_response
        foot_effective_mass = effective_mass_from_point_response(
            foot_response, normal
        )
        ball_effective_mass = effective_mass_from_point_response(
            ball_response, normal
        )
        reduced_mass = effective_mass_from_point_response(
            combined_response, normal
        )
        self.operational_response_evaluations += 1
        self.foot_directional_effective_masses_kg.append(foot_effective_mass)
        self.combined_directional_reduced_masses_kg.append(reduced_mass)
        self.operational_response_sources.add(source)
        audit = {
            "source": source,
            "evidence_status": evidence_status,
            "foot_directional_effective_mass_kg": foot_effective_mass,
            "ball_directional_effective_mass_kg": ball_effective_mass,
            "combined_directional_reduced_mass_kg": reduced_mass,
            "foot_point_inverse_mass_world_per_kg": foot_response.tolist(),
            "ball_point_inverse_mass_world_per_kg": ball_response.tolist(),
            "combined_relative_point_inverse_mass_world_per_kg": (
                combined_response.tolist()
            ),
            "provider_audit": dict(provider_audit),
        }
        return combined_response, reduced_mass, audit

    def _stored_contact_energy_j(
        self,
        compression_m: float,
        state: ContactState,
    ) -> float:
        normal_energy = (
            self.model.pressure_energy_j(compression_m)
            + self.model.shell_energy_j(compression_m)
        )
        tangential_displacement = np.asarray(
            state.tangential_displacement_m, dtype=np.float64
        )
        tangential_energy = (
            0.5
            * self.model.prior.tangential_stiffness_n_m
            * float(tangential_displacement @ tangential_displacement)
        )
        return normal_energy + tangential_energy

    def current_geometry(self) -> SphereFootContactGeometry:
        """Re-evaluate live outer-step geometry without changing contact state."""

        ball_position, _ = self.ball_view.get_world_pose()
        foot_position, foot_orientation = self.foot_view.get_world_pose()
        return self._contact_geometry(
            ball_center_world_m=_vec(ball_position),
            foot_center_world_m=_vec(foot_position),
            foot_orientation_world_wxyz=tuple(
                float(value) for value in foot_orientation
            ),
        )

    def step(self, time_s: float, dt_s: float) -> dict[str, Any]:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("coupler timestep must be positive and finite")

        # Query each live rigid body exactly once.  The internal impact solve
        # evolves only the local relative mode; PhysX remains responsible for
        # the complete articulated-body response to the resulting impulse.
        ball_position, _ = self.ball_view.get_world_pose()
        foot_position, foot_orientation = self.foot_view.get_world_pose()
        ball_center = _vec(ball_position)
        foot_origin = _vec(foot_position)
        foot_orientation_tuple = tuple(float(value) for value in foot_orientation)
        foot_com = add(
            foot_origin,
            quat_rotate(foot_orientation_tuple, self.foot_center_of_mass_local_m),
        )
        geometry = self._contact_geometry(
            ball_center_world_m=ball_center,
            foot_center_world_m=foot_origin,
            foot_orientation_world_wxyz=foot_orientation_tuple,
        )

        ball_velocity = _vec(self.ball_view.get_linear_velocity())
        ball_omega = _vec(self.ball_view.get_angular_velocity())
        foot_velocity = _vec(self.foot_view.get_linear_velocity())
        foot_omega = _vec(self.foot_view.get_angular_velocity())
        contact_point = geometry.closest_point_on_foot_world_m
        ball_arm = sub(contact_point, ball_center)
        foot_arm = sub(contact_point, foot_com)
        ball_point_velocity = add(ball_velocity, cross(ball_omega, ball_arm))
        foot_point_velocity = add(foot_velocity, cross(foot_omega, foot_arm))
        relative_velocity = sub(foot_point_velocity, ball_point_velocity)
        normal = geometry.normal_foot_to_ball_world
        incident_closing_speed = dot(relative_velocity, normal)

        # The internal mode sees only the shoe force and the two bodies'
        # inertia.  Actuators, gravity, the pitch under the ball and native
        # contacts also move the bodies during the outer step; their effect is
        # the difference between the measured contact-point velocity and the
        # internal prediction of it.  The internal integrator would have
        # turned that step-constant acceleration into dt * dv / 2 of
        # indentation, so add it, and hold the total acceleration of those
        # loads over the next step (a zero-order-hold interface predictor).
        # Sustained presses then load the ball as the live geometry does,
        # instead of the internal compression drifting open.
        unmodelled_acceleration = ZERO
        if self.engaged and self._predicted_relative_velocity is not None:
            assert self._predicted_dt_s is not None
            velocity_change = sub(
                relative_velocity, self._predicted_relative_velocity
            )
            closing_speed_change = dot(velocity_change, normal)
            correction_m = 0.5 * self._predicted_dt_s * closing_speed_change
            before_j = self._stored_contact_energy_j(self.compression_m, self.state)
            self.compression_m = max(0.0, self.compression_m + correction_m)
            self.boundary_stored_energy_j += (
                self._stored_contact_energy_j(self.compression_m, self.state)
                - before_j
            )
            unmodelled_acceleration = add(
                self._unmodelled_relative_acceleration,
                scale(1.0 / self._predicted_dt_s, velocity_change),
            )
            self.unmodelled_compression_correction_m += abs(correction_m)
            self.maximum_unmodelled_closing_speed_change_m_s = max(
                self.maximum_unmodelled_closing_speed_change_m_s,
                abs(closing_speed_change),
            )
            self.maximum_unmodelled_relative_acceleration_m_s2 = max(
                self.maximum_unmodelled_relative_acceleration_m_s2,
                norm(unmodelled_acceleration),
            )
        self._predicted_relative_velocity = None
        self._unmodelled_relative_acceleration = unmodelled_acceleration

        # The internal contact mode can reach zero indentation during an outer
        # 2 ms PhysX interval even though the as-yet-unadvanced rigid poses
        # still overlap.  Re-seeding a new compressed spring from that stale
        # outer geometry creates energy and contact chatter.  Once the local
        # mode opens, the overlap still present on the next step is stale and
        # only shrinks while the bodies separate; overlap beyond it is a new
        # compression of the ball and engages the contact law at once.  This
        # is a partitioned-coupling release latch, not a pose, velocity, or
        # outcome assignment.
        if not self.engaged and self.awaiting_geometric_separation:
            overlap = geometry.compression_m
            if overlap <= 0.0:
                self.awaiting_geometric_separation = False
                self.stale_overlap_m = None
            else:
                self.stale_overlap_m = (
                    overlap
                    if self.stale_overlap_m is None
                    else min(self.stale_overlap_m, overlap)
                )
                if overlap <= self.stale_overlap_m:
                    self.release_hysteresis_steps += 1
                    self.maximum_release_hysteresis_overlap_m = max(
                        self.maximum_release_hysteresis_overlap_m, overlap
                    )
                    return self._zero_sample(
                        time_s=time_s,
                        geometry=geometry,
                        foot_com=foot_com,
                        relative_velocity=relative_velocity,
                        engaged=False,
                    )
                self.awaiting_geometric_separation = False
                self.engaged = True
                self.compression_m = overlap - self.stale_overlap_m
                self.stale_overlap_m = None
                self.reengagements_beyond_stale_overlap += 1
                self.boundary_stored_energy_j += self._stored_contact_energy_j(
                    self.compression_m, self.state
                )

        # Continuous onset prediction prevents an outer 500 Hz step from
        # discovering millimetres of overlap that a 2 kHz step never saw.
        # The surface is locally frozen only for the sub-millisecond onset
        # estimate; after that the constitutive subsolver updates velocity.
        active_delay_s = 0.0
        if not self.engaged:
            if geometry.compression_m > 0.0:
                self.engaged = True
                self.compression_m = geometry.compression_m
            else:
                gap_m = max(0.0, geometry.signed_distance_m - self.ball.radius_m)
                if incident_closing_speed <= 0.0:
                    return self._zero_sample(
                        time_s=time_s,
                        geometry=geometry,
                        foot_com=foot_com,
                        relative_velocity=relative_velocity,
                        engaged=False,
                    )
                active_delay_s = gap_m / incident_closing_speed
                if active_delay_s >= dt_s:
                    return self._zero_sample(
                        time_s=time_s,
                        geometry=geometry,
                        foot_com=foot_com,
                        relative_velocity=relative_velocity,
                        engaged=False,
                    )
                self.engaged = True
                self.compression_m = 0.0
            if self.contact_started_s is None:
                self.contact_started_s = time_s + active_delay_s
                self.contact_initial_stored_energy_j = (
                    self._stored_contact_energy_j(
                        self.compression_m, self.state
                    )
                )
            else:
                self.boundary_stored_energy_j += self._stored_contact_energy_j(
                    self.compression_m, self.state
                )

        active_duration_s = dt_s - active_delay_s
        substep_count = max(
            1, int(math.ceil(active_duration_s / self.contact_substep_dt_s))
        )
        substep_dt_s = active_duration_s / substep_count
        local_relative_velocity = relative_velocity
        ball_force_impulse = np.zeros(3, dtype=np.float64)
        foot_force_impulse = np.zeros(3, dtype=np.float64)
        ball_torque_impulse = np.zeros(3, dtype=np.float64)
        foot_torque_impulse = np.zeros(3, dtype=np.float64)
        outer_peak_force_n = 0.0
        outer_peak_compression_m = self.compression_m
        sliding = False
        performed_substeps = 0
        (
            combined_point_inverse_mass,
            reduced_mass,
            point_inverse_mass_audit,
        ) = self._point_inverse_mass_response(
            contact_point=contact_point,
            ball_arm=ball_arm,
            normal=normal,
        )

        for substep in range(substep_count):
            compression_rate = dot(local_relative_velocity, normal)
            if self.compression_m <= 0.0 and compression_rate <= 0.0:
                self.engaged = False
                self.awaiting_geometric_separation = True
                self.stale_overlap_m = None
                self.state = ContactState()
                if self.contact_ended_s is None:
                    self.contact_ended_s = (
                        time_s + active_delay_s + substep * substep_dt_s
                    )
                break

            old_compression_m = self.compression_m
            # Midpoint indentation avoids the artificial zero-force first
            # sample at exact time of impact while remaining causal: the seed
            # comes only from the measured incident relative velocity.
            evaluation_compression_m = max(
                0.0,
                old_compression_m + 0.5 * substep_dt_s * compression_rate,
            )
            old_stored_contact_energy_j = self._stored_contact_energy_j(
                old_compression_m, self.state
            )
            wrench = self.model.contact_wrench(
                compression_m=evaluation_compression_m,
                relative_velocity_other_minus_ball_world_m_s=local_relative_velocity,
                normal_other_to_ball_world=normal,
                contact_point_world_m=contact_point,
                ball_center_world_m=ball_center,
                other_center_world_m=foot_com,
                reduced_mass_kg=reduced_mass,
                dt_s=substep_dt_s,
                state=self.state,
            )
            force_ball = np.asarray(wrench.force_on_ball_world_n, dtype=np.float64)
            force_foot = np.asarray(wrench.force_on_other_world_n, dtype=np.float64)
            torque_ball = np.asarray(wrench.torque_on_ball_world_n_m, dtype=np.float64)
            torque_foot = np.asarray(wrench.torque_on_other_world_n_m, dtype=np.float64)
            ball_force_impulse += substep_dt_s * force_ball
            foot_force_impulse += substep_dt_s * force_foot
            ball_torque_impulse += substep_dt_s * torque_ball
            foot_torque_impulse += substep_dt_s * torque_foot
            performed_substeps += 1
            self.contact_substeps += 1
            outer_peak_force_n = max(outer_peak_force_n, wrench.normal.total_n)
            outer_peak_compression_m = max(
                outer_peak_compression_m, evaluation_compression_m
            )
            sliding = sliding or wrench.state.sliding

            # Equal-and-opposite impulse changes the full foot-minus-ball
            # contact-point velocity by -(W_foot + W_ball) J.  W_ball includes
            # the hollow sphere's rotational response and W_foot is the live
            # articulated operational response.  This is an internal
            # integration variable only; no rigid-body velocity is assigned.
            contact_relative_velocity = (
                np.asarray(local_relative_velocity, dtype=np.float64)
                - substep_dt_s * (combined_point_inverse_mass @ force_ball)
            )
            # The other loads on the two bodies, held from the previous outer
            # step (zero on the first step of a contact).
            next_relative_velocity = tuple(
                float(value)
                for value in (
                    contact_relative_velocity
                    + substep_dt_s * np.asarray(unmodelled_acceleration)
                )
            )
            next_compression_rate = dot(next_relative_velocity, normal)
            next_compression_m = old_compression_m + 0.5 * substep_dt_s * (
                compression_rate + next_compression_rate
            )
            contact_opens = (
                next_compression_m <= 0.0 and next_compression_rate < 0.0
            )
            if contact_opens:
                committed_compression_m = 0.0
                committed_state = ContactState()
            else:
                committed_compression_m = max(0.0, next_compression_m)
                committed_state = wrench.state

            # For constant W over this internal substep, midpoint force work
            # exactly equals the loss of the local relative-mode kinetic
            # energy due to the contact impulse alone.  The second residual
            # separates recoverable gas, shell, and tangential-spring energy
            # from constitutive/numerical loss; there the work is taken over
            # the full relative displacement, which also sets the indentation.
            old_relative_velocity_array = np.asarray(
                local_relative_velocity, dtype=np.float64
            )
            next_relative_velocity_array = np.asarray(
                next_relative_velocity, dtype=np.float64
            )
            midpoint_relative_velocity = 0.5 * (
                old_relative_velocity_array + next_relative_velocity_array
            )
            contact_work_extracted_j = (
                substep_dt_s
                * float(force_ball @ midpoint_relative_velocity)
            )
            impulse_work_j = substep_dt_s * float(
                force_ball
                @ (0.5 * (old_relative_velocity_array + contact_relative_velocity))
            )
            old_relative_kinetic_j = 0.5 * float(
                old_relative_velocity_array
                @ np.linalg.solve(
                    combined_point_inverse_mass,
                    old_relative_velocity_array,
                )
            )
            next_relative_kinetic_j = 0.5 * float(
                contact_relative_velocity
                @ np.linalg.solve(
                    combined_point_inverse_mass,
                    contact_relative_velocity,
                )
            )
            relative_kinetic_loss_j = (
                old_relative_kinetic_j - next_relative_kinetic_j
            )
            next_stored_contact_energy_j = self._stored_contact_energy_j(
                committed_compression_m, committed_state
            )
            stored_energy_change_j = (
                next_stored_contact_energy_j - old_stored_contact_energy_j
            )
            constitutive_dissipation_j = (
                contact_work_extracted_j - stored_energy_change_j
            )
            self.subcycle_contact_work_extracted_j += contact_work_extracted_j
            self.subcycle_relative_kinetic_loss_j += relative_kinetic_loss_j
            self.subcycle_constitutive_dissipation_j += (
                constitutive_dissipation_j
            )
            self.subcycle_numerical_energy_creation_j += max(
                0.0, -constitutive_dissipation_j
            )
            self.minimum_incremental_constitutive_dissipation_j = min(
                self.minimum_incremental_constitutive_dissipation_j,
                constitutive_dissipation_j,
            )
            self.maximum_subcycle_work_closure_error_j = max(
                self.maximum_subcycle_work_closure_error_j,
                abs(impulse_work_j - relative_kinetic_loss_j),
            )

            self.state = committed_state
            self.compression_m = committed_compression_m
            local_relative_velocity = next_relative_velocity
            if contact_opens:
                self.engaged = False
                self.awaiting_geometric_separation = True
                self.stale_overlap_m = None
                if self.contact_ended_s is None:
                    denominator = old_compression_m - next_compression_m
                    crossing_fraction = (
                        1.0 if denominator <= 0.0
                        else min(
                            1.0,
                            max(0.0, old_compression_m / denominator),
                        )
                    )
                    self.contact_ended_s = (
                        time_s
                        + active_delay_s
                        + (substep + crossing_fraction) * substep_dt_s
                    )
                break

        if self.engaged:
            self._predicted_relative_velocity = local_relative_velocity
            self._predicted_dt_s = dt_s

        impulse_magnitude = float(np.linalg.norm(ball_force_impulse))
        if impulse_magnitude > 0.0:
            # PhysX integrates these average wrenches for dt_s, recovering the
            # exact subsolver impulse without a post-impact state assignment.
            average_ball_force = ball_force_impulse / dt_s
            average_foot_force = foot_force_impulse / dt_s
            average_ball_torque = ball_torque_impulse / dt_s
            average_foot_torque = foot_torque_impulse / dt_s
            self.ball_view._rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=np.asarray([average_ball_force], dtype=np.float32),
                torques=np.asarray([average_ball_torque], dtype=np.float32),
                is_global=True,
            )
            self.foot_view._rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=np.asarray([average_foot_force], dtype=np.float32),
                torques=np.asarray([average_foot_torque], dtype=np.float32),
                is_global=True,
            )
            self.impulse_on_ball_n_s += ball_force_impulse
            self.impulse_on_foot_n_s += foot_force_impulse
            force_error = np.linalg.norm(average_ball_force + average_foot_force)
            moment_ball = average_ball_torque + np.cross(
                np.asarray(ball_center), average_ball_force
            )
            moment_foot = average_foot_torque + np.cross(
                np.asarray(foot_com), average_foot_force
            )
            moment_error = np.linalg.norm(moment_ball + moment_foot)
            self.maximum_force_closure_error_n = max(
                self.maximum_force_closure_error_n, float(force_error)
            )
            self.maximum_origin_moment_closure_error_n_m = max(
                self.maximum_origin_moment_closure_error_n_m, float(moment_error)
            )
            self.peak_force_n = max(self.peak_force_n, outer_peak_force_n)
            self.peak_compression_m = max(
                self.peak_compression_m, outer_peak_compression_m
            )
            self.contact_steps += 1
        else:
            average_ball_force = np.zeros(3, dtype=np.float64)
            average_foot_force = np.zeros(3, dtype=np.float64)
            average_ball_torque = np.zeros(3, dtype=np.float64)
            average_foot_torque = np.zeros(3, dtype=np.float64)

        return {
            "time_s": time_s,
            "geometry": asdict(geometry),
            "foot_center_of_mass_world_m": foot_com,
            "relative_velocity_foot_minus_ball_world_m_s": relative_velocity,
            "compression_m": outer_peak_compression_m,
            "normal_force_n": outer_peak_force_n,
            "force_on_ball_world_n": _vec(average_ball_force),
            "torque_on_ball_world_n_m": _vec(average_ball_torque),
            "force_on_foot_world_n": _vec(average_foot_force),
            "torque_on_foot_world_n_m": _vec(average_foot_torque),
            "sliding": sliding,
            "engaged": self.engaged,
            "contact_substeps": performed_substeps,
            "point_inverse_mass_response": point_inverse_mass_audit,
        }

    def report(self) -> dict[str, Any]:
        return {
            "patch": asdict(self.patch),
            "contact_geometry": self.contact_geometry_audit,
            "foot_center_of_mass_local_m": self.foot_center_of_mass_local_m,
            "contact_substep_hz": self.contact_substep_hz,
            "point_response_model": {
                "tensor_subsolver": True,
                "runtime_foot_provider": self.foot_inverse_mass_provider is not None,
                "fallback_effective_mass_prior_kg": (
                    self.patch.effective_mass_prior_kg
                ),
                "ball_inertia_model": "isotropic thin-shell 2/3 m r^2 prior",
                "operational_response_sources": sorted(
                    self.operational_response_sources
                ),
            },
            "operational_response_evaluations": (
                self.operational_response_evaluations
            ),
            "foot_directional_effective_mass_range_kg": (
                None
                if not self.foot_directional_effective_masses_kg
                else [
                    min(self.foot_directional_effective_masses_kg),
                    max(self.foot_directional_effective_masses_kg),
                ]
            ),
            "combined_directional_reduced_mass_range_kg": (
                None
                if not self.combined_directional_reduced_masses_kg
                else [
                    min(self.combined_directional_reduced_masses_kg),
                    max(self.combined_directional_reduced_masses_kg),
                ]
            ),
            "subcycle_energy_audit": {
                "initial_geometric_overlap_energy_j": (
                    self.contact_initial_stored_energy_j
                ),
                "contact_work_extracted_j": (
                    self.subcycle_contact_work_extracted_j
                ),
                "relative_mode_kinetic_loss_j": (
                    self.subcycle_relative_kinetic_loss_j
                ),
                "constitutive_dissipation_j": (
                    self.subcycle_constitutive_dissipation_j
                ),
                "summed_negative_increment_energy_creation_j": (
                    self.subcycle_numerical_energy_creation_j
                ),
                "negative_increment_fraction_of_total_dissipation": (
                    self.subcycle_numerical_energy_creation_j
                    / max(
                        self.subcycle_constitutive_dissipation_j,
                        1.0e-12,
                    )
                ),
                "residual_stored_contact_energy_j": (
                    self._stored_contact_energy_j(
                        self.compression_m, self.state
                    )
                ),
                "outer_boundary_stored_energy_j": self.boundary_stored_energy_j,
                "constitutive_accounting_residual_j": (
                    self.contact_initial_stored_energy_j
                    + self.subcycle_contact_work_extracted_j
                    + self.boundary_stored_energy_j
                    - self.subcycle_constitutive_dissipation_j
                    - self._stored_contact_energy_j(
                        self.compression_m, self.state
                    )
                ),
                "minimum_incremental_constitutive_dissipation_j": (
                    None
                    if not math.isfinite(
                        self.minimum_incremental_constitutive_dissipation_j
                    )
                    else self.minimum_incremental_constitutive_dissipation_j
                ),
                "maximum_work_vs_relative_kinetic_closure_error_j": (
                    self.maximum_subcycle_work_closure_error_j
                ),
                "scope": (
                    "local constant-W contact substeps; outer PhysX boundary "
                    "and actuator work are excluded"
                ),
            },
            "contact_started_s": self.contact_started_s,
            "contact_ended_s": self.contact_ended_s,
            "partitioned_release_hysteresis": {
                "enabled": True,
                "suppressed_stale_overlap_steps": self.release_hysteresis_steps,
                "maximum_suppressed_stale_overlap_m": (
                    self.maximum_release_hysteresis_overlap_m
                ),
                "reengagements_beyond_stale_overlap": (
                    self.reengagements_beyond_stale_overlap
                ),
                "release_condition": (
                    "after internal indentation reaches zero, the live overlap "
                    "on the next outer step is stale and ratchets down as the "
                    "bodies separate; any overlap beyond it re-engages the "
                    "contact law, measured from the stale remainder"
                ),
            },
            "unmodelled_load_correction": {
                "maximum_closing_speed_change_m_s": (
                    self.maximum_unmodelled_closing_speed_change_m_s
                ),
                "maximum_held_relative_acceleration_m_s2": (
                    self.maximum_unmodelled_relative_acceleration_m_s2
                ),
                "summed_absolute_compression_correction_m": (
                    self.unmodelled_compression_correction_m
                ),
                "rule": (
                    "measured minus predicted contact-point velocity at each "
                    "outer boundary: added to the internal indentation as "
                    "dt * dv / 2 and held as an acceleration over the next step"
                ),
            },
            "contact_steps": self.contact_steps,
            "contact_substeps": self.contact_substeps,
            "peak_force_n": self.peak_force_n,
            "peak_compression_m": self.peak_compression_m,
            "impulse_on_ball_n_s": self.impulse_on_ball_n_s.tolist(),
            "impulse_on_foot_n_s": self.impulse_on_foot_n_s.tolist(),
            "impulse_pair_closure_error_n_s": float(np.linalg.norm(
                self.impulse_on_ball_n_s + self.impulse_on_foot_n_s
            )),
            "maximum_force_closure_error_n": self.maximum_force_closure_error_n,
            "maximum_origin_moment_closure_error_n_m": (
                self.maximum_origin_moment_closure_error_n_m
            ),
        }
