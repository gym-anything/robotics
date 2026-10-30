"""NumPy execution backend for the reference :mod:`football.net_model` graph.

The scalar ``CableNetPatch`` remains the readable reference oracle.  This
backend evaluates the same edge, contact, anchor, and semi-implicit integration
equations in bulk so a regulation-width graph can run at 24 kHz.  It exposes
the same runtime attributes needed by ``NetBallCoupler`` and synchronises its
state back to the source nodes once per enclosing PhysX step for rendering.
"""

from __future__ import annotations

import numpy as np

from .football_specs import BallSpec
from .net_model import CableNetPatch, NetStepResult


def _tuple(values) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)


class VectorizedCableNetPatch:
    def __init__(self, source: CableNetPatch) -> None:
        self.source = source
        self.prior = source.prior
        self.width_m = source.width_m
        self.height_m = source.height_m
        self.plane_x_m = source.plane_x_m
        self.center_y_m = source.center_y_m
        self.bottom_z_m = source.bottom_z_m
        self.columns = source.columns
        self.rows = source.rows
        self.reference_normal_world = source.reference_normal_world
        self.surface_corners_world_m = source.surface_corners_world_m
        self.nodes = source.nodes
        self.edges = source.edges
        self.rest = np.asarray(
            [node.rest_position_m for node in source.nodes], dtype=np.float64
        )
        self.position = np.asarray(
            [node.position_m for node in source.nodes], dtype=np.float64
        )
        self.velocity = np.asarray(
            [node.velocity_m_s for node in source.nodes], dtype=np.float64
        )
        self.mass = np.asarray([node.mass_kg for node in source.nodes], dtype=np.float64)
        self.anchored = np.asarray(
            [node.anchored for node in source.nodes], dtype=np.bool_
        )
        self.dynamic = ~self.anchored
        self.edge_a = np.asarray([edge.node_a for edge in source.edges], dtype=np.int64)
        self.edge_b = np.asarray([edge.node_b for edge in source.edges], dtype=np.int64)
        self.rest_length = np.asarray(
            [edge.rest_length_m for edge in source.edges], dtype=np.float64
        )
        reduced_mass = (
            self.mass[self.edge_a] * self.mass[self.edge_b]
            / (self.mass[self.edge_a] + self.mass[self.edge_b])
        )
        self.axial_damping = (
            2.0
            * self.prior.axial_damping_ratio
            * np.sqrt(self.prior.axial_stiffness_n * reduced_mass)
        )
        self.transverse_damping = (
            2.0
            * self.prior.junction_transverse_damping_ratio
            * np.sqrt(self.prior.axial_stiffness_n * reduced_mass)
        )

    @property
    def total_mass_kg(self) -> float:
        return float(self.mass.sum())

    def reset(self) -> None:
        self.position[:] = self.rest
        self.velocity.fill(0.0)
        self.sync_nodes()

    def sync_nodes(self) -> None:
        for index, node in enumerate(self.nodes):
            node.position_m = _tuple(self.position[index])
            node.velocity_m_s = _tuple(self.velocity[index])

    def total_momentum_kg_m_s(self) -> tuple[float, float, float]:
        return _tuple((self.mass[:, None] * self.velocity).sum(axis=0))

    def step(
        self,
        *,
        ball_center_world_m=None,
        ball_velocity_world_m_s=(0.0, 0.0, 0.0),
        ball_angular_velocity_world_rad_s=(0.0, 0.0, 0.0),
        ball: BallSpec = BallSpec(),
        dt_s: float,
        gravity_world_m_s2=(0.0, 0.0, -9.80665),
    ) -> NetStepResult:
        if dt_s <= 0.0:
            raise ValueError("net timestep must be positive")
        p = self.prior
        forces = self.mass[:, None] * np.asarray(
            gravity_world_m_s2, dtype=np.float64
        )
        edge_delta = self.position[self.edge_b] - self.position[self.edge_a]
        edge_length = np.linalg.norm(edge_delta, axis=1)
        valid_edge = edge_length > 1.0e-12
        direction = np.zeros_like(edge_delta)
        direction[valid_edge] = (
            edge_delta[valid_edge] / edge_length[valid_edge, None]
        )
        extension = edge_length - self.rest_length
        relative_velocity = (
            self.velocity[self.edge_b] - self.velocity[self.edge_a]
        )
        relative_speed = np.einsum("ij,ij->i", relative_velocity, direction)
        tension = np.maximum(
            0.0,
            p.installed_pretension_n
            + p.axial_stiffness_n * extension
            + self.axial_damping * relative_speed,
        )
        tension[~valid_edge] = 0.0
        edge_force = tension[:, None] * direction
        np.add.at(forces, self.edge_a, edge_force)
        np.add.at(forces, self.edge_b, -edge_force)
        transverse_velocity = (
            relative_velocity - relative_speed[:, None] * direction
        )
        transverse_force = self.transverse_damping[:, None] * transverse_velocity
        transverse_force[~valid_edge] = 0.0
        np.add.at(forces, self.edge_a, transverse_force)
        np.add.at(forces, self.edge_b, -transverse_force)
        positive_extension = np.maximum(0.0, extension)
        elastic_energy = float(np.sum(
            p.installed_pretension_n * positive_extension
            + 0.5 * p.axial_stiffness_n * positive_extension**2
        ))

        force_on_ball = np.zeros(3, dtype=np.float64)
        torque_on_ball = np.zeros(3, dtype=np.float64)
        contact_count = 0
        maximum_contact_force = 0.0
        if ball_center_world_m is not None:
            center = np.asarray(ball_center_world_m, dtype=np.float64)
            ball_velocity = np.asarray(ball_velocity_world_m_s, dtype=np.float64)
            ball_omega = np.asarray(
                ball_angular_velocity_world_rad_s, dtype=np.float64
            )
            length_squared = np.einsum("ij,ij->i", edge_delta, edge_delta)
            fraction = np.zeros(len(self.edges), dtype=np.float64)
            contact_valid_edge = length_squared > 1.0e-16
            center_from_a = center - self.position[self.edge_a]
            fraction[contact_valid_edge] = np.clip(
                np.einsum(
                    "ij,ij->i",
                    center_from_a[contact_valid_edge],
                    edge_delta[contact_valid_edge],
                )
                / length_squared[contact_valid_edge],
                0.0,
                1.0,
            )
            point = self.position[self.edge_a] + fraction[:, None] * edge_delta
            point_to_ball = center - point
            distance = np.linalg.norm(point_to_ball, axis=1)
            compression = (
                ball.radius_m + 0.5 * p.cord_diameter_m - distance
            )
            candidate = contact_valid_edge & (compression > 0.0)
            if np.any(candidate):
                indices = np.flatnonzero(candidate)
                normal = np.empty((len(indices), 3), dtype=np.float64)
                candidate_distance = distance[indices]
                nonzero = candidate_distance > 1.0e-12
                normal[nonzero] = (
                    point_to_ball[indices[nonzero]]
                    / candidate_distance[nonzero, None]
                )
                normal[~nonzero] = -np.asarray(
                    self.reference_normal_world, dtype=np.float64
                )
                f = fraction[indices]
                cord_velocity = (
                    (1.0 - f)[:, None] * self.velocity[self.edge_a[indices]]
                    + f[:, None] * self.velocity[self.edge_b[indices]]
                )
                arm = point[indices] - center
                ball_patch_velocity = ball_velocity + np.cross(ball_omega, arm)
                relative_cord_minus_ball = cord_velocity - ball_patch_velocity
                compression_rate = np.einsum(
                    "ij,ij->i", relative_cord_minus_ball, normal
                )
                delta = compression[indices]
                elastic = p.contact_stiffness_n_per_m_pow_n * delta**p.contact_exponent
                tangent_stiffness = (
                    p.contact_exponent
                    * p.contact_stiffness_n_per_m_pow_n
                    * delta ** (p.contact_exponent - 1.0)
                )
                damping = (
                    2.0
                    * p.contact_damping_ratio
                    * np.sqrt(np.maximum(1.0, tangent_stiffness) * ball.mass_kg)
                    * compression_rate
                )
                normal_magnitude = np.maximum(0.0, elastic + damping)
                active = normal_magnitude > 0.0
                if np.any(active):
                    indices = indices[active]
                    normal = normal[active]
                    relative_cord_minus_ball = relative_cord_minus_ball[active]
                    normal_magnitude = normal_magnitude[active]
                    f = fraction[indices]
                    normal_component = np.einsum(
                        "ij,ij->i", relative_cord_minus_ball, normal
                    )[:, None] * normal
                    tangent_velocity = relative_cord_minus_ball - normal_component
                    tangent_speed = np.linalg.norm(tangent_velocity, axis=1)
                    tangent_direction = np.zeros_like(tangent_velocity)
                    moving = tangent_speed > 1.0e-12
                    tangent_direction[moving] = (
                        tangent_velocity[moving] / tangent_speed[moving, None]
                    )
                    contact_force = (
                        normal_magnitude[:, None] * normal
                        + p.ball_cord_dynamic_friction
                        * normal_magnitude[:, None]
                        * tangent_direction
                    )
                    force_on_ball = contact_force.sum(axis=0)
                    torque_on_ball = np.cross(
                        point[indices] - center, contact_force
                    ).sum(axis=0)
                    reaction = -contact_force
                    np.add.at(
                        forces,
                        self.edge_a[indices],
                        (1.0 - f)[:, None] * reaction,
                    )
                    np.add.at(
                        forces,
                        self.edge_b[indices],
                        f[:, None] * reaction,
                    )
                    contact_count = int(len(indices))
                    maximum_contact_force = float(
                        np.linalg.norm(contact_force, axis=1).max()
                    )

        anchor_reaction = -forces[self.anchored]
        anchor_positions = self.position[self.anchored]
        anchor_force = anchor_reaction.sum(axis=0)
        anchor_torque = np.cross(anchor_positions, anchor_reaction).sum(axis=0)
        acceleration = forces[self.dynamic] / self.mass[self.dynamic, None]
        self.velocity[self.dynamic] += dt_s * acceleration
        self.position[self.dynamic] += dt_s * self.velocity[self.dynamic]
        self.position[self.anchored] = self.rest[self.anchored]
        self.velocity[self.anchored] = 0.0
        kinetic_energy = float(np.sum(
            0.5
            * self.mass[self.dynamic]
            * np.einsum(
                "ij,ij->i", self.velocity[self.dynamic], self.velocity[self.dynamic]
            )
        ))
        normal = np.asarray(self.reference_normal_world, dtype=np.float64)
        deflection = np.abs((self.position[self.dynamic] - self.rest[self.dynamic]) @ normal)
        maximum_deflection = float(deflection.max()) if deflection.size else 0.0
        return NetStepResult(
            _tuple(force_on_ball),
            _tuple(torque_on_ball),
            _tuple(anchor_force),
            _tuple(anchor_torque),
            contact_count,
            maximum_contact_force,
            elastic_energy,
            kinetic_energy,
            maximum_deflection,
        )
