"""Tension-only cable graph and two-way ball/net contact.

The net is not an animated surface.  Every rendered cord has a corresponding
tension-only edge; internal node masses move under gravity, cable tension,
damping, and ball contact.  Contact returns and applies equal/opposite forces.
The default material data are engineering-scale priors because public football
goal standards do not identify installed-net dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .football_math import Vec3, ZERO, add, cross, dot, norm, reject, scale, sub, unit
from .football_specs import BallSpec


@dataclass(frozen=True)
class CableNetPrior:
    evidence_status: str = "OPEN NET-SPECIMEN PRIOR / NOT DYNAMICALLY CALIBRATED"
    mesh_spacing_m: float = 0.22
    cord_diameter_m: float = 0.0045
    cord_linear_density_kg_m: float = 0.0060
    axial_stiffness_n: float = 2_500.0
    axial_damping_ratio: float = 0.08
    junction_transverse_damping_ratio: float = 0.28
    installed_pretension_n: float = 6.0
    contact_exponent: float = 1.50
    contact_stiffness_n_per_m_pow_n: float = 180_000.0
    contact_damping_ratio: float = 0.22
    ball_cord_dynamic_friction: float = 0.24

    def __post_init__(self) -> None:
        positive = (
            self.mesh_spacing_m,
            self.cord_diameter_m,
            self.cord_linear_density_kg_m,
            self.axial_stiffness_n,
            self.contact_stiffness_n_per_m_pow_n,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("net geometry/material scales must be positive")
        if self.contact_exponent <= 1.0 or self.installed_pretension_n < 0.0:
            raise ValueError("net contact exponent/pretension are invalid")
        if (
            self.axial_damping_ratio < 0.0
            or self.junction_transverse_damping_ratio < 0.0
            or self.contact_damping_ratio < 0.0
        ):
            raise ValueError("net damping must be non-negative")
        if self.ball_cord_dynamic_friction < 0.0:
            raise ValueError("ball-cord friction must be non-negative")


@dataclass
class CableNode:
    rest_position_m: Vec3
    position_m: Vec3
    velocity_m_s: Vec3
    mass_kg: float
    anchored: bool


@dataclass(frozen=True)
class CableEdge:
    node_a: int
    node_b: int
    rest_length_m: float


@dataclass(frozen=True)
class NetStepResult:
    force_on_ball_world_n: Vec3
    torque_on_ball_world_n_m: Vec3
    anchor_force_world_n: Vec3
    anchor_torque_world_n_m: Vec3
    contact_count: int
    maximum_contact_force_n: float
    elastic_energy_j: float
    kinetic_energy_j: float
    maximum_deflection_m: float


def _add_in_place(values: list[Vec3], index: int, increment: Vec3) -> None:
    values[index] = add(values[index], increment)


class CableNetPatch:
    """Rectangular, vertical, square-mesh goal-net patch."""

    def __init__(
        self,
        *,
        width_m: float,
        height_m: float,
        plane_x_m: float = 0.0,
        center_y_m: float = 0.0,
        bottom_z_m: float = 0.0,
        prior: CableNetPrior = CableNetPrior(),
        anchor_boundary: bool = True,
    ) -> None:
        if width_m <= 0.0 or height_m <= 0.0:
            raise ValueError("net patch dimensions must be positive")
        self.prior = prior
        self.width_m = width_m
        self.height_m = height_m
        self.plane_x_m = plane_x_m
        self.center_y_m = center_y_m
        self.bottom_z_m = bottom_z_m
        self.reference_normal_world: Vec3 = (1.0, 0.0, 0.0)
        self.surface_corners_world_m: tuple[Vec3, Vec3, Vec3, Vec3] = (
            (plane_x_m, center_y_m - 0.5 * width_m, bottom_z_m),
            (plane_x_m, center_y_m + 0.5 * width_m, bottom_z_m),
            (plane_x_m, center_y_m - 0.5 * width_m, bottom_z_m + height_m),
            (plane_x_m, center_y_m + 0.5 * width_m, bottom_z_m + height_m),
        )
        self.columns = max(2, int(round(width_m / prior.mesh_spacing_m)) + 1)
        self.rows = max(2, int(round(height_m / prior.mesh_spacing_m)) + 1)
        dy = width_m / (self.columns - 1)
        dz = height_m / (self.rows - 1)
        # Each interior knot owns four half-segments; boundary nodes own the
        # segments that actually meet them.  This preserves total cord mass.
        incident_length = [[0.0 for _ in range(self.columns)] for _ in range(self.rows)]
        for row in range(self.rows):
            for column in range(self.columns - 1):
                incident_length[row][column] += 0.5 * dy
                incident_length[row][column + 1] += 0.5 * dy
        for row in range(self.rows - 1):
            for column in range(self.columns):
                incident_length[row][column] += 0.5 * dz
                incident_length[row + 1][column] += 0.5 * dz
        self.nodes: list[CableNode] = []
        for row in range(self.rows):
            z = bottom_z_m + row * dz
            for column in range(self.columns):
                y = center_y_m - 0.5 * width_m + column * dy
                position = (plane_x_m, y, z)
                boundary = row in (0, self.rows - 1) or column in (0, self.columns - 1)
                mass = max(
                    1.0e-5,
                    prior.cord_linear_density_kg_m * incident_length[row][column],
                )
                self.nodes.append(CableNode(position, position, ZERO, mass, anchor_boundary and boundary))
        self.edges: list[CableEdge] = []
        for row in range(self.rows):
            for column in range(self.columns - 1):
                a = self.index(row, column)
                b = self.index(row, column + 1)
                self.edges.append(CableEdge(a, b, dy))
        for row in range(self.rows - 1):
            for column in range(self.columns):
                a = self.index(row, column)
                b = self.index(row + 1, column)
                self.edges.append(CableEdge(a, b, dz))

    @classmethod
    def from_corners(
        cls,
        *,
        bottom_left_world_m: Vec3,
        bottom_right_world_m: Vec3,
        top_left_world_m: Vec3,
        top_right_world_m: Vec3,
        prior: CableNetPrior = CableNetPrior(),
        anchor_boundary: bool = True,
    ) -> "CableNetPatch":
        """Build a bilinear cable patch in an arbitrary 3-D orientation.

        The four boundary curves remain straight for a planar quadrilateral,
        which covers the sloped back of the selected goal specimen.  A
        non-planar quadrilateral is also represented explicitly by bilinear
        rest positions; no render-only warp is introduced.
        """

        bottom_width = norm(sub(bottom_right_world_m, bottom_left_world_m))
        top_width = norm(sub(top_right_world_m, top_left_world_m))
        left_height = norm(sub(top_left_world_m, bottom_left_world_m))
        right_height = norm(sub(top_right_world_m, bottom_right_world_m))
        width = 0.5 * (bottom_width + top_width)
        height = 0.5 * (left_height + right_height)
        if width <= 0.0 or height <= 0.0:
            raise ValueError("net surface corners must span two non-zero dimensions")
        patch = cls(
            width_m=width,
            height_m=height,
            prior=prior,
            anchor_boundary=anchor_boundary,
        )
        corners = (
            bottom_left_world_m,
            bottom_right_world_m,
            top_left_world_m,
            top_right_world_m,
        )
        patch.surface_corners_world_m = corners
        tangent_u = sub(bottom_right_world_m, bottom_left_world_m)
        tangent_v = sub(top_left_world_m, bottom_left_world_m)
        patch.reference_normal_world = unit(cross(tangent_u, tangent_v), (1.0, 0.0, 0.0))
        patch.plane_x_m = sum(corner[0] for corner in corners) / 4.0
        patch.center_y_m = sum(corner[1] for corner in corners) / 4.0
        patch.bottom_z_m = 0.5 * (
            bottom_left_world_m[2] + bottom_right_world_m[2]
        )

        def bilinear(u: float, v: float) -> Vec3:
            bottom = add(
                scale(1.0 - u, bottom_left_world_m),
                scale(u, bottom_right_world_m),
            )
            top = add(
                scale(1.0 - u, top_left_world_m),
                scale(u, top_right_world_m),
            )
            return add(scale(1.0 - v, bottom), scale(v, top))

        for row in range(patch.rows):
            v = row / (patch.rows - 1)
            for column in range(patch.columns):
                u = column / (patch.columns - 1)
                node = patch.nodes[patch.index(row, column)]
                position = bilinear(u, v)
                node.rest_position_m = position
                node.position_m = position
                node.velocity_m_s = ZERO

        patch.edges = [
            CableEdge(edge.node_a, edge.node_b, norm(sub(
                patch.nodes[edge.node_b].rest_position_m,
                patch.nodes[edge.node_a].rest_position_m,
            )))
            for edge in patch.edges
        ]
        incident_length = [0.0 for _ in patch.nodes]
        for edge in patch.edges:
            incident_length[edge.node_a] += 0.5 * edge.rest_length_m
            incident_length[edge.node_b] += 0.5 * edge.rest_length_m
        for node, length in zip(patch.nodes, incident_length):
            node.mass_kg = max(1.0e-5, prior.cord_linear_density_kg_m * length)
        return patch

    def index(self, row: int, column: int) -> int:
        return row * self.columns + column

    @property
    def total_mass_kg(self) -> float:
        return sum(node.mass_kg for node in self.nodes)

    def reset(self) -> None:
        for node in self.nodes:
            node.position_m = node.rest_position_m
            node.velocity_m_s = ZERO

    def total_momentum_kg_m_s(self) -> Vec3:
        momentum = ZERO
        for node in self.nodes:
            momentum = add(momentum, scale(node.mass_kg, node.velocity_m_s))
        return momentum

    def _edge_forces(self, forces: list[Vec3]) -> tuple[float, float]:
        elastic_energy = 0.0
        maximum_tension = 0.0
        p = self.prior
        for edge in self.edges:
            a = self.nodes[edge.node_a]
            b = self.nodes[edge.node_b]
            delta = sub(b.position_m, a.position_m)
            length = norm(delta)
            if length <= 1.0e-12:
                continue
            direction = scale(1.0 / length, delta)
            extension = length - edge.rest_length_m
            relative_speed = dot(sub(b.velocity_m_s, a.velocity_m_s), direction)
            relative_velocity = sub(b.velocity_m_s, a.velocity_m_s)
            reduced_mass = a.mass_kg * b.mass_kg / (a.mass_kg + b.mass_kg)
            damping = 2.0 * p.axial_damping_ratio * math.sqrt(p.axial_stiffness_n * reduced_mass)
            tension = max(
                0.0,
                p.installed_pretension_n + p.axial_stiffness_n * extension + damping * relative_speed,
            )
            force = scale(tension, direction)
            _add_in_place(forces, edge.node_a, force)
            _add_in_place(forces, edge.node_b, scale(-1.0, force))
            # Braided cords and knots dissipate transverse relative motion even
            # before it has become pure axial strain.  The equal/opposite pair
            # preserves net momentum and is preferable to world-space node drag.
            transverse_damping = (
                2.0 * p.junction_transverse_damping_ratio
                * math.sqrt(p.axial_stiffness_n * reduced_mass)
            )
            transverse_force = scale(
                transverse_damping,
                reject(relative_velocity, direction),
            )
            _add_in_place(forces, edge.node_a, transverse_force)
            _add_in_place(forces, edge.node_b, scale(-1.0, transverse_force))
            positive_extension = max(0.0, extension)
            elastic_energy += (
                p.installed_pretension_n * positive_extension
                + 0.5 * p.axial_stiffness_n * positive_extension * positive_extension
            )
            maximum_tension = max(maximum_tension, tension)
        return elastic_energy, maximum_tension

    def step(
        self,
        *,
        ball_center_world_m: Vec3 | None,
        ball_velocity_world_m_s: Vec3 = ZERO,
        ball_angular_velocity_world_rad_s: Vec3 = ZERO,
        ball: BallSpec = BallSpec(),
        dt_s: float,
        gravity_world_m_s2: Vec3 = (0.0, 0.0, -9.80665),
    ) -> NetStepResult:
        if dt_s <= 0.0:
            raise ValueError("net timestep must be positive")
        forces = [scale(node.mass_kg, gravity_world_m_s2) for node in self.nodes]
        elastic_energy, _ = self._edge_forces(forces)
        force_on_ball = ZERO
        torque_on_ball = ZERO
        contact_count = 0
        maximum_contact_force = 0.0
        p = self.prior
        if ball_center_world_m is not None:
            for edge in self.edges:
                node_a = self.nodes[edge.node_a]
                node_b = self.nodes[edge.node_b]
                edge_delta = sub(node_b.position_m, node_a.position_m)
                length_squared = dot(edge_delta, edge_delta)
                if length_squared <= 1.0e-16:
                    continue
                fraction = max(
                    0.0,
                    min(1.0, dot(sub(ball_center_world_m, node_a.position_m), edge_delta) / length_squared),
                )
                point = add(node_a.position_m, scale(fraction, edge_delta))
                point_to_ball = sub(ball_center_world_m, point)
                distance = norm(point_to_ball)
                compression = ball.radius_m + 0.5 * p.cord_diameter_m - distance
                if compression <= 0.0:
                    continue
                normal = unit(point_to_ball, scale(-1.0, self.reference_normal_world))
                cord_velocity = add(
                    scale(1.0 - fraction, node_a.velocity_m_s),
                    scale(fraction, node_b.velocity_m_s),
                )
                ball_patch_velocity = add(
                    ball_velocity_world_m_s,
                    cross(ball_angular_velocity_world_rad_s, sub(point, ball_center_world_m)),
                )
                relative_cord_minus_ball = sub(cord_velocity, ball_patch_velocity)
                compression_rate = dot(relative_cord_minus_ball, normal)
                elastic = p.contact_stiffness_n_per_m_pow_n * compression**p.contact_exponent
                tangent_stiffness = (
                    p.contact_exponent * p.contact_stiffness_n_per_m_pow_n
                    * compression ** (p.contact_exponent - 1.0)
                )
                damping = 2.0 * p.contact_damping_ratio * math.sqrt(
                    max(1.0, tangent_stiffness) * ball.mass_kg
                ) * compression_rate
                normal_force_magnitude = max(0.0, elastic + damping)
                if normal_force_magnitude <= 0.0:
                    continue
                tangent_velocity = reject(relative_cord_minus_ball, normal)
                tangent_force = scale(
                    p.ball_cord_dynamic_friction * normal_force_magnitude,
                    unit(tangent_velocity),
                )
                contact_force = add(scale(normal_force_magnitude, normal), tangent_force)
                force_on_ball = add(force_on_ball, contact_force)
                torque_on_ball = add(
                    torque_on_ball,
                    cross(sub(point, ball_center_world_m), contact_force),
                )
                reaction = scale(-1.0, contact_force)
                _add_in_place(forces, edge.node_a, scale(1.0 - fraction, reaction))
                _add_in_place(forces, edge.node_b, scale(fraction, reaction))
                contact_count += 1
                maximum_contact_force = max(maximum_contact_force, norm(contact_force))

        anchor_force = ZERO
        anchor_torque = ZERO
        kinetic_energy = 0.0
        maximum_deflection = 0.0
        for index, node in enumerate(self.nodes):
            if node.anchored:
                reaction = scale(-1.0, forces[index])
                anchor_force = add(anchor_force, reaction)
                anchor_torque = add(anchor_torque, cross(node.position_m, reaction))
                node.position_m = node.rest_position_m
                node.velocity_m_s = ZERO
                continue
            acceleration = scale(1.0 / node.mass_kg, forces[index])
            node.velocity_m_s = add(node.velocity_m_s, scale(dt_s, acceleration))
            node.position_m = add(node.position_m, scale(dt_s, node.velocity_m_s))
            kinetic_energy += 0.5 * node.mass_kg * dot(node.velocity_m_s, node.velocity_m_s)
            maximum_deflection = max(
                maximum_deflection,
                abs(dot(
                    sub(node.position_m, node.rest_position_m),
                    self.reference_normal_world,
                )),
            )
        return NetStepResult(
            force_on_ball,
            torque_on_ball,
            anchor_force,
            anchor_torque,
            contact_count,
            maximum_contact_force,
            elastic_energy,
            kinetic_energy,
            maximum_deflection,
        )
