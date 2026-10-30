"""One canonical classic 32-panel topology for rendering and aerodynamics.

The familiar football is a truncated icosahedron: twelve pentagons, twenty
hexagons, ninety edges.  This module constructs that topology analytically.
The renderer and the aerodynamic orientation model consume the same vertices,
so the visible seam orientation cannot drift away from the force model.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable

from .football_math import Vec3, add, cross, dot, norm, reject, scale, unit


@dataclass(frozen=True)
class Panel:
    kind: str
    vertices: tuple[Vec3, ...]

    @property
    def center(self) -> Vec3:
        return unit(tuple(sum(v[i] for v in self.vertices) for i in range(3)))  # type: ignore[arg-type]


@dataclass(frozen=True)
class SeamExposure:
    projected_density: float
    asymmetry_strength: float
    bias_body: Vec3


def _icosahedron_vertices() -> tuple[Vec3, ...]:
    phi = 0.5 * (1.0 + math.sqrt(5.0))
    raw: list[Vec3] = []
    for a in (-1.0, 1.0):
        for b in (-phi, phi):
            raw.append((0.0, a, b))
            raw.append((a, b, 0.0))
            raw.append((b, 0.0, a))
    # The loop creates each of the twelve standard vertices exactly once.
    return tuple(unit(vertex) for vertex in raw)


def _edges(vertices: tuple[Vec3, ...]) -> tuple[tuple[int, int], ...]:
    distances = [
        norm((vertices[i][0] - vertices[j][0], vertices[i][1] - vertices[j][1], vertices[i][2] - vertices[j][2]))
        for i, j in itertools.combinations(range(len(vertices)), 2)
    ]
    edge_length = min(value for value in distances if value > 1.0e-12)
    tolerance = 1.0e-8
    return tuple(
        (i, j)
        for i, j in itertools.combinations(range(len(vertices)), 2)
        if abs(norm((
            vertices[i][0] - vertices[j][0],
            vertices[i][1] - vertices[j][1],
            vertices[i][2] - vertices[j][2],
        )) - edge_length) < tolerance
    )


def _triangles(edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, int], ...]:
    edge_set = {tuple(sorted(edge)) for edge in edges}
    return tuple(
        (i, j, k)
        for i, j, k in itertools.combinations(range(12), 3)
        if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set
    )


def _truncated_point(vertices: tuple[Vec3, ...], source: int, neighbor: int) -> Vec3:
    # One third of the way along an icosahedron edge, projected to the ball.
    return unit(add(scale(2.0 / 3.0, vertices[source]), scale(1.0 / 3.0, vertices[neighbor])))


def _ordered_around_axis(points: Iterable[Vec3], axis: Vec3) -> tuple[Vec3, ...]:
    normal = unit(axis)
    reference = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (0.0, 1.0, 0.0)
    tangent_x = unit(cross(reference, normal))
    tangent_y = unit(cross(normal, tangent_x))
    ordered = sorted(points, key=lambda p: math.atan2(dot(p, tangent_y), dot(p, tangent_x)))
    return _orient_outward(tuple(ordered))


def _orient_outward(points: tuple[Vec3, ...]) -> tuple[Vec3, ...]:
    if len(points) < 3:
        return points
    normal = (0.0, 0.0, 0.0)
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        normal = add(normal, cross(current, following))
    center = unit(tuple(sum(v[i] for v in points) for i in range(3)))  # type: ignore[arg-type]
    return tuple(reversed(points)) if dot(normal, center) < 0.0 else points


def build_panels() -> tuple[Panel, ...]:
    vertices = _icosahedron_vertices()
    edges = _edges(vertices)
    triangles = _triangles(edges)
    neighbors: dict[int, list[int]] = {index: [] for index in range(len(vertices))}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)

    pentagons = [
        Panel(
            "pentagon",
            _ordered_around_axis(
                (_truncated_point(vertices, source, neighbor) for neighbor in neighbors[source]),
                vertices[source],
            ),
        )
        for source in range(len(vertices))
    ]

    hexagons: list[Panel] = []
    for i, j, k in triangles:
        triangle = (i, j, k)
        # Make the original triangle outward-facing before walking its edges.
        a, b, c = (vertices[index] for index in triangle)
        if dot(cross((b[0] - a[0], b[1] - a[1], b[2] - a[2]),
                     (c[0] - a[0], c[1] - a[1], c[2] - a[2])), add(add(a, b), c)) < 0.0:
            triangle = (i, k, j)
        i2, j2, k2 = triangle
        points = (
            _truncated_point(vertices, i2, j2),
            _truncated_point(vertices, j2, i2),
            _truncated_point(vertices, j2, k2),
            _truncated_point(vertices, k2, j2),
            _truncated_point(vertices, k2, i2),
            _truncated_point(vertices, i2, k2),
        )
        hexagons.append(Panel("hexagon", _orient_outward(points)))

    panels = tuple(pentagons + hexagons)
    if len(pentagons) != 12 or len(hexagons) != 20:
        raise AssertionError("invalid truncated-icosahedron face count")
    return panels


PANELS: tuple[Panel, ...] = build_panels()


def unique_seams(panels: tuple[Panel, ...] = PANELS, digits: int = 10) -> tuple[tuple[Vec3, Vec3], ...]:
    """Return the ninety undirected polygon edges without float duplicates."""

    seams: dict[tuple[tuple[float, ...], tuple[float, ...]], tuple[Vec3, Vec3]] = {}
    for panel in panels:
        for index, start in enumerate(panel.vertices):
            end = panel.vertices[(index + 1) % len(panel.vertices)]
            a = tuple(round(value, digits) for value in start)
            b = tuple(round(value, digits) for value in end)
            key = (a, b) if a < b else (b, a)
            seams.setdefault(key, (start, end))
    result = tuple(seams.values())
    if len(result) != 90:
        raise AssertionError(f"expected 90 seams, constructed {len(result)}")
    return result


SEAMS: tuple[tuple[Vec3, Vec3], ...] = unique_seams()


def seam_exposure(flow_direction_body: Vec3) -> SeamExposure:
    """Compute a smooth, deterministic seam-orientation signature.

    ``flow_direction_body`` points along ball velocity relative to the air.
    Only the upstream hemisphere contributes.  This is not a CFD solution: it
    is an orientation state fed into the explicitly-prior coefficient/wake
    model.  Crucially, it is continuous, geometry-derived, and reproducible.
    """

    flow = unit(flow_direction_body, (1.0, 0.0, 0.0))
    density_sum = 0.0
    weight_sum = 0.0
    bias = (0.0, 0.0, 0.0)
    for start, end in SEAMS:
        midpoint = unit(add(start, end))
        facing = max(0.0, dot(midpoint, flow))
        if facing <= 0.0:
            continue
        tangent = unit((end[0] - start[0], end[1] - start[1], end[2] - start[2]))
        crossflow = norm(reject(tangent, flow))
        weight = facing * facing
        density_sum += weight * crossflow
        weight_sum += weight
        # A cubic facing weight emphasizes asymmetric seams near stagnation;
        # projection ensures no artificial thrust component is introduced.
        bias = add(bias, scale(facing**3 * (0.35 + 0.65 * crossflow), reject(midpoint, flow)))

    projected_density = 0.0 if weight_sum == 0.0 else density_sum / weight_sum
    raw_strength = norm(bias) / max(weight_sum, 1.0e-12)
    bias_direction = unit(bias)
    return SeamExposure(
        projected_density=projected_density,
        asymmetry_strength=min(1.0, 6.0 * raw_strength),
        bias_body=bias_direction,
    )
