"""Actual through-bored rotary stopcock, not a sliding seal proxy.

The analytic solid is a Y-axis cylinder minus a Z-axis 2 mm bore. Meshing
error is bounded by the declared sample spacing; an analytic signed-distance
query is also supplied for geometry inspection and hydraulic qualification.
"""

import numpy as np
from .geometry import MeshData
from .runtime import require_newton


def capped_cylinder_sdf(points, radius, half_length, axis):
    p = np.asarray(points)
    radial = np.linalg.norm(np.delete(p, axis, axis=-1), axis=-1)-radius
    axial = np.abs(p[..., axis])-half_length
    return np.minimum(np.maximum(radial, axial), 0.) + np.hypot(np.maximum(radial, 0.), np.maximum(axial, 0.))


def plug_sdf(points, radius=.007, half_length=.014, bore=.001):
    outer = capped_cylinder_sdf(points, radius, half_length, 1)
    hole = np.linalg.norm(np.asarray(points)[..., :2], axis=-1)-bore
    return np.maximum(outer, -hole)


def stopcock_plug(radius=.007, half_length=.014, bore=.001, spacing=.000125):
    if not 0 < bore < radius < half_length or spacing <= 0:
        raise ValueError("Invalid bored plug dimensions")
    _, wp = require_newton()
    lower = np.array([-radius, -half_length, -radius])-2*spacing
    counts = np.ceil(-2*lower/spacing).astype(int)+1
    upper = lower+(counts-1)*spacing
    if np.prod(counts) > 5_000_000:
        raise ValueError("Declared plug mesh resolution exceeds 5M sample budget")
    axes = [lower[a]+np.arange(counts[a])*spacing for a in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    field = wp.array(plug_sdf(points, radius, half_length, bore).astype(np.float32), device="cpu")
    mc = wp.MarchingCubes(*map(int, counts), domain_bounds_lower_corner=lower.tolist(),
                          domain_bounds_upper_corner=upper.tolist(), device="cpu")
    mc.surface(field, 0.)
    return MeshData(mc.verts.numpy(), mc.indices.numpy().reshape(-1, 3)).outward()


def socket_sdf(points, inner=.007, outer=.010, half_length=.010, bore=.001):
    p = np.asarray(points)
    sleeve = capped_cylinder_sdf(p, outer, half_length, 1)
    cavity = np.sqrt(p[..., 0]**2+p[..., 2]**2)-inner
    ports = np.sqrt(p[..., 0]**2+p[..., 1]**2)-bore
    return np.maximum.reduce([sleeve, -cavity, -ports])


def stopcock_socket(inner=.007, outer=.010, half_length=.010, bore=.001, spacing=.000125):
    _, wp = require_newton()
    lower = np.array([-outer, -half_length, -outer])-2*spacing
    counts = np.ceil(-2*lower/spacing).astype(int)+1
    upper = lower+(counts-1)*spacing
    points = np.stack(np.meshgrid(*[lower[a]+np.arange(counts[a])*spacing for a in range(3)], indexing="ij"), axis=-1)
    field = wp.array(socket_sdf(points, inner, outer, half_length, bore).astype(np.float32), device="cpu")
    mc = wp.MarchingCubes(*map(int, counts), domain_bounds_lower_corner=lower.tolist(),
                          domain_bounds_upper_corner=upper.tolist(), device="cpu")
    mc.surface(field, 0.)
    return MeshData(mc.verts.numpy(), mc.indices.numpy().reshape(-1, 3)).outward()


def socket_cells(inner=.007, outer=.010, half_length=.010, bore=.001):
    from scipy.spatial import ConvexHull

    ys = np.r_[-half_length, bore*np.sin(np.linspace(-np.pi/2, np.pi/2, 9)), half_length]
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for sign in (-1, 1):
            for u0, u1 in zip(np.linspace(-1., 1., 17)[:-1], np.linspace(-1., 1., 17)[1:]):
                points = []
                for y in (y0, y1):
                    cut = np.sqrt(max(0., bore*bore-y*y))
                    for r in (inner, outer):
                        angle = np.arccos(cut/r)
                        points.extend([(sign*r*np.cos(u*angle), y, r*np.sin(u*angle)) for u in (u0, u1)])
                points = np.array(points)
                hull = ConvexHull(points)
                faces = hull.simplices.copy()
                for index, (f, normal) in enumerate(zip(faces, hull.equations[:, :3])):
                    a, b, c = points[f]
                    if np.dot(np.cross(b-a, c-a), normal) < 0:
                        faces[index] = f[::-1]
                yield MeshData(points, faces)


def plug_cells(radius=.007, half_length=.014, bore=.001, bore_segments=32):
    """Convex cells retaining the through bore; no full-cylinder contact proxy.

    Bore chord error <= bore*(1-cos(pi/(2*bore_segments))) (1.21 micrometres
    at default dimensions). Outer arcs use 64 samples per full circle.
    """
    from scipy.spatial import ConvexHull

    def cell(points):
        hull = ConvexHull(points)
        faces = hull.simplices.copy()
        for i, (f, n) in enumerate(zip(faces, hull.equations[:, :3])):
            a, b, c = points[f]
            if np.dot(np.cross(b-a, c-a), n) < 0:
                faces[i] = f[::-1]
        return MeshData(points, faces)

    for lo, hi in ((-half_length, -bore), (bore, half_length)):
        angles = np.linspace(0., 2*np.pi, 64, endpoint=False)
        yield cell(np.array([(radius*np.cos(t), y, radius*np.sin(t)) for y in (lo, hi) for t in angles]))
    ys = bore*np.sin(np.linspace(-np.pi/2, np.pi/2, bore_segments+1))
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for sign in (-1, 1):
            points = []
            for y in (y0, y1):
                cut = np.sqrt(max(0., bore*bore-y*y))
                angle = np.arccos(cut/radius)
                points.extend([(sign*radius*np.cos(t), y, radius*np.sin(t))
                               for t in np.linspace(-angle, angle, 33)])
            yield cell(np.array(points))
