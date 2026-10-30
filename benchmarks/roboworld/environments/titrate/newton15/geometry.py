"""Single dimensional mesh source for collision, optics and API geometry.

Coordinates and radii are metres. Hollow objects retain their real cavities;
the optional convex wall cells are only the rigid-contact decomposition.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MeshData:
    vertices: np.ndarray
    triangles: np.ndarray
    meridian: np.ndarray | None = None
    angular_segments: int = 0

    @property
    def signed_volume_m3(self):
        t = self.vertices[self.triangles].astype(float)
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()/6)

    def outward(self):
        if self.signed_volume_m3 < 0:
            self.triangles = self.triangles[:, ::-1].copy()
        return self


def revolve(profile, segments=96, max_meridian_edge=.005):
    """Closed meridian polygon -> closed manifold solid of revolution."""
    profile = np.asarray(profile, dtype=float)
    if profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 3:
        raise ValueError("A closed r,z meridian polygon is required")
    if not np.isfinite(profile).all() or np.any(profile[:, 0] < 0) or segments < 12:
        raise ValueError("Invalid profile or angular resolution")
    original_profile = profile.copy()
    if max_meridian_edge is not None:
        if not np.isfinite(max_meridian_edge) or max_meridian_edge <= 0:
            raise ValueError("Positive meridian edge bound required")
        # Subdivide straight meridian edges without changing their surface.
        # A 655 mm by 0.34 mm bore triangle loses closest-point accuracy in
        # float32. Adding coplanar rings keeps the same cavity, wall and volume
        # while conditioning the collision query's barycentric calculation.
        rings = []
        for p, q in zip(profile, np.roll(profile, -1, axis=0)):
            n = max(1, int(np.ceil(np.linalg.norm(q-p)/max_meridian_edge)))
            rings.extend(p+(q-p)*j/n for j in range(n))
        profile = np.asarray(rings)
    theta = np.arange(segments)*2*np.pi/segments
    vertices = np.array([(r*np.cos(t), r*np.sin(t), z) for r, z in profile for t in theta])
    faces = []
    for ring in range(len(profile)):
        nxt = (ring+1) % len(profile)
        for k in range(segments):
            a, b = ring*segments+k, ring*segments+(k+1) % segments
            c, d = nxt*segments+(k+1) % segments, nxt*segments+k
            for triangle in ((a, b, c), (a, c, d)):
                t = vertices[list(triangle)]
                if np.linalg.norm(np.cross(t[1]-t[0], t[2]-t[0])) > 1e-15:
                    faces.append(triangle)
    # Weld coincident axis vertices so the API mesh is actually manifold.
    vertices, inverse = np.unique(np.round(vertices, 12), axis=0, return_inverse=True)
    return MeshData(vertices, inverse[np.asarray(faces, dtype=int)], original_profile, segments).outward()


def record_meridian(builder, shape, mesh):
    """Retain construction data for exact, efficient interior queries."""
    if mesh.meridian is not None:
        if not hasattr(builder, "titration_meridians"):
            builder.titration_meridians = {}
        builder.titration_meridians[int(shape)] = (mesh.meridian.copy(), mesh.angular_segments)


def tube(inner, outer, z0, z1, segments=96):
    if not 0 < inner < outer or z1 <= z0:
        raise ValueError("Tube needs an open bore and positive wall thickness/length")
    return revolve([(inner, z0), (outer, z0), (outer, z1), (inner, z1)], segments)


def convex_wall_cells(levels, segments=32):
    """Closed convex cells for piecewise-linear axisymmetric hollow walls.

    levels are (z, inside radius, outside radius). For each band a 96-sided
    optical/fluid mesh can coexist with these separately flagged rigid hulls.
    """
    from scipy.spatial import ConvexHull

    for bottom, top in zip(levels[:-1], levels[1:]):
        if bottom[1] == 0. and top[1] == 0.:
            # A solid floor band is one convex frustum. Splitting it into
            # needle-like radial wedges adds internal collision seams even
            # though their UNION is exactly this same convex solid.
            angles = np.arange(segments)*2*np.pi/segments
            points = np.array([(ro*np.cos(t), ro*np.sin(t), z)
                               for z, _, ro in (bottom, top) for t in angles])
            hull = ConvexHull(points)
            faces = hull.simplices.copy()
            for index, (tri, normal) in enumerate(zip(faces, hull.equations[:, :3])):
                a, b, c = points[tri]
                if np.dot(np.cross(b-a, c-a), normal) < 0:
                    faces[index] = tri[::-1]
            yield MeshData(points, faces)
            continue
        for k in range(segments):
            t0, t1 = np.array([k, k+1])*2*np.pi/segments
            points = np.unique(np.array([(r*np.cos(t), r*np.sin(t), z)
                                        for z, ri, ro in (bottom, top)
                                        for r in (ri, ro) for t in (t0, t1)]), axis=0)
            hull = ConvexHull(points)
            faces = hull.simplices.copy()
            for index, (tri, normal) in enumerate(zip(faces, hull.equations[:, :3])):
                a, b, c = points[tri]
                if np.dot(np.cross(b-a, c-a), normal) < 0:
                    faces[index] = tri[::-1]
            yield MeshData(points, faces)


# Corning 4980-125: published 67 mm maximum OD and 114 mm height.
# Wall/shoulder/rim details are explicit engineering reconstructions, not CAD
# supplied by Corning. A 2 mm bottom and 1.5 mm side wall remain in collision.
FLASK_PROFILE = [
    (0., 0.), (.025, 0.), (.031, .002), (.0335, .006), (.0335, .015),
    (.029, .040), (.022, .067), (.015, .085), (.014, .091), (.014, .110),
    (.015, .112), (.015, .114), (.0125, .114), (.0125, .091),
    (.0135, .086), (.0205, .067), (.0275, .040), (.032, .015),
    (.032, .007), (.029, .003), (.025, .002), (0., .002)]

FLASK_WALL_LEVELS = [
    (.003, .029, .032), (.007, .032, .0335), (.015, .032, .0335),
    (.040, .0275, .029), (.067, .0205, .022), (.086, .0135, .015),
    (.091, .0125, .014), (.110, .0125, .014), (.114, .0125, .015)]


def flask():
    return revolve(FLASK_PROFILE)


def profile_wall_cells(profile, segments=32):
    """Decompose exactly the same meridian polygon into convex radial cells.

    Horizontal floors and rolled rims keep their actual heights. No filled
    convex hull is substituted for a vessel, and no independent floor proxy.
    """
    poly = np.asarray(profile, dtype=float)
    heights = np.unique(poly[:, 1])
    for z0, z1 in zip(heights[:-1], heights[1:]):
        middle = (z0+z1)/2
        crossings = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            if min(a[1], b[1]) < middle < max(a[1], b[1]):
                radii = [a[0]+(z-a[1])*(b[0]-a[0])/(b[1]-a[1]) for z in (z0, middle, z1)]
                crossings.append(radii)
        crossings.sort(key=lambda row: row[1])
        if len(crossings) % 2:
            raise ValueError("Invalid meridian section")
        for inside, outside in zip(crossings[::2], crossings[1::2]):
            yield from convex_wall_cells([(z0, inside[0], outside[0]), (z1, inside[2], outside[2])], segments)


def beaker(radius=.023, height=.070, wall=.0015):
    return revolve([(0., 0.), (radius-wall, 0.), (radius, wall),
                    (radius, height-.002), (radius+.001, height),
                    (radius-wall, height), (radius-wall, .002), (0., .002)])


def sample_cylinder(center, radius, z0, z1, spacing):
    """Initial physical charge; no motion-time emitters or volume resetting."""
    from .fluid import lattice_box

    center = np.asarray(center)
    points = lattice_box((-radius, -radius, z0), (radius, radius, z1), spacing)
    return points[np.linalg.norm(points[:, :2], axis=1) < radius] + center
