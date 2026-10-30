"""Continuous surface reconstructed from actual material-point positions.

No repositioning/respawning of fluid, no prescribed flask fill plane. The fixed
0.5 volume-fraction isosurface is a visualization approximation: mesh volume
must be reported against material volume, not silently rescaled to fit it.
"""

from dataclasses import dataclass
import numpy as np

from .runtime import require_newton


@dataclass
class Surface:
    vertices: np.ndarray
    triangles: np.ndarray
    volume_m3: float
    material_volume_m3: float
    unresolved_particle_count: int = 0
    approximation: str = "Fixed 0.5 Wendland volume field; no volume fitting"
    pink_molar: np.ndarray | None = None
    pink_volume: dict | None = None


def reconstruct(positions, volume_m3, support_m, grid_m, device="cuda:0"):
    _, wp = require_newton()
    from .surface_kernels import sample_volume

    q = np.asarray(positions, dtype=np.float32)
    lower = np.floor((q.min(axis=0)-support_m)/grid_m)*grid_m
    upper = np.ceil((q.max(axis=0)+support_m)/grid_m)*grid_m
    counts = np.rint((upper-lower)/grid_m).astype(int)+1
    if np.prod(counts) > 2_000_000:
        raise ValueError("Surface sampling exceeds the explicit two-million-voxel budget")
    points = wp.array(q, dtype=wp.vec3, device=device)
    grid = wp.HashGrid(64, 64, 64, device=device)
    grid.build(points, support_m)
    field = wp.empty(tuple(int(x) for x in counts), dtype=float, device=device)
    wp.launch(sample_volume, dim=tuple(counts),
              inputs=[grid.id, points, float(volume_m3), float(support_m),
                      wp.vec3(*lower), float(grid_m)], outputs=[field], device=device)
    mc = wp.MarchingCubes(*[int(x) for x in counts], device=device,
                          domain_bounds_lower_corner=lower.tolist(),
                          domain_bounds_upper_corner=upper.tolist())
    mc.surface(field, .5)
    vertices = mc.verts.numpy()
    triangles = mc.indices.numpy().reshape(-1, 3)
    if len(triangles):
        tri = vertices[triangles].astype(float)
        mesh_volume = abs(np.einsum("ij,ij->i", tri[:, 0],
                                    np.cross(tri[:, 1], tri[:, 2])).sum()/6.0)
    else:
        mesh_volume = 0.0
    return Surface(vertices, triangles, float(mesh_volume), len(q)*volume_m3)


def reconstruct_sparse(positions, volume_m3, support_m, grid_m, device="cuda:0",
                       *, brick_cells=16, max_bricks=4096):
    """Same field/threshold on occupied bricks, without a huge empty air grid.

    Every particle activates all bricks touched by its support. Brick surfaces
    are stitched by shared positions, not smoothed or fitted to material mass.
    Isolated parcels whose support does not reach any reconstructed surface are
    explicitly shown by equal-volume sphere glyphs and counted in metadata.
    Surface-adjacent parcels still contribute to the continuous volume field.
    """
    _, wp = require_newton()
    from .surface_kernels import sample_bricks, at_particles
    q = np.asarray(positions, np.float32)
    if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
        raise ValueError("Finite nonempty positions required for surface reconstruction")
    if min(volume_m3, support_m, grid_m) <= 0 or type(brick_cells) is not int or brick_cells < 2:
        raise ValueError("Positive surface dimensions and integer brick size required")
    size = grid_m*brick_cells
    # Enumerate support-overlapping neighbours of occupied bricks; this can
    # conservatively activate empty neighbours, never omit a fluid parcel.
    occupied = np.unique(np.floor(q/size).astype(np.int64), axis=0)
    reach = int(np.ceil(support_m/size))
    offsets = np.stack(np.meshgrid(*([np.arange(-reach, reach+1)]*3), indexing="ij"), axis=-1).reshape(-1, 3)
    bricks = np.unique((occupied[:, None, :]+offsets).reshape(-1, 3), axis=0)
    if len(bricks) > max_bricks:
        raise ValueError(f"Surface needs {len(bricks)} bricks; budget is {max_bricks}; no resolution substitution")
    points = wp.array(q, dtype=wp.vec3, device=device)
    grid = wp.HashGrid(64, 64, 64, device=device)
    grid.build(points, support_m)
    origins = wp.array(bricks*size, dtype=wp.vec3, device=device)
    n = brick_cells+1
    field = wp.empty((len(bricks), n, n, n), dtype=float, device=device)
    wp.launch(sample_bricks, dim=field.shape, inputs=[grid.id, points, volume_m3, support_m,
        origins, grid_m], outputs=[field], device=device)
    mc = wp.MarchingCubes(n, n, n, device=device, domain_bounds_lower_corner=[0., 0., 0.],
        domain_bounds_upper_corner=[size, size, size])
    vertices, faces, used = [], [], 0
    # Read only brick ranges to avoid invoking marching cubes on empty fields.
    limits = field.numpy().reshape(len(bricks), -1)
    active = np.flatnonzero((limits.min(1) <= .5) & (limits.max(1) >= .5))
    for index in active:
        mc.surface(field[int(index)], .5)
        v, f = mc.verts.numpy(), mc.indices.numpy().reshape(-1, 3)
        if len(f):
            vertices.append(v+bricks[index]*size)
            faces.append(f+used)
            used += len(v)
    if vertices:
        mv, mf = np.concatenate(vertices), np.concatenate(faces)
        t = mv[mf].astype(float)
        if np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() < 0:
            faces = [f[:, ::-1] for f in faces]
    values = wp.empty(len(q), dtype=float, device=device)
    wp.launch(at_particles, dim=len(q), inputs=[grid.id, points, volume_m3, support_m], outputs=[values], device=device)
    unresolved = np.flatnonzero(values.numpy() < .5)
    if len(unresolved) and vertices:
        from scipy.spatial import cKDTree
        distance, _ = cKDTree(np.concatenate(vertices)).query(q[unresolved])
        unresolved = unresolved[distance > support_m]
    if len(unresolved):
        import trimesh
        sphere = trimesh.creation.icosphere(subdivisions=1,
            radius=(3*volume_m3/(4*np.pi))**(1/3))
        # The sphere glyph's polygonal volume equals its one parcel; this is
        # not a rescaling of the reconstructed continuous surface.
        sphere.vertices *= (volume_m3/sphere.volume)**(1/3)
        if used+len(unresolved)*len(sphere.vertices) > 2_000_000:
            raise ValueError("Unresolved parcel glyphs exceed surface vertex budget; no parcels omitted")
        for index in unresolved:
            vertices.append(sphere.vertices+q[index])
            faces.append(sphere.faces+used)
            used += len(sphere.vertices)
    v = np.concatenate(vertices) if vertices else np.empty((0, 3))
    f = np.concatenate(faces).astype(np.int32) if faces else np.empty((0, 3), np.int32)
    if len(v):
        # Sub-micrometre key only identifies shared MC seam vertices. Retain
        # original coordinates; do not round/reposition the liquid surface.
        _, indices, inverse = np.unique(np.round(v, 8), axis=0, return_index=True, return_inverse=True)
        v, f = v[indices], inverse[f]
        t = v[f].astype(float)
        measured = abs(np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()/6)
    else:
        measured = 0.
    return Surface(v, f, measured, len(q)*volume_m3, len(unresolved),
        "Fixed 0.5 Wendland field, stitched bricks; under-resolved parcels shown as equal-volume sphere glyphs; no smoothing or volume fitting")
