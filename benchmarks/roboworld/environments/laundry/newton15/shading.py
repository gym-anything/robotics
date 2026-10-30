"""Interpolate shading normals without altering the physical or render surface.

Linear subdivision introduces interior vertices in each coarse triangle. If
normals are averaged *after* subdivision, those vertices see only that face,
and undo smooth shading. Use the coarse vertices' existing per-panel normals
and interpolate through the same barycentric embedding as the UVs instead.
Thickness offsets remain exactly those of the original finite-thickness shell.
"""
import numpy as np
from ..garment_render_mesh import deform_thick_garment_panel


def deform(topology, particles, *, thickness_m, shading="smooth"):
    if shading not in ("smooth", "legacy"):
        raise ValueError("shading must be smooth or legacy")
    points, normals = deform_thick_garment_panel(topology, particles, thickness_m=thickness_m)
    if shading == "legacy":
        return points, normals
    embedding = topology.embedding
    indices = embedding.coarse_vertex_indices
    original = np.all(indices == indices[:, :1], axis=1)
    coarse = np.zeros_like(particles, dtype=np.float64)
    coarse[indices[original, 0]] = normals[:embedding.vertex_count][original]
    smooth = embedding.interpolate(coarse)
    lengths = np.linalg.norm(smooth, axis=1)
    valid = lengths > 1e-10
    smooth[valid] /= lengths[valid, None]
    smooth[~valid] = normals[:embedding.vertex_count][~valid]
    return points, np.concatenate((smooth, -smooth))
