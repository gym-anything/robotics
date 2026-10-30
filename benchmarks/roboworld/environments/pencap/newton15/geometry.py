"""The existing revolved pen/cap solids, including the real open cap cavity."""

import numpy as np
from .runtime import CACHE, require_engine


def revolve(profile, segments=64):
    profile = np.asarray(profile, float)
    angles = np.arange(segments)*2*np.pi/segments
    vertices = np.array([(r*np.cos(a), y, r*np.sin(a)) for r, y in profile for a in angles])
    faces = []
    for i in range(len(profile)):
        j = (i+1) % len(profile)
        for k in range(segments):
            k2 = (k+1) % segments
            a, b, c, d = i*segments+k, i*segments+k2, j*segments+k, j*segments+k2
            faces.extend(((a, c, d), (a, d, b)))
    faces = np.asarray(faces, np.int32)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    if np.einsum("ij,ij->", v0, np.cross(v1, v2)) < 0:
        faces = faces[:, [0, 2, 1]]
    # Weld identical polar vertices and remove zero-area pole triangles. This
    # changes topology only, not the surface, and gives SDF sign a closed mesh.
    import trimesh
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or mesh.volume <= 0:
        raise ValueError("Revolved collision solid is not closed/outward")
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)


def pen_profile(c):
    r = c.shaft_radius
    return [(0., 0.), (.0043*.25, .002), (.0043, .018), (r, .028),
            (r, c.bead_u-c.bead_flat/2-c.lead_run), (c.bead_radius, c.bead_u-c.bead_flat/2),
            (c.bead_radius, c.bead_u+c.bead_flat/2), (r, c.bead_u+c.bead_flat/2+c.return_run),
            (r, c.shoulder_u), (.0065, c.shoulder_u), (.0065, c.shoulder_u+.004),
            (r, c.shoulder_u+.006), (r, c.pen_length-.004), (.0042, c.pen_length), (0., c.pen_length)]


def cap_profile(c):
    r, length = c.cap_radius, c.cap_length
    return [(c.bore_radius, 0.), (c.bore_radius, length-.003), (0., length-.002),
            (0., length), (r, length), (r, .003), (r+.0005, .0015), (r+.0005, 0.)]


def collision_mesh(profile, c, *, sdf=True):
    newton, _ = require_engine()
    v, f = revolve(profile)
    mesh = newton.Mesh(v, f.ravel())
    if sdf:
        mesh.build_sdf(device=c.device, max_resolution=c.sdf_resolution,
                       narrow_band_range=(-.0005, .0005), margin=c.contact_gap,
                       cache_dir=CACHE / "sdf")
    return mesh


def set_mass(builder, body, mass):
    """Normalize volume-derived inertia to the baseline's explicit total mass."""
    old = builder.body_mass[body]
    if old <= 0:
        raise ValueError("Cannot assign mass to an empty dynamic solid")
    builder.body_inertia[body] = builder.body_inertia[body]*(mass/old)
    builder.body_mass[body] = mass


def add_cap(builder, c, transform, *, profile=None, material=None, as_link=False):
    newton, wp = require_engine()
    body = (builder.add_link if as_link else builder.add_body)(xform=transform, label="cap")
    cfg = material or newton.ModelBuilder.ShapeConfig(
        mu=c.plastic_friction, ke=1e7, kd=5000., gap=c.contact_gap, mu_torsional=0., mu_rolling=0.)
    builder.add_shape_mesh(body, mesh=collision_mesh(profile or cap_profile(c), c), cfg=cfg,
                           label="cap_shell", color=(.025, .12, .50))
    nub = newton.ModelBuilder.ShapeConfig(mu=c.plastic_friction, ke=c.nub_stiffness,
        kd=c.nub_damping, gap=c.contact_gap, mu_torsional=0., mu_rolling=0.)
    radius = c.bead_radius-c.interference+c.nub_radius
    for i in range(c.nub_count):
        a = 2*np.pi*i/c.nub_count
        builder.add_shape_sphere(body, radius=c.nub_radius, cfg=nub, label=f"cap_nub_{i}",
            xform=wp.transform((radius*np.cos(a), c.nub_offset, radius*np.sin(a)), wp.quat_identity()),
            color=(.025, .12, .50))
    set_mass(builder, body, c.cap_mass)
    return body
