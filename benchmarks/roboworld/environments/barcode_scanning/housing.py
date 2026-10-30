"""Render-only shapes of the scanner housing and its cradle (the physics uses the boxes in scene.py).

Rounded hulls: the convex hull of spheres at the corners of an inner box (a sphere-swept box), so every edge has
the moulded radius of a real housing. The head has a window tunnel cut into its nose (manifold3d boolean) that
stays outside the imager's field of view: the entrance pupil is 50 mm behind the window, where the 32.8 x 24.8 deg
field is 29.4 x 22.0 mm wide at the window plane; the tunnel is 33 x 27 mm and also encloses the two illumination
LEDs and the aim source. Sizes follow scene.py (HEAD_HALF, HANDLE_HALF, handle rake) so visual and collider agree
to within the edge radii.
"""

import numpy as np


def rounded_box(half, radius, taper=None, n=10):
    """Sphere-swept box: `half` (x, y, z) outer half-extents, edge `radius`; optional `taper` = (sy, sz) scale of
    the -x end relative to the +x end (a nose narrower/wider than the rear). Returns trimesh.Trimesh."""
    import trimesh
    hx, hy, hz = (float(h) - radius for h in half)
    sph = trimesh.creation.icosphere(subdivisions=2, radius=radius).vertices
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                ky, kz = (1., 1.) if (taper is None or sx > 0) else taper
                pts.append(sph + (sx*hx, sy*hy*ky, sz*hz*kz))
    return trimesh.convex.convex_hull(np.concatenate(pts))


def _transform(mesh, centre, quat_xyzw):
    from scipy.spatial.transform import Rotation
    m = mesh.copy()
    T = np.eye(4); T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix(); T[:3, 3] = centre
    m.apply_transform(T)
    return m


def scanner_meshes(head_half, handle_centre, handle_quat, handle_half, window=(.033, .027), window_z=-.001):
    """dict(head=, window_inner=, handle=) meshes in the scanner frame (+x out of the window, +z up)."""
    import trimesh
    hh = np.asarray(head_half, float)
    head = rounded_box(hh, .011, taper=(1.0, 1.0))
    head.apply_translation((-hh[0], 0., 0.))
    ww, wh = window
    tunnel = trimesh.creation.box(extents=(.070, ww, wh))
    tunnel.apply_translation((.035 - .065, 0., window_z))            # from 65 mm inside the head to 5 mm beyond the nose
    head = head.difference(tunnel, engine="manifold")
    # dark interior lining of the tunnel (so the opening reads as a window, not a hole into a lit shell)
    lining = trimesh.creation.box(extents=(.066, ww - .0004, wh - .0004))
    lining.apply_translation((.033 - .066 - .0005, 0., window_z))
    lining = lining.difference(trimesh.creation.box(extents=(.080, ww - .0024, wh - .0024)).apply_translation((.03 - .066, 0., window_z)),
                               engine="manifold")
    dh = np.asarray(handle_half, float)
    grip = rounded_box(dh, .010, taper=(1.0, 1.0))
    # the handle's bottom end is slightly slimmer than its top (moulded taper)
    v = grip.vertices.copy()
    t = (v[:, 2] + dh[2])/(2*dh[2])                                      # 0 at the bottom, 1 at the top
    v[:, 0] *= .92 + .08*t; v[:, 1] *= .92 + .08*t
    grip = trimesh.Trimesh(v, grip.faces)
    grip = _transform(grip, handle_centre, handle_quat)
    # the moulded neck where the grip meets the head (a 6 mm-radius blend over the handle's top 22 mm and the head's
    # underside above it), so the housing reads as one pistol-grip moulding rather than a box on a stick
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(handle_quat)
    top = [np.asarray(handle_centre) + R.apply((sx*(dh[0] - .006), sy*(dh[1] - .006), dh[2] - z))
           for sx in (-1, 1) for sy in (-1, 1) for z in (.006, .022)]
    xs = [p[0] for p in top]
    under = [np.array([x, sy*(dh[1] - .003), -hh[2] + .006]) for x in (min(xs) - .006, max(xs) - .004) for sy in (-1, 1)]
    sph = trimesh.creation.icosphere(subdivisions=2, radius=.006).vertices
    neck = trimesh.convex.convex_hull(np.concatenate([sph + p for p in top + under]))
    grip = grip.union(neck, engine="manifold")
    return dict(head=head, lining=lining, handle=grip)


def trigger_mesh(arm):
    """Trigger blade (6 x 16 x 20 mm face) with rounded edges, in the trigger body frame (pivot at the origin)."""
    b = rounded_box((.003, .008, .010), .0025)
    b.apply_translation((.007, 0., -arm))
    return b


def cradle_meshes(base_half, cup_outer_half, cup_inner_half, cup_floor):
    """Base block and cup (outer rounded box minus the handle pocket), in their own frames."""
    import trimesh
    base = rounded_box(base_half, .008)
    outer = rounded_box(cup_outer_half, .006)
    inner = trimesh.creation.box(extents=2*np.asarray(cup_inner_half, float))
    inner.apply_translation((0., 0., cup_floor + cup_inner_half[2]))
    cup = outer.difference(inner, engine="manifold")
    return dict(base=base, cup=cup)
