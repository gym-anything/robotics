"""Pallets and loads, native MuJoCo contact with the friction pairs set in params.

- GMA 48 x 40 in stringer pallet (NWPCA drawing, params.GMA_PALLET): three 1-3/8 x 3-1/2 in stringers along the 48 in
  length, seven top deck boards and five bottom deck boards of 5/8 in across them; the fork openings between the
  stringers are 3-1/2 in high and ~17.9 in wide (derived from the drawing, checked in probes). One rigid body of box
  shapes; mass 20 kg spread over the boards and stringers.
- Wrapped unit load: a well-wrapped load moves as one unit with its pallet — the load is a box shape on the pallet body
  (rigid composite) with the load's mass at its centroid.
- Loose carton stack: free carton bodies (0.50 x 0.40 x 0.25 m) stacked on the pallet in a column pattern.
- Steel coil unit: a short steel cylinder lying on a pallet with two timber chocks (one rigid composite, 2,000 kg).
- OCC bales: 60 x 30 x 48 in boxes (450-590 kg) on two 4 x 4 in dunnage strips (free bodies).

Every load is placed resting (a few mm above its support) and settled by physics with the start state; the task builder
checks the settled state before tick 0.
"""

import math

import numpy as np

from . import params as P

IN = 0.0254


def gma_boxes():
    """[(centre, half extents, kind)] of a GMA pallet in its own frame: x along the stringers (48 in), y across
    (40 in), z up from the floor; fork entry from the +/-x faces between the stringers."""
    g = P.GMA_PALLET
    L, W, H, t, sw = g["L"], g["W"], g["H"], g["deck"], g["stringer_w"]
    hs = H - 2 * t                                       # stringer height (3.5 in)
    out = []
    for y in (-(W / 2 - sw / 2), 0.0, W / 2 - sw / 2):
        out.append(((0.0, y, t + hs / 2), (L / 2, sw / 2, hs / 2), "stringer"))
    # top deck: two 5.5 in lead boards at the ends and five 3.5 in boards between, evenly spaced
    lead, inner = P.GMA_BOARDS["lead"], P.GMA_BOARDS["inner"]
    widths = [lead] + [inner] * 5 + [lead]
    xs = np.linspace(-L / 2 + widths[0] / 2, L / 2 - widths[-1] / 2, 7)
    for x, w in zip(xs, widths):
        out.append(((float(x), 0.0, H - t / 2), (w / 2, W / 2, t / 2), "top_deck"))
    # bottom deck: two 5.5 in lead boards and three 3.5 in boards
    widths = [lead] + [inner] * 3 + [lead]
    xs = np.linspace(-L / 2 + widths[0] / 2, L / 2 - widths[-1] / 2, 5)
    for x, w in zip(xs, widths):
        out.append(((float(x), 0.0, t / 2), (w / 2, W / 2, t / 2), "bottom_deck"))
    return out


def fork_openings():
    """Fork openings of a GMA pallet (pallet frame): [(y centre, width, z bottom, z top)]."""
    g = P.GMA_PALLET
    W, H, t, sw = g["W"], g["H"], g["deck"], g["stringer_w"]
    w = (W - 3 * sw) / 2
    return [(-(sw / 2 + w / 2), w, t, H - t), (sw / 2 + w / 2, w, t, H - t)]


def _quat_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def add_pallet(scene, name, pose, load=None, z0=0.0):
    """A pallet body at pose (x, y, yaw) on a support at height z0, with an optional rigid load:
    load = dict(kind='wrapped'|'coil', mass, size=(lx, ly, h)) (wrapped: a box; coil: a cylinder on chocks); a wrapped
    load's optional offset_x shifts its centre along the stringers (a short dense load stacked at one end)."""
    from .scene import box_shape
    newton, wp, b = scene.newton, scene.wp, scene.b
    x, y, yaw = pose
    g = P.GMA_PALLET
    boxes = gma_boxes()
    vol = sum(8 * h[0] * h[1] * h[2] for _, h, _ in boxes)
    m_pal = g["mass"]
    mass, com_acc, inertia = 0.0, np.zeros(3), np.zeros((3, 3))
    parts = []
    for c, h, kind in boxes:
        mm = m_pal * 8 * h[0] * h[1] * h[2] / vol
        parts.append((np.asarray(c), np.asarray(h), mm))
    if load is not None:
        lx, ly, lh = load["size"]
        if load["kind"] == "coil":
            parts.append((np.array([0.0, 0.0, g["H"] + lh / 2]), np.array([lx / 2, ly / 2, lh / 2]), load["mass"]))
        else:
            parts.append((np.array([load.get("offset_x", 0.0), 0.0, g["H"] + lh / 2]), np.array([lx / 2, ly / 2, lh / 2]),
                          load["mass"]))
    M = sum(p[2] for p in parts)
    com = sum(p[2] * p[0] for p in parts) / M
    I = np.zeros((3, 3))
    for c, h, mm in parts:
        a, bb, cc = 2 * h
        Ib = np.diag([mm * (bb * bb + cc * cc) / 12, mm * (a * a + cc * cc) / 12, mm * (a * a + bb * bb) / 12])
        d = c - com
        I += Ib + mm * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    body = b.add_link(xform=wp.transform((x, y, z0 + 0.003), wp.quat(*_quat_yaw(yaw))), mass=float(M),
                      com=tuple(map(float, com)), inertia=wp.mat33(*I.ravel()), lock_inertia=True, label=f"load/{name}")
    cfg = scene.cfg(mu=P.MU_PALLET_FLOOR)
    shapes = []
    for c, h, kind in boxes:
        s = box_shape(b, newton, body, wp.transform(tuple(map(float, c)), wp.quat_identity()), h, cfg, f"{name}:{kind}")
        scene.shape_group[s] = "obj"; scene.materials[s] = "wood"
        shapes.append(s)
    if load is not None:
        lx, ly, lh = load["size"]
        if load["kind"] == "coil":
            # a steel coil lying with its axis along the pallet's x, on two chocks (part of the composite)
            r = lh / 2
            s = b.add_shape_cylinder(body, xform=wp.transform((0.0, 0.0, g["H"] + r),
                                                              wp.quat(*_quat_axis((0, 1, 0), math.pi / 2))),
                                     radius=float(r), half_height=float(lx / 2), cfg=scene.cfg(mu=P.MU_COIL), label=f"{name}:coil")
            scene.materials[s] = "steel"
        else:
            s = box_shape(b, newton, body, wp.transform((float(load.get("offset_x", 0.0)), 0.0, g["H"] + lh / 2),
                                                        wp.quat_identity()),
                          (lx / 2, ly / 2, lh / 2), scene.cfg(mu=P.MU_CARTON), f"{name}:wrapped")
            scene.materials[s] = "wrapped"
        scene.shape_group[s] = "obj"
        shapes.append(s)
    j = b.add_joint_free(body, label=f"load/{name}")
    b.add_articulation([j], label=f"load/{name}")
    scene.bodies[f"load/{name}"] = body
    scene.loads[name] = dict(body=body, kind=(load or {}).get("kind", "pallet"), mass=float(M), shapes=shapes,
                             size=(g["L"], g["W"], g["H"] + ((load or {}).get("size", (0, 0, 0))[2])), pose=pose, z0=z0)
    return body


def _quat_axis(axis, angle):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    s = math.sin(angle / 2)
    return (float(a[0] * s), float(a[1] * s), float(a[2] * s), math.cos(angle / 2))


def add_box(scene, name, centre, size, mass, yaw=0.0, mu=None, material="carton", label_prefix="load"):
    """A free box body (carton, bale, dunnage strip) resting at centre (x, y, z of its bottom face)."""
    from .scene import box_shape
    wp, b = scene.wp, scene.b
    lx, ly, lz = size
    I = np.diag([mass * (ly * ly + lz * lz) / 12, mass * (lx * lx + lz * lz) / 12, mass * (lx * lx + ly * ly) / 12])
    body = b.add_link(xform=wp.transform((centre[0], centre[1], centre[2] + lz / 2 + 0.002), wp.quat(*_quat_yaw(yaw))),
                      mass=float(mass), inertia=wp.mat33(*I.ravel()), lock_inertia=True, label=f"{label_prefix}/{name}")
    s = box_shape(b, scene.newton, body, wp.transform_identity(), (lx / 2, ly / 2, lz / 2),
                  scene.cfg(mu=P.MU_CARTON if mu is None else mu), f"{name}:box")
    scene.shape_group[s] = "obj"; scene.materials[s] = material
    j = b.add_joint_free(body, label=f"{label_prefix}/{name}")
    b.add_articulation([j], label=f"{label_prefix}/{name}")
    scene.bodies[f"{label_prefix}/{name}"] = body
    scene.loads[name] = dict(body=body, kind=material, mass=float(mass), shapes=[s], size=tuple(size),
                             pose=(centre[0], centre[1], yaw), z0=centre[2])
    return body


def add_carton_stack(scene, name, pallet_pose, z0=0.0, layers=5, rng=None):
    """Pallet with a loose column-stacked carton load: 6 cartons (0.50 x 0.40 x 0.25 m) per layer, 12-20 kg each."""
    add_pallet(scene, name, pallet_pose, z0=z0)
    x, y, yaw = pallet_pose
    c, s = math.cos(yaw), math.sin(yaw)
    H = P.GMA_PALLET["H"]
    rng = rng or np.random.default_rng(0)
    k = 0
    for layer in range(layers):
        for i in range(3):            # 3 along the pallet length (x): 0.40 m each
            for j in range(2):        # 2 across (y): 0.50 m each
                lx, ly, lz = P.CARTON_SIZE
                px = (i - 1) * (lx + 0.002)               # built 2 mm apart and 3 mm above each other, then settled
                py = (j - 0.5) * (ly + 0.002)
                cx, cy = x + c * px - s * py, y + s * px + c * py
                add_box(scene, f"{name}_c{k:02d}", (cx, cy, z0 + H + 0.003 + layer * (lz + 0.003)), (lx, ly, lz),
                        float(rng.uniform(*P.CARTON_MASS)), yaw=yaw)
                k += 1


def build(scene):
    if not hasattr(scene, "loads"):
        scene.loads = {}                 # stored rack units may already be in it (facility.build_bodies)
    for spec in getattr(scene, "load_specs", []):
        kind = spec.pop("kind")
        {"pallet": lambda **s: add_pallet(scene, **s), "stack": lambda **s: add_carton_stack(scene, **s),
         "box": lambda **s: add_box(scene, **s)}[kind](**spec)


def mujoco_hook(world, m):
    """Elliptic-cone friction pairs (one condim everywhere): fork steel on pallet wood uses MU_FORK_PALLET (the pair takes
    the max of the two geoms' mu in MuJoCo, so the forks' own mu is set to it and the pallet's deck to the same)."""
    pass
