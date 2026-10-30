"""Explicit rigid-contact regularization, separate from compliant nib/pad laws.

Newton 1.5's MuJoCo CPU path interprets generic ke/kd through the legacy
acceleration-space conversion, NOT as N/m and N s/m. Use authored MuJoCo
solver parameters for rigid surfaces. Do not label them measured compliance.
"""

import numpy as np


def configure_rigid_contact(model):
    from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW

    # 4 ms critically damped contact regularization, high impedance. Nib/pad
    # against porcelain remain separately filtered and physically compliant.
    model.mujoco.solref_mode.assign(np.full(model.shape_count, SOLREF_MODE_RAW, np.int32))
    model.mujoco.solref.assign(np.tile([0.004, 1.0], (model.shape_count, 1)).astype(np.float32))
    model.mujoco.geom_solimp.assign(
        np.tile([0.99, 0.995, 0.001, 0.5, 2.0], (model.shape_count, 1)).astype(np.float32)
    )


def configure_robot_contact(model, robot_shapes, dt):
    """Robot-robot and robot-static contact at the stiffest stable regularization.

    Robot shells, the floor slab, walls and board frame stop each other far
    faster than the 4 ms scene regularization: it lets a link closing at v sink
    about v*4ms/2 (15-30 mm at the 10-18 m/s a falling, thrashing G1 reaches).
    2*dt is MuJoCo's lower bound for a soft constraint's time constant.
    Solmix 0 on robot and static shapes leaves every pair with a movable tool
    or furniture body on that body's unchanged solref/solimp; friction and
    condim mixing are unchanged.
    """
    static = np.flatnonzero(model.shape_body.numpy() < 0)
    shapes = np.union1d(np.asarray(robot_shapes, int), static)
    solref = model.mujoco.solref.numpy()
    solref[shapes] = (2 * dt, 1.0)
    model.mujoco.solref.assign(solref)
    solmix = model.mujoco.geom_solmix.numpy()
    solmix[shapes] = 0.0
    model.mujoco.geom_solmix.assign(solmix)


def configure_surface_masks(model, shape_ids):
    """Exact masks including imported partial self-exclusions; never approximate.

    Collapse shapes by IDENTICAL forbidden-neighbor sets after removing native
    whole-body exclusions. The classroom uses fewer than 32 such classes, unlike
    the generic all-shapes graph-color fallback. Both allowed and forbidden
    relationships are preserved. This is collision compilation, not geometry.
    """
    from collections import Counter, defaultdict
    from newton import ShapeFlags

    flags = model.shape_flags.numpy()
    selected = set(np.flatnonzero(flags & int(ShapeFlags.COLLIDE_SHAPES)))
    bodies = model.shape_body.numpy()
    count = Counter(int(bodies[s]) for s in selected)
    pairs = [
        (int(a), int(b))
        for a, b in model.shape_collision_filter_pairs_array()
        if a in selected and b in selected and bodies[a] != bodies[b]
    ]
    body_count = Counter(tuple(sorted((int(bodies[a]), int(bodies[b])))) for a, b in pairs)
    excluded = {p for p, c in body_count.items() if c == count[p[0]] * count[p[1]]}
    neighbors = defaultdict(set)
    for a, b in pairs:
        if tuple(sorted((int(bodies[a]), int(bodies[b])))) not in excluded:
            neighbors[a].add(b)
            neighbors[b].add(a)
    classes = {}
    membership = {}
    for shape in sorted(selected):
        key = tuple(sorted(neighbors[shape]))
        if key not in classes:
            classes[key] = len(classes)
        membership[shape] = classes[key]
    if len(classes) > 32:
        raise RuntimeError(f"Exact collision graph needs {len(classes)} classes, exceeding 32")
    types = np.zeros(model.shape_count, np.int64)
    affinity = np.zeros(model.shape_count, np.int64)
    for shape, c in membership.items():
        types[shape] = 1 << c
        allowed = (1 << len(classes)) - 1
        for forbidden in neighbors[shape]:
            allowed &= ~(1 << membership[forbidden])
        affinity[shape] = allowed
    model.mujoco.contype.assign(types)
    model.mujoco.conaffinity.assign(affinity)
    model.mujoco.collision_mask_domain.assign(np.zeros(model.shape_count, np.int64))
