"""Enforce exact custom/native ball contact ownership in the CPU adapter.

Newton 1.5's large-graph coloring fallback can over-approximate shape filters.
Body-wide excludes cannot express ball/pitch exclusion while preserving ball/posts
because those static shapes share the world body. Reserve one unused MuJoCo type
bit for the ball's native partners; leave every non-ball pair unchanged.
"""

import numpy as np


def enforce_ball_pairs(world):
    m = world.solver.mj_model
    mapping = world.solver.mjc_geom_to_newton_shape.numpy().reshape(-1)
    ids = np.flatnonzero(mapping == world.ball_shape)
    if len(ids) != 1 or np.any(mapping < 0):
        raise RuntimeError("Cannot establish exact ball collision geometry mapping")
    ball = int(ids[0])
    if np.any(m.pair_geom1 == ball) or np.any(m.pair_geom2 == ball):
        raise RuntimeError("Explicit ball contact pairs bypass ownership masks")
    expected = ~world.model.shape_collision_filter_mask(
        np.column_stack((np.full(m.ngeom, world.ball_shape), mapping))
    )
    expected[ball] = False
    types = m.geom_contype.view(np.uint32).copy()
    affinity = m.geom_conaffinity.view(np.uint32).copy()

    def pairs(t, a):
        return ((t[:, None] & a[None, :]) | (a[:, None] & t[None, :])) != 0

    before = pairs(types, affinity)
    used = int(np.bitwise_or.reduce(types))
    available = [bit for bit in range(32) if not used & (1 << bit)]
    if not available:
        raise RuntimeError(
            "No unused collision type bit; cannot qualify exact ownership"
        )
    bit = np.uint32(1 << available[-1])
    affinity &= ~bit
    types[expected] |= bit
    types[ball] = 0
    affinity[ball] = bit
    after = pairs(types, affinity)
    other = np.arange(m.ngeom) != ball
    if not np.array_equal(after[np.ix_(other, other)], before[np.ix_(other, other)]):
        raise RuntimeError("Ownership correction changed unrelated contact pairs")
    if not np.array_equal(after[ball], expected):
        raise RuntimeError("Ball/native collision ownership mismatch")
    m.geom_contype[:] = types.view(np.int32)
    m.geom_conaffinity[:] = affinity.view(np.int32)
    # MuJoCo also uses body-level OR masks during broad phase.
    m.body_contype[:] = 0
    m.body_conaffinity[:] = 0
    for geom, body in enumerate(m.geom_bodyid):
        m.body_contype[body] |= m.geom_contype[geom]
        m.body_conaffinity[body] |= m.geom_conaffinity[geom]
    return {
        "ball_geom_id": ball,
        "reserved_type_bit": available[-1],
        "corrected_geom_ids": np.flatnonzero(before[ball] != after[ball]).tolist(),
        "native_partner_geom_ids": np.flatnonzero(expected).tolist(),
        "unrelated_pair_masks_unchanged": True,
        "exact_ball_masks": True,
    }
