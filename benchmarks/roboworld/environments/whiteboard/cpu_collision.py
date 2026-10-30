"""Rebuild derived collision bounds after Newton 1.5's CPU inertia-frame sync.

Upstream issue: https://github.com/newton-physics/newton/issues/3805
No geometry, mass, inertial tensor, collision mask, or contact law is changed.
Only the acceleration structure is refitted to the model's CURRENT frames.
Tests compare against MuJoCo's unpruned narrow-phase on identical states.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def refit_cpu_bvh(model):
    """Conservative AABB union in each body's inertial frame, including leaves.

    MuJoCo 3.11 mj_collideTree uses body-inertial BVH bounds until both nodes
    are leaves; only then does it use geom-local geom_aabb directly. Child
    indices in bvh_child are relative to the body's body_bvhadr.
    """
    original = model.bvh_aabb.copy()
    for body in range(model.nbody):
        start, count = int(model.body_bvhadr[body]), int(model.body_bvhnum[body])
        if count == 0:
            continue
        inertial = Rotation.from_quat(np.roll(model.body_iquat[body], -1)).as_matrix()
        for offset in sorted(
            range(count), key=lambda k: int(model.bvh_depth[start + k]), reverse=True
        ):
            node = start + offset
            geom = int(model.bvh_nodeid[node])
            if geom >= 0:
                rotation = (
                    inertial.T @ Rotation.from_quat(np.roll(model.geom_quat[geom], -1)).as_matrix()
                )
                center = (
                    inertial.T @ (model.geom_pos[geom] - model.body_ipos[body])
                    + rotation @ model.geom_aabb[geom, :3]
                )
                half = np.abs(rotation) @ model.geom_aabb[geom, 3:]
            else:
                children = [start + int(c) for c in model.bvh_child[node] if c >= 0]
                if not children:
                    raise RuntimeError("Invalid CPU collision-tree node")
                bounds = model.bvh_aabb[children]
                lo = np.min(bounds[:, :3] - bounds[:, 3:], axis=0)
                hi = np.max(bounds[:, :3] + bounds[:, 3:], axis=0)
                center = (lo + hi) / 2
                half = (hi - lo) / 2
            model.bvh_aabb[node, :3] = center
            model.bvh_aabb[node, 3:] = half + 1e-12
    return {
        "nodes": int(model.nbvhstatic),
        "changed_nodes": int(np.any(np.abs(original - model.bvh_aabb) > 1e-9, axis=1).sum()),
        "max_bound_change_m": float(np.max(np.abs(original - model.bvh_aabb), initial=0)),
    }
