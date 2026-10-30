"""Verified prismatic-coordinate gradients and opposed-hardware-jaw topology."""

import numpy as np
import newton
import warp as wp


@wp.func
def coordinate(
    j: int,
    q: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    parents: wp.array[int],
    children: wp.array[int],
    xp: wp.array[wp.transform],
    xc: wp.array[wp.transform],
    starts: wp.array[int],
    axes: wp.array[wp.vec3],
):
    p = parents[j]
    c = children[j]
    fp = q[p] * xp[j]
    fc = q[c] * xc[j]
    axis = wp.transform_vector(fp, axes[starts[j]])
    a = wp.transform_get_translation(fp)
    b = wp.transform_get_translation(fc)
    rp = b - wp.transform_point(q[p], com[p])
    rc = b - wp.transform_point(q[c], com[c])
    return (
        wp.dot(b - a, axis),
        wp.spatial_vector(-axis, -wp.cross(rp, axis)),
        wp.spatial_vector(axis, wp.cross(rc, axis)),
    )


def joint_pairs(m, pairs=None):
    if pairs is None:
        pairs = []
        for arm in ("left", "right"):
            ids = []
            for i in (1, 2):
                suffix = f"wsg_finger_joint{i}"
                found = [
                    j
                    for j, label in enumerate(m.joint_label)
                    if label.startswith(arm + "/") and label.endswith(suffix)
                ]
                if len(found) != 1:
                    raise ValueError(
                        "Expected one hardware jaw joint: " + arm + "/" + suffix
                    )
                ids.append(found[0])
            pairs.append(ids)
    pairs = np.asarray(pairs)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or not len(pairs)
        or pairs.dtype.kind not in "iu"
    ):
        raise ValueError("Nonempty integer joint pairs required")
    if np.any(pairs < 0) or np.any(pairs >= m.joint_count):
        raise ValueError("Jaw joint index outside model")
    parent = m.joint_parent.numpy()
    child = m.joint_child.numpy()
    kind = m.joint_type.numpy()
    axis = m.joint_axis.numpy()
    starts = m.joint_qd_start.numpy()
    used = set()
    for a, b in pairs:
        if (
            kind[a] != newton.JointType.PRISMATIC
            or kind[b] != newton.JointType.PRISMATIC
        ):
            raise ValueError("Synchronizer supports only prismatic jaws")
        if parent[a] < 0 or parent[a] != parent[b] or child[a] == child[b]:
            raise ValueError("Jaws require a shared physical housing")
        bodies = {int(parent[a]), int(child[a]), int(child[b])}
        if used & bodies:
            raise ValueError("Belt groups must have disjoint bodies")
        used |= bodies
        if not np.allclose(axis[starts[a]], -axis[starts[b]], atol=1e-7):
            raise ValueError("Expected opposed hardware jaw axes")
    return wp.array(pairs, dtype=int, device=m.device)
