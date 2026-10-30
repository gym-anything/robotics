"""Resolved tool/porcelain contact. This owns that pair's wrench exactly once.

The Newton nib/pad-to-porcelain pairs are filtered; all other rigid collisions
(frame, tools, fingers, floor) stay enabled. Wall is an immovable architectural
support. Nib/pad internal deformation is a bonded elastic solid, not a weld.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def tool_contacts(env, q, qd, external_forces=None):
    from .deformable_contact import board_patch, caps

    forces = np.zeros((env.model.body_count, 6), np.float32)
    external = np.zeros_like(forces) if external_forces is None else np.asarray(external_forces)
    caps(env, q, qd, forces)
    events = []
    b, p = env.config.board, env.config.materials
    for color in env.ink.reservoir:
        name = "marker_" + color
        i = env.body_ids[name]
        # Conservative broad phase: no nib can touch the writing surface when
        # its body origin is outside this expanded board bounding box.
        # This only skips impossible pairs; it does not drop shallow contact.
        origin = q[i, :3]
        if (
            origin[1] > b.face_y + 0.067
            or origin[1] < b.face_y - 0.067
            or abs(origin[0]) > b.writing_width / 2 + 0.067
            or abs(origin[2] - b.center_z) > b.writing_height / 2 + 0.067
        ):
            env.deformations.pop(name, None)
            events.append((name, color, None))
            continue
        # A cap prevents geometric access to the nib, even in external fixtures.
        cap = env.body_ids["cap_" + color]
        cap_rel = Rotation.from_quat(q[i, 3:]).inv().apply(q[cap, :3] - q[i, :3])
        capped = np.linalg.norm(cap_rel[:2]) < 0.012 and 0.035 < cap_rel[2] < 0.076
        facing = Rotation.from_quat(q[i, 3:]).apply([0, 0, 1])[1] < -0.20
        if capped or not facing:
            env.deformations.pop(name, None)
        patch = (
            None
            if capped or not facing
            else board_patch(
                env,
                "nib",
                name,
                q[i],
                qd[i],
                (0, 0, 0.063),
                ((0.0025, 0, 0), (0, 0.001, 0)),
                p.nib_friction,
                p.nib_damping_ns_m,
                applied_wrench=forces[i] + external[i],
            )
        )
        if patch:
            forces[i] += patch["wrench"]
        events.append((name, color, patch))
    i = env.body_ids["eraser"]
    rot = Rotation.from_quat(q[i, 3:])
    normal = rot.apply([0, 0, -1])
    point = q[i, :3] + rot.apply([0, 0, -0.010])
    facing = normal[1] < -0.6
    gap = point[1] - b.face_y
    if (
        facing
        and -0.004 < gap < 0.015
        and abs(point[0]) < b.writing_width / 2
        and abs(point[2] - b.center_z) < b.writing_height / 2
    ):
        magnetic = -p.magnet_surface_force_n / (1 + max(0, gap) / p.magnet_gap_scale_m) ** 2
        forces[i, 1] += magnetic
        forces[i, 3:] += np.cross(
            rot.apply([0, 0, -0.010]) - rot.apply(env.com[i]), [0, magnetic, 0]
        )
    if not facing:
        env.deformations.pop("eraser", None)
    patch = (
        board_patch(
            env,
            "felt",
            "eraser",
            q[i],
            qd[i],
            (0, 0, -0.010),
            ((0.0515, 0, 0), (0, 0.0265, 0)),
            p.pad_friction,
            p.pad_damping_ns_m,
            applied_wrench=forces[i] + external[i],
        )
        if facing
        else None
    )
    if patch:
        env._pad_contact_state = patch["tangential_state"]
        forces[i] += patch["wrench"]
    else:
        env._pad_contact_state = None
    events.append(("eraser", None, patch))
    return forces, events
