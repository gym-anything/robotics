"""Elastic material/rigid-body wrench coupling; all positions are SI.

Board-backed nib and felt and cap/barrel interior contact use tetrahedral solid
deformation. Other tool contacts retain explicitly documented rigid collision
proxies. An immovable board reacts the opposite wrench through its wall mount.
"""

import numpy as np
from functools import lru_cache
from scipy.spatial.transform import Rotation
from .elasticity import tool_solid


@lru_cache(maxsize=3)
def surface(kind):
    s = tool_solid(kind)
    return s, np.array([i for i in np.unique(s.faces) if not s.fixed[i]], int)


@lru_cache(maxsize=1)
def cap_surface():
    """Immutable reference-node selection; deformed poses are never cached."""
    s = tool_solid("cap")
    radii = np.linalg.norm(s.rest[:, :2], axis=1)
    inner = np.array([i for i in np.unique(s.faces) if radii[i] < 0.008 and not s.fixed[i]], int)
    return s, inner


def cross3(a, b):
    """Cross product for known (..., 3) operands, preserving NumPy arithmetic order."""
    a, b = np.asarray(a), np.asarray(b)
    return np.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        axis=-1,
    )


def _wrench(points, forces, com):
    return np.concatenate((forces.sum(axis=0), cross3(points - com, forces).sum(axis=0)))


def board_patch(env, kind, name, q, qd, center, axes, friction, damping, applied_wrench=None):
    s, nodes = surface(kind)
    rot = Rotation.from_quat(q[3:])
    b = env.config.board
    points = q[:3] + rot.apply(s.rest[nodes])
    penetration = b.face_y - points[:, 1]
    inside = (np.abs(points[:, 0]) <= b.writing_width / 2) & (
        np.abs(points[:, 2] - b.center_z) <= b.writing_height / 2
    )
    nodes = nodes[inside]
    points = points[inside]
    penetration = penetration[inside]
    if not len(nodes) or np.max(penetration) <= 0:
        env.deformations.pop(name, None)
        return None
    depth = 0.0075 if kind == "nib" else 0.005
    if np.max(penetration) > depth * 0.6:
        raise FloatingPointError(
            f"{kind} compression exceeds resolved small-strain material domain"
        )
    normal = rot.inv().apply([0, 1, 0])
    normals = np.tile(normal, (len(nodes), 1))
    world_com = q[:3] + rot.apply(env.com[env.body_ids[name]])
    positions = points.copy()
    positions[:, 1] = b.face_y
    velocity = qd[:3] + cross3(qd[3:], positions - world_com)
    body = env.body_ids[name]
    mass_value = float(env.model.body_mass.numpy()[body])
    M = np.zeros((6, 6))
    M[:3, :3] = mass_value * np.eye(3)
    M[3:, 3:] = rot.as_matrix() @ env.model.body_inertia.numpy()[body] @ rot.as_matrix().T
    base_wrench = np.zeros(6) if applied_wrench is None else np.array(applied_wrench, float)
    base_wrench[:3] += mass_value * np.asarray(env.solver.mj_model.opt.gravity)
    directions = np.broadcast_to([[1.0, 0, 0], [0, 0, 1.0]], (len(nodes), 2, 3))
    J = np.concatenate(
        (directions, cross3((positions - world_com)[:, None, :], directions)), axis=2
    )
    dt = env.config.physics_dt
    from .friction import implicit_regularized

    shear = np.zeros_like(s.rest)
    for iteration in range(30):
        u, elastic, quality = s.contact(nodes, normals, penetration, external_force=shear)
        loads = elastic @ normal
        weights = loads / max(float(loads.sum()), 1e-20)
        loads = np.maximum(0, loads - damping * velocity[:, 1] * weights)
        normal_forces = np.zeros_like(points)
        normal_forces[:, 1] = loads
        free_velocity = qd + dt * np.linalg.solve(
            M, base_wrench + _wrench(positions, normal_forces, world_com)
        )
        tangential_2d, friction_quality = implicit_regularized(
            free_velocity, M, J, friction * loads, dt
        )
        tangential = np.einsum("nij,ni->nj", directions, tangential_2d)
        candidate = np.zeros_like(s.rest)
        np.add.at(candidate, nodes, rot.inv().apply(tangential))
        residual = float(np.max(np.abs(candidate - shear)))
        if residual < 1e-7:
            break
        shear = candidate
    else:
        raise FloatingPointError("Friction/elastic contact fixed point did not converge")
    forces = tangential.copy()
    forces[:, 1] = loads
    positions = q[:3] + rot.apply(s.rest[nodes] + u[nodes])
    strain = s.strain(u)
    if np.max(np.abs(strain)) > 0.60:
        raise FloatingPointError(
            f"{kind} strain exceeds .60; material model cannot represent crushing/tearing"
        )
    env.deformations[name] = {
        "kind": kind,
        "u": u,
        "energy_j": s.energy(u),
        "max_strain": float(np.max(np.abs(strain))),
        "quality": quality,
        "friction": friction_quality,
        "board_reaction_world": -_wrench(positions, forces, np.zeros(3)),
    }
    load = float(loads.sum())
    if load <= 0:
        return None
    projected = rot.apply(np.asarray(axes))
    return {
        "wrench": _wrench(positions, forces, world_com),
        "load": load,
        "center": np.average(positions[:, [0, 2]], axis=0, weights=loads),
        "axes": np.array([[projected[0, 0], projected[1, 0]], [projected[0, 2], projected[1, 2]]]),
        "compression": float(np.max(penetration)),
        "tangential_state": None,
    }


def caps(env, q, qd, forces):
    s, inner = cap_surface()
    for color in env.ink_colors:
        name = "cap_" + color
        cap = env.body_ids[name]
        marker = env.body_ids["marker_" + color]
        if np.linalg.norm(q[cap, :3] - q[marker, :3]) > 0.16:
            env.deformations.pop(name, None)
            continue
        rc, rm = Rotation.from_quat(q[cap, 3:]), Rotation.from_quat(q[marker, 3:])
        positions = q[cap, :3] + rc.apply(s.rest[inner])
        local = rm.inv().apply(positions - q[marker, :3])
        r = np.linalg.norm(local[:, :2], axis=1)
        valid = (local[:, 2] >= -0.065) & (local[:, 2] <= 0.0555) & (r > 1e-6)
        nodes = inner[valid]
        positions = positions[valid]
        local = local[valid]
        r = r[valid]
        if not len(nodes) or np.max(0.007 - r) <= 0:
            env.deformations.pop(name, None)
            continue
        radial = np.c_[local[:, :2] / r[:, None], np.zeros(len(nodes))]
        world_normal = rm.apply(radial)
        cc = q[cap, :3] + rc.apply(env.com[cap])
        cm = q[marker, :3] + rm.apply(env.com[marker])
        # Implicit compliant impulse: C + dt^2 J M^-1 J^T. The resolved PP
        # material is too stiff for an explicit force evaluated at old poses.
        # This adds no physics steps and does not soften the material modulus.
        dt = env.config.physics_dt
        mass = env.model.body_mass.numpy()
        inertia = env.model.body_inertia.numpy()
        inverse = np.zeros((12, 12))
        for offset, body, rot in ((0, cap, rc), (6, marker, rm)):
            inverse[offset : offset + 3, offset : offset + 3] = np.eye(3) / mass[body]
            inverse[offset + 3 : offset + 6, offset + 3 : offset + 6] = (
                rot.as_matrix() @ np.linalg.inv(inertia[body]) @ rot.as_matrix().T
            )
        J = np.c_[
            world_normal,
            cross3(positions - cc, world_normal),
            -world_normal,
            -cross3(positions - cm, world_normal),
        ]
        velocity = np.concatenate((qd[cap], qd[marker]))
        u, force, quality = s.contact(
            nodes,
            rc.inv().apply(world_normal),
            0.007 - r - dt * (J @ velocity),
            mobility=(J, dt * dt * inverse),
            resolve_displacement=False,
            warm_start_nodes=env.deformations.get(name, {}).get("force_nodes"),
        )
        world_force = rc.apply(force)
        normal_load = np.linalg.norm(world_force, axis=1)
        normal_wrench = np.concatenate(
            (_wrench(positions, world_force, cc), _wrench(positions, -world_force, cm))
        )
        predicted = velocity + dt * (inverse @ normal_wrench)
        friction = np.zeros_like(world_force)
        friction_work = 0.0
        active = np.flatnonzero(normal_load > 0)
        normals = world_normal[active]
        reference = np.zeros_like(normals)
        reference[:, 2] = np.abs(normals[:, 2]) < 0.8
        reference[:, 0] = np.abs(normals[:, 2]) >= 0.8
        tangent0 = cross3(normals, reference)
        tangent0 /= np.linalg.norm(tangent0, axis=1)[:, None]
        directions = np.stack((tangent0, cross3(normals, tangent0)), axis=1)
        T = np.concatenate(
            (
                directions,
                cross3((positions[active] - cc)[:, None, :], directions),
                -directions,
                -cross3((positions[active] - cm)[:, None, :], directions),
            ),
            axis=2,
        )
        transpose = T.transpose(0, 2, 1)
        effective = T @ inverse @ transpose
        from .friction import solve

        impulses, (friction_work, iterations, maximum) = solve(
            predicted,
            T,
            effective,
            np.linalg.inv(effective),
            inverse @ transpose,
            0.45 * normal_load[active] * dt,
        )
        friction[active] = np.einsum("nij,ni->nj", directions, impulses) / dt
        world_force += friction
        # Apply exactly the same points used in J so net force/moment cancel.
        forces[cap] += _wrench(positions, world_force, cc)
        forces[marker] += _wrench(positions, -world_force, cm)
        env.deformations[name] = {
            "kind": "cap",
            "force_nodes": nodes[normal_load > 0],
            "force_local_n": force[normal_load > 0],
            "reconstruction": "u = K_free^-1 f; pinned P2 material, root displacement zero",
            "energy_j": quality.get("elastic_energy_j", 0.0),
            "quality": quality,
            "normal_load_n": float(normal_load.sum()),
            "friction_power_w": friction_work / dt,
            "friction_iterations": int(iterations),
            "friction_last_impulse_change_ns": float(maximum),
            "internal_force_residual_n": 0.0,
            "internal_moment_residual_nm": float(
                np.linalg.norm(
                    cross3(positions, world_force).sum(axis=0)
                    + cross3(positions, -world_force).sum(axis=0)
                )
            ),
        }


def displacement(s, record):
    if "u" in record:
        return np.asarray(record["u"])
    f = np.zeros_like(s.rest)
    if "force_nodes" in record:
        np.add.at(f, record["force_nodes"], record["force_local_n"])
    return s.displacement(f) if np.any(f) else f


def render_meshes(env, *, full=False):
    meshes = {}
    for obj in env.objects:
        kind = (
            "nib"
            if obj.name.startswith("marker_")
            else "cap"
            if obj.name.startswith("cap_")
            else "felt"
            if obj.name == "eraser"
            else None
        )
        if kind is None:
            continue
        s = tool_solid(kind)
        record = env.deformations.get(obj.name)
        if not hasattr(env, "_deformation_render_cache"):
            env._deformation_render_cache = {}
        cached = env._deformation_render_cache.get(obj.name)
        if cached is None or cached[0] is not record:
            u = displacement(s, record or {})
            env._deformation_render_cache[obj.name] = (record, u)
        else:
            u = cached[1]
        nodes = np.arange(len(s.rest)) if full else np.unique(s.faces)
        remap = np.full(len(s.rest), -1, int)
        remap[nodes] = np.arange(len(nodes))
        meshes[obj.name] = {"kind": kind, "vertices": (s.rest + u)[nodes], "faces": remap[s.faces]}
    return meshes
