"""Newton 1.5 builder of the driving-robot kit (catalogue model drive_kit) inside a host vehicle's articulation.

The host environment supplies the kit fit: a list of parts in the host's frame (the vehicle frame at its build pose),
each a rigid body with
  name, parent       a host body index (int) or an earlier part's name;
  anchor             the body's origin in the host frame (its joint's anchor for a moving part);
  joint              None: bolted to the parent (a fixed joint: the kit's frame, mounts, brackets and the ballast are
                     installed before the run, as a real kit's are); or dict(kind 'revolute' | 'prismatic', axis (host
                     frame), lower, upper, armature, friction, motor (True: a commanded actuator with POSITION_VELOCITY
                     targets that the host's motor model drives), ke, kd);
  mass, inertia      kg and the diagonal of the inertia about the body origin's axes (kg m^2); com (body frame, m);
  collide            end-effector colliders [dict(kind 'box' | 'sphere' | 'capsule', pos, quat (xyzw), size, label)]
                     in the body frame (the host frame's orientation, origin at the anchor);
  visual             drawn-only parts [dict(kind 'box' | 'cylinder' | 'ring', pos, quat, size, finish)] (no collision, no
                     mass); the finish (a name of the model's drawing finishes) ends the shape's label, for the renderer.
Every body frame has the host frame's orientation at the build pose. The host decides which shapes collide with
what (forklift: each end effector only with the control it works).

Returns the shared RobotRuntime of the instance (joints, frames, body and shape ids) and an info dict (per part: body,
joint, colliding shapes).
"""

import math

import numpy as np

from ...runtime import RobotRuntime


def _box_mesh(h):
    hx, hy, hz = h
    V = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
         (1, 5, 7), (1, 7, 3)]
    return V, np.asarray(F, np.int32).ravel()


def _ring_mesh(r_in, r_out, half_h, n=32):
    """Annulus about z (r_in = 0: a solid cylinder)."""
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    if r_in <= 0.0:
        V = np.concatenate([np.stack([r_out * c, r_out * s, np.full(n, z)], 1) for z in (-half_h, half_h)]
                           + [np.array([[0, 0, -half_h], [0, 0, half_h]])]).astype(np.float32)
        F = []
        for i in range(n):
            j = (i + 1) % n
            F += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
        return V, np.asarray(F, np.int32).ravel()
    rings = [np.stack([r * c, r * s, np.full(n, z)], 1) for r in (r_out, r_in) for z in (-half_h, half_h)]
    V = np.concatenate(rings).astype(np.float32)          # 0: out bottom, n: out top, 2n: in bottom, 3n: in top
    F = []
    for i in range(n):
        j = (i + 1) % n
        ob, ot, ib, it = 0, n, 2 * n, 3 * n
        F += [(ob + i, ob + j, ot + j), (ob + i, ot + j, ot + i),          # outer wall
              (ib + i, it + j, ib + j), (ib + i, it + i, it + j),          # inner wall
              (ot + i, ot + j, it + j), (ot + i, it + j, it + i),          # top
              (ob + i, ib + j, ob + j), (ob + i, ib + i, ib + j)]          # bottom
    return V, np.asarray(F, np.int32).ravel()


def _visual_mesh(newton, kind, size):
    if kind == "box":
        V, F = _box_mesh(size)
    elif kind == "cylinder":
        V, F = _ring_mesh(0.0, size[0], size[1])
    elif kind == "ring":
        V, F = _ring_mesh(size[0], size[1], size[2])
    else:
        raise ValueError(kind)
    return newton.Mesh(V, F)


def build(builder, *, instance, spec, setup_profile, parts, host_origin, host_quat, host_anchors, frames, shape_cfg):
    """Add the kit's parts to `builder`. host_origin, host_quat (xyzw): the host frame's world pose at build;
    host_anchors {host body index: its origin in the host frame}; frames {name: dict(part, pos, quat)} (body-frame
    poses of the kit's frames, e.g. the head camera); shape_cfg: callable(label) -> newton ShapeConfig of a collider."""
    import newton
    import warp as wp
    from scipy.spatial.transform import Rotation
    R0 = Rotation.from_quat(host_quat)
    o = np.asarray(host_origin, float)
    world = lambda p: o + R0.apply(np.asarray(p, float))
    qw = wp.quat(*map(float, host_quat))
    first_body, first_shape, first_joint = builder.body_count, builder.shape_count, builder.joint_count
    body, anchor, joints, info = {}, {}, {}, {}
    vis_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False,
                                              is_visible=True)
    for part in parts:
        name = part["name"]
        lab = f"{instance}/{name}"
        I = np.diag(np.asarray(part.get("inertia", (1e-4, 1e-4, 1e-4)), float))
        b = builder.add_link(xform=wp.transform(tuple(map(float, world(part["anchor"]))), qw), mass=float(part["mass"]),
                             com=tuple(map(float, part.get("com", (0.0, 0.0, 0.0)))), inertia=wp.mat33(*I.ravel()),
                             lock_inertia=True, label=lab)
        body[name], anchor[name] = b, np.asarray(part["anchor"], float)
        parent = part["parent"]
        if isinstance(parent, str):
            pb, pa = body[parent], anchor[parent]
        else:
            pb, pa = int(parent), np.asarray(host_anchors[int(parent)], float)
        rel = tuple(map(float, anchor[name] - pa))
        jd = part.get("joint")
        if jd is None:
            j = builder.add_joint_fixed(pb, b, parent_xform=wp.transform(rel, wp.quat_identity()),
                                        child_xform=wp.transform_identity(), label=lab)
        else:
            kw = dict(parent_xform=wp.transform(rel, wp.quat_identity()), child_xform=wp.transform_identity(),
                      axis=tuple(map(float, jd["axis"])), limit_lower=float(jd["lower"]), limit_upper=float(jd["upper"]),
                      armature=float(jd.get("armature", 0.0)), friction=float(jd.get("friction", 0.0)), label=lab,
                      collision_filter_parent=True)
            if jd.get("motor"):
                kw.update(actuator_mode=newton.JointTargetMode.POSITION_VELOCITY, target_ke=float(jd.get("ke", 0.0)),
                          target_kd=float(jd.get("kd", 0.0)), effort_limit=float(jd.get("effort", 1e6)))
            f = builder.add_joint_revolute if jd["kind"] == "revolute" else builder.add_joint_prismatic
            j = f(pb, b, **kw)
        joints[name] = j
        coll = []
        for c in part.get("collide", ()):
            tf = wp.transform(tuple(map(float, c["pos"])), wp.quat(*map(float, c.get("quat", (0, 0, 0, 1)))))
            cfg = shape_cfg(c["label"])
            if c["kind"] == "box":
                s = builder.add_shape_box(b, xform=tf, hx=float(c["size"][0]), hy=float(c["size"][1]),
                                          hz=float(c["size"][2]), cfg=cfg, label=f"{lab}:{c['label']}")
            elif c["kind"] == "sphere":
                s = builder.add_shape_sphere(b, xform=tf, radius=float(c["size"][0]), cfg=cfg, label=f"{lab}:{c['label']}")
            elif c["kind"] == "capsule":
                s = builder.add_shape_capsule(b, xform=tf, radius=float(c["size"][0]), half_height=float(c["size"][1]),
                                              cfg=cfg, label=f"{lab}:{c['label']}")
            else:
                raise ValueError(c["kind"])
            coll.append(s)
        for k, v in enumerate(part.get("visual", ())):
            tf = wp.transform(tuple(map(float, v["pos"])), wp.quat(*map(float, v.get("quat", (0, 0, 0, 1)))))
            builder.add_shape_mesh(b, xform=tf, mesh=_visual_mesh(newton, v["kind"], v["size"]), cfg=vis_cfg,
                                   label=f"{lab}:vis{k}:{v.get('finish', 'black')}")
        info[name] = dict(body=b, joint=j, colliders=coll, part=part)
    builder.add_articulation(list(range(first_joint, builder.joint_count)), label=instance)
    prof = spec["actuation"]["profiles"][setup_profile]["joints"]
    addresses = {}
    for name, j in joints.items():
        jd = info[name]["part"].get("joint")
        if jd is None:
            continue
        q, v = builder.joint_q_start[j], builder.joint_qd_start[j]
        addresses[f"{instance}/{name}"] = dict(
            q=q, dof=v, lower=float(jd["lower"]), upper=float(jd["upper"]),
            effort_limit=float(jd.get("effort", 0.0)), velocity_limit=float(jd.get("velocity", 0.0)),
            unit="rad" if jd["kind"] == "revolute" else "m", native_label=builder.joint_label[j],
            motor=bool(jd.get("motor")), product=prof.get(name, {}).get("product"))
    out_frames = {}
    for fname, fr in frames.items():
        b = body[fr["part"]]
        out_frames[fname] = dict(body=b, body_label=builder.body_label[b], position_m=list(map(float, fr["pos"])),
                                 orientation_wxyz=[float(fr["quat"][3]), *map(float, fr["quat"][:3])])
    groups = {g: {**e, "joints": [f"{instance}/{n}" for n in e["joints"]]} for g, e in spec.get("groups", {}).items()}
    runtime = RobotRuntime(instance, spec["id"], None, addresses, groups, out_frames,
                           tuple(range(first_body, builder.body_count)), tuple(range(first_shape, builder.shape_count)),
                           dict(path="robots/models/drive_kit/robot.json", builder=__name__),
                           metadata=dict(controller_profile=setup_profile, parameters={},
                                         initial_pose=spec["actuation"]["profiles"][setup_profile]["initial_pose"],
                                         body_labels=list(builder.body_label[first_body:]),
                                         shape_labels=list(builder.shape_label[first_shape:]), first_joint=first_joint))
    return runtime, info
