"""Newton 1.5 composition, preserving each explicit controller profile.

No clock, solver, world objects, renderer or task policy is owned here.
"""

import copy
import hashlib
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from ..catalog import model as definition, resolve
from ..runtime import RobotRuntime, RobotRig
from .. import assets


PANDA_HAND_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")


def _panda_without_hand(path):
    """The pinned panda_arm_hand.urdf without the Franka Hand (its links and the joints to them), the mesh URIs made
    absolute (the result is parsed from a string, so package:// cannot resolve against the file). Bytes."""
    xml = ET.parse(path).getroot()
    for joint in list(xml.findall("joint")):
        if joint.find("child").get("link") in PANDA_HAND_LINKS:
            xml.remove(joint)
    for link in list(xml.findall("link")):
        if link.get("name") in PANDA_HAND_LINKS:
            xml.remove(link)
    package = Path(path).resolve().parent.parent
    for mesh in xml.iter("mesh"):
        name = mesh.get("filename", "")
        if name.startswith("package://"):
            mesh.set("filename", str(package / name[len("package://"):].split("/", 1)[1]))
    return ET.tostring(xml)


def mount(position, yaw=0.0):
    import warp as wp

    # Preserve the legacy native float32 axis-angle conversion exactly.
    q = list(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw))
    return {"position_m": list(position), "orientation_wxyz": [q[3], *q[:3]]}


def _native_pose(value, wp):
    from ..spec import pose

    value = pose(value)
    q = value["orientation_wxyz"]
    return wp.transform(value["position_m"], wp.quat(*q[1:], q[0]))


def _frame(parent, xml, bodies, prefix, wp, world):
    """Walk only collapsed fixed ancestors; retain their exact authored transform."""
    if parent not in {node.get("name") for node in xml.findall("link")}:
        raise ValueError(f"Frame references a missing authored link: {parent}")
    local = wp.transform_identity()
    by_child = {j.find("child").get("link"): j for j in xml.findall("joint")}

    def matching(name):
        return [
            label for label in bodies if label == name or label.endswith("/" + name)
        ]

    while not matching(parent):
        joint = by_child.get(parent)
        if joint is None:
            local = wp.transform_multiply(world, local)
            q = list(wp.transform_get_rotation(local))
            return {
                "body": -1,
                "body_label": None,
                "position_m": list(wp.transform_get_translation(local)),
                "orientation_wxyz": [q[3], *q[:3]],
            }
        if joint.get("type") != "fixed":
            raise ValueError(f"Cannot resolve physical frame {prefix + parent}")
        origin = joint.find("origin")
        xyz = [
            float(x)
            for x in (
                origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            ).split()
        ]
        rpy = [
            float(x)
            for x in (
                origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            ).split()
        ]
        local = wp.transform_multiply(wp.transform(xyz, wp.quat_rpy(*rpy)), local)
        parent = joint.find("parent").get("link")
    q = list(wp.transform_get_rotation(local))
    labels = matching(parent)
    if len(labels) != 1:
        raise ValueError(f"Ambiguous physical frame {parent}")
    return {
        "body": bodies[labels[0]],
        "body_label": labels[0],
        "position_m": list(wp.transform_get_translation(local)),
        "orientation_wxyz": [q[3], *q[:3]],
    }


def add_robot(
    builder,
    model_id,
    profile_name,
    *,
    instance,
    placement,
    prefix=None,
    shape_config=None,
    sdf=False,
    parameters=None,
    initial_pose=None,
    asset_path=None,
    isolated=False,
):
    import newton
    import warp as wp
    from newton.solvers import SolverMuJoCo

    if newton.__version__ != "1.5.0":
        raise RuntimeError(
            f"Shared robot adapter requires Newton 1.5.0; found {newton.__version__}"
        )
    if instance in getattr(builder, "_gar_robots", {}):
        raise ValueError(f"Duplicate robot instance: {instance}")
    spec = definition(model_id)
    data = spec.to_dict()
    if data["asset"]["format"] != "urdf":
        raise ValueError("Newton shared builder currently supports URDF components")
    profile = copy.deepcopy(data["actuation"]["profiles"][profile_name])
    if not newton.use_coord_layout_targets:
        raise RuntimeError(
            "Shared robots require newton.use_coord_layout_targets = True before "
            "creating the ModelBuilder; legacy DOF-shaped position targets are "
            "not supported. The adapter does not change global engine settings."
        )
    settings = parameters or {}
    allowed = {"finger_effort_limit", "finger_velocity_limit", "finger_kp"}
    if set(settings) - allowed:
        raise ValueError(
            f"Unsupported explicit robot settings: {sorted(set(settings) - allowed)}"
        )
    for key, value in settings.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Positive finite robot setting required: {key}")
    path = assets.resolve(spec.id, path=asset_path)
    source = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    tool = profile.get("default_tool")
    bare_xml = None
    if tool == "wsg50":
        from ..attachments.wsg50.geometry import generated_urdf

        payload, source = generated_urdf(spec.id)
    elif spec.id == "panda" and tool is None:
        # a Panda profile without a tool (a torch mounted by the environment on link8): the locked package has only
        # panda_arm_hand.urdf, so the Franka Hand's links and joints are removed in memory (the FR3 has a bare-arm file)
        payload = _panda_without_hand(path)
        bare_xml = payload.decode()
        source = dict(source, derived="panda_arm_hand.urdf without panda_hand and its fingers")
    else:
        if spec.id == "fr3" and tool == "franka-hand":
            # a profile without a tool (e.g. a torch mounted by the environment on link8) keeps the bare-arm URDF
            path = path.with_name("fr3_franka_hand.urdf")
            source = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        payload = path.read_bytes()
    xml = ET.fromstring(payload)
    # Continuous joints (wheels, steering, casters of a mobile base) are actuated or passive DOFs like any revolute.
    active = {
        j.get("name"): j
        for j in xml.findall("joint")
        if j.get("type") in ("revolute", "prismatic", "continuous")
    }
    if set(profile["joints"]) != set(active):
        raise ValueError(
            "Controller joint inventory does not match the assembled robot"
        )
    if settings and not any("finger" in name for name in active):
        raise ValueError(
            "Finger overrides supplied to a robot with no parallel-jaw fingers"
        )
    initial = data["poses"][initial_pose or profile["initial_pose"]]
    if set(initial) != set(active):
        raise ValueError("Initial pose joint inventory mismatch")
    prefix = instance if prefix is None else prefix
    label_prefix = prefix + "/" if prefix else ""
    xform = _native_pose(placement, wp)
    direct = bool(profile.get("inherit_shape_config")) and not isolated
    if isolated and profile.get("inherit_shape_config") and shape_config is None:
        shape_config = builder.default_shape_cfg
    local = (
        builder
        if direct
        else newton.ModelBuilder(
            gravity=(0, 0, -9.80665)
            if profile_name in ("deposit_cans", "wiping")
            else (0, 0, -9.81)
        )
    )
    if not direct:
        SolverMuJoCo.register_custom_attributes(local)
        if shape_config is not None:
            local.default_shape_cfg = copy.deepcopy(shape_config)
        if profile.get("default_gap") is not None:
            local.default_shape_cfg.gap = profile["default_gap"]
        if sdf:
            local.default_shape_cfg.configure_sdf(max_resolution=128, force_sdf=True)
    qb, vb, bb, sb, jb = (
        len(builder.joint_q),
        len(builder.joint_qd),
        builder.body_count,
        builder.shape_count,
        builder.joint_count,
    )
    first = local.joint_count
    kwargs = dict(
        floating=data["root"]["mounting"] == "floating",
        enable_self_collisions=True,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    if not direct:
        kwargs["parse_visuals_as_colliders"] = False
    if direct or profile.get("transform_stage") == "import":
        kwargs["xform"] = xform
    if direct and prefix:
        raise ValueError(
            "Direct compatibility import is unprefixed; use isolated composition for named instances"
        )
    if tool == "wsg50":
        with tempfile.TemporaryDirectory(prefix="gar-robot-") as tmp:
            generated = Path(tmp) / "robot.urdf"
            generated.write_bytes(payload)
            local.add_urdf(generated, **kwargs)
    elif bare_xml is not None:
        local.add_urdf(bare_xml, **kwargs)
    else:
        local.add_urdf(str(path), **kwargs)
    addresses, legacy = {}, {}
    for j in range(first, local.joint_count):
        short = local.joint_label[j].split("/")[-1]
        if short not in active:
            continue
        q, v = local.joint_q_start[j], local.joint_qd_start[j]
        joint = active[short]
        limit = joint.find("limit").attrib
        values = dict(profile["joints"][short])
        finger = "finger" in short
        if finger:
            for param, key in [
                ("finger_effort_limit", "effort_limit"),
                ("finger_velocity_limit", "velocity_limit"),
                ("finger_kp", "kp"),
            ]:
                if param in settings:
                    values[key] = settings[param]
        lower, upper = float(limit.get("lower", "-inf")), float(limit.get("upper", "inf"))
        if not lower <= initial[short] <= upper:
            raise ValueError(f"Invalid initial joint position: {short}")
        local.joint_q[q] = local.joint_target_q[q] = initial[short]
        for key, field in [
            ("kp", "joint_target_ke"),
            ("kd", "joint_target_kd"),
            ("effort_limit", "joint_effort_limit"),
            ("limit_kp", "joint_limit_ke"),
            ("limit_kd", "joint_limit_kd"),
        ]:
            if key in values:
                getattr(local, field)[v] = values[key]
        local.joint_target_mode[v] = int(newton.JointTargetMode.POSITION_VELOCITY)
        if profile.get("actuator_gravity_compensation"):
            attr = local.custom_attributes["mujoco:jnt_actgravcomp"]
            if attr.values is None:
                attr.values = {}
            attr.values[v] = True
        name = label_prefix + short
        entry = dict(
            q=q + (0 if direct else qb),
            dof=v + (0 if direct else vb),
            lower=lower,
            upper=upper,
            effort_limit=local.joint_effort_limit[v],
            velocity_limit=values.get(
                "velocity_limit", float(limit.get("velocity", "inf"))
            ),
        )
        legacy[name] = dict(entry)
        if profile_name in ("whiteboard", "football"):
            legacy[name].pop("velocity_limit")
            legacy[name]["effort_nm"] = legacy[name].pop("effort_limit")
            if profile_name == "football":
                legacy[name].update(kp=values["kp"], kd=values["kd"])
        addresses[name] = dict(
            entry,
            unit="m" if joint.get("type") == "prismatic" else "rad",
            native_label=(
                local.joint_label[j] if direct else label_prefix + local.joint_label[j]
            ),
        )
    if profile.get("gravity_compensation"):
        local.custom_attributes["mujoco:gravcomp"].values = {
            i: 1.0 for i in range(local.body_count)
        }
    if not direct:
        for i, body in enumerate(local.shape_body):
            if profile.get("all_shape_gap") is not None:
                local.shape_gap[i] = profile["all_shape_gap"]
            if profile.get("zero_rolling_torsional"):
                local.shape_material_mu_rolling[i] = 0.0
                local.shape_material_mu_torsional[i] = 0.0
            if body >= 0 and "finger" in local.body_label[body]:
                for key, value in profile.get("finger_material", {}).items():
                    getattr(local, "shape_material_" + key)[i] = value
                if profile.get("finger_solimp") is not None:
                    attr = local.custom_attributes["mujoco:geom_solimp"]
                    if attr.values is None:
                        attr.values = {}
                    attr.values[i] = attr.dtype(*profile["finger_solimp"])
        compose = {} if profile.get("transform_stage") == "import" else {"xform": xform}
        if prefix:
            compose["label_prefix"] = prefix
        builder.add_builder(local, **compose)
    bodies = {builder.body_label[i]: i for i in range(bb, builder.body_count)}
    frames = {}
    for name, frame in data.get("frames", {}).items():
        entry = _frame(frame["parent"], xml, bodies, label_prefix, wp, xform)
        authored = _native_pose({k: v for k, v in frame.items() if k != "parent"}, wp)
        base = _native_pose(
            {k: v for k, v in entry.items() if k in ("position_m", "orientation_wxyz")},
            wp,
        )
        combined = wp.transform_multiply(base, authored)
        q = list(wp.transform_get_rotation(combined))
        entry.update(
            position_m=list(wp.transform_get_translation(combined)),
            orientation_wxyz=[q[3], *q[:3]],
        )
        frames[name] = entry
    if tool:
        tcp = _frame(
            "wsg_base"
            if tool == "wsg50"
            else "panda_hand"
            if spec.id == "panda"
            else spec.id + "_hand_tcp",
            xml,
            bodies,
            label_prefix,
            wp,
            xform,
        )
        offset_z = 0.142 if tool == "wsg50" else 0.1034 if spec.id == "panda" else 0.0
        if offset_z:
            # Tip-pad center, same authored TCP used by the original environment.
            # Panda's URDF has no hand_tcp link; retain the existing 103.4 mm
            # hand offset from laundry/pencap/lockkey kinematics.
            q = tcp["orientation_wxyz"]
            offset = wp.quat_rotate(wp.quat(*q[1:], q[0]), wp.vec3(0, 0, offset_z))
            tcp["position_m"] = [
                a + float(b) for a, b in zip(tcp["position_m"], offset)
            ]
        frames["tcp"] = tcp
    groups = {
        name: {**entry, "joints": [label_prefix + n for n in entry["joints"]]}
        for name, entry in data.get("groups", {}).items()
    }
    fingers = [name for name in addresses if "finger" in name]
    if fingers:
        groups["hand"] = {"joints": fingers, "role": "parallel_jaw"}
    runtime = RobotRuntime(
        instance,
        spec.id,
        spec.fingerprint,
        addresses,
        groups,
        frames,
        tuple(range(bb, builder.body_count)),
        tuple(range(sb, builder.shape_count)),
        source,
        metadata={
            "controller_profile": profile_name,
            "parameters": settings,
            "initial_pose": initial_pose or profile["initial_pose"],
            "legacy_addresses": legacy,
            "urdf": payload,
            "body_labels": list(builder.body_label[bb:]),
            "shape_labels": list(builder.shape_label[sb:]),
            "first_joint": jb,
        },
    )
    handles = getattr(builder, "_gar_robots", {})
    if instance in handles:
        raise ValueError(f"Duplicate robot instance: {instance}")
    handles[instance] = runtime
    builder._gar_robots = handles
    return runtime


def add_to(builder, setup, *, mounts):
    setup = resolve(setup)
    data = setup.to_dict()
    # Preflight every component before building anything. Unsupported mechanisms
    # are explicit errors, never silently omitted accessories.
    for name, item in data["instances"].items():
        if "mount" in item and item["mount"] not in mounts:
            raise ValueError(f"Missing mount: {item['mount']}")
        profile = definition(item["model"]).to_dict()["actuation"]["profiles"][
            item["controller_profile"]
        ]
        if name in getattr(builder, "_gar_robots", {}):
            raise ValueError(f"Duplicate robot instance: {name}")

        children = [
            child
            for child in data.get("attachments", {}).values()
            if child["parent"].split("/")[0] == name
        ]
        expected = profile.get("default_tool")
        if (
            len(children) != (1 if expected else 0)
            or expected
            and children[0]["model"].split("@")[0] != expected
        ):
            raise ValueError(
                f"{name}: profile requires exactly its physical {expected or 'no'} tool"
            )
        if children and children[0]["parent"] != name + "/flange":
            raise ValueError("Tool must mount to the declared flange")
        if expected == "wsg50":
            raise ValueError(
                "WSG50 requires the environment native bounded-drive belt mechanism; use its integrated environment adapter"
            )
    instances = {}
    for name, item in data["instances"].items():
        instances[name] = add_robot(
            builder,
            item["model"],
            item["controller_profile"],
            instance=name,
            placement=mounts[item["mount"]] if "mount" in item else item["pose"],
            parameters=item.get("parameters"),
            initial_pose=item.get("initial_pose"),
            isolated=True,
        )
    return RobotRig(setup, instances)


def remount(builder, runtime, parent_body, parent_world_xform):
    """Ride a fixed-mounted catalogue robot on an existing body (a vehicle cab) instead of the world frame.

    `add_robot` composes a fixed robot welded to the world: its base link collapses into the world and its first
    joint's parent is -1. This re-parents those joints and the collapsed base shapes onto `parent_body` (given its
    world transform at build time) and merges the robot's articulation into the parent's, which must be the most
    recently added articulation before the robot (Newton's articulations must be contiguous). Geometry, masses and
    controller settings are unchanged; the robot becomes part of the vehicle's kinematic tree, so the vehicle's
    accelerations act on it physically.
    """
    import warp as wp

    inv = wp.transform_inverse(parent_world_xform)
    bodies = set(runtime.body_ids)
    first = runtime.metadata["first_joint"]
    moved = 0
    for j in range(first, builder.joint_count):
        if builder.joint_parent[j] == -1 and builder.joint_child[j] in bodies:
            builder.joint_parent[j] = parent_body
            builder.joint_X_p[j] = wp.transform_multiply(inv, builder.joint_X_p[j])
            moved += 1
    for s in runtime.shape_ids:
        if builder.shape_body[s] == -1:
            builder.shape_body[s] = parent_body
            builder.shape_transform[s] = wp.transform_multiply(inv, builder.shape_transform[s])
            builder.body_shapes.setdefault(-1, [])
            if s in builder.body_shapes[-1]:
                builder.body_shapes[-1].remove(s)
            builder.body_shapes.setdefault(parent_body, []).append(s)
    # merge: drop articulation entries that start inside the robot's joints
    # (several instances composed in one add_to call are merged by the first remount; later calls find none left)
    keep = [k for k, start in enumerate(builder.articulation_start) if start < first]
    n_art = len(builder.articulation_start)
    names = [a for a in vars(builder) if a.startswith("articulation_") and isinstance(getattr(builder, a), list)
             and len(getattr(builder, a)) == n_art and a != "articulation_world_start"]
    for name in names:
        values = getattr(builder, name)
        setattr(builder, name, [values[k] for k in keep])
    if keep and "articulation_end" in names:
        # the parent articulation now ends where the robot's last joint ends
        builder.articulation_end[keep[-1]] = builder.joint_count
    if moved == 0:
        raise ValueError("remount: no world-parented robot joint found")
    runtime.metadata["mounted_on_body"] = int(parent_body)
    return moved
