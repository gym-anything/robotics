"""Revision-pinned physical G1; no floor weld, balance oracle, or writing policy."""

from functools import lru_cache
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from .config import G1_URDF

EXPECTED_SHA256 = "555edb931dbe8780a82c778f7c7f53fe7f4f6d73c7e791845631cda9e8042302"


def reflected_inertia(rotor, gears):
    """Two-stage planetary output inertia J0*(N1*N2)^2 + J1*N2^2 + J2, kg m^2."""
    return rotor[0] * (gears[1] * gears[2]) ** 2 + rotor[1] * gears[2] ** 2 + rotor[2]


# Unitree's published G1 motor rotor/carrier/output inertias and stage ratios
# (unitreerobotics/unitree_rl_mjlab, src/assets/robots/unitree_g1/g1_constants.py,
# "Motor specs (from Unitree)"). Ankles and waist roll/pitch are linkages driven
# by two 5020 motors; Unitree approximates them as both armatures at 1:1.
MOTORS = {
    "5020": reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16)),
    "7520_14": reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22)),
    "7520_22": reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)),
    "4010": reflected_inertia((0.068e-4, 0.0, 0.0), (1, 5, 5)),
}
JOINT_MOTOR = {
    "hip_pitch": "7520_14",
    "hip_yaw": "7520_14",
    "waist_yaw": "7520_14",
    "hip_roll": "7520_22",
    "knee": "7520_22",
    "shoulder_pitch": "5020",
    "shoulder_roll": "5020",
    "shoulder_yaw": "5020",
    "elbow": "5020",
    "wrist_roll": "5020",
    "wrist_pitch": "4010",
    "wrist_yaw": "4010",
}
LINKAGE = ("ankle_pitch", "ankle_roll", "waist_pitch", "waist_roll")


def armature(name):
    """Reflected rotor inertia of a G1 body joint. Dex3 joints: no published value, 0."""
    key = name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    if key in LINKAGE:
        return 2 * MOTORS["5020"]
    return MOTORS[JOINT_MOTOR[key]] if key in JOINT_MOTOR else 0.0


@lru_cache(maxsize=1)
def velocity_limits():
    """Published joint speed limits (rad/s) of the pinned G1/Dex3 URDF."""
    root = ET.parse(G1_URDF).getroot()
    return {
        j.get("name"): float(j.find("limit").get("velocity"))
        for j in root.findall("joint")
        if j.get("type") != "fixed"
    }


def motor_physics(builder, joints):
    """Electric servo limits a free PD law omits, applied to every G1/Dex3 joint.

    A voltage-limited motor's driving torque falls linearly from its peak (effort)
    to zero at its no-load speed (the URDF velocity): the DC-motor envelope with
    stall torque equal to the effort limit. Back-EMF damping D = effort/velocity is
    a passive (implicit) joint damper and the clamped servo keeps kp and kd - D:
    torque = clamp(PD + D*w, +-effort) - D*w. Driving torque is bounded by
    effort - D*|w|, so no command drives a joint past its rated speed; braking
    may exceed the drive's current limit by D*|w|. Unsaturated damping stays kd.
    Where D exceeds kd (Dex3) servo damping is D instead. The exact negative
    servo velocity gain kd - D stays in MuJoCo's implicit linearization while
    Newton's joint-level clamp is active, so the clamped remainder is explicit:
    a 1.8e-5 kg m^2 finger link overshot to 48.7 rad/s (limit 12) that way.
    Reflected rotor inertia is joint armature.
    """
    for name, joint in joints.items():
        dof = joint["dof"]
        back_emf = builder.joint_effort_limit[dof] / velocity_limits()[name]
        builder.joint_damping[dof] = back_emf
        builder.joint_target_kd[dof] = max(builder.joint_target_kd[dof] - back_emf, 0.0)
        builder.joint_armature[dof] = armature(name)


# Unitree's MJCF sole for this G1 revision (unitree_rl_mjlab, src/assets/robots/
# unitree_g1/xmls/g1.xml, <side>_foot1..7_collision): seven 10 mm capsules
# (fromto, ankle-roll frame) covering the pinned foot mesh's footprint (x -0.066
# to 0.142 m, |y| <= 0.038 m). Their underside is the URDF sole plane, z=-0.035 m.
SOLE = {
    "left": (
        ((0.1, -0.026, -0.025), (0.05, -0.027, -0.025)),
        ((-0.044, -0.018, -0.025), (0.123, -0.018, -0.025)),
        ((-0.052, -0.01, -0.025), (0.13, -0.01, -0.025)),
        ((-0.054, 0, -0.025), (0.132, 0, -0.025)),
        ((-0.052, 0.01, -0.025), (0.13, 0.01, -0.025)),
        ((-0.044, 0.018, -0.025), (0.123, 0.018, -0.025)),
        ((0.1, 0.026, -0.025), (0.05, 0.026, -0.025)),
    ),
    "right": (
        ((0.1, -0.026, -0.025), (0.05, -0.026, -0.025)),
        ((-0.044, -0.018, -0.025), (0.123, -0.018, -0.025)),
        ((-0.052, -0.01, -0.025), (0.13, -0.01, -0.025)),
        ((-0.054, 0, -0.025), (0.132, 0, -0.025)),
        ((-0.052, 0.01, -0.025), (0.13, 0.01, -0.025)),
        ((-0.044, 0.018, -0.025), (0.123, 0.018, -0.025)),
        ((0.1, 0.026, -0.025), (0.05, 0.026, -0.025)),
    ),
}
SOLE_RADIUS = 0.01


def solid_soles(builder, wp, robot):
    """Replace the URDF's four 5 mm corner spheres per foot by the solid sole.

    The spheres were the whole foot: a link closing at ~10 m/s crosses a 5 mm
    sphere in one 0.5 ms tick and passes into it. The capsules keep the sole
    plane, friction and contact settings; they add no mass (density 0).
    """
    from newton import GeoType, ShapeFlags
    from scipy.spatial.transform import Rotation

    for side, segments in SOLE.items():
        body = next(
            b for b in robot.body_ids if builder.body_label[b].endswith(f"/{side}_ankle_roll_link")
        )
        spheres = [
            s for s in robot.shape_ids
            if builder.shape_body[s] == body and builder.shape_type[s] == GeoType.SPHERE
            and builder.shape_flags[s] & ShapeFlags.COLLIDE_SHAPES
        ]
        if len(spheres) != 4:
            raise RuntimeError(f"Expected the pinned URDF's four {side} sole spheres")
        cfg = builder.default_shape_cfg.copy()
        cfg.density, cfg.is_visible = 0.0, False
        cfg.mu, cfg.gap = builder.shape_material_mu[spheres[0]], builder.shape_gap[spheres[0]]
        for s in spheres:
            builder.shape_flags[s] &= ~int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)
        for i, (a, b) in enumerate(segments):
            a, b = np.asarray(a), np.asarray(b)
            axis = (b - a) / np.linalg.norm(b - a)
            q = Rotation.align_vectors([axis], [[0, 0, 1]])[0].as_quat()
            builder.add_shape_capsule(
                body,
                xform=wp.transform(wp.vec3(*((a + b) / 2)), wp.quat(*q)),
                radius=SOLE_RADIUS,
                half_height=float(np.linalg.norm(b - a) / 2),
                cfg=cfg,
                label=f"{side}_sole_{i + 1}",
            )


# Self-collision pairs two joints apart that Unitree's MJCF for this revision
# never brings into contact: it excludes pelvis/hip-roll and elbow/wrist-pitch,
# and gives hip-pitch, ankle-pitch, shoulder-pitch, shoulder-roll and wrist-roll
# links no collision geometry. Their URDF mesh hulls overlap across the
# intervening joint and held normal forces of up to 70 kN at ~1 mm depth for
# hundreds of ticks under legal commands. Everything else still collides.
UNITREE_NEIGHBOURS = (
    ("pelvis", "{side}_hip_roll_link"),
    ("{side}_elbow_link", "{side}_wrist_pitch_link"),
    ("{side}_hip_pitch_link", "{side}_hip_yaw_link"),
    ("{side}_hip_yaw_link", "{side}_ankle_pitch_link"),
    ("torso_link", "{side}_shoulder_roll_link"),
    ("{side}_shoulder_pitch_link", "{side}_shoulder_yaw_link"),
    ("{side}_shoulder_roll_link", "{side}_elbow_link"),
    ("{side}_shoulder_yaw_link", "{side}_wrist_roll_link"),
    ("{side}_wrist_roll_link", "{side}_wrist_yaw_link"),
)


def neighbour_exclusions(builder, robot):
    from newton import ShapeFlags

    def shapes(link):
        body = next(b for b in robot.body_ids if builder.body_label[b].endswith("/" + link))
        found = [
            s for s in range(builder.shape_count)
            if builder.shape_body[s] == body and builder.shape_flags[s] & ShapeFlags.COLLIDE_SHAPES
        ]
        if not found:
            raise RuntimeError(f"No collision geometry on {link}")
        return found

    for side in ("left", "right"):
        for a, b in UNITREE_NEIGHBOURS:
            for sa in shapes(a.format(side=side)):
                for sb in shapes(b.format(side=side)):
                    builder.add_shape_collision_filter_pair(sa, sb)


def add_robot(builder, wp, layout=None):
    from robots.backends.newton import add_robot as build, mount
    from .config import SceneLayout
    layout=layout or SceneLayout()
    robot = build(builder,'g1','whiteboard',instance='robot',prefix='',asset_path=G1_URDF,
        placement=mount((layout.robot_x,layout.robot_y,.79),layout.robot_yaw_rad))
    motor_physics(builder, robot.joints)
    solid_soles(builder, wp, robot)
    neighbour_exclusions(builder, robot)
    return robot.metadata['legacy_addresses']
