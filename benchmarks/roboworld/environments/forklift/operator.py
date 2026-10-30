"""The seated operator: a floating humanoid from the shared robots catalog with the model's `forklift` controller profile,
seated on the truck's suspension seat by contact only — no weld, no fixed base — and held in the seat by its own weight,
the cushion's friction, its posture and the lap belt. Rig `forklift-h1-2` (the Unitree H1-2 with Inspire hands) is the
robot offered (robot_setup.ALIASES); rig `forklift-g1` (the Unitree G1 with Dex3 hands) seats the same way for probes
and is refused as an operator with probe E14g's measurements (robots/catalog.py ENVIRONMENTS['forklift']).

Everything robot-specific comes from the catalogue model (robots/models/<model>/robot.json): its root frame, the hand,
foot and head-camera frames, its groups' roles, the `forklift` controller profile (gains, limits; optional per-joint
reflected rotor inertia `armature`) and its start pose, plus the kinematic tree of its URDF (Skeleton below: which joints
are the legs, the trunk, the arms and the hands, and which links are the thighs, the shanks, the palms, the thumbs). The
few seat-fit values that are measured per model and cannot be derived (the lap belt's wrap points on the pelvis, the
collision pairs filtered as convex-hull artifacts, the thumb's rest angle, where the hands rest on the thighs) are in
SEAT_FIT, keyed by catalogue model id; a humanoid without an entry is not seated.

Start posture (every task): buttocks on the cushion and back on the backrest, hands resting on the thighs, feet flat on
the footwell floor plate (right foot beside the accelerator, left foot left of the inching pedal). It is solved by
inverse kinematics on MuJoCo's own geometry at construction, placed with a small clearance, and settled by physics
before tick 0 (world.settle in the task builder); nothing holds the robot afterwards.

Lap belt: two webbing halves from anchors on the seat frame beside the hips to wrap points on the front
of the pelvis, tension-only springs of polyester webbing. The emergency-locking retractor (UN R16 type 4, multiple
sensitivity) pays out and retracts freely against its retraction spring (BELT_RETRACT_N) until one of its two sensors
trips: the vehicle sensor, a pendulum on the seat, when the apparent gravity in the seat frame (specific force: the
truck's acceleration and gravity) tilts more than atan(BELT lock_g) from the seat's vertical, which is a deceleration of
lock_g on level ground and a static tilt of the same angle (R16 6.2.5.3.1.1, .3, .4); the webbing sensor when the
webbing's payout acceleration exceeds BELT_WEBBING_LOCK_G (R16 6.2.5.3.2), sensed through BELT_SENSOR_TAU. Locked, the
webbing holds; the pawl releases once both sensors have reset and the webbing is no longer loaded. Forces act on the
pelvis and, equal and opposite, on the seat body.
"""

import math

import numpy as np

from . import params as P
from .mathx import quat_to_R, cross3
from .motors import Motors, ARMATURE, GAIN_BOUNDS

PELVIS_TRUCK = (-1.00, 0.04, 1.37)        # build-time placement (truck frame); refined by start()
SEAT_CLEARANCE = 0.003                    # m above the cushion at construction
HAND_REST_BACK = -0.10                    # m: palm target behind the knee (at -0.16 the thumbs settled onto the robot's
HAND_REST_CLEAR = 0.025                   # own hip links at 6-13 N and stick-slipped; the start state never settled)
START_LEAN = math.radians(5.0)            # forward lean of the H1-2's start posture: pelvis and trunk pitched about the hips
# (the H1-2 has no waist pitch), thighs kept level. Authored from the cab's geometry, not measured on a driver. With the
# overhead guard at the sheet's h6 (params.GUARD_RAISE) an upright H1-2's head sits 27 mm under the rear slat
# (guard_slat_c, chassis x -0.997..-0.919) and straddles its front edge, and a hard brake application lifts the seated
# operator ~32 mm on the suspension (the pedal force's reaction unloads it), so the head met the slat at 2.8-3.1 kN in
# the E13 stop. A tall driver under a low guard leans forward: at 5 deg as built (8.9 deg once settled on the robot's
# default gains) the head's top 10 cm lie in the guard's open bay (settled x -0.843..-0.691: 76 mm in front of the slat,
# 167 mm behind the front bar) and the E13 stop and ride meet the guard with 0 N. A forward trunk inclination of 0-20 deg
# is the "optimal" band of ISO 11226 / EN 1005-4, a posture that may be held for a long time (read: Continental AG,
# ergonomics requirement specification for machine manufacturers, 2024, p. 3, after BGIA-Report 2/2007)
BACK_CLEARANCE = 0.004
SOLE_OFFSET = 0.010                       # sole target above the visual floor plate: the cab collider (convex decomposition
                                          # of the mesh) lies up to ~6 mm above it; the clearance pairs keep it out
BELT_RETRACT_N = P.BELT_RETRACT_N         # retractor's retraction force (UN R16 lap-belt minimum)
BELT_LOCK_TILT = math.atan(P.BELT["lock_g"])   # vehicle sensor: pendulum trip angle of the apparent gravity (24.2 deg)
BELT_WEBBING_LOCK_G = P.BELT_WEBBING_LOCK_G  # webbing sensor: payout acceleration that locks (g)
BELT_SENSOR_TAU = P.BELT_SENSOR_TAU       # webbing sensor response time (s)
BELT_ANCHOR_SEAT = P.BELT_ANCHOR_SEAT     # seat-body frame offsets of the anchors (left, right)
MU_ROBOT_SHELL = P.MU_ROBOT_SHELL         # contact friction of the robot's links (not the hands)
MU_ROBOT_PAD = P.MU_ROBOT_PAD             # finger and palm pads: sliding mu and torsional friction
BELT_WRAP_PELVIS = ((0.10, 0.13, -0.12), (0.10, -0.13, -0.12))   # H1-2 pelvis frame (front of the hips, left, right)

# The robots this module builds, by public name: every offered robot (robot_setup.ALIASES, from the catalogue's
# ENVIRONMENTS['forklift'] aliases) and the probe-only G1 (seated for the E14g probes; refused as an operator).
# Seated humanoids (RIGS) are those whose catalogue model has a SEAT_FIT entry; the driving-robot kit (KIT_RIGS) is built
# by the kit fit below. A further humanoid that fits the seat needs a SEAT_FIT entry and a catalogue alias, nothing else.
PROBE_RIGS = {"g1": "forklift-g1"}
KIT_MODEL = "drive_kit"


def _rig_model(rig):
    import robots
    models = {i["model"].split("@")[0] for i in robots.resolve(rig).to_dict()["instances"].values()}
    if len(models) != 1:
        raise ValueError(f"forklift builds one robot per rig, not {sorted(models)} ({rig})")
    return models.pop()


def _all_rigs():
    from .robot_setup import ALIASES
    return dict(ALIASES, **PROBE_RIGS)

# Seat fit per catalogue model: the values measured on that robot in the seat that its catalogue metadata cannot give.
#   belt_wrap_pelvis  lap belt wrap points (pelvis frame: front of the hips, left, right)
#   filtered          collision pairs filtered as convex-hull artifacts (Skeleton link sets; "thumbK": the thumb chain's
#                     K-th link from the palm), each measured: hulls in contact in the seated posture, meshes apart
#   thumb_rest        angle of the thumb's base joint in the start posture (rotated in, clear of the thighs), or None
#   hand_rest_back, hand_rest_clear   palm target behind the knee and its clearance from the legs (start posture)
#   feet_y            the soles' lateral targets in the truck frame (left, right)
#   name, hands       how the policy's notes name the robot and its hands
SEAT_FIT = {
    "h1_2": dict(
        belt_wrap_pelvis=BELT_WRAP_PELVIS,
        # The H1-2's thigh links against its pelvis and torso: MuJoCo collides the convex hulls of the link meshes, and
        # in the seated hip flexion the right thigh's hull (4790 cm3 around a concave link) met the pelvis hull with
        # 1.36 kN when the leg turned inboard toward the brake pedal, while the real meshes stayed 9.5 mm apart with no
        # point of either inside the other (E14 self-collision probe). The Inspire thumb's proximal link sits in the
        # palm: its hull overlaps the palm's (wrist_yaw_link) hull in every thumb pose, so the pair pushed the thumb with
        # ~13 N and it chattered at 20-95 mm/s against the thigh at rest (E0). The legs still collide with each other,
        # the seat and the cab; the thumb's tip links still touch the palm, the knob and the controls.
        filtered=(("trunk", "thigh"), ("palm", "thumb1")),
        thumb_rest=0.35, hand_rest_back=HAND_REST_BACK, hand_rest_clear=HAND_REST_CLEAR, feet_y=(0.20, -0.17),
        name="H1-2", hands="Inspire hands"),
    "g1": dict(
        # wrap points: estimate, the H1-2's scaled to the G1's hips (lateral 0.8 x the hip-roll joint's 0.116 m offset,
        # vertical 0.74 x the hip joints' 0.103 m drop, 8 cm in front of the pelvis origin); not tuned on a ride
        belt_wrap_pelvis=((0.08, 0.09, -0.08), (0.08, -0.09, -0.08)),
        # no hull artifact (measured in the settled seated pose: pelvis and hip-roll hulls 3.1-3.4 mm apart with the
        # meshes 2.8-3.2 mm apart; the resting thumb tips touch the thighs at 1.2-4.2 N with the meshes 0.1-0.2 mm apart:
        # real contacts, kept)
        filtered=(),
        thumb_rest=None, hand_rest_back=HAND_REST_BACK, hand_rest_clear=HAND_REST_CLEAR, feet_y=(0.20, -0.17),
        # upright: START_LEAN is the H1-2's (its head under the overhead guard); the 1.32 m G1 sits far below the guard
        lean=0.0,
        name="G1", hands="Dex3-1 hands"),
}

# public name -> catalogue rig: the seated humanoids (a SEAT_FIT model) and the driving-robot kit
RIGS = {a: r for a, r in _all_rigs().items() if _rig_model(r) in SEAT_FIT}
KIT_RIGS = {a: r for a, r in _all_rigs().items() if _rig_model(r) == KIT_MODEL}

SIDES = ("left", "right")
LEGACY_HAND_KEYS = ("wrist", "thumb", "index", "middle", "ring", "pinky")     # seated_posture: see there


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


class Skeleton:
    """The seated humanoid's anatomy from its catalogue model and its URDF tree; nothing here names a robot's links.

    legs[side]   the movable joints from the root to the foot frame's link (six: three hip joints, knee, two ankle);
    torso        the joints that carry both hands (H1-2: torso_joint; G1: waist yaw, roll, pitch);
    arms[side]   the rest of the chain to the hand frame's link (seven: three shoulder joints, elbow, three wrist);
    hands[side]  the movable joints below the hand frame's link (the fingers); thumb[side]: the thumb's chain.
    The hip, ankle joints are told apart by their axes in the root frame at the zero pose (pitch: lateral, roll:
    forward, yaw: vertical). Link sets (link names; body() gives the body a link was merged into): trunk (the root and
    the torso chain's links), chest (the link that carries the arms and the head), thigh (the two leg links between the
    first hip link and the knee), knee, shank (knee and ankle links), wrist, palm (the hand frame's body), fingers,
    thumbK (the thumb chain's K-th link). The catalogue groups' roles (legs, torso, arm, hand) are checked against it.
    """

    def __init__(self, spec, urdf, body_links):
        import xml.etree.ElementTree as ET
        x = ET.fromstring(urdf)
        self.id = spec["id"]
        self.joint, self.by_child = {}, {}
        self.order = []
        for j in x.findall("joint"):
            o = j.find("origin")
            ax = j.find("axis")
            rec = dict(type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                       axis=np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]),
                       rpy=[float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()])
            self.joint[j.get("name")] = rec
            self.by_child[rec["child"]] = j.get("name")
            if rec["type"] in ("revolute", "prismatic", "continuous"):
                self.order.append(j.get("name"))
        frames = spec.get("frames", {})
        need = [f"{s}_{k}" for k in ("hand", "foot") for s in SIDES] + ["head_camera"]
        missing = [k for k in need if k not in frames]
        if missing:
            raise ValueError(f"robot model {self.id!r} has no {', '.join(missing)} frame(s): the seated operator needs "
                             "both hands, both feet and a head camera")
        self.root = spec["root"]["frame"]
        self.body_links = set(body_links)
        self.hand_link = {s: frames[f"{s}_hand"]["parent"] for s in SIDES}
        self.foot_link = {s: frames[f"{s}_foot"]["parent"] for s in SIDES}
        self.camera_link = frames["head_camera"]["parent"]
        self.legs = {s: self._chain(self.foot_link[s]) for s in SIDES}
        to_hand = {s: self._chain(self.hand_link[s]) for s in SIDES}
        self.torso = [j for j in to_hand["left"] if j in to_hand["right"]]
        self.arms = {s: [j for j in to_hand[s] if j not in self.torso] for s in SIDES}
        self.hands = {s: [j for j in self.order if self._below(self.joint[j]["parent"], self.hand_link[s])] for s in SIDES}
        for s in SIDES:
            if len(self.legs[s]) != 6 or len(self.arms[s]) != 7 or not self.hands[s]:
                raise ValueError(f"robot model {self.id!r}: the seated operator needs six-joint legs, seven-joint arms "
                                 f"and fingered hands ({s}: {len(self.legs[s])} leg, {len(self.arms[s])} arm, "
                                 f"{len(self.hands[s])} hand joints)")
        self.role = {}
        for s in SIDES:
            L = self.legs[s]
            hip = self._by_axis(L[:3], ("roll", "pitch", "yaw"))
            ank = self._by_axis(L[4:6], ("roll", "pitch"))
            for r, j in hip.items():
                self.role[j] = ("leg", s, "hip_" + r)
            self.role[L[3]] = ("leg", s, "knee")
            for r, j in ank.items():
                self.role[j] = ("leg", s, "ankle_" + r)
            A = self.arms[s]
            for k, j in enumerate(A):
                self.role[j] = ("arm", s, "shoulder" if k < 3 else "elbow" if k == 3 else "wrist")
            for j in self.hands[s]:
                self.role[j] = ("hand", s, "thumb" if "thumb" in j else "finger")
        for j in self.torso:
            self.role[j] = ("torso", None, "torso")
        self.thumb = {s: [j for j in self.hands[s] if self.role[j][2] == "thumb"] for s in SIDES}
        child = lambda js: [self.joint[j]["child"] for j in js]
        self.links = dict(trunk=[self.root] + child(self.torso), chest=(child(self.torso) or [self.root])[-1:])
        for s in SIDES:
            L, A = self.legs[s], self.arms[s]
            self.links.update({f"thigh:{s}": child(L[1:3]), f"knee:{s}": child(L[3:4]), f"shank:{s}": child(L[3:6]),
                               f"wrist:{s}": child(A[4:7]), f"palm:{s}": [self.hand_link[s]],
                               f"fingers:{s}": child(self.hands[s])})
            for k, l_ in enumerate(child(self.thumb[s])):
                self.links[f"thumb{k}:{s}"] = [l_]
        for name, g in spec.get("groups", {}).items():
            want = {"legs": "leg", "torso": "torso", "arm": "arm", "hand": "hand"}.get(g.get("role"))
            if want and any(self.role.get(j, (None,))[0] != want for j in g["joints"]):
                raise ValueError(f"robot model {self.id!r}: catalogue group {name!r} (role {g['role']}) disagrees with "
                                 "its URDF tree")

    def _chain(self, link):
        out = []
        while link != self.root:
            jn = self.by_child.get(link)
            if jn is None:
                raise ValueError(f"robot model {self.id!r}: link {link} is not below the root {self.root}")
            if self.joint[jn]["type"] in ("revolute", "prismatic", "continuous"):
                out.append(jn)
            link = self.joint[jn]["parent"]
        return out[::-1]

    def _below(self, link, top):
        while True:
            if link == top:
                return True
            jn = self.by_child.get(link)
            if jn is None:
                return False
            link = self.joint[jn]["parent"]

    def _axis_root(self, jn):
        """The joint's axis in the root frame at the zero pose."""
        R = np.eye(3)
        path, link = [], self.joint[jn]["child"]
        while link != self.root:
            j = self.by_child[link]
            path.append(j)
            link = self.joint[j]["parent"]
        for j in path[::-1]:
            R = R @ _rpy(*self.joint[j]["rpy"])
        return R @ self.joint[jn]["axis"]

    def _by_axis(self, joints, roles):
        dirs = dict(roll=0, pitch=1, yaw=2)
        out = {}
        for r in roles:
            j = max(joints, key=lambda j: abs(self._axis_root(j)[dirs[r]]))
            if j in out.values():
                raise ValueError(f"robot model {self.id!r}: cannot tell the {roles} joints apart by their axes")
            out[r] = j
        return out

    def body(self, link):
        """The body a link was merged into (fixed joints collapse into their parent)."""
        while link not in self.body_links:
            link = self.joint[self.by_child[link]]["parent"]
        return link

    def bodies(self, key, sides=SIDES):
        """Body names (link names of the merged bodies) of a link set; per-side sets for both sides unless given."""
        keys = [key] if key in self.links else [f"{key}:{s}" for s in sides]
        out = []
        for k in keys:
            for l_ in self.links[k]:
                b = self.body(l_)
                if b not in out:
                    out.append(b)
        return out

    def is_role(self, joint, *roles, side=None):
        r = self.role.get(joint.split("/")[-1])
        return r is not None and r[2] in roles and (side is None or r[1] == side)

    def motor_group(self, joint):
        """The motors.py gain-bound and armature class of a joint: leg, torso, arm, wrist or hand."""
        kind, _, part = self.role[joint.split("/")[-1]]
        return dict(leg="leg", torso="torso", hand="hand")[kind] if kind != "arm" else ("wrist" if part == "wrist" else "arm")


def skeleton_of(handle):
    """Skeleton of a robots runtime handle (one instance)."""
    import robots
    spec = robots.model(handle.model_id).to_dict()
    links = [lab.split("/")[-1] for lab in handle.metadata["body_labels"]]
    return Skeleton(spec, handle.metadata["urdf"], links)


def seat_fit(model_id):
    fit = SEAT_FIT.get(model_id)
    if fit is None:
        raise ValueError(f"robot model {model_id!r} is not fitted to the forklift seat (operator.SEAT_FIT)")
    return fit


def add(scene):
    if scene.c.robot in KIT_RIGS:
        return add_kit(scene)
    import robots
    T = scene.truck
    x, y, yaw = scene.truck_pose
    z0 = scene.truck_origin[2]
    c, s = math.cos(yaw), math.sin(yaw)
    px, py, pz = PELVIS_TRUCK
    pos = (x + c * px - s * py, y + s * px + c * py, z0 + pz)
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    rig_id = RIGS.get(scene.c.robot)
    if rig_id is None:
        raise ValueError(f"forklift seats robots {sorted(RIGS)}, not {scene.c.robot!r}")
    setup = robots.resolve(rig_id)
    first_body, first_shape = scene.b.body_count, scene.b.shape_count
    first_dof = len(scene.b.joint_qd)
    scene.robot_pending = robots.add_to(scene.b, setup, mounts={"operator_seat": dict(position_m=list(pos),
                                                                                      orientation_wxyz=list(q))})
    scene.robot_setup = setup
    scene.robot_bodies = list(range(first_body, scene.b.body_count))
    scene.robot_shapes = list(range(first_shape, scene.b.shape_count))
    scene.robot_mass = float(sum(scene.b.body_mass[i] for i in scene.robot_bodies))
    handle = next(iter(scene.robot_pending.instances.values()))
    scene.robot_skeleton = sk = skeleton_of(handle)
    fit = seat_fit(handle.model_id)
    for i in scene.robot_shapes:
        scene.shape_group[i] = "obj"
        scene.materials[i] = "robot"
        scene.b.shape_gap[i] = 0.002          # rigid_gap on the robot's shapes (barcode_scanning)
    # collision pairs filtered as measured convex-hull artifacts of this robot in the seat (SEAT_FIT)
    blink = lambda i: str(scene.b.body_label[scene.b.shape_body[i]]).split("/")[-1]
    n_pairs = 0
    for set_a, set_b in fit["filtered"]:
        per_side = not (set_a in sk.links or set_b in sk.links)
        for sides in ((("left",), ("right",)) if per_side else (SIDES,)):
            A_, B_ = set(sk.bodies(set_a, sides)), set(sk.bodies(set_b, sides))
            sa = [i for i in scene.robot_shapes if blink(i) in A_]
            sb = [i for i in scene.robot_shapes if blink(i) in B_]
            for a in sa:
                for b in sb:
                    scene.b.add_shape_collision_filter_pair(a, b)
            n_pairs += len(sa) * len(sb)
    scene.robot_filtered_pairs = n_pairs
    # reflected rotor inertia on the robot's dofs: the profile's per-joint value, else the motors.ARMATURE class estimate
    prof = robots.model(handle.model_id).to_dict()["actuation"]["profiles"][handle.metadata["controller_profile"]]["joints"]
    for name, a in handle.joints.items():
        short = name.split("/")[-1]
        scene.b.joint_armature[a["dof"]] = prof[short].get("armature", ARMATURE[sk.motor_group(short)])


def mujoco_hook(world, m):
    """Spin resistance on the hand links (torsional friction of a finger pad on the knob) and the robot's own
    contact with the seat and controls at the shared rigid settings."""
    if getattr(world.scene, "kit", None) is not None:
        return kit_mujoco_hook(world, m)
    sk = world.scene.robot_skeleton
    sb = world.model.shape_body.numpy()
    robot = set(world.scene.robot_bodies)
    hand = set(sk.bodies("palm") + sk.bodies("fingers"))
    for g in range(m.ngeom):
        s = int(world._geom_shape[g]) if g < len(world._geom_shape) else -1
        if s < 0:
            continue
        owner = int(sb[s])
        if owner in robot:
            lab = world.model.body_label[owner]
            if lab.split("/")[-1] in hand:
                m.geom_condim[g] = 4
                m.geom_friction[g][0] = MU_ROBOT_PAD["slide"]      # finger and palm pads (Inspire RH56 silicone tips,
                m.geom_friction[g][1] = MU_ROBOT_PAD["torsion"]    # Dex3-1 tactile pads)
            else:
                m.geom_friction[g][0] = MU_ROBOT_SHELL             # the rest of the robot (Newton's default, made explicit)
    # finger linkage followers (URDF mimic joints, MuJoCo joint equalities; the Inspire hands have them, the Dex3 none):
    # a steel linkage, not the default 20 ms spring, which lets a follower swing against its driver when the hand is
    # struck (E17)
    import mujoco
    jb = np.asarray(m.jnt_bodyid)
    robot_mj = {world.mj_body(b) for b in robot}
    for e in range(m.neq):
        if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT) and int(jb[int(m.eq_obj1id[e])]) in robot_mj:
            m.eq_solref[e] = (P.ROBOT_LINKAGE_TICKS * world.config.dt, 1.0)


class Operator:
    def __init__(self, world):
        self.w = world
        rig = world.rig
        self.handle = h = next(iter(rig.instances.values()))
        self.joints = dict(h.joints)
        import robots
        spec = robots.model(h.model_id).to_dict()
        self.model_id = h.model_id
        self.skeleton = sk = world.scene.robot_skeleton
        self.fit = seat_fit(h.model_id)
        self.profile = spec["actuation"]["profiles"][h.metadata["controller_profile"]]["joints"]
        self.seated = spec["poses"][h.metadata["initial_pose"]]
        self.motors = mo = Motors(world, self.joints, self.profile)
        # gain bounds by the catalogue skeleton's motor groups (motors.group_of reads H1-2 joint names; identical there)
        groups = [sk.motor_group(n) for n in mo.names]
        if groups != mo.group:
            mo.group = groups
            mo.kp_max = np.array([GAIN_BOUNDS[g][0] for g in groups])
            mo.kd_max = np.array([GAIN_BOUNDS[g][1] for g in groups])
        labels = list(world.model.body_label)
        self.body = {lab.split("/")[-1]: i for i, lab in enumerate(labels) if i in set(world.scene.robot_bodies)}
        self.pelvis = self.body[sk.root]
        jc = world.model.joint_child.numpy()
        self.free_joint = [j for j in range(world.model.joint_count) if int(jc[j]) == self.pelvis]
        self.belt = dict(fastened=True, locked=False, rest=None, length=None, tension=(0.0, 0.0), payout_v=0.0,
                         sensed_v=0.0, payout_a=0.0, tilt=0.0)
        self.events = []

    # ------------------------------------------------------------------ per tick
    def write(self, world, jf):
        ctrl = world.control.joint_target_q.numpy()
        qd = world.state.joint_qd.numpy()
        self.motors.write(ctrl, qd)
        world.control.joint_target_q.assign(ctrl)
        if self.belt["fastened"]:
            self._belt(world)

    def _belt(self, world):
        bq = world.state.body_q.numpy().astype(float)
        bqd = world.state.body_qd.numpy().astype(float)
        com = world._com_world(bq)
        seat = world.T.bodies["seat"]
        Rs = quat_to_R(bq[seat:seat + 1, 3:7])[0]
        Rp = quat_to_R(bq[self.pelvis:self.pelvis + 1, 3:7])[0]
        anchors = [bq[seat, :3] + Rs @ np.asarray(a) for a in BELT_ANCHOR_SEAT]
        wraps = [bq[self.pelvis, :3] + Rp @ np.asarray(p) for p in self.fit["belt_wrap_pelvis"]]
        # path: left anchor -> left wrap -> right wrap -> right anchor (the webbing slides over the lap)
        segs = [(anchors[0], wraps[0]), (wraps[0], wraps[1]), (wraps[1], anchors[1])]
        L = sum(float(np.linalg.norm(b - a)) for a, b in segs)
        b = self.belt
        dt = world.config.dt
        if b["length"] is None:
            b["length"] = L
            b["rest"] = L
        b["payout_v"] = (L - b["length"]) / dt
        b["length"] = L
        # retractor sensors. Vehicle: the pendulum reads the apparent gravity (specific force) in the seat frame
        ach = np.asarray(getattr(world.events, "chassis_acc", np.zeros(3)), float)
        f = Rs.T @ (ach + np.array([0.0, 0.0, P.G]))
        b["tilt"] = math.atan2(math.hypot(f[0], f[1]), f[2])
        # webbing: payout acceleration through the sensor's response (first-order on the payout speed, then on its rate)
        k = min(1.0, dt / BELT_SENSOR_TAU)
        v_prev = b["sensed_v"]
        b["sensed_v"] += k * (b["payout_v"] - b["sensed_v"])
        b["payout_a"] += k * ((b["sensed_v"] - v_prev) / dt - b["payout_a"])
        tripped = b["tilt"] > BELT_LOCK_TILT or b["payout_a"] > BELT_WEBBING_LOCK_G * P.G
        if not b["locked"] and tripped:
            b["locked"] = True
            b["rest"] = min(b["rest"], L)
            b["lock_time"] = world.sim_time
        elif b["locked"] and not tripped and L <= b["rest"]:
            b["locked"] = False                              # sensors reset and the webbing unloaded: the pawl drops
        if b["locked"]:
            stretch = L - b["rest"]
            T = max(0.0, P.BELT["k"] * stretch + P.BELT["c"] * max(b["payout_v"], 0.0)) + BELT_RETRACT_N
        else:
            b["rest"] = L                                    # free payout / retraction
            T = BELT_RETRACT_N
        b["tension"] = T
        wrench = world.wrench if world.wrench is not None else None
        F_pel = np.zeros(3); M_pel = np.zeros(3)
        F_seat = np.zeros(3); M_seat = np.zeros(3)
        # force on each wrap point = T (unit toward the neighbours); on each anchor = T toward the wrap
        u0 = (anchors[0] - wraps[0]); u0 /= np.linalg.norm(u0)
        u1 = (wraps[1] - wraps[0]); u1 /= np.linalg.norm(u1)
        u2 = (anchors[1] - wraps[1]); u2 /= np.linalg.norm(u2)
        f_w0 = T * (u0 + u1)
        f_w1 = T * (-u1 + u2)
        for p, f in ((wraps[0], f_w0), (wraps[1], f_w1)):
            F_pel += f; M_pel += cross3(p - com[self.pelvis], f)
        for p, f in ((anchors[0], -T * u0), (anchors[1], -T * u2)):
            F_seat += f; M_seat += cross3(p - com[seat], f)
        self.belt_wrench = (F_pel, M_pel, F_seat, M_seat)

    def apply_belt(self, wrench):
        if self.belt["fastened"] and getattr(self, "belt_wrench", None) is not None:
            F_pel, M_pel, F_seat, M_seat = self.belt_wrench
            seat = self.w.T.bodies["seat"]
            wrench[self.pelvis, :3] += F_pel; wrench[self.pelvis, 3:] += M_pel
            wrench[seat, :3] += F_seat; wrench[seat, 3:] += M_seat

    # ------------------------------------------------------------------ helpers (owner / assisted)
    def gravity_ff(self):
        """Joint torques that hold the current posture against gravity (MuJoCo's bias forces at zero velocity),
        for the commanded motors; a program on the robot would compute these from its own model."""
        import mujoco
        m, d0 = self.w.solver.mj_model, self.w.solver.mj_data
        d = mujoco.MjData(m)
        d.qpos[:] = d0.qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        mjdof = self._mj_dofs()
        tau = np.asarray(d.qfrc_bias)[mjdof]
        return np.clip(tau, -self.motors.effort, self.motors.effort)

    def _mj_dofs(self):
        if getattr(self, "_mjdof", None) is None:
            import mujoco
            m = self.w.solver.mj_model
            names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "": j for j in range(m.njnt)}
            out = []
            for n in self.motors.names:
                short = n.split("/")[-1]
                j = [v for k, v in names.items() if k.endswith(short)][0]
                out.append(int(m.jnt_dofadr[j]))
            self._mjdof = np.array(out, int)
        return self._mjdof

    def proprioception(self):
        q = self.w.state.joint_q.numpy(); qd = self.w.state.joint_qd.numpy()
        tau = self.motors.torques()
        return {n: dict(position=float(q[self.joints[n]["q"]]), velocity=float(qd[self.joints[n]["dof"]]),
                        effort=float(tau[i]) if i is not None else 0.0)
                for n, i in ((n, self.motors.index.get(n)) for n in self.joints)}


# ------------------------------------------------------------------ start posture
def start(world):
    if getattr(world.scene, "kit", None) is not None:
        return kit_start(world)
    world.operator = op = Operator(world)
    q, pose = seated_posture(world)
    qa = world.model.joint_q.numpy()
    for n, v in q.items():
        qa[op.joints[n]["q"]] = v
    # free joint of the pelvis
    qs = world.model.joint_q_start.numpy()
    a = int(qs[op.free_joint[0]])
    qa[a:a + 3] = pose[0]
    qa[a + 3:a + 7] = pose[1]
    world.model.joint_q.assign(qa)
    world.state.joint_q.assign(qa)
    world.state.joint_qd.zero_()
    world.newton.eval_fk(world.model, world.state.joint_q, world.state.joint_qd, world.state)
    world.next_state.assign(world.state)
    world.sync_solver()
    op.motors.hold_current()
    op.motors.cmd["tau"][:] = op.gravity_ff()
    op.start_command = op.motors.checkpoint()


ACCEL_READY = False        # start posture with the right foot over the accelerator: off. With the thighs near level the
                           # sole cannot lie on the 55-deg plate (ankle pitch at its -0.9 rad limit, ball 10 cm short; E14)
ACCEL_HOVER = 0.012        # ball of the foot above the plate (m)


def accel_pad(world):
    """World point and outward normal of the accelerator press point (scripted.Scripted.pedal_pad's model): the ball of
    the foot 3 cm above the plate centre, the sole 30 deg flatter than the 55-deg plate."""
    from .geometry import body_meshes, body_vertices
    from .mathx import quat_to_R
    T = world.T
    V = body_vertices(body_meshes(), "pedal_accel").astype(float)
    piv = np.asarray(T.anchors["pedal_accel"])
    cm = V.mean(0)
    _, _, Wt = np.linalg.svd(V - cm, full_matrices=False)
    along = Wt[0] * np.sign(Wt[0][2])
    plate_n = Wt[2] * (1.0 if Wt[2][0] < 0 else -1.0)
    c = cm + along * 0.03 + plate_n * 0.004
    a = math.radians(-30.0)
    Ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    nrm = Ry @ plate_n
    b = T.bodies["pedal_accel"]
    bq = world.state.body_q.numpy().astype(float)
    R = quat_to_R(bq[b:b + 1, 3:7])[0]
    return bq[b, :3] + R @ (c - piv), R @ nrm


def seated_posture(world, lean=None, feet=None):
    """Joint angles and pelvis pose of the start posture, on the world's MuJoCo model and geometry (Kin):
    1. pelvis pitched forward by `lean` (SEAT_FIT's, else START_LEAN), hip pitch so the thighs keep their direction;
    2. pelvis lowered until its seat links are SEAT_CLEARANCE above the cushion collider and moved back until the
       trunk or the hip links are BACK_CLEARANCE from the backrest (MuJoCo geom distances through Kin.distance);
    3. knees and ankles (hip pitch within +-0.2 rad) put the soles flat on the toe board or floor plate where the shins
       reach, clear of the cab, pedals and levers (clearance pairs);
    4. pelvis re-placed; arms rest the hands on the thighs, clear of the legs.
    Returns (joint dict, (pelvis position, quaternion xyzw)); metrics in world.scene.start_posture."""
    from scipy.spatial.transform import Rotation
    from .kin import Kin
    op = world.operator
    sk, fit = op.skeleton, op.fit
    kin = Kin(world)
    m = kin.model
    pch, Rch = world.body_pose("chassis")
    to_w = lambda p: pch + Rch @ np.asarray(p, float)
    lean = fit.get("lean", START_LEAN) if lean is None else lean
    q0 = {n: float(op.seated[n.split("/")[-1]]) for n in op.joints}
    for n in q0:
        if sk.is_role(n, "hip_pitch"):
            # the pelvis pitches forward by `lean` about its lateral axis; the hip pitch turns the thigh back by the same
            # angle about that axis, so the thighs keep the seated pose's direction (the joint's sign from its axis: the
            # H1-2's hip pitch axis is +y and flexion negative, so this lowers q; `q0 += lean` would turn the thighs
            # down by twice the lean, which the leg IK's +-0.2 rad window cannot level)
            q0[n] -= lean * float(np.sign(sk._axis_root(n.split("/")[-1])[1]))
    kin.sync(q0)
    qp = (Rotation.from_matrix(Rch) * Rotation.from_euler("y", lean)).as_quat()
    kin.data.qpos[kin.free_adr + 3:kin.free_adr + 7] = (qp[3], qp[0], qp[1], qp[2])
    kin.mj.mj_kinematics(m, kin.data)
    B = lambda n: kin.body(n)
    footL, footR = B(sk.body(sk.foot_link["left"])), B(sk.body(sk.foot_link["right"]))
    robot_b = {world.mj_body(b) for b in world.scene.robot_bodies}
    geoms_of = lambda bodies: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies]
    seat_b = world.mj_body(world.T.bodies["seat"])
    cushion = [g for g in geoms_of({seat_b}) if "cushion" in world.mj_geom_label(g)]
    backrest = [g for g in geoms_of({seat_b}) if "backrest" in world.mj_geom_label(g)]
    seat_links = geoms_of({B(n) for n in [sk.body(sk.root)] + sk.bodies("thigh")})        # buttocks: pelvis and thighs
    # what can meet the backrest: the trunk and the hip links. A forward-pitched pelvis carries the hip links (the H1-2's
    # hip-pitch housings sit behind the hip axis) back into the backrest; with the trunk links alone they are placed
    # 18-50 mm inside it at a 5-9 deg lean
    back_links = geoms_of({B(n) for n in sk.bodies("trunk") + sk.bodies("thigh")})
    cab_b = {world.mj_body(world.T.bodies[n]) for n in ("chassis", "pedal_inch", "pedal_brake", "pedal_accel",
                                                         "park_lever", "dir_lever", "steering_wheel", "knob",
                                                         "lever_lift", "lever_tilt", "lever_shift", "key", "mast_outer")}
    leg_b = {B(n) for n in sk.bodies("shank")}
    leg_pairs = kin.geom_pairs(leg_b, cab_b)

    placed = []

    def place():
        # distances through Kin.distance, not plain mj_geomDistance: the latter reads exactly 0.0 for some separated hull
        # pairs whenever its distmax exceeds the true distance (the pelvis against the backrest at every distmax from 5 mm
        # to 0.3 m with the true gap 4 mm), which the placement would read as touching, moving the
        # robot by the clearance on every pass
        dz = dx = None
        for _ in range(40):
            kin.mj.mj_kinematics(m, kin.data)
            dz = min(kin.distance(g, h, 0.3) for g in seat_links for h in cushion) - SEAT_CLEARANCE
            kin.data.qpos[kin.free_adr:kin.free_adr + 3] -= Rch @ np.array([0.0, 0.0, dz])
            kin.mj.mj_kinematics(m, kin.data)
            dx = min(kin.distance(g, h, 0.3) for g in back_links for h in backrest) - BACK_CLEARANCE
            kin.data.qpos[kin.free_adr:kin.free_adr + 3] -= Rch @ np.array([dx, 0.0, 0.0])
            if abs(dz) < 2e-4 and abs(dx) < 2e-4:
                break
        kin.mj.mj_kinematics(m, kin.data)
        placed.append((round(1000 * float(dz), 3), round(1000 * float(dx), 3)))   # last correction (mm): converged if ~0

    place()
    # soles: lowest collision vertices of the ankle-roll links (body frame)
    sole = {}
    for side, b in (("l", footL), ("r", footR)):
        V = _body_vertices_local(m, kin.data, b)
        sole[side] = V[V[:, 2] < V[:, 2].min() + 0.004].mean(0)
    feet = feet or {"l": fit["feet_y"][0], "r": fit["feet_y"][1]}
    legs = [n for n in op.joints if sk.is_role(n, "knee", "ankle_pitch", "ankle_roll", "hip_pitch")]
    if ACCEL_READY:
        legs += [n for n in op.joints if sk.is_role(n, "hip_yaw", "hip_roll", side="right")]
    slope = 0.321
    n_tb = Rch @ np.array([-math.sin(math.atan(slope)), 0.0, math.cos(math.atan(slope))])
    tgt = []
    for side, b in (("l", footL), ("r", footR)):
        if side == "r" and ACCEL_READY:
            # a driver's right foot rests on the accelerator: the ball of the foot hovers ACCEL_HOVER above the organ
            # pedal's plate, the sole 30 deg flatter than the plate, toes forward (E14 posture search; reaching this
            # pose from a foot on the floor wedges the H1-2's foot between the toe board and the cab)
            pa, na = accel_pad(world)
            V = _body_vertices_local(m, kin.data, b)
            S = V[V[:, 2] < V[:, 2].min() + 0.004]
            ball = np.array([S[:, 0].max() - 0.09, S[:, 1].mean(), V[:, 2].min()])
            tgt.append((b, ball, pa + ACCEL_HOVER * na, 1.0))
            tgt.append((b, "axis", (0.0, 0.0, 1.0), tuple(na), 0.3))
            tgt.append((b, "axis", (1.0, 0.0, 0.0), tuple(Rch @ np.array([1.0, 0.0, 0.0])), 0.05))
            continue
        x_nat = float(((kin.point(b, sole[side]) - pch) @ Rch)[0])
        if x_nat > -0.42:      # on the toe board
            p0, nrm = to_w((-0.40, 0.0, 0.67 + SOLE_OFFSET)), n_tb
        else:
            p0, nrm = to_w((-0.50, 0.0, 0.665 + SOLE_OFFSET)), Rch @ np.array([0.0, 0.0, 1.0])
        tgt.append((b, sole[side], ("plane", p0, nrm), 1.0))
        tgt.append((b, sole[side], to_w((0.0, feet[side], 0.0)), np.array([0.0, 0.3, 0.0]), Rch))
        tgt.append((b, "axis", (0.0, 0.0, 1.0), nrm, 0.3))
    lo = {n: q0[n] - 0.2 for n in legs if sk.is_role(n, "hip_pitch")}
    # unconstrained first, then with the cab clearance pairs seeded from it (the clearance terms otherwise hold the
    # solve in the basin of the initial feet-on-the-floor posture)
    q_free, _ = kin.solve(legs, tgt, nominal=q0, reg=0.02)
    kin.sync(dict(q0, **q_free), keep_base=True)
    qleg, err_leg = kin.solve(legs, tgt, nominal=dict(q0, **q_free), reg=0.02, pairs=leg_pairs, clearance=0.004)
    for n, v in qleg.items():                       # keep the thighs near level (hip pitch within +-0.2 rad)
        if n in lo and not (ACCEL_READY and sk.is_role(n, "hip_pitch", side="right")):
            qleg[n] = float(np.clip(v, lo[n], lo[n] + 0.4))
    # thumbs rotated in (clear of the thighs; they rest there otherwise) before the arms are solved, so the hands'
    # clearance covers the thumbs' final pose: set after the solve, a thumb swung into the robot's own hip link and
    # stick-slipped there at ~14 mm/s, and the start state never settled
    thumbs = {} if fit["thumb_rest"] is None else \
        {n: fit["thumb_rest"] for n in op.joints if n.split("/")[-1] in (sk.thumb["left"][:1] + sk.thumb["right"][:1])}
    kin.sync(dict(q0, **qleg, **thumbs), keep_base=True)
    place()
    # hands resting on the thighs, clear of the legs
    (handL, palmL), (handR, palmR) = hand_frame(world, "left"), hand_frame(world, "right")
    arms = [n for n in op.joints if sk.is_role(n, "shoulder", "elbow", "wrist")]
    kneeL, kneeR = (B(sk.bodies("knee", (s_,))[0]) for s_ in SIDES)
    htgt = []
    for b_h, palm, b_k in ((handL, palmL, kneeL), (handR, palmR, kneeR)):
        pk = kin.pose(b_k)[0]
        htgt.append((b_h, palm, pk + Rch @ np.array([fit["hand_rest_back"], 0.0, 0.13]), 1.0))
    hand_bodies = {B(n) for n in sk.bodies("wrist") + sk.bodies("palm") + sk.bodies("fingers")}
    # Bit for bit with the H1-2 start states the release was qualified on: their hand set was a substring match on body
    # names, which also caught bodies of the truck and the scene (the steering wheel: "steeRING"). Their pairs do not
    # depend on the arm joints (constant clearance residuals) but they change the least-squares path (dropped, the
    # H1-2's resting arms moved by up to 0.010 rad). Kept until every start state is re-qualified.
    hand_bodies |= {i for n, i in kin.bname.items() if i not in robot_b and any(k in n for k in LEGACY_HAND_KEYS)}
    # the thighs' distal links, the knees and the trunk
    body_b = {B(n) for n in [sk.body(sk.links[f"thigh:{s_}"][-1]) for s_ in SIDES] + sk.bodies("knee") + sk.bodies("trunk")}
    pairs = kin.geom_pairs(hand_bodies, body_b | cab_b)
    qarm, err_arm = kin.solve(arms, htgt, nominal=dict(q0, **thumbs), reg=0.05, pairs=pairs, clearance=fit["hand_rest_clear"])
    q = dict(q0, **qleg, **thumbs, **qarm)
    kin.sync(q, keep_base=True)
    place()
    # audit: every robot geom against everything it can touch
    worst = []
    for g in geoms_of(robot_b):
        for h in range(m.ngeom):
            if int(m.geom_bodyid[h]) in robot_b:
                continue
            ct, ca = int(m.geom_contype[g]), int(m.geom_conaffinity[g])
            if not ((ct & int(m.geom_conaffinity[h])) or (int(m.geom_contype[h]) & ca)):
                continue
            dd = kin.mj.mj_geomDistance(m, kin.data, g, h, 0.01, None)
            if dd < 0.0:
                worst.append((round(dd * 1000, 1), world.mj_geom_label(g), world.mj_geom_label(h)))
    worst.sort()
    tor = B(sk.body(sk.links["chest"][0]))
    head = (_world_vertices(m, kin.data, {tor}) - pch) @ Rch
    world.scene.start_posture = dict(lean_deg=math.degrees(lean), leg_err=err_leg, arm_err_m=err_arm, place_residual_mm=placed,
                                     penetrations_mm=worst[:12], head_top_z=float(head[:, 2].max()),
                                     pelvis_truck=((kin.data.qpos[kin.free_adr:kin.free_adr + 3] - pch) @ Rch).tolist())
    p = kin.data.qpos[kin.free_adr:kin.free_adr + 3].copy()
    wxyz = kin.data.qpos[kin.free_adr + 3:kin.free_adr + 7].copy()
    return q, (p, np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]))


def hand_frame(world, side):
    """(MuJoCo body id, local point) of the catalog's hand frame (palm base) of one side."""
    fr = world.operator.handle.frames[f"{side}_hand"]
    return world.mj_body(fr["body"]), np.asarray(fr.get("position_m", (0.0, 0.0, 0.0)), float)


def _body_vertices_local(m, d, b):
    """Collision vertices of a body in its own frame: mesh vertices, and for primitive geoms (the G1's soles are four
    5 mm spheres) the extreme points along the geom's axes."""
    out = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != b:
            continue
        t = int(m.geom_type[g])
        if t == 7:                                                         # mjGEOM_MESH
            mid = int(m.geom_dataid[g])
            V = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
        elif t in (2, 3, 5, 6):                                            # sphere, capsule, cylinder, box
            sz = np.asarray(m.geom_size[g], float)
            if t == 2:
                V = np.vstack([np.eye(3), -np.eye(3)]) * sz[0]
            elif t == 6:
                V = np.array([(i * sz[0], j * sz[1], k * sz[2]) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
            else:
                ends = [np.array([0.0, 0.0, h * sz[1]]) for h in (-1, 1)]
                V = np.vstack([e + sz[0] * np.vstack([np.eye(3), -np.eye(3)]) for e in ends])
        else:
            continue
        R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]
        Rb = d.xmat[b].reshape(3, 3); pb = d.xpos[b]
        Vw = p + V @ R.T
        out.append((Vw - pb) @ Rb)
    return np.concatenate(out, 0)


def _world_vertices(m, d, bodies):
    out = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in bodies or int(m.geom_type[g]) != 7:
            continue
        mid = int(m.geom_dataid[g])
        V = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
        out.append(d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T)
    return np.concatenate(out, 0) if out else np.zeros((0, 3))


def _geom_vertices_truck(m, d, mj_body, pch, Rch, tag, world):
    out = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != mj_body:
            continue
        if tag not in world.mj_geom_label(g):
            continue
        if int(m.geom_type[g]) == 7:
            mid = int(m.geom_dataid[g])
            V = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
            out.append(d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T)
    W = np.concatenate(out, 0)
    return (W - pch) @ Rch


# ====================================================================================================================
# The driving-robot kit (rig forklift-drive-kit, catalogue model drive_kit)
#
# A vehicle-testing kit fitted to the truck's own controls, each actuator modelled at its output: an arm turning about
# the axis of the control it works, so its end effector stays on the same point of the control through the whole travel.
#   steer        steering robot (SR 15 Orbit figures): a rotor about the steering column's axis, carried by a stator
#                whose reaction arm is bolted to the cab; its fork straddles the spinner knob (the drive passes through
#                the knob's bearing into the wheel). Real SR robots clamp to the rim; this rim's collider is a smooth ring
#                without spokes, so a rim clamp would be either a weld or a friction clamp: the fork is the form-fit
#                coupling that acts by contact only.
#   accelerator, brake, inching   pedal robots: an arm about each pedal's pivot with a flat foot over the pad, a few
#                millimetres off it at rest; it pushes (never pulls: the pedal's own spring returns it).
#   lift, tilt, shift, park      lever robots: an arm about each lever's pivot with a fork straddling its knob (handle).
#   direction    selector robot: an arm about the stalk's pivot axis (the column axis) with a fork round its tip.
#   key          key robot: a rotor coaxial with the key gripping its head with two spring-loaded jaws (the key's
#                collider is a 23 mm round head: only a friction grip can turn it, as a hand's pinch does).
#   horn         button robot: a rocker on a bridge over the steering wheel, its rounded tip on the column axis over
#                the horn button.
# Bolted to the truck before the run (fixed joints to the chassis; params.DRIVE_KIT_MOUNTS): the kit frame (with the
# IMU), the pedal unit, the lever bank, the steering robot's stator, the key robot's mount, the horn bridge and the
# camera mast (a D435i-class camera hanging from the overhead guard at an operator's eye point); a ballast strapped to
# the seat closes the operator-presence switch (params.DRIVE_KIT_BALLAST). Only the end effectors collide, and each only
# with the control it works (MuJoCo contype/conaffinity bits, kit_mujoco_hook); housings, arms, mounts and the ballast
# carry mass and are drawn but have no colliders. Nothing of the kit is attached to a control: every force on a control
# is a contact force. The kit's construction (end effectors, clearances, masses, drawing) is its catalogue model's
# (robots/models/drive_kit/robot.json, "mechanisms"); the product limits are its actuation profile's.
# kit joint -> (control joint, the control body its end effector touches, the target collider tags or None for all)
KIT_CONTROLS = {"steer": ("steering_wheel", "knob", None), "accelerator": ("pedal_accel", "pedal_accel", None),
                "brake": ("pedal_brake", "pedal_brake", ("pad",)), "inching": ("pedal_inch", "pedal_inch", ("pad",)),
                "lift": ("lever_lift", "lever_lift", ("knob",)), "tilt": ("lever_tilt", "lever_tilt", ("knob",)),
                "shift": ("lever_shift", "lever_shift", ("knob",)), "direction": ("dir_lever", "dir_lever", None),
                "park": ("park_lever", "park_lever", None), "key": ("key", "key", None), "horn": ("horn", "horn", None)}
KIT_ORDER = tuple(KIT_CONTROLS)
KIT_GROUP = {"steer": "steering", "accelerator": "pedal", "brake": "pedal", "inching": "pedal", "lift": "lever",
             "tilt": "lever", "shift": "lever", "direction": "selector", "park": "hand_brake", "key": "key",
             "horn": "button"}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _quat_cols(a, b):
    """xyzw quaternion of the frame whose x, y axes are a, b (z = a x b)."""
    from scipy.spatial.transform import Rotation
    a = _unit(a); b = _unit(b - (b @ a) * a)
    return tuple(Rotation.from_matrix(np.column_stack([a, b, np.cross(a, b)])).as_quat())


def _shape_vertices(scene, body, tags=None, dense=0, prefix=False):
    """Truck-frame vertices of a truck body's convex-hull colliders (the colliders the kit's end effectors touch); with
    dense > 0 also points on every triangle (a barycentric grid of `dense` steps), for contact searches. prefix: the tags
    are tag prefixes."""
    T, b = scene.truck, scene.b
    a0 = np.asarray(T.anchors[body], float) - np.asarray(T.shifts.get(body, (0.0, 0.0, 0.0)), float)
    out = []
    for i, (bn, tag, _) in T.shapes.items():
        if bn != body or (tags is not None and not any(tag.startswith(t) if prefix else tag == t for t in tags)):
            continue
        src = b.shape_source[i]
        if src is not None and hasattr(src, "vertices"):
            V = np.asarray(src.vertices, float) * np.asarray(b.shape_scale[i], float) + a0
            out.append(V)
            if dense and getattr(src, "indices", None) is not None:
                F = np.asarray(src.indices, int).reshape(-1, 3)
                g = [(i_ / dense, j_ / dense) for i_ in range(dense + 1) for j_ in range(dense + 1 - i_)]
                A = np.array([(1 - u - v, u, v) for u, v in g])
                out.append(np.einsum("kb,fbd->fkd", A, V[F]).reshape(-1, 3))
    return np.concatenate(out, 0)


def _fork(kc, V, u_t, u_r, u_z, fork, cover):
    """Two tines straddling the vertex cloud V (centre kc) along u_t, `fork` the model's fork mechanism (clearance, tine
    half thickness, smallest half size), `cover` the share of the cloud's extent along u_r and u_z the tines cover:
    [(centre, half sizes, quaternion)] and the cloud's extents (truck frame; a tine box's x along u_t, y along u_r)."""
    d = V - kc
    et, er, ez = d @ u_t, d @ u_r, d @ u_z
    rm, zm = 0.5 * (er.max() + er.min()), 0.5 * (ez.max() + ez.min())
    th, lo = fork["tine_half_thickness_m"], fork["tine_min_half_m"]
    half = (th, max(lo, cover * 0.5 * (er.max() - er.min())), max(lo, cover * 0.5 * (ez.max() - ez.min())))
    q = _quat_cols(u_t, u_r)
    tines = []
    for side, e in ((+1, et.max()), (-1, et.min())):
        c = kc + u_t * (e + side * (fork["clearance_m"] + th)) + u_r * rm + u_z * zm
        tines.append((c, half, q))
    return tines, dict(t=(float(et.min()), float(et.max())), r=(float(er.min()), float(er.max())),
                       z=(float(ez.min()), float(ez.max())))


def _angular_gap(V, piv, axis, s, face, n_face, u, v, hu, hv, search):
    """The arm angle (rad, toward pressing) at which a pedal foot's face (the plane through `face` with normal n_face,
    the rectangle +-hu along u and +-hv along v) first meets the pad's collider points V, the pedal held still: the pad
    turned the other way about the common axis (the arm turns about the pedal's own pivot). search: the model's search
    settings (step, largest angle)."""
    from scipy.spatial.transform import Rotation
    d = V - piv
    for th in np.arange(0.0, search["gap_max_rad"], search["gap_step_rad"]):
        Rm = Rotation.from_rotvec(-s * th * np.asarray(axis, float)).as_matrix()
        W = piv + d @ Rm.T - face
        inside = (np.abs(W @ u) <= hu) & (np.abs(W @ v) <= hv)
        if (inside & (W @ n_face >= 0.0)).any():
            return float(th)
    raise ValueError("drive kit: a pedal foot never meets its pad")


def _bar(p0, p1, w, finish="black"):
    """A drawn bar (box) from p0 to p1 (body frame), half width w, in one of the kit's finishes."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L <= 0.0:
        return None
    ref = np.eye(3)[int(np.argmin(np.abs(d)))]
    return dict(kind="box", pos=tuple((p0 + p1) / 2), quat=_quat_cols(d, ref), size=(L / 2, w, w), finish=finish)


def _box_inertia(mass, half):
    a, b, c = (2 * np.asarray(half, float)) ** 2
    return (mass * (b + c) / 12, mass * (a + c) / 12, mass * (a + b) / 12)


def kit_fit(scene, spec, profile="forklift"):
    """The kit's parts in the truck frame (robots/models/drive_kit/newton.py), its frames, and per actuator joint the
    data the motor model needs: the output radius (the end effector's contact point from the arm's axis), the joint-unit
    limits derived from the product's output limits, the contact target and the press sign. spec: the kit's catalogue
    model (its actuation profile's product limits, its mechanisms); the mounts: params.DRIVE_KIT_MOUNTS."""
    from . import truck as TR
    from scipy.spatial.transform import Rotation
    prof = spec["actuation"]["profiles"][profile]["joints"]
    mech = spec["mechanisms"]
    A, B, D = mech["arms"], mech["bodies"], mech["drawing"]
    FT, FK, KJ, HN = mech["pedal_foot"], mech["fork"], mech["key_jaws"], mech["horn"]
    M = P.DRIVE_KIT_MOUNTS
    BL = P.DRIVE_KIT_BALLAST
    T = scene.truck
    ex, ey, ez = np.eye(3)
    n = _unit(T.column_axis)
    ch, seat = T.bodies["chassis"], T.bodies["seat"]
    parts, arms = [], {}
    bar, thin = D["bar_m"], D["thin_bar_m"]

    def arm(name, parent, anchor, axis, lower, upper, radius, collide, visual, mass, extra=None):
        spec_j = prof[name]
        out = spec_j["output"]
        r = 1.0 if out == "rotary" else float(radius)
        w = np.asarray([p[0] for p in spec_j["envelope"]], float) / r
        f = np.asarray([p[1] for p in spec_j["envelope"]], float) * r
        w_nl = float(w[-1] * (1.0 + spec_j["cutoff_fraction"]))
        v_out = np.asarray([p[0] for p in spec_j["envelope"]], float)
        f_out = np.asarray([p[1] for p in spec_j["envelope"]], float)
        arms[name] = dict(radius=float(radius), output=out, env_w=np.append(w, w_nl).tolist(), env_t=np.append(f, 0.0).tolist(),
                          env_v=np.append(v_out, v_out[-1] * (1.0 + spec_j["cutoff_fraction"])).tolist(),
                          env_f=np.append(f_out, 0.0).tolist(), peak=float(spec_j["peak"]),
                          effort=float(spec_j["peak"] * r), w_nl=w_nl, kp=spec_j["kp"], kd=spec_j["kd"], kp_max=spec_j["kp_max"],
                          kd_max=spec_j["kd_max"], control=KIT_CONTROLS[name][0], target=KIT_CONTROLS[name][1],
                          tags=KIT_CONTROLS[name][2], group=KIT_GROUP[name], lower=float(lower), upper=float(upper),
                          axis=np.asarray(axis, float).tolist(), anchor=np.asarray(anchor, float).tolist(),
                          product=spec_j["product"], **(extra or {}))
        gyr = 0.5 * float(radius) if out == "linear" else A["rotor_gyration_m"]
        parts.append(dict(name=name, parent=parent, anchor=tuple(anchor), mass=mass,
                          inertia=(mass * gyr ** 2 / 3 + A["min_inertia_kg_m2"],) * 3,
                          joint=dict(kind="revolute", axis=tuple(axis), lower=lower, upper=upper,
                                     armature=spec_j.get("armature", 0.0), friction=spec_j.get("friction", 0.0), motor=True,
                                     ke=spec_j["kp"], kd=spec_j["kd"], effort=float(spec_j["peak"] * r), velocity=w_nl),
                          collide=collide, visual=[v for v in visual if v is not None]))

    def fork_parts(V, kc, piv, ut, ur, uz, cover):
        tines, ext = _fork(kc, V, ut, ur, uz, FK, cover)
        coll, vis = [], []
        for k, (tc, half, q) in enumerate(tines):
            coll.append(dict(kind="box", pos=tuple(tc - piv), quat=q, size=half, label=f"tine{k}"))
            vis.append(dict(kind="box", pos=tuple(tc - piv), quat=q, size=half, finish="alu"))
        return coll, vis, ext

    # ---- bolted parts: kit frame (IMU) on the footwell floor, pedal unit, lever bank, camera mast, seat ballast
    frame_at = np.asarray(M["frame"], float)
    parts.append(dict(name="frame", parent=ch, anchor=tuple(frame_at), mass=B["frame_mass_kg"],
                      inertia=(B["frame_inertia_kg_m2"],) * 3,
                      visual=[dict(kind="box", pos=(0.0, 0.0, 0.0), size=tuple(D["frame_box_m"]), finish="alu"),
                              dict(kind="box", pos=tuple(D["frame_rail_at_m"]), size=tuple(D["frame_rail_m"]), finish="alu")]))
    pedal_kg = sum(prof[k]["mass_kg"] for k in ("accelerator", "brake", "inching")) - 3 * A["mass_kg"]
    parts.append(dict(name="pedal_unit", parent="frame", anchor=tuple(frame_at + M["pedal_unit_offset"]), mass=pedal_kg,
                      inertia=(B["pedal_unit_inertia_kg_m2"],) * 3,
                      visual=[dict(kind="box", pos=(0.0, 0.0, 0.0), size=tuple(D["pedal_box_m"]), finish="black"),
                              dict(kind="cylinder", pos=tuple(-ex * (D["pedal_box_m"][0] + D["logo_disc_m"][1])),
                                   quat=_quat_cols(ey, ez), size=tuple(D["logo_disc_m"]), finish="blue")]))
    lever_kg = sum(prof[k]["mass_kg"] for k in ("lift", "tilt", "shift", "direction", "park")) - 5 * A["mass_kg"]
    parts.append(dict(name="lever_bank", parent=ch, anchor=tuple(M["lever_bank"]), mass=lever_kg,
                      inertia=(B["lever_bank_inertia_kg_m2"],) * 3,
                      visual=[dict(kind="box", pos=(0.0, 0.0, 0.0), size=tuple(D["lever_box_m"]), finish="black"),
                              dict(kind="cylinder", pos=tuple(-ex * (D["lever_box_m"][0] + D["logo_disc_m"][1])),
                                   quat=_quat_cols(ey, ez), size=tuple(D["logo_disc_m"]), finish="blue")]))
    cam = np.asarray(M["camera"], float)
    top = cam + np.asarray(D["camera_box_at_m"], float) - ex * D["mast_top_back_m"]
    G = _shape_vertices(scene, "chassis", ("guard_",), prefix=True)
    over = G[(np.abs(G[:, 0] - top[0]) <= D["mast_top_back_m"]) & (G[:, 2] > cam[2])]
    guard_z = float(over[:, 2].min()) if len(over) else float(cam[2])
    parts.append(dict(name="camera_mast", parent=ch, anchor=tuple(cam), mass=B["camera_mass_kg"],
                      inertia=(B["camera_inertia_kg_m2"],) * 3,
                      visual=[dict(kind="box", pos=tuple(D["camera_box_at_m"]), size=tuple(D["camera_box_m"]), finish="black"),
                              _bar(np.asarray(D["camera_box_at_m"]) + ez * D["camera_box_m"][2],
                                   (top[0] - cam[0], 0.0, guard_z - cam[2]), D["mast_bar_m"], "alu")]))
    cush = _shape_vertices(scene, "seat", ("cushion",))
    ctop = np.array([0.5 * (cush[:, 0].min() + cush[:, 0].max()), BL["y"], cush[:, 2].max()])
    fx, fy = BL["footprint"]
    bal_half = (fx / 2, fy / 2, BL["mass"] / BL["density"] / (fx * fy) / 2)
    parts.append(dict(name="ballast", parent=seat, anchor=tuple(ctop + ez * bal_half[2]), mass=BL["mass"],
                      inertia=_box_inertia(BL["mass"], bal_half),
                      visual=[dict(kind="box", pos=(0.0, 0.0, 0.0), size=bal_half, finish="steel")]))

    # ---- steering robot: stator (reaction arm to the cowl) and rotor with a fork round the spinner knob
    c = np.asarray(T.anchors["steering_wheel"], float)
    kc = np.asarray(T.anchors["knob"], float)
    K = _shape_vertices(scene, "knob")
    u_r = _unit((kc - c) - ((kc - c) @ n) * n)
    u_t = np.cross(n, u_r)
    stator_at = c - n * M["stator_behind_rim"]
    st_kg = prof["steer"]["mass_kg"] - A["steer_rotor_mass_kg"]
    ring_q = _quat_cols(u_t, u_r)
    ra = np.asarray(D["reaction_arm_m"], float) - u_t * D["reaction_arm_offset_m"]
    parts.append(dict(name="steer_stator", parent=ch, anchor=tuple(stator_at), mass=st_kg,
                      inertia=tuple(B["stator_inertia_kg_m2"]),
                      visual=[dict(kind="ring", pos=tuple(n * D["stator_ring_at_m"]), quat=ring_q, size=tuple(D["stator_ring_m"]),
                                   finish="alu"),
                              _bar(-u_t * D["reaction_arm_offset_m"], ra, D["reaction_arm_bar_m"], "alu")]))
    r_knob = float(np.linalg.norm((kc - c) - ((kc - c) @ n) * n))
    coll, vis, ext = fork_parts(K, kc, c, u_t, u_r, n, FK["cover_fraction"])
    vis.append(dict(kind="ring", pos=tuple(-n * D["rotor_ring_at_m"]), quat=ring_q, size=tuple(D["rotor_ring_m"]), finish="alu"))
    for cl in coll:
        p_ = np.asarray(cl["pos"], float)
        vis.append(_bar(p_ - n * cl["size"][2], (p_ @ u_t) * u_t + u_r * D["rotor_post_radius_m"] - n * D["rotor_ring_at_m"],
                        thin, "alu"))
    arm("steer", "steer_stator", c, n, -A["steer_range_rad"], A["steer_range_rad"], r_knob, coll, vis,
        A["steer_rotor_mass_kg"], extra=dict(fork=ext))

    # ---- pedal robots: an arm about each pedal's pivot, a flat foot a few millimetres off the pad
    for name, pedal, tags in (("accelerator", "pedal_accel", None), ("brake", "pedal_brake", ("pad",)),
                              ("inching", "pedal_inch", ("pad",))):
        piv = np.asarray(T.anchors[pedal], float)
        s = TR.PEDAL_PRESS_SIGN[pedal]
        # a pedal that rests turned from the asset's pose (the inching pedal, PEDAL_INCH_RAISE; truck.py turns its joint
        # frame by -s * T.pedal_rest[pedal] about the pivot axis): its asset-pose vertices are turned the same way here
        rest = -s * float(getattr(T, "pedal_rest", {}).get(pedal, 0.0))
        cr, sr = math.cos(rest), math.sin(rest)
        R_rest = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
        V = piv + (_shape_vertices(scene, pedal, tags) - piv) @ R_rest.T
        if pedal == "pedal_accel":
            # the organ pedal's plate: its long axis in the pedal's plane of motion, the face normal across it facing
            # away from the press; the press point up the plate from its centroid (operator.accel_pad's point)
            cm = V.mean(0)
            xz = (V - cm)[:, [0, 2]]
            _, _, Wt = np.linalg.svd(xz, full_matrices=False)
            along = _unit((Wt[0][0], 0.0, Wt[0][1]))
            along = along * np.sign(along[2])
            cp = cm + along * FT["organ_press_point_m"]
            press = _unit(s * np.cross(ey, cp - piv))
            n_out = _unit(np.cross(along, ey))
            n_out = -n_out if n_out @ press > 0 else n_out
        else:
            cp = 0.5 * (V.min(0) + V.max(0))
            r = cp - piv
            r = r - (r @ ey) * ey
            n_out = -_unit(s * np.cross(ey, r))
        w_ = _unit(np.cross(n_out, ey))
        hw = min(FT["max_half_width_m"], FT["pad_cover_fraction"] * float(V[:, 1].max() - V[:, 1].min()))
        hl = min(FT["max_half_length_m"], FT["pad_cover_fraction"] * float(((V - cp) @ w_).max() - ((V - cp) @ w_).min()))
        # the pad's surface under the foot: the highest collider point along n_out inside the foot's footprint
        Vd = piv + (_shape_vertices(scene, pedal, tags, dense=mech["search"]["dense_steps"]) - piv) @ R_rest.T
        under = (np.abs((Vd - cp) @ ey) <= hw) & (np.abs((Vd - cp) @ w_) <= hl)
        h = float((((Vd[under] if under.any() else Vd) - cp) @ n_out).max())
        face = cp + n_out * (h + FT["gap_m"])
        foot = face + n_out * FT["half_thickness_m"]
        contact = cp + n_out * h
        rr = float(np.linalg.norm((contact - piv) - ((contact - piv) @ ey) * ey))
        gap_rad = _angular_gap(Vd, piv, ey, s, face, n_out, ey, w_, hw, hl, mech["search"])
        travel = {"pedal_accel": P.PEDAL_ACCEL, "pedal_brake": P.PEDAL_BRAKE, "pedal_inch": P.PEDAL_INCH}[pedal]["travel"]
        hi_ = s * (travel + gap_rad + A["overtravel_rad"])
        lo_ = -s * A["retract_rad"]
        q = _quat_cols(n_out, ey)
        side = ey * (D["arm_offset_m"] if pedal != "pedal_accel" else -D["arm_offset_m"])
        size = (FT["half_thickness_m"], hw, hl)
        vis = [dict(kind="box", pos=tuple(foot - piv), quat=q, size=size, finish="alu"),
               _bar(side, foot - piv + side, bar, "black"), _bar(foot - piv + side, foot - piv, thin, "steel")]
        arm(name, "pedal_unit", piv, ey, min(lo_, hi_), max(lo_, hi_), rr,
            [dict(kind="box", pos=tuple(foot - piv), quat=q, size=size, label="foot")], vis, A["mass_kg"],
            extra=dict(press_sign=s, gap_rad=gap_rad, travel=travel))

    # ---- lever robots: an arm about each lever's pivot, a fork round its knob
    for name, lever in (("lift", "lever_lift"), ("tilt", "lever_tilt"), ("shift", "lever_shift")):
        piv = np.asarray(T.anchors[lever], float)
        V = _shape_vertices(scene, lever, ("knob",))
        kc_ = 0.5 * (V.min(0) + V.max(0))
        r = kc_ - piv
        r = r - (r @ ey) * ey
        ur = _unit(r)
        coll, vis, ext = fork_parts(V, kc_, piv, np.cross(ey, ur), ur, ey, FK["lever_cover_fraction"])
        vis.append(_bar((0.0, 0.0, 0.0), kc_ - piv - ur * D["fork_arm_back_m"], thin))
        travel = P.LEVER["travel"]
        arm(name, "lever_bank", piv, ey, -(travel + A["overtravel_rad"]), travel + A["overtravel_rad"],
            float(np.linalg.norm(r)), coll, vis, A["mass_kg"], extra=dict(fork=ext, travel=travel))

    # ---- selector robot: an arm about the direction stalk's axis (the column axis), a fork round the stalk's tip
    piv = np.asarray(T.anchors["dir_lever"], float)
    V = _shape_vertices(scene, "dir_lever")
    d = V - piv
    rad = np.linalg.norm(d - np.outer(d @ n, n), axis=1)
    tip = V[rad > rad.max() - FK["stalk_tip_region_m"]]
    kc_ = 0.5 * (tip.min(0) + tip.max(0))
    ur = _unit((kc_ - piv) - ((kc_ - piv) @ n) * n)
    coll, vis, ext = fork_parts(tip, kc_, piv, np.cross(n, ur), ur, n, FK["cover_fraction"])
    vis.append(_bar(n * D["stalk_arm_up_m"], kc_ - piv + n * D["stalk_arm_up_m"], thin))
    travel = P.DIR_LEVER["travel"]
    arm("direction", "lever_bank", piv, n, -(travel + A["overtravel_rad"]), travel + A["overtravel_rad"],
        float(np.linalg.norm((kc_ - piv) - ((kc_ - piv) @ n) * n)), coll, vis, A["mass_kg"],
        extra=dict(fork=ext, travel=travel))

    # ---- hand-brake lever robot: an arm about the parking-brake lever's pivot, a fork round its handle
    # (truck.PARK_PIVOT and the lever's colliders are read at build: the lever's geometry is the truck's)
    piv = np.asarray(T.anchors["park_lever"], float)
    V = _shape_vertices(scene, "park_lever")
    d = V - piv
    rad = np.linalg.norm(d - np.outer(d @ ey, ey), axis=1)
    hand = V[rad > rad.max() - FK["handle_region_m"]]
    kc_ = 0.5 * (hand.min(0) + hand.max(0))
    r = kc_ - piv
    r = r - (r @ ey) * ey
    ur = _unit(r)
    coll, vis, ext = fork_parts(hand, kc_, piv, np.cross(ey, ur), ur, ey, FK["cover_fraction"])
    vis.append(_bar(ey * D["park_arm_offset_m"], kc_ - piv + ey * D["park_arm_offset_m"], bar))
    travel = P.PARK_LEVER["travel"]
    arm("park", "lever_bank", piv, ey, -(travel + A["overtravel_rad"]), A["overtravel_rad"], float(np.linalg.norm(r)),
        coll, vis, A["mass_kg"], extra=dict(fork=ext, travel=travel))

    # ---- key robot: a rotor coaxial with the key, two jaws pressed on the key's head by their springs
    kcen = np.asarray(T.anchors["key"], float)
    V = _shape_vertices(scene, "key")
    rk = float(np.linalg.norm((V - kcen)[:, 1:], axis=1).max())
    jaw_x = float(V[:, 0].min()) + KJ["inset_m"] - kcen[0]          # in from the head's end toward the operator
    ja = np.asarray(D["key_mount_at_m"], float)
    parts.append(dict(name="key_mount", parent=ch, anchor=tuple(kcen + ex * ja[0]),
                      mass=prof["key"]["mass_kg"] - A["key_rotor_mass_kg"] - 2 * KJ["mass_kg"],
                      inertia=(B["key_mount_inertia_kg_m2"],) * 3,
                      visual=[dict(kind="cylinder", pos=tuple(ex * ja[1]), quat=_quat_cols(ey, ez), size=tuple(D["key_mount_m"]),
                                   finish="alu")]))
    sl = D["key_sleeve_m"]
    arm("key", "key_mount", kcen, ex, -A["key_overtravel_rad"], P.KEY["start"] + A["key_overtravel_rad"], rk, [],
        [dict(kind="ring", pos=tuple(ex * (jaw_x - KJ["pad_half_m"][0])), quat=_quat_cols(ey, ez),
              size=(rk + sl[0], rk + sl[1], sl[2]), finish="black"),
         _bar(ex * (jaw_x - KJ["pad_half_m"][0]), ex * D["key_post_end_m"], thin, "steel")], A["key_rotor_mass_kg"],
        extra=dict(jaw_preload=KJ["preload_N"]))
    for k, sg in enumerate((+1.0, -1.0)):
        c_j = ex * jaw_x + ez * sg * (rk + KJ["pad_half_m"][2] + KJ["build_clearance_m"])
        parts.append(dict(name=f"key_jaw{k}", parent="key", anchor=tuple(kcen + c_j), mass=KJ["mass_kg"],
                          inertia=(KJ["inertia_kg_m2"],) * 3,
                          joint=dict(kind="prismatic", axis=tuple(-sg * ez), lower=KJ["travel_m"][0], upper=KJ["travel_m"][1],
                                     armature=0.0, friction=0.0, motor=False),
                          collide=[dict(kind="box", pos=(0.0, 0.0, 0.0), size=tuple(KJ["pad_half_m"]), label="jaw")],
                          visual=[dict(kind="box", pos=(0.0, 0.0, 0.0), size=tuple(KJ["pad_half_m"]))]))

    # ---- button robot: a rocker on a bridge over the wheel, its rounded tip on the column axis over the horn button
    hc = np.asarray(T.anchors["horn"], float)
    u = _unit(ex - (ex @ n) * n)
    a = np.cross(n, u)
    tipp = hc + n * (TR.HORN["thick"] / 2 + HN["tip_radius_m"] + HN["gap_m"])
    pv = hc + n * HN["rocker_height_m"] + u * HN["rocker_reach_m"]
    rel = tipp - pv
    sign = -1.0 if np.cross(a, rel) @ n > 0 else 1.0       # the joint direction that moves the tip toward -n
    hb = np.asarray(D["horn_bridge_m"], float)
    parts.append(dict(name="horn_bridge", parent=ch, anchor=tuple(pv + u * D["horn_bridge_at_m"]), mass=HN["bridge_mass_kg"],
                      inertia=(HN["bridge_inertia_kg_m2"],) * 3,
                      visual=[_bar((0.0, 0.0, 0.0), u * hb[0] + n * hb[1], bar, "alu")]))
    arm("horn", "horn_bridge", pv, a, -HN["range_rad"], HN["range_rad"], float(np.linalg.norm(rel - (rel @ a) * a)),
        [dict(kind="sphere", pos=tuple(rel), size=(HN["tip_radius_m"],), label="tip")],
        [_bar((0.0, 0.0, 0.0), rel + n * D["horn_rise_m"], thin), _bar(rel + n * D["horn_rise_m"], rel, thin, "steel")],
        HN["rocker_mass_kg"], extra=dict(press_sign=sign))

    frames = {"head_camera": dict(part="camera_mast", pos=(0.0, 0.0, 0.0),
                                  quat=tuple(Rotation.from_euler("y", math.radians(M["camera_pitch_deg"])).as_quat())),
              "imu": dict(part="frame", pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0))}
    return parts, frames, arms


def add_kit(scene):
    """Build the driving-robot kit into the truck (robots/models/drive_kit/newton.py) from the kit fit."""
    import newton
    import robots
    from robots.runtime import RobotRig
    from robots.models.drive_kit import newton as KB
    from .world import GROUPS
    T = scene.truck
    setup = robots.resolve(KIT_RIGS[scene.c.robot])
    inst, item = next(iter(setup.to_dict()["instances"].items()))
    spec_obj = robots.model(item["model"])
    spec = spec_obj.to_dict()
    parts, frames, arms = kit_fit(scene, spec, item["controller_profile"])
    yaw = scene.truck_pose[2]
    quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    first_body, first_shape = scene.b.body_count, scene.b.shape_count
    gap = spec["mechanisms"]["contact"]["gap_m"]
    cfg = lambda label: newton.ModelBuilder.ShapeConfig(density=0.0, mu=P.MU_ROBOT_SHELL, gap=gap, margin=0.0)
    runtime, info = KB.build(scene.b, instance=inst, spec=spec, setup_profile=item["controller_profile"], parts=parts,
                             host_origin=scene.truck_origin, host_quat=quat,
                             host_anchors={T.bodies["chassis"]: T.anchors["chassis"], T.bodies["seat"]: T.anchors["seat"]},
                             frames=frames, shape_cfg=cfg)
    runtime.definition_fingerprint = spec_obj.fingerprint
    scene.robot_pending = RobotRig(setup, {inst: runtime})
    scene.robot_setup = setup
    scene.robot_bodies = list(range(first_body, scene.b.body_count))
    scene.robot_shapes = list(range(first_shape, scene.b.shape_count))
    scene.robot_mass = float(sum(scene.b.body_mass[i] for i in scene.robot_bodies))
    for i in scene.robot_shapes:
        scene.shape_group[i] = "obj"
        scene.materials[i] = "robot"
    # contact bits above world.GROUPS': each actuator's end effectors collide only with the colliders of its control
    bit0 = max(max(ct, ca) for ct, ca in GROUPS.values()).bit_length()
    ee, targets = {}, {}
    for k, name in enumerate(KIT_ORDER):
        shapes = list(info[name]["colliders"])
        if name == "key":
            shapes += info["key_jaw0"]["colliders"] + info["key_jaw1"]["colliders"]
        ee[name] = (1 << (bit0 + k), shapes)
        _, body, tags = KIT_CONTROLS[name]
        targets[name] = [i for i, (bn, tag, _) in T.shapes.items() if bn == body and (tags is None or tag in tags)]
        if not shapes or not targets[name]:
            raise ValueError(f"drive kit: {name} has no end effector or no target collider")
    scene.kit = dict(instance=inst, info={k: dict(body=v["body"], joint=v["joint"], colliders=v["colliders"])
                                          for k, v in info.items()},
                     arms=arms, ee=ee, targets=targets, imu_body=info["frame"]["body"],
                     jaws=[info["key_jaw0"]["joint"], info["key_jaw1"]["joint"]], ballast_kg=P.DRIVE_KIT_BALLAST["mass"],
                     finishes={k: v for k, v in spec["mechanisms"]["drawing"]["finishes"].items() if isinstance(v, dict)},
                     radius_every=spec["mechanisms"]["search"]["contact_radius_every_ticks"])


def kit_mujoco_hook(world, m):
    """Each kit end effector collides only with the colliders of the control it works (one contype bit per actuator,
    set on the control's conaffinity); the end effectors' friction is the robot shell's (params MU_ROBOT_SHELL), the
    key jaws' the pads' (MU_ROBOT_PAD)."""
    kit = world.scene.kit
    shape_geom = {int(s): g for g, s in enumerate(world._geom_shape[:m.ngeom]) if s >= 0}
    kit_shapes = set(world.scene.robot_shapes)
    for g in range(m.ngeom):                                # no kit geom collides unless listed below
        s = int(world._geom_shape[g]) if g < len(world._geom_shape) else -1
        if s in kit_shapes:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
    kit["geoms"] = {}
    for name, (bit, shapes) in kit["ee"].items():
        gs = [shape_geom[s] for s in shapes]
        for g in gs:
            m.geom_contype[g] = bit
            m.geom_conaffinity[g] = 0
            m.geom_friction[g][0] = MU_ROBOT_PAD["slide"] if name == "key" else MU_ROBOT_SHELL
        tg = [shape_geom[s] for s in kit["targets"][name] if s in shape_geom]
        for g in tg:
            m.geom_conaffinity[g] = int(m.geom_conaffinity[g]) | bit
        kit["geoms"][name] = dict(ee=gs, target=tg)


class KitMotors(Motors):
    """The kit's actuators as position servos (q*, dq*, kp, kd, tau per actuator, as motors.Motors) saturated by each
    product's published force-speed envelope AT ITS OUTPUT: for a rotary output (the steering and key robots) the torque
    about the arm's axis; for a linear output (pedal, lever, selector, hand-brake and button robots) the force the end
    effector drives along its path, i.e. the arm's torque over the radius at which it touches its control. That radius
    is the contact's (the normal-force-weighted distance of the contact points from the arm's axis, re-read every few
    steps), not a nominal one (at the nominal radius the accelerator pushed 191.6 N against its 150 N
    and the brake 798 N against its 750 N); without contact, the nominal radius. The force across the arm's path (along
    its radius) is a constraint force of the two coaxial bearings, not the actuator's.
    Motoring force is interpolated between the published points at the output's speed and falls to zero just above the
    last published speed; braking force (against the motion) is limited by the peak."""

    def __init__(self, world, joints, arms, kit):
        prof = {n.split("/")[-1]: dict(kp=arms[n.split("/")[-1]]["kp"], kd=arms[n.split("/")[-1]]["kd"],
                                       effort_limit=arms[n.split("/")[-1]]["effort"],
                                       velocity_limit=arms[n.split("/")[-1]]["w_nl"]) for n in joints}
        super().__init__(world, joints, prof)
        short = [n.split("/")[-1] for n in self.names]
        self.short = short
        self.group = [arms[s]["group"] for s in short]
        self.kp_max = np.array([arms[s]["kp_max"] for s in short], float)
        self.kd_max = np.array([arms[s]["kd_max"] for s in short], float)
        self.linear = np.array([arms[s]["output"] == "linear" for s in short])
        self.radius = np.array([arms[s]["radius"] if arms[s]["output"] == "linear" else 1.0 for s in short], float)
        self.peak = np.array([arms[s]["peak"] for s in short], float)
        self.env_v = [np.asarray(arms[s]["env_v"], float) for s in short]
        self.env_f = [np.asarray(arms[s]["env_f"], float) for s in short]
        self.axis_local = [np.asarray(arms[s]["axis"], float) for s in short]
        self.arm_body = [kit["info"][s]["body"] for s in short]
        self.ee = [np.asarray(kit["geoms"][s]["ee"], int) for s in short]
        self.tgt = [np.asarray(kit["geoms"][s]["target"], int) for s in short]
        self.arm = self.radius.copy()                       # the contact radius in use (m; 1 for rotary outputs)
        self.every = int(kit["radius_every"])
        self.contact_force = np.zeros(len(short))           # the end effector's force along its path, last step (N)

    def contact_arms(self):
        """Per linear actuator, from the last step's contacts between its end effectors and its control: the
        normal-force-weighted distance of the contact points from the arm's axis (m), or the nominal radius without
        contact."""
        import mujoco
        w = self.w
        m, d = w.solver.mj_model, w.solver.mj_data
        n = int(d.ncon)
        arm = self.radius.copy()
        self.contact_force[:] = 0.0
        if n == 0:
            return arm
        if getattr(self, "_ee_of", None) is None:          # geom -> linear actuator of its end effector; its targets
            self._ee_of = np.full(m.ngeom, -1, int)
            self._tgt_of = np.zeros((len(self.names), m.ngeom), bool)
            for i in np.flatnonzero(self.linear):
                self._ee_of[self.ee[i]] = i
                self._tgt_of[i, self.tgt[i]] = True
        con = d.contact
        g1, g2 = np.asarray(con.geom1[:n]), np.asarray(con.geom2[:n])
        k1, k2 = self._ee_of[g1], self._ee_of[g2]
        cand = np.flatnonzero((k1 >= 0) | (k2 >= 0))
        if not len(cand):
            return arm
        bq = w.state.body_q.numpy()
        f6 = np.zeros(6)
        acc = {}
        frames, poss, fric = np.asarray(con.frame), np.asarray(con.pos), np.asarray(con.friction)
        for c in cand:
            i, other, ctrl_is_2 = (k1[c], g2[c], True) if k1[c] >= 0 else (k2[c], g1[c], False)
            if not self._tgt_of[i, other]:
                continue
            if i not in acc:
                b = self.arm_body[i]
                ax = quat_to_R(bq[b:b + 1, 3:7].astype(float))[0] @ self.axis_local[i]
                acc[i] = [bq[b, :3].astype(float), ax, 0.0, 0.0, 0.0]
            a, ax = acc[i][0], acc[i][1]
            mujoco.mj_contactForce(m, d, int(c), f6)
            fn = abs(float(f6[0]))
            ft, cap = float(np.hypot(f6[1], f6[2])), float(fric[c][0]) * fn
            if ft > cap:                                            # read back outside its friction cone: to the cone
                f6[1:3] *= cap / ft
            Fw = frames[c].reshape(3, 3).T @ f6[:3]                 # on geom2 (the normal points 1 -> 2)
            Fc = Fw if ctrl_is_2 else -Fw                           # on the control
            rv = poss[c].astype(float) - a
            acc[i][2] += float(np.cross(rv, Fc) @ ax)
            acc[i][3] += fn
            acc[i][4] += fn * float(np.linalg.norm(rv - (rv @ ax) * ax))
        for i, (_, _, M, wsum, rsum) in acc.items():
            if wsum > 0.0:
                arm[i] = rsum / wsum
                self.contact_force[i] = abs(M) / arm[i]             # along the end effector's path
        return arm

    def write(self, control_q, qd):
        m = self.w.solver.mj_model
        c = self.cmd
        if self._gains_dirty:
            m.actuator_biasprm[self.act_pos, 1] = -c["kp"]
            m.actuator_biasprm[self.act_pos, 2] = -c["kd"]
            self._gains_dirty = False
        control_q[self.q_idx] = c["kp"] * c["q"] + c["kd"] * c["dq"] + c["tau"]
        v = qd[self.dof]
        if self.w.tick % self.every == 0:                   # the contact radius, re-read every few steps
            self.arm = self.contact_arms()
        v_out = np.abs(v) * self.radius
        motoring = np.array([np.interp(float(v_out[i]), self.env_v[i], self.env_f[i], right=0.0)
                             for i in range(len(self.names))]) * self.arm
        braking = self.peak * self.arm
        m.actuator_forcerange[self.act_pos, 0] = np.where(v >= 0.0, -braking, -motoring)
        m.actuator_forcerange[self.act_pos, 1] = np.where(v >= 0.0, motoring, braking)


class KitOperator:
    """The kit as the session's robot: its motors (KitMotors: the policy's command channels), the jaw springs of the key
    robot (a force element on the jaw joints), and its IMU on the kit frame (env.robot_status reads `pelvis`)."""

    def __init__(self, world):
        self.w = world
        self.handle = h = next(iter(world.rig.instances.values()))
        self.joints = dict(h.joints)
        self.model_id = h.model_id
        kit = world.scene.kit
        self.kit = kit
        motor = {n: a for n, a in h.joints.items() if a.get("motor")}
        self.motors = KitMotors(world, motor, kit["arms"], kit)
        self.pelvis = kit["imu_body"]
        qds = world.model.joint_qd_start.numpy()
        self.jaw_dofs = [int(qds[j]) for j in kit["jaws"]]
        self.jaw_preload = float(kit["arms"]["key"]["jaw_preload"])
        self.belt = dict(fastened=False, locked=False, tension=(0.0, 0.0))
        self.events = []

    def write(self, world, jf):
        ctrl = world.control.joint_target_q.numpy()
        qd = world.state.joint_qd.numpy()
        self.motors.write(ctrl, qd)
        world.control.joint_target_q.assign(ctrl)
        for d in self.jaw_dofs:                              # the jaws' springs close them on the key's head
            jf[d] += self.jaw_preload

    def apply_belt(self, wrench):
        return None

    def proprioception(self):
        q = self.w.state.joint_q.numpy(); qd = self.w.state.joint_qd.numpy()
        tau = self.motors.torques()
        return {n: dict(position=float(q[self.joints[n]["q"]]), velocity=float(qd[self.joints[n]["dof"]]),
                        effort=float(tau[self.motors.index[n]])) for n in self.motors.names}


def kit_start(world):
    """Installation pose: every actuator's arm at the angle of the control it works (the pedal feet their gap off their
    pads), set before tick 0 like the humanoid's seated posture, then held by the motors' start command."""
    world.operator = op = KitOperator(world)
    qa = world.model.joint_q.numpy()
    for name in KIT_ORDER:
        jn = f"{op.handle.instance}/{name}"
        ctrl = KIT_CONTROLS[name][0]
        qa[op.joints[jn]["q"]] = 0.0 if name == "horn" else float(qa[world.q_of[ctrl]])
    world.model.joint_q.assign(qa)
    world.state.joint_q.assign(qa)
    world.state.joint_qd.zero_()
    world.newton.eval_fk(world.model, world.state.joint_q, world.state.joint_qd, world.state)
    world.next_state.assign(world.state)
    world.sync_solver()
    op.motors.hold_current()
    op.start_command = op.motors.checkpoint()


# ====================================================================================================================
def policy_notes(world, min_ticks, max_s):
    """The robot-specific policy notes of a session (the backend's capabilities 'control' and 'cameras'), from the
    robot the session runs: a seated humanoid (its SEAT_FIT name and hands) or the driving-robot kit."""
    op = world.operator
    if isinstance(op, KitOperator):
        arms = world.scene.kit["arms"]
        inst = op.handle.instance
        acts = "; ".join(f"{n} ({a['product'].split(' (')[0]}: {a['effort']:.3g} N m"
                         + (f" = {a['effort'] / a['radius']:.0f} N at {a['radius'] * 1000:.0f} mm from its axis" if a["output"] == "linear" else "")
                         + (f"; press_sign {a['press_sign']:+.0f}, gap {a['gap_rad']:.4f} rad, travel {a['travel']:.4f} rad"
                            if "gap_rad" in a else "") + ")" for n, a in arms.items())
        control = (f"Driving-robot kit: one position servo per actuator, named_joint_target names '{inst}/<actuator>' "
                   "(q*, rad: the actuator's arm turns about the axis of the control it works, so q* is that control's "
                   "angle; a pedal's foot meets its pad after a small gap, so a pedal at fraction f of its travel needs "
                   "q* = press_sign (f travel + gap), each pedal's sign, gap and travel listed below), '.dq' (rad/s), '.kp' (N m/rad), '.kd' (N m s/rad), "
                   "'.tau' (N m feed-forward); unnamed channels keep their last value. tau = kp (q* - q) + kd (dq* - dq) "
                   "+ tau_ff, clipped to the product's published force-speed envelope at its output (the motor's "
                   f"velocity limit is where its force reaches zero). A command lasts at least {min_ticks} ticks "
                   f"({min_ticks * world.config.dt * 1e3:g} ms) and at most {max_s:g} s. The key robot's jaws are "
                   "spring-closed, not motors. The start command holds every actuator at its control's start angle. "
                   f"Actuators: {acts}. Observation: each actuator's angle, rate and output torque, and the IMU on the "
                   "kit's frame")
        cameras = ("Ideal pinhole RGB of the kit's cab camera (848 x 480, 69.4 deg horizontal: a D435i-class colour "
                   "stream) on a mast hanging from the overhead guard at an operator's eye point, looking forward "
                   f"{P.DRIVE_KIT_MOUNTS['camera_pitch_deg']:g} deg down, rendered by Isaac RTX from the simulated state at a 30 Hz frame "
                   "clock; no depth, noise or motion blur")
        return dict(control=control, cameras=cameras)
    fit = op.fit
    name = fit.get("name", op.model_id)
    followers = f" The {fit.get('hands', 'hands')}' mimic followers are not motors." if op.motors.followers else ""
    control = (f"Unitree low-level motor command per {name} joint: named_joint_target names '<joint>' (q*, rad), "
               "'<joint>.dq' (rad/s), '<joint>.kp' (N m/rad), '<joint>.kd' (N m s/rad), '<joint>.tau' (N m "
               "feed-forward); unnamed channels keep their last value. tau = kp (q* - q) + kd (dq* - dq) + tau_ff, "
               f"clipped to the motor's torque-speed envelope. A command lasts at least {min_ticks} ticks "
               f"({min_ticks * world.config.dt * 1e3:g} ms) and at most {max_s:g} s.{followers} The start command "
               "holds the seated posture with a gravity feed-forward")
    cameras = (f"Ideal pinhole RGB of the {name} head camera (848 x 480, 69.4 deg horizontal: the D435i colour "
               "stream) rendered by Isaac RTX from the simulated state at a 30 Hz frame clock; no depth, noise or "
               "motion blur")
    return dict(control=control, cameras=cameras)
