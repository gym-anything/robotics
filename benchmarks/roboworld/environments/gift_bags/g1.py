"""Unitree G1 (29 body joints, two Dex3-1 hands) configuration of gift-bags (rig robots/rigs/gift-g1.json).

The humanoid stands on the floor in front of the counter or the dining table (scene.G1_LAYOUT) and works on the same
packaging as the Franka configurations. Nothing here is task logic: this module holds the robot-specific pieces the
rig, the coupled world and the GAR adapter need -- the head camera mount, the Dex3 hand postures that gripper commands
map to, which links are Dex3 finger surfaces, the arm IK that turns Cartesian hand targets into joint targets (the
physical position servos then move the arm), and the standing measurement.

Robot model: robots/models/g1/robot.json (pinned URDF g1_29dof_with_hand_rev_1_0, sha256 555edb93...), controller
profile "barcode_scanning" used unchanged (the repository's standing G1 gains: legs kp 800 / kd 20, waist and
shoulders/elbows 120 / 6, wrists 80 / 3, Dex3 fingers 1.8 / 0.035; torques saturate at the URDF effort limits). The robot
is floating: no weld to the floor, no pelvis support, no balance controller -- it stands on its PD servos as in the
whiteboard and barcode-scanning environments, with the arm/waist gravity feed-forward Unitree's manipulation stack sends
(rig.RobotRig._g1_gravity_feedforward). Its links collide in MuJoCo with their URDF collision geometry (meshes' convex
hulls, the shoulder cylinders, four 5 mm spheres per foot); in the paper contact world every link is a driven obstacle
(world.World), the Dex3 finger links with the friction class "dex3_finger" (params.FRICTION_DEX3).

Dex3-1 hands: seven revolute joints each (thumb 3, index 2, middle 2), position servos with the URDF effort limits
(thumb_0 2.45 N m, the others 1.4 N m). A finger presses only through its joint servo against the contact reaction:
there is no grasp primitive, force mode or parallel closing axis. A gripper command maps to a posture between HAND_OPEN
and HAND_CLOSED (closure = 1 - half_width / 0.04 m), stated in the adapter's capabilities.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

INSTANCE = "g1"
SETUP = "gift-g1"
MOUNT = "g1_floor_mount"
PELVIS_Z = 0.79                     # m, pelvis height at import (whiteboard/robot.py, barcode_scanning/g1.py): feet ~5 mm
                                    # above the floor, the robot settles onto them in the first tenths of a second
SIDES = ("left", "right")
# MuJoCo step of the floating G1: 0.5 ms. Barcode scanning measured it falling within 15 s of hands-off rest at 1 ms and
# standing 30 s at 0.5 ms, and its Dex3 grip slipping 3.7 / 0.7 mm (barcode_scanning/config.py G1_RIGID_DT_S).
# In this rig (robot-only, 10 s hands-off) it stood at both steps, the feet creeping more at 1 ms. World refuses a G1
# configuration with a longer step; Config(rigid_dt=5e-4, substeps=8) keeps the 4 ms contact step.
RIGID_DT_MAX = 5e-4
SUBSTEPS = 8

# Head camera: the D435 of the G1 head (URDF d435_joint on torso_link, 47.6 deg down), zero lens offset, 69 deg HFOV
# (whiteboard/sensors.py g1_rgb mount, as in barcode_scanning/g1.py; D435 RGB field of view from Intel's datasheet, not
# re-measured here)
HEAD_CAMERA = dict(link="torso_link", xyz=(0.0576235, 0.01753, 0.42987), pitch=0.8307767239493009, hfov_deg=69.)

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
WAIST_JOINTS = ("waist_yaw", "waist_roll", "waist_pitch")
TCP_LOCAL_M = np.array([0.08, 0.0, 0.0])        # whiteboard/ik.py HandIK: 80 mm along the wrist-yaw link +X (palm centre)
# Dex3 links whose surfaces are finger surfaces (friction class dex3_finger); the palm and every other link are "link"
FINGER_LINKS = tuple(f"{s}_hand_{n}_link" for s in SIDES for n in HAND_JOINTS)

# Hand postures (joint values per side, rad). OPEN: the URDF zero pose, fingers straight (thumb-to-index hull distance
# 110 mm). TIP_PINCH: the thumb opposed to the index and middle fingers with the closest points of the distal links near
# their tips, 10.8 mm apart (forward kinematics grid on MuJoCo's collision hulls). CLOSED: TIP_PINCH with the thumb's
# distal joint and both fingers' distal joints flexed 0.25 rad further, so the distal links meet and the servos squeeze
# (the force a closed hand presses with has NOT been measured: the G1 is development-only here). Right hand: index/middle flex
# positive, thumb_2 negative; the left hand mirrors the signs (URDF limits).
HAND_OPEN = {s: {n: 0. for n in HAND_JOINTS} for s in SIDES}
_TIP_PINCH_RIGHT = dict(thumb_0=0., thumb_1=0., thumb_2=-1.10, index_0=1.40, index_1=.60, middle_0=1.40, middle_1=.60)
_CLOSED_RIGHT = dict(thumb_0=0., thumb_1=0., thumb_2=-1.35, index_0=1.40, index_1=.85, middle_0=1.40, middle_1=.85)
_mirror = lambda d: {n: (v if n == "thumb_0" else -v) for n, v in d.items()}
TIP_PINCH = {"right": dict(_TIP_PINCH_RIGHT), "left": _mirror(_TIP_PINCH_RIGHT)}
HAND_CLOSED = {"right": dict(_CLOSED_RIGHT), "left": _mirror(_CLOSED_RIGHT)}
HALF_WIDTH_OPEN = .04               # m, the Franka Hand's half-gap scale the GAR Gripper action uses


# Paper contact world geometry of the G1 links: the URDF collision meshes are the visual STL meshes, whose convex hulls
# (what MuJoCo collides) carry 44,236 vertices over the 42 links against 2,260 for the whole FR3 pair, which made one
# 4 ms contact step take ~1.4 s. Each hull is replaced by the hull of a subset of its own vertices, grown greedily until
# every original hull vertex lies within HULL_TOL_M of a face plane of the subset hull: an inner approximation (never
# larger than the MuJoCo hull), 5,127 vertices at 0.5 mm.
HULL_TOL_M = 5e-4


def inner_hull(V, tol=HULL_TOL_M):
    """(vertices, faces) of a convex hull of a subset of the convex hull vertices of V such that every hull vertex of V is
    within `tol` of a face plane of it (greedy farthest-point growth from the axis and 20 seeded direction extremes)."""
    import trimesh
    P = np.asarray(trimesh.convex.convex_hull(np.asarray(V, float)).vertices)
    idx = set()
    for k in range(3):
        idx |= {int(np.argmin(P[:, k])), int(np.argmax(P[:, k]))}
    for d in np.random.default_rng(0).normal(size=(20, 3)):
        idx.add(int(np.argmax(P @ d)))
    while True:
        h = trimesh.convex.convex_hull(P[sorted(idx)])
        n = h.face_normals
        out = P @ n.T - (n*h.triangles[:, 0]).sum(1)
        best = np.argmax(out, 0)
        add = {int(best[f]) for f in np.flatnonzero(out[best, np.arange(len(n))] > tol)} - idx
        if not add:
            return np.asarray(h.vertices, float), np.asarray(h.faces)
        idx |= add


def joint(side, name):
    """Policy joint name: joint('right', 'elbow') -> 'g1/right_elbow_joint', joint('right', 'index_0') ->
    'g1/right_hand_index_0_joint', joint(None, 'waist_pitch') -> 'g1/waist_pitch_joint'."""
    if side is None:
        return f"{INSTANCE}/{name}_joint"
    mid = "hand_" if name in HAND_JOINTS else ""
    return f"{INSTANCE}/{side}_{mid}{name}_joint"


def hand_posture(side, closure):
    """Joint targets {policy name: value} of a Dex3 hand at `closure` in [0, 1] (0 open, 1 closed)."""
    if side not in SIDES:
        raise ValueError(f"No Dex3 hand '{side}' on the G1 (left, right)")
    c = float(closure)
    if not math.isfinite(c) or not 0. <= c <= 1.:
        raise ValueError("hand closure must be in [0, 1]")
    return {joint(side, n): (1. - c)*HAND_OPEN[side][n] + c*HAND_CLOSED[side][n] for n in HAND_JOINTS}


def closure_of_half_width(half_width_m):
    w = float(half_width_m)
    if not math.isfinite(w) or not 0. <= w <= HALF_WIDTH_OPEN:
        raise ValueError(f"half_width_m must be in [0, {HALF_WIDTH_OPEN}] m (G1: mapped to the Dex3 hand posture)")
    return 1. - w/HALF_WIDTH_OPEN


def is_g1(robot):
    """True when the gift-bags robot selection resolves to the G1 configuration."""
    if robot is None:
        return False
    if isinstance(robot, str) and robot in ("g1", SETUP):
        return True
    from robots.catalog import for_environment
    try:
        setup = for_environment("gift-bags", robot)
    except ValueError:
        return False
    return any(i.get("model", "").split("@")[0] == "g1" for i in setup.to_dict()["instances"].values())


def body_index(model_or_builder, suffix):
    labels = list(model_or_builder.body_label)
    hits = [i for i, lab in enumerate(labels) if lab.startswith(INSTANCE + "/") and lab.split("/")[-1] == suffix]
    if len(hits) != 1:
        raise RuntimeError(f"G1 body {suffix}: {len(hits)} matches")
    return hits[0]


def head_camera(rig):
    """Register the head RGB (policy camera) on the torso link, like RobotRig._wrist_camera does for the FR3."""
    body = body_index(rig.b, HEAD_CAMERA["link"])
    R = Rotation.from_euler("xyz", (0., HEAD_CAMERA["pitch"], 0.))
    eye = np.asarray(HEAD_CAMERA["xyz"], float)
    rig.cameras["head_rgb"] = dict(body=body, eye=eye.tolist(), target=(eye + .3*R.apply((1., 0., 0.))).tolist(),
                                   up=R.apply((0., 0., 1.)).tolist(), fov=HEAD_CAMERA["hfov_deg"], role="policy")


def pelvis_state(rig):
    """Pelvis position and tilt from vertical (deg): what the robot's IMU would report."""
    i = body_index(rig.model, "pelvis")
    q = rig.state.body_q.numpy()[i]
    z = Rotation.from_quat(q[3:]).apply((0., 0., 1.))
    return q[:3].astype(float), math.degrees(math.acos(max(-1., min(1., float(z[2])))))


REG_M_PER_RAD = 1e-3       # IK pull toward the seed: 1 mm of residual per rad of joint change (resolves the redundancy)


class G1Kinematics:
    """Seven-joint arm IK of each G1 arm (whiteboard/ik.py HandIK and barcode_scanning/g1.py G1HandIK: same TCP, bounded
    least squares) on a PRIVATE copy of the robot's MuJoCo data: it reads the current state (floating base, waist, the
    other arm) and returns joint targets; it never writes the simulation state. The targets are then tracked by the
    physical position servos. Same interface as kinematics.ArmKinematics (arms, joints_now, forward, solve,
    cartesian_path) so the GAR adapter plans G1 Cartesian commands like the Franka ones."""

    def __init__(self, rig):
        import mujoco
        self.rig, self.mj = rig, mujoco
        self.model = rig.solver.mj_model
        self.data = mujoco.MjData(self.model)
        names = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(self.model.njnt)}
        bnames = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(self.model.nbody)}
        self.arms, self._chains = {}, {}
        for side in SIDES:
            js = [joint(side, n) for n in ARM_JOINTS]
            ids = []
            for n in js:
                short = n.split("/")[-1]
                hit = [j for k, j in names.items() if k and k.endswith("_" + short)]
                if len(hit) != 1:
                    raise RuntimeError(f"G1 IK: MuJoCo joint for {n}: {hit}")
                ids.append(hit[0])
            body = [i for k, i in bnames.items() if k and k.endswith(f"{side}_wrist_yaw_link")]
            if len(body) != 1:
                raise RuntimeError(f"G1 IK: wrist body for {side}: {body}")
            lo = np.array([rig.joints[n]["lower"] for n in js])
            hi = np.array([rig.joints[n]["upper"] for n in js])
            self._chains[side] = (js, np.asarray(self.model.jnt_qposadr[ids]), body[0], lo, hi)
            self.arms[side] = (None, js)

    def _sync(self):
        self.rig.solver._update_mjc_data(self.data, self.rig.model, self.rig.state)
        self.mj.mj_kinematics(self.model, self.data)

    def joints_now(self, arm):
        q = self.rig.state.joint_q.numpy()
        return np.array([q[self.rig.joints[n]["q"]] for n in self._chains[arm][0]])

    def _tcp(self, body):
        R = self.data.xmat[body].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.data.xpos[body] + R @ TCP_LOCAL_M
        return T

    def forward(self, arm, q=None):
        """4x4 world pose of the hand TCP now, or with this arm at joint values q (rest of the robot as it is now)."""
        if arm not in self._chains:
            raise ValueError(f"No arm '{arm}' on the G1 (left, right)")
        js, adr, body, lo, hi = self._chains[arm]
        self._sync()
        if q is not None:
            self.data.qpos[adr] = np.asarray(q, float)
            self.mj.mj_kinematics(self.model, self.data)
        return self._tcp(body)

    def _solve(self, arm, position, target_R, seed, rot_weight=.15, reg=REG_M_PER_RAD):
        """Bounded least squares from `seed`. The arm has 7 joints for a 6-D target: without the small pull `reg` toward
        the seed the solver drifted along the redundant direction, so a 1 cm Cartesian step from the arms-down start
        came back 0.37 rad away (a refused branch change)."""
        from scipy.optimize import least_squares
        js, adr, body, lo, hi = self._chains[arm]
        q_seed = np.clip(np.asarray(seed, float), lo + 1e-6, hi - 1e-6)

        def err(q):
            self.data.qpos[adr] = q
            self.mj.mj_kinematics(self.model, self.data)
            T = self._tcp(body)
            R = Rotation.from_matrix(T[:3, :3])
            return np.r_[T[:3, 3] - position, rot_weight*(target_R*R.inv()).as_rotvec(), reg*(q - q_seed)]
        r = least_squares(err, q_seed, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300)
        res = err(r.x)
        return r.x.copy(), float(np.linalg.norm(res[:3])), float(np.linalg.norm(res[3:6])/rot_weight)

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=2e-3, tol_rad=.03, restarts=0):
        """Joint targets {name: value} putting the TCP at `position` (world, m) with orientation `orientation_xyzw` (None:
        keep the current one). Bounded to the URDF limits; raises ValueError (no action executed) when the residual
        exceeds the tolerances (whiteboard HandIK acceptance: 2 mm, 0.03 rad). With `restarts`, seeds from a fixed-seed
        generator are tried too and the solution closest (max joint change) to the current configuration is kept."""
        if arm not in self._chains:
            raise ValueError(f"No arm '{arm}' on the G1 (left, right)")
        js, adr, body, lo, hi = self._chains[arm]
        position = np.asarray(position, float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("IK target must be three finite metre coordinates")
        self._sync()
        now = self.data.qpos[adr].copy()
        start = now if seed is None else np.asarray(seed, float)
        target = Rotation.from_matrix(self._tcp(body)[:3, :3]) if orientation_xyzw is None \
            else Rotation.from_quat(orientation_xyzw)
        rng = np.random.default_rng(12345)
        best, first = None, None
        for s0 in [start] + [rng.uniform(lo, hi) for _ in range(int(restarts))]:
            q, pe, ae = self._solve(arm, position, target, s0)
            first = first or (pe, ae)
            if pe <= tol_m and ae <= tol_rad:
                dq = float(np.abs(q - now).max())
                if best is None or dq < best[0]:
                    best = (dq, q)
                if s0 is start:
                    break
        if best is None:
            pe, ae = first
            raise ValueError(f"IK unreachable ({pe*1e3:.1f} mm, {math.degrees(ae):.1f} deg); no action executed")
        return dict(zip(js, map(float, best[1])))

    def cartesian_path(self, arm, start_q, poses, step_m=.002, step_rad=math.radians(2.), max_jump_rad=.35):
        """kinematics.ArmKinematics.cartesian_path for a G1 arm: joint targets along the straight-line, slerped TCP motion
        through `poses` [(position, quat_xyzw), ...] from joint vector `start_q`, every sample solved locally from the
        previous one (one IK branch); ValueError before anything moves if a pose is unreachable (2 mm / 0.03 rad) or a
        joint would jump more than `max_jump_rad` between samples. The rest of the robot (floating base, waist, the other
        arm) is taken as it is now. Samples are 2 mm / 2 deg apart (the FR3's are 1 cm / 8 deg): from the arms-down start a
        vertical TCP move turns the elbow 0.07 rad per mm, which at 1 cm samples read as a 0.37 rad jump although the
        motion is continuous."""
        from scipy.spatial.transform import Slerp
        if arm not in self._chains:
            raise ValueError(f"No arm '{arm}' on the G1 (left, right)")
        js, adr, body, lo, hi = self._chains[arm]
        q = np.clip(np.asarray(start_q, float), lo + 1e-6, hi - 1e-6)
        T = self.forward(arm, q)
        p0, r0 = T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3])
        out = []
        for k, (pos, quat) in enumerate(poses):
            p1, r1 = np.asarray(pos, float), Rotation.from_quat(quat)
            ang = (r0.inv()*r1).magnitude()
            n = max(1, int(math.ceil(max(np.linalg.norm(p1 - p0)/step_m, ang/step_rad))))
            sl = Slerp([0., 1.], Rotation.concatenate([r0, r1]))
            for i in range(1, n + 1):
                s = i/n
                qn, pe, ae = self._solve(arm, p0 + s*(p1 - p0), sl([s])[0], q)
                if pe > 2e-3 or ae > .03:
                    raise ValueError(f"path pose {k} step {i}/{n} unreachable ({pe*1e3:.1f} mm, {math.degrees(ae):.1f} deg)")
                jump = float(np.abs(qn - q).max())
                if jump > max_jump_rad:
                    raise ValueError(f"path pose {k} step {i}/{n}: joint jump {jump:.2f} rad (IK branch change)")
                q = qn
                out.append((dict(zip(js, map(float, q))), k))
            p0, r0 = p1, r1
        return out
