"""Unitree G1 (29 body joints, two Dex3-1 hands) configuration of percussion.

The humanoid stands on the floor: behind the drum in the gait preset, at the instrument table's near edge in the table
presets (the drum then stands at its right hand). It plays with its Dex3 fingertips (vibro-contact proxies on the index,
middle and thumb tips) or with a stick or mallet held in a Dex3 grasp. Nothing here is task logic: this module holds the
placement per preset, the head camera and head microphone, the fingertip proxies, the arm IK that turns Cartesian hand
targets into joint targets (the physical servos then move the arm) and the pelvis state.

Robot model: robots/models/g1/robot.json (pinned URDF g1_29dof_with_hand_rev_1_0), controller profile "barcode_scanning"
(the repository's standing G1 profile, reused unchanged: legs kp 800 / kd 20, waist, shoulders and elbows 120 / 6,
wrists 80 / 3, Dex3 fingers 1.8 / 0.035; torques saturate at the URDF effort limits). Floating base: no weld, no pelvis
support, no balance controller; the waist and arms get the model's gravity torque as feed-forward
(Unitree's teleoperation stack practice, as in barcode_scanning). The arm IK is barcode_scanning's G1HandIK
(whiteboard HandIK convention), copied here so the environment owns it.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

INSTANCE = "g1"
SETUP = "percussion-g1"
MOUNT = "g1_floor_mount"
PELVIS_Z = 0.79                     # m, pelvis height at import (whiteboard/barcode): feet ~5 mm above the floor
RIGID_SUBSTEPS = 2                  # MuJoCo steps of 0.5 ms per 1 ms tick: the G1 falls at 1 ms (barcode_scanning 6.5.3)

# D435 on the torso (URDF d435_joint), 69 deg HFOV: the whiteboard / barcode mount
HEAD_CAMERA = dict(link="torso_link", xyz=(0.0576235, 0.01753, 0.42987), pitch=0.8307767239493009, hfov_deg=69.)
# the G1's four-microphone array sits in the head (Unitree G1 datasheet [S37]); one omnidirectional capsule at the head
# centre stands for it (torso_link frame: above the D435, estimate from the URDF head geometry)
HEAD_MIC = dict(link="torso_link", xyz=(0.03, 0.0, 0.47))

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
TCP_LOCAL_M = np.array([0.08, 0.0, 0.0])        # whiteboard/ik.py HandIK: 80 mm along the wrist-yaw link +X
PALM_IN_WRIST_M = {"right": np.array([0.0415, -0.003, 0.]), "left": np.array([0.0415, 0.003, 0.])}   # URDF palm joints
FINGERTIP_LINKS = ("index_1", "middle_1", "thumb_2")
FINGERTIP_R = 0.008                 # m, the Dex3 fingertip pad's radius (proxy sphere at the distal link's far end)

# arms down at the sides, hands open (barcode_scanning's ARMS_DOWN: shoulders pitched 0.1 rad forward and rolled 0.3 rad
# out, elbows 1.3 rad; every arm and hand link clear of the body). A forearms-forward start put the hands inside the
# drum and onto the trap tables (MuJoCo distance test at build); the policy raises the arms to play
ARMS_READY = {**{f"{INSTANCE}/{s}_shoulder_pitch_joint": -.1 for s in ("left", "right")},
              f"{INSTANCE}/left_shoulder_roll_joint": .3, f"{INSTANCE}/right_shoulder_roll_joint": -.3,
              **{f"{INSTANCE}/{s}_elbow_joint": 1.3 for s in ("left", "right")}}
# where the humanoid stands (pelvis at import, facing +y), per preset; the table presets move the drum to its right hand
LAYOUT = {
    # table presets: 0.25 m from the table edge; the drum front-right of the robot, 2 cm clear of the table's edge
    "therapy": dict(pelvis=(0.0, 0.12), yaw=math.pi/2, drum=(0.52, 0.16)),
    "assessment": dict(pelvis=(0.0, 0.12), yaw=math.pi/2, drum=(0.52, 0.16)),
    "pentatonic": dict(pelvis=(0.0, 0.12), yaw=math.pi/2, drum=(0.52, 0.16)),
    "gait": dict(pelvis=(0.0, -0.02), yaw=math.pi/2, drum=None),
}


def joint(side, name):
    """Policy joint name, e.g. joint('right', 'elbow') -> 'g1/right_elbow_joint'."""
    mid = "hand_" if name in HAND_JOINTS else ""
    return f"{INSTANCE}/{side}_{mid}{name}_joint"


def body_index(model_or_builder, suffix):
    labels = list(model_or_builder.body_label)
    hits = [i for i, lab in enumerate(labels) if lab.startswith(INSTANCE + "/") and lab.split("/")[-1] == suffix]
    if len(hits) != 1:
        raise RuntimeError(f"G1 body {suffix}: {len(hits)} matches")
    return hits[0]


def head_camera(scene):
    """The head RGB (policy camera) on the torso link."""
    body = body_index(scene.b, HEAD_CAMERA["link"])
    R = Rotation.from_euler("xyz", (0., HEAD_CAMERA["pitch"], 0.))
    eye = np.asarray(HEAD_CAMERA["xyz"], float)
    scene.cameras["head_rgb"] = dict(body=body, eye=eye.tolist(), target=(eye + .3*R.apply((1., 0., 0.))).tolist(),
                                     up=R.apply((0., 0., 1.)).tolist(), fov=HEAD_CAMERA["hfov_deg"], role="policy")


def fingertip_points(builder, body):
    """Far end of a Dex3 distal link: the vertex of its collision meshes farthest from the link origin (link frame), and
    the unit direction to it."""
    best, d_best = None, -1.
    for i in range(builder.shape_count):
        if builder.shape_body[i] != body:
            continue
        src = builder.shape_source[i]
        if src is None or not hasattr(src, "vertices"):
            continue
        tf = np.asarray(builder.shape_transform[i], float)
        R = Rotation.from_quat(tf[3:])
        v = R.apply(np.asarray(src.vertices, float)*np.asarray(builder.shape_scale[i], float)) + tf[:3]
        k = int(np.argmax(np.linalg.norm(v, axis=1)))
        if np.linalg.norm(v[k]) > d_best:
            best, d_best = v[k], float(np.linalg.norm(v[k]))
    if best is None:
        raise RuntimeError(f"no collision mesh on G1 body {builder.body_label[body]}")
    return best, best/d_best


class G1HandIK:
    """Seven-joint arm IK of the G1 (whiteboard/ik.py HandIK, same TCP and bounded least squares; barcode_scanning g1.py)
    on a PRIVATE copy of the world's MuJoCo data: it reads the current state and returns joint targets; it never writes
    the simulation state. Joint targets are then tracked by the physical position servos."""

    ARMS = ("left", "right")

    def __init__(self, world):
        import mujoco
        self.world, self.mj = world, mujoco
        self.model = world.solver.mj_model
        self.data = mujoco.MjData(self.model)
        self._chains = {}
        names = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(self.model.njnt)}
        bnames = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(self.model.nbody)}
        for arm in self.ARMS:
            js = [joint(arm, n) for n in ARM_JOINTS]
            ids = []
            for n in js:
                short = n.split("/")[-1]
                hit = [j for k, j in names.items() if k and k.endswith("_" + short)]
                if len(hit) != 1:
                    raise RuntimeError(f"G1 IK: MuJoCo joint for {n}: {hit}")
                ids.append(hit[0])
            body = [i for k, i in bnames.items() if k and k.endswith(f"{arm}_wrist_yaw_link")]
            if len(body) != 1:
                raise RuntimeError(f"G1 IK: wrist body for {arm}: {body}")
            lo = np.array([world.joints[n]["lower"] for n in js]); hi = np.array([world.joints[n]["upper"] for n in js])
            self._chains[arm] = (js, np.asarray(self.model.jnt_qposadr[ids]), body[0], lo, hi)

    def sync(self):
        self.world.solver._update_mjc_data(self.data, self.world.model, self.world.state)
        self.mj.mj_kinematics(self.model, self.data)

    def names(self, arm):
        return list(self._chains[arm][0])

    def pose(self, arm, q=None, sync=True):
        """TCP (position, Rotation) of an arm now, or at arm joint values q."""
        js, adr, body, lo, hi = self._chains[arm]
        if sync:
            self.sync()
        if q is not None:
            self.data.qpos[adr] = q
            self.mj.mj_kinematics(self.model, self.data)
        R = Rotation.from_matrix(self.data.xmat[body].reshape(3, 3))
        return self.data.xpos[body] + R.apply(TCP_LOCAL_M), R

    def joints_now(self, arm):
        q = self.world.state.joint_q.numpy()
        return np.array([q[self.world.joints[n]["q"]] for n in self._chains[arm][0]])

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=2e-3, tol_rad=0.03, rot_weight=0.15,
              restarts=0):
        """Joint targets putting the TCP at `position` (world, m) with orientation `orientation_xyzw` (None: keep the
        current one). Bounded to the URDF limits; raises ValueError (no action executed) beyond the tolerances."""
        from scipy.optimize import least_squares
        if arm not in self._chains:
            raise ValueError(f"No arm '{arm}' on the G1 (left, right)")
        js, adr, body, lo, hi = self._chains[arm]
        position = np.asarray(position, float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("IK target must be three finite metre coordinates")
        self.sync()
        base = self.data.qpos.copy()
        now = self.data.qpos[adr].copy()
        start = now if seed is None else np.asarray(seed, float)
        target = Rotation.from_matrix(self.data.xmat[body].reshape(3, 3)) if orientation_xyzw is None \
            else Rotation.from_quat(orientation_xyzw)

        def err(q):
            self.data.qpos[adr] = q
            self.mj.mj_kinematics(self.model, self.data)
            R = Rotation.from_matrix(self.data.xmat[body].reshape(3, 3))
            return np.r_[self.data.xpos[body] + R.apply(TCP_LOCAL_M) - position, rot_weight*(target*R.inv()).as_rotvec()]
        rng = np.random.default_rng(12345)
        seeds = [start] + [rng.uniform(lo, hi) for _ in range(int(restarts))]
        best, first = None, None
        for s0 in seeds:
            r = least_squares(err, np.clip(s0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), xtol=1e-10, ftol=1e-10, gtol=1e-10,
                              max_nfev=200)
            res = err(r.x)
            pe, ae = float(np.linalg.norm(res[:3])), float(np.linalg.norm(res[3:])/rot_weight)
            first = first or (pe, ae)
            if pe <= tol_m and ae <= tol_rad:
                dq = float(np.abs(r.x - now).max())
                if best is None or dq < best[0]:
                    best = (dq, r.x.copy(), pe, ae)
                if s0 is start:
                    break
        self.data.qpos[:] = base
        if best is None:
            pe, ae = first
            raise ValueError(f"IK unreachable ({pe*1e3:.1f} mm, {math.degrees(ae):.1f} deg); no action executed")
        return dict(zip(js, map(float, best[1]))), dict(position_error_m=best[2], orientation_error_rad=best[3])


def pelvis_state(world):
    """Pelvis height and tilt from vertical (deg): what the robot's IMU would report."""
    i = world.scene.g1_hw["pelvis"]
    q = world.state.body_q.numpy()[i]
    z = Rotation.from_quat(q[3:]).apply((0., 0., 1.))
    return float(q[2]), math.degrees(math.acos(max(-1., min(1., float(z[2])))))


class G1Kinematics:
    """ArmKinematics-compatible adapter (forward -> 4x4 TCP pose, solve -> joint-target dict) over G1HandIK, so the
    backend's Cartesian commands and the demos address the G1's hands like the FR3 arms ('left', 'right')."""

    def __init__(self, world):
        self.ik = G1HandIK(world)
        self.world = world
        self.arms = {a: (None, self.ik.names(a)) for a in G1HandIK.ARMS}

    def joints_now(self, arm):
        return self.ik.joints_now(arm)

    def forward(self, arm, q=None):
        p, R = self.ik.pose(arm, q)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.as_matrix(), p
        return T

    def solve(self, arm, position, orientation_xyzw=None, seed=None, restarts=8, **kw):
        """Bounded IK from the current (or given) seed; if that misses the 2 mm / 0.03 rad acceptance, fixed-seed
        restarts are tried and the solution nearest the current configuration is kept (no action on failure)."""
        return self.ik.solve(arm, position, orientation_xyzw, seed=seed, restarts=restarts)[0]
