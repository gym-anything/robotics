"""Unitree G1 (29 body joints, two Dex3-1 hands) configuration of barcode scanning.

The humanoid stands on the floor in front of the work surface and operates the same scanner, cradle, keypad and
shelving as the FR3 configurations. Nothing here is task logic: this module holds the G1 placement per preset (where
the robot stands, where its utility cart is parked), the head camera mount, the arm IK that turns Cartesian hand
targets into joint targets (the physical servos then move the arm), and the robot's protective behaviour.

Robot model: robots/models/g1/robot.json (pinned URDF g1_29dof_with_hand_rev_1_0, sha256 555edb93...), controller
profile "barcode_scanning" (a copy of the repository's standing G1 profile "whiteboard", owned by this environment: legs
kp 800 / kd 20, waist and shoulders/elbow 120 / 6, wrists 80 / 3, Dex3 fingers 1.8 / 0.035; torques saturate at the URDF
effort limits). The robot is floating: no weld to the floor, no pelvis support, no balance controller -- it stands on
its PD servos as in the whiteboard environment.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

INSTANCE = "g1"
SETUP = "scan-g1"
MOUNT = "g1_floor_mount"
PELVIS_Z = 0.79                     # m, pelvis height at import (whiteboard/robot.py): feet ~5 mm above the floor, settle

# Head camera: the D435 of the G1 head (URDF d435_joint on torso_link, 47.6 deg down), zero lens offset, 69 deg HFOV
# (the whiteboard environment's g1_rgb mount, whiteboard/sensors.py; D435 RGB field of view from Intel's datasheet, not
# re-measured here)
HEAD_CAMERA = dict(link="torso_link", xyz=(0.0576235, 0.01753, 0.42987), pitch=0.8307767239493009, hfov_deg=69.)

# Where the humanoid stands and where its utility cart is parked, per preset (world frame of scene.py: shelf / counter
# face at y = 0, aisle y < 0). The FR3 cart stands 0.28 m from the shelf face, exactly where a standing operator has to
# be, and its accessory tray is at the cart's shelf-side edge: a person (or a G1) beside the unmoved cart is 0.5-0.6 m
# from the cradle. The G1 configuration therefore parks the SAME cart turned 90 deg at the robot's right, its tray (same
# tray, cradle, keypad and display) facing the robot, the cradle at the tray's shelf end within reach of the right hand.
#   pelvis: (x, y) of the pelvis at import, facing +y (yaw pi/2)
#   cart:   (x, y) of the cart centre and its yaw (pi/2: the cart's tray edge faces -x, towards the robot)
#   arms:   initial arm posture over the profile's 'whiteboard' pose (forearms forward at 0.885 m): at the 0.915 m
#           counter the forearms would start inside the counter top, so there the arms start down at the sides, which
#           lets the robot stand close to the counter edge. Straight down (elbows 1.5 rad,
#           no shoulder roll) put each hand 34 mm inside its own hip links: the first physics step threw the arms apart
#           and the robot fell at reset (tilt 102 deg, pelvis on the floor). Shoulders rolled out 0.3 rad and pitched
#           0.1 rad forward, elbows 1.3 rad: every arm and hand link >= 22.7 mm from the body, the other arm, the
#           counter, the cart tray and the stock at the counter and library placements (MuJoCo distance test, measured)
ARMS_DOWN = {**{f"{INSTANCE}/{s}_shoulder_pitch_joint": -.1 for s in ("left", "right")},
             f"{INSTANCE}/left_shoulder_roll_joint": .3, f"{INSTANCE}/right_shoulder_roll_joint": -.3,
             **{f"{INSTANCE}/{s}_elbow_joint": 1.3 for s in ("left", "right")}}
LAYOUT = {
    "stockroom": dict(pelvis=(0.0, -0.55), yaw=math.pi/2, cart=(0.65, -0.56), cart_yaw=math.pi/2, arms=None),
    "counter": dict(pelvis=(0.0, -0.22), yaw=math.pi/2, cart=(0.65, -0.56), cart_yaw=math.pi/2, arms=ARMS_DOWN),
    "library": dict(pelvis=(0.0, -0.22), yaw=math.pi/2, cart=(0.65, -0.56), cart_yaw=math.pi/2, arms=ARMS_DOWN),
}
# Items on the tray for an operator standing at the tray's long edge instead of behind it (cart frame of scene.cart:
# tray along x, operator on the +y side): the cradle where the FR3 layout has it (x = +0.19 m, the tray end nearest the
# shelf) turned so its window faces the shelf (world +y), the keypad in the middle with its top row away from the
# operator, the display where the FR3 layout has it (x = -0.20 m) tilted back towards the operator's head camera.
# x (m) along the tray, yaw (deg / rad), tilt (deg).
TRAY = dict(cradle_x=0.19, cradle_yaw_deg=0., keypad_yaw=math.pi, display_x=-0.20, display_tilt_deg=-60.)

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
TCP_LOCAL_M = np.array([0.08, 0.0, 0.0])        # whiteboard/ik.py HandIK: 80 mm along the wrist-yaw link +X
PALM_IN_WRIST_M = {"right": np.array([0.0415, -0.003, 0.]), "left": np.array([0.0415, 0.003, 0.])}   # URDF palm joints


def joint(side, name):
    """Policy joint name, e.g. joint('right', 'elbow') -> 'g1/right_elbow_joint', joint('right', 'index_0') ->
    'g1/right_hand_index_0_joint'."""
    mid = "hand_" if name in HAND_JOINTS else ""
    return f"{INSTANCE}/{side}_{mid}{name}_joint"


def cart_transform(preset):
    """(centre, yaw) of the cart for the G1 layout, or None."""
    lay = LAYOUT[preset]
    return tuple(lay["cart"]), float(lay["cart_yaw"])


def body_index(model_or_builder, suffix):
    labels = list(model_or_builder.body_label)
    hits = [i for i, lab in enumerate(labels) if lab.startswith(INSTANCE + "/") and lab.split("/")[-1] == suffix]
    if len(hits) != 1:
        raise RuntimeError(f"G1 body {suffix}: {len(hits)} matches")
    return hits[0]


def head_camera(scene, handle):
    """Register the head RGB (policy camera) on the torso link, like Scene._wrist_camera does for the FR3."""
    b = scene.b
    body = body_index(b, HEAD_CAMERA["link"])
    R = Rotation.from_euler("xyz", (0., HEAD_CAMERA["pitch"], 0.))
    eye = np.asarray(HEAD_CAMERA["xyz"], float)
    scene.cameras["head_rgb"] = dict(body=body, eye=eye.tolist(), target=(eye + .3*R.apply((1., 0., 0.))).tolist(),
                                     up=R.apply((0., 0., 1.)).tolist(), fov=HEAD_CAMERA["hfov_deg"], role="policy")


class G1HandIK:
    """Seven-joint arm IK of the G1 (whiteboard/ik.py HandIK, same TCP and bounded least squares) on a PRIVATE copy of
    the world's MuJoCo data: it reads the current state (floating base, waist, the other arm) and returns joint targets;
    it never writes the simulation state. Joint targets are then tracked by the physical position servos."""

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
            # the same arm with the three waist joints in front of it (whole upper body; moves the other arm too)
            ws = [f"{INSTANCE}/waist_{n}_joint" for n in ("yaw", "roll", "pitch")]
            wid = [[j for k, j in names.items() if k and k.endswith("_" + n.split("/")[-1])][0] for n in ws]
            wlo = np.array([world.joints[n]["lower"] for n in ws]); whi = np.array([world.joints[n]["upper"] for n in ws])
            self._chains[arm + "+waist"] = (ws + js, np.r_[np.asarray(self.model.jnt_qposadr[wid]), self._chains[arm][1]],
                                            body[0], np.r_[wlo, lo], np.r_[whi, hi])

    def sync(self, q_override=None):
        self.world.solver._update_mjc_data(self.data, self.world.model, self.world.state)
        if q_override:
            for arm, q in q_override.items():
                self.data.qpos[self._chains[arm][1]] = q
        self.mj.mj_kinematics(self.model, self.data)

    def names(self, arm):
        return list(self._chains[arm][0])

    def pose(self, arm, q=None, sync=True):
        """TCP (position, Rotation) of an arm now, or at arm joint values q (other joints at their current values)."""
        js, adr, body, lo, hi = self._chains[arm]
        if sync:
            self.sync()
        if q is not None:
            self.data.qpos[adr] = q
            self.mj.mj_kinematics(self.model, self.data)
        R = Rotation.from_matrix(self.data.xmat[body].reshape(3, 3))
        return self.data.xpos[body] + R.apply(TCP_LOCAL_M), R

    def palm(self, arm, q=None, sync=True):
        p, R = self.pose(arm, q, sync)
        return p - R.apply(TCP_LOCAL_M) + R.apply(PALM_IN_WRIST_M[arm]), R

    def collisions(self, arm, q, extra=None, margin=0.003):
        """Kinematic collision check (private data): the arm chain at joint values q (and optionally other joints,
        `extra` {policy name: value}); returns the contacts closer than `margin` between a body of this arm/hand and
        anything that is not the same arm: (arm body, other shape label, distance m)."""
        js, adr, body, lo, hi = self._chains[arm]
        side = "left" if arm.startswith("left") else "right"
        self.sync()
        self.data.qpos[adr] = q
        for n, v in (extra or {}).items():
            short = n.split("/")[-1]
            jid = [j for j in range(self.model.njnt)
                   if (self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_JOINT, j) or "").endswith("_" + short)]
            self.data.qpos[self.model.jnt_qposadr[jid[0]]] = v
        self.mj.mj_forward(self.model, self.data)
        w = self.world
        sh = w._geom_shape
        sb = w.model.shape_body.numpy()
        labels = w.model.body_label
        mine = lambda lab: lab.startswith(INSTANCE + "/") and (f"/{side}_hand_" in lab or any(
            f"/{side}_{p}" in lab for p in ("shoulder_", "elbow_", "wrist_")))
        out = []
        for c in self.data.contact[:self.data.ncon]:
            if c.dist > margin:
                continue
            s = [int(sh[g]) if 0 <= g < len(sh) else -1 for g in c.geom]
            b = [labels[int(sb[x])] if x >= 0 and sb[x] >= 0 else "world" for x in s]
            m = [mine(x) for x in b]
            if m[0] == m[1]:
                continue
            k = 0 if m[0] else 1
            out.append((b[k].split("/")[-1], w.model.shape_label[s[1 - k]] if s[1 - k] >= 0 else "?", float(c.dist)))
        return out

    def joints_now(self, arm):
        q = self.world.state.joint_q.numpy()
        return np.array([q[self.world.joints[n]["q"]] for n in self._chains[arm][0]])

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=2e-3, tol_rad=0.03, rot_weight=0.15,
              restarts=0):
        """Joint targets putting the TCP at `position` (world, m) with orientation `orientation_xyzw` (None: keep the
        current one). Bounded to the URDF limits; raises ValueError (no action executed) when the residual exceeds the
        tolerances (whiteboard HandIK acceptance: 2 mm, 0.03 rad). With `restarts`, seeds drawn from a fixed-seed
        generator are tried too and the solution closest (max joint change) to the current configuration is kept."""
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
