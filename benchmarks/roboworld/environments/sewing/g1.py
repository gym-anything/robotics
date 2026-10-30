"""Unitree G1 (29 body joints, two Dex3-1 hands) at the sewing table: the humanoid configuration of sewing.

A standing operator at the machine. Sewing rooms seat their operators at the 762 mm table; stand-up sewing lines
exist too (recalled, not surveyed). The G1 standing at the 762 mm table has its shoulders ~0.3 m above the table top,
about where a seated adult's are (measured in reach study E0b). The robot stands on the floor in front of the table,
facing the needle (+y), on its position servos: no weld to the floor, no pelvis support, no balance controller, as in barcode scanning and the
whiteboard environment. Nothing here is task logic: this module holds the placement, the ready posture, the head
camera, the hands' cloth pads (the Dex3 links as the cloth solver sees them) and the arm IK that turns Cartesian hand
targets into joint targets (the physical servos then move the arm).

Robot model: robots/models/g1/robot.json (pinned URDF g1_29dof_with_hand_rev_1_0) with its controller profile "sewing"
(legs kp 800 / kd 20, waist and shoulders/elbows 120 / 6, wrists 80 / 3, Dex3 fingers 1.8 / 0.035; torques saturate at
the URDF effort limits): the gains and standing pose of the repository's standing G1 profile "whiteboard", copied under
sewing's own name, rig robots/rigs/sewing-g1.json.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from . import layout as LY

INSTANCE = "g1"
SETUP = "sewing-g1"
MOUNT = "g1_floor_mount"
PELVIS_ABOVE_FLOOR = 0.79           # m, pelvis height at import (whiteboard/robot.py): feet ~5 mm above the floor
# where it stands: centred on the needle line a little to the left (as an operator sits, the needle in front of the
# body's centre line), the pelvis 0.13 m behind the table's front edge, facing +y [authored; reach study E0b: from 0.08,
# 0.13, 0.18 and 0.23 m the hands reach all 110 key points flat on the work, 36 of them leaning on the waist at
# 0.13 m, and the ready posture touches nothing from any of these places]
STAND = dict(x=-0.04, y=LY.TABLE["y"][0] - 0.13, yaw_deg=90.0)

# ready posture: both hands flat (palm down, fingers toward the machine) 0.13 m over the table, the left 0.17 m and the
# right 0.20 m in front of the needle, either side of the head (palms at x -0.19 and +0.15 m), the Dex3 thumbs
# folded toward the palm as far as they go (with the palm down the Dex3 thumb points at the work; folded, it still
# hangs below the palm). IK of g1.G1HandIK for those palm targets, checked at the commanded configuration for contact
# with the table, the machine and the robot's own links (none deeper than 0.1 mm; the Dex3 asset's own 0.2 mm overlap
# at the thumb's root, present in every posture, aside). Palms 0.12 m in front of the needle put the right hand's
# fingers into the machine's head: the head pushes the hand out and the robot falls onto the table and the machine
# within 2 s. The left palm sits 40 mm further out so the left wrist clears the torso (otherwise 10.8 mm inside it,
# held pressed there by the servo, which a check after settling, not at the command, misses).
READY = {**{f"{INSTANCE}/left_{j}_joint": v for j, v in (("shoulder_pitch", .298), ("shoulder_roll", .730),
                                                         ("shoulder_yaw", -.868), ("elbow", -.613),
                                                         ("wrist_roll", 1.161), ("wrist_pitch", .933),
                                                         ("wrist_yaw", .250), ("hand_thumb_1", 1.047))},
         **{f"{INSTANCE}/right_{j}_joint": v for j, v in (("shoulder_pitch", .684), ("shoulder_roll", -.705),
                                                          ("shoulder_yaw", .424), ("elbow", -.622),
                                                          ("wrist_roll", -1.117), ("wrist_pitch", .499),
                                                          ("wrist_yaw", -.359), ("hand_thumb_1", -1.047))}}

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
TCP_LOCAL_M = np.array([0.08, 0.0, 0.0])        # whiteboard/ik.py HandIK: 80 mm along the wrist-yaw link +X (the palm)
# Head camera: the D435 of the G1 head (URDF d435_joint on torso_link, 47.6 deg down), 69 deg HFOV (barcode scanning
# g1.HEAD_CAMERA, from the whiteboard environment's mount; D435 RGB field of view from Intel's datasheet)
HEAD_CAMERA = dict(link="torso_link", xyz=(0.0576235, 0.01753, 0.42987), pitch=0.8307767239493009, hfov_deg=69.)
# Dex3 links the cloth sees (the box around each link's URDF collision mesh, in the link frame) and the palm (the
# wrist-yaw link's second collision mesh, x 41.5-128.2 mm). Boxes, like the Franka Hand's URDF finger boxes.
CLOTH_LINKS = ("hand_thumb_1_link", "hand_thumb_2_link", "hand_index_0_link", "hand_index_1_link",
               "hand_middle_0_link", "hand_middle_1_link")
PALM_X_MIN = 0.0415                 # m: the palm's mesh starts here along the wrist-yaw link's x


def joint(side, name):
    """Policy joint name, e.g. joint('right', 'elbow') -> 'g1/right_elbow_joint', joint('right', 'index_0') ->
    'g1/right_hand_index_0_joint'."""
    mid = "hand_" if name in HAND_JOINTS else ""
    return f"{INSTANCE}/{side}_{mid}{name}_joint"


def body_index(builder_or_model, suffix):
    labels = list(builder_or_model.body_label)
    hits = [i for i, lab in enumerate(labels) if lab.startswith(INSTANCE + "/") and lab.split("/")[-1] == suffix]
    if len(hits) != 1:
        raise RuntimeError(f"G1 body {suffix}: {len(hits)} matches")
    return hits[0]


def mount_pose():
    """(x, y, z) of the pelvis at import and its yaw (rad), world frame (table top z = 0)."""
    return (STAND["x"], STAND["y"], PELVIS_ABOVE_FLOOR - LY.TABLE_H), math.radians(STAND["yaw_deg"])


def head_camera(scene, handle):
    """Register the head RGB on the torso link (policy camera)."""
    body = body_index(scene.b, HEAD_CAMERA["link"])
    R = Rotation.from_euler("xyz", (0., HEAD_CAMERA["pitch"], 0.))
    eye = np.asarray(HEAD_CAMERA["xyz"], float)
    scene.cameras["head_rgb"] = dict(body=body, eye=eye.tolist(), target=(eye + .3*R.apply((1., 0., 0.))).tolist(),
                                     up=R.apply((0., 0., 1.)).tolist(), fov=HEAD_CAMERA["hfov_deg"], role="policy")


def _mesh_box(builder, shape):
    """Centre, half extents (link frame) of the box around a collision mesh shape's vertices."""
    src = builder.shape_source[shape]
    v = np.asarray(src.vertices, float)*np.asarray(builder.shape_scale[shape], float)
    tf = np.asarray(builder.shape_transform[shape], float)
    vv = Rotation.from_quat(tf[3:]).apply(v) + tf[:3]
    lo, hi = vv.min(0), vv.max(0)
    return 0.5*(lo + hi), 0.5*(hi - lo)


def cloth_pads(builder, handle, newton):
    """{side: [(body, centre, quat_xyzw, half)]} for the Dex3 finger links and {side: (body, centre, half)} for the
    palm, from the URDF collision meshes (link frames)."""
    fingers, palms = {}, {}
    for side in ("left", "right"):
        fingers[side] = []
        for link in CLOTH_LINKS:
            body = body_index(builder, f"{side}_{link}")
            meshes = [i for i in handle.shape_ids if builder.shape_body[i] == body
                      and builder.shape_type[i] == newton.GeoType.MESH and builder.shape_source[i] is not None]
            c, h = _mesh_box(builder, meshes[0])
            fingers[side].append((body, c, np.array([0., 0., 0., 1.]), tuple(h)))
        body = body_index(builder, f"{side}_wrist_yaw_link")
        meshes = [i for i in handle.shape_ids if builder.shape_body[i] == body
                  and builder.shape_type[i] == newton.GeoType.MESH and builder.shape_source[i] is not None]
        boxes = [_mesh_box(builder, i) for i in meshes]
        palm = [b for b in boxes if b[0][0] - b[1][0] >= PALM_X_MIN - 1e-3]
        if not palm:
            raise RuntimeError(f"G1 {side} palm mesh not found on the wrist-yaw link")
        c, h = palm[0]
        palms[side] = (body, c, tuple(h))
    return fingers, palms


class G1HandIK:
    """Seven-joint arm IK of the G1 (whiteboard/ik.py HandIK and barcode scanning's G1HandIK: same TCP, bounded least
    squares) on a PRIVATE copy of the world's MuJoCo data: it reads the current state (floating base, waist, the other
    arm) and returns joint targets; it never writes the simulation state."""

    ARMS = ("left", "right")

    def __init__(self, world):
        import mujoco
        self.world, self.mj = world, mujoco
        self.model = world.solver.mj_model
        self.data = mujoco.MjData(self.model)
        self._chains = {}
        self.last_error = None
        self.arms = {}             # Kinematics-compatible: arm -> (None, joint names)
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
            lo = np.array([world.joints[n]["lower"] for n in js])
            hi = np.array([world.joints[n]["upper"] for n in js])
            self._chains[arm] = (js, np.asarray(self.model.jnt_qposadr[ids]), body[0], lo, hi)
            # the same arm with the three waist joints in front of it (leaning; moves the other arm too)
            ws = [f"{INSTANCE}/waist_{n}_joint" for n in ("yaw", "roll", "pitch")]
            wid = [[j for k, j in names.items() if k and k.endswith("_" + n.split("/")[-1])][0] for n in ws]
            wlo = np.array([world.joints[n]["lower"] for n in ws])
            whi = np.array([world.joints[n]["upper"] for n in ws])
            self.arms[arm] = (None, js)
            self.arms[arm + "+waist"] = (None, ws + js)
            self._chains[arm + "+waist"] = (ws + js, np.r_[np.asarray(self.model.jnt_qposadr[wid]),
                                                           self._chains[arm][1]], body[0], np.r_[wlo, lo], np.r_[whi, hi])

    def sync(self):
        self.world.solver._update_mjc_data(self.data, self.world.model, self.world.state)
        self.mj.mj_kinematics(self.model, self.data)

    def names(self, arm):
        return list(self._chains[arm][0])

    def joints_now(self, arm):
        q = self.world.state.joint_q.numpy()
        return np.array([q[self.world.joints[n]["q"]] for n in self._chains[arm][0]])

    def pose(self, arm, q=None, sync=True):
        """TCP (position, Rotation) now, or at arm joint values q (other joints at their current values)."""
        js, adr, body, lo, hi = self._chains[arm]
        if sync:
            self.sync()
        if q is not None:
            self.data.qpos[adr] = q
            self.mj.mj_kinematics(self.model, self.data)
        R = Rotation.from_matrix(self.data.xmat[body].reshape(3, 3))
        return self.data.xpos[body] + R.apply(TCP_LOCAL_M), R

    def forward(self, arm, q=None):
        p, R = self.pose(arm, q)
        T = np.eye(4)
        T[:3, :3] = R.as_matrix()
        T[:3, 3] = p
        return T

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=2e-3, tol_rad=0.03, rot_weight=0.15,
              restarts=0):
        """Joint targets putting the TCP at `position` (world, m) with orientation `orientation_xyzw` (None: keep the
        current one); bounded to the URDF limits; raises ValueError (no action executed) beyond the tolerances
        (whiteboard HandIK acceptance: 2 mm, 0.03 rad). Returns {joint name: value}; the residuals are in
        self.last_error."""
        from scipy.optimize import least_squares
        if arm not in self._chains:
            raise ValueError(f"No arm '{arm}' on the G1 (left, right, left+waist, right+waist)")
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
            return np.r_[self.data.xpos[body] + R.apply(TCP_LOCAL_M) - position,
                         rot_weight*(target*R.inv()).as_rotvec()]
        rng = np.random.default_rng(12345)
        seeds = [start] + [rng.uniform(lo, hi) for _ in range(int(restarts))]
        best, first = None, None
        for s0 in seeds:
            r = least_squares(err, np.clip(s0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), xtol=1e-10, ftol=1e-10,
                              gtol=1e-10, max_nfev=200)
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
        self.last_error = dict(position_error_m=best[2], orientation_error_rad=best[3])
        return dict(zip(js, map(float, best[1])))


def palm_down(side, yaw_deg=0.0):
    """Wrist-yaw link orientation (xyzw) with the hand flat: fingers (link +x) along world +y turned by yaw_deg, palm
    down. The Dex3 fingers close toward the wrist link's +y on the right hand and -y on the left (measured: flexing
    the index moves its tip 46 mm that way, toward the thumb), so the palm face is link +y (right) / -y (left)."""
    sign = 1.0 if side == "right" else -1.0
    x = np.array([0., 1., 0.])
    y = np.array([0., 0., -1.])*sign
    R = Rotation.from_matrix(np.column_stack([x, y, np.cross(x, y)]))
    return (Rotation.from_euler("z", math.radians(yaw_deg))*R).as_quat()


def pelvis_state(world):
    """Pelvis height above the floor and tilt from vertical (deg): what the robot's IMU would report."""
    i = body_index(world.model, "pelvis")
    q = world.state.body_q.numpy()[i]
    z = Rotation.from_quat(q[3:]).apply((0., 0., 1.))
    return float(q[2]) + LY.TABLE_H, math.degrees(math.acos(max(-1., min(1., float(z[2])))))
