"""The Unitree G1 (29 body joints, two Dex3-1 hands) at the welding fixture table: the humanoid deployment `g1`
(rig robots/rigs/welding-g1.json).

The robot stands on the floor behind the table, facing the seam (-y), on its position servos: no weld to the floor,
no pelvis support, no balance controller (as in the sewing, barcode-scanning and whiteboard environments). The light
torch (welding/torch.py RM2 on G1_TORCH_MOUNT) is clamped to the back of the right Dex3 palm; the left Dex3 hand
handles the part. Nothing here is task logic: the standing place, the ready posture over the controller profile's
standing pose, the arm kinematics (IK computes joint targets only; the servos move the arm).

Robot model: robots/models/g1/robot.json (pinned URDF g1_29dof_with_hand_rev_1_0, the Unitree URDF limits: shoulders
and elbow 25 N m, wrist roll 25 N m, wrist pitch and yaw 5 N m) with the repository's standing G1 controller profile
"whiteboard" (legs kp 800 / kd 20, waist and shoulders/elbows 120 / 6, wrists 80 / 3, Dex3 fingers 1.8 / 0.035;
torques saturate at the URDF effort limits) plus the gravity feed-forward on the waist and arm joints that Unitree's
manipulation stack sends (welding/world.py, as sewing/world.py).
"""

import copy
import math

import numpy as np
from scipy.spatial.transform import Rotation

INSTANCE = "g1"
SETUP = "welding-g1"
MOUNT = "g1_floor_mount"
# the stance: the legs sit back (hips -0.30, knees 0.50, ankles -0.20 rad: pelvis upright, soles flat) so the pelvis is
# 40 mm behind the ankles instead of 6 mm in the profile's standing pose (URDF FK). Reaching over the table with the
# torch put the robot's centre of mass 70 mm in front of the ankles, 50 mm behind the toes, and from the standing pose
# it rocked onto its toes and fell onto the table within 3 s (E23 check: pelvis tilt 15 deg); sitting back
# moves the feet 34 mm forward under the same upper body. Pelvis height: the soles' contact spheres (5 mm radius,
# 30 mm under the ankle) 1 mm above the floor at import, ankle 735.9 mm under the pelvis in this stance (URDF FK)
LEGS = dict(hip_pitch=-.30, hip_roll=0., hip_yaw=0., knee=.50, ankle_pitch=-.20, ankle_roll=0.)
PELVIS_ABOVE_FLOOR = .7359 + .035 + .001
# where it stands: behind the table (its back edge at y = 0.40 m), centred on the seam, facing -y (the weld face of the
# bracket looks toward it), the pelvis 0.10 m behind the table's edge, the toes under the table top [authored; reach
# study E23: with the waist leaning 0.2-0.4 rad the torch reached every seam pose from 0.53-0.57 m, without a lean 1-8
# of 9; stance study: of y 0.50/0.52/0.55 m and leans 0.2/0.3 rad, y 0.50 m with 0.2 rad loads the waist pitch least
# (62 % of its 35 N m, E18) and sways least, nothing touching the table]
STAND = dict(x=0.0, y=0.50, yaw_deg=-90.0)

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
WAIST_JOINTS = ("waist_yaw", "waist_roll", "waist_pitch")
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
TORCH_SIDE, HAND_SIDE = "right", "left"
# the handling hand's tool point: 80 mm along the wrist-yaw link's +x (the palm), as the whiteboard, sewing and barcode
# scanning G1 IK (whiteboard/ik.py HandIK)
HAND_TCP_LOCAL_M = (0.08, 0.0, 0.0)

# ready posture over the profile's standing pose:
# the waist leaning 0.2 rad toward the table; the torch arm holding the torch at the procedure's angles 120 mm back
# along the wire from the seam start (x = 10 mm), as the other deployments' start poses: of 80 seeded IK solutions the
# one farthest from the joint limits (0.42 rad, the wrist yaw); the handling arm hanging plumb beside the hip; the torch
# hand's index and middle fingers curled and its thumb folded, clear of the torch on the back of the palm; the legs in
# the stance LEGS (added below)
READY = {
         "g1/right_shoulder_pitch_joint": -2.4893,
         "g1/right_shoulder_roll_joint": -1.3098,
         "g1/right_shoulder_yaw_joint": 1.5575,
         "g1/right_elbow_joint": -0.4942,
         "g1/right_wrist_roll_joint": 0.7648,
         "g1/right_wrist_pitch_joint": 1.1457,
         "g1/right_wrist_yaw_joint": 1.197,
         "g1/waist_pitch_joint": 0.2,
         "g1/left_shoulder_pitch_joint": 0.3,
         "g1/left_shoulder_roll_joint": 0.15,
         "g1/left_shoulder_yaw_joint": 0.0,
         "g1/left_elbow_joint": 0.3,
         "g1/left_wrist_roll_joint": 0.0,
         "g1/left_wrist_pitch_joint": 0.0,
         "g1/left_wrist_yaw_joint": 0.0,
         "g1/right_hand_index_0_joint": 1.0,
         "g1/right_hand_index_1_joint": 1.0,
         "g1/right_hand_middle_0_joint": 1.0,
         "g1/right_hand_middle_1_joint": 1.0,
         "g1/right_hand_thumb_1_joint": -1.047}


def joint(side, name):
    """Joint name, e.g. joint('right', 'elbow') -> 'g1/right_elbow_joint', joint(None, 'waist_pitch') ->
    'g1/waist_pitch_joint', joint('left', 'index_0') -> 'g1/left_hand_index_0_joint'."""
    if side is None:
        return f"{INSTANCE}/{name}_joint"
    mid = "hand_" if name in HAND_JOINTS else ""
    return f"{INSTANCE}/{side}_{mid}{name}_joint"


READY.update({joint(side, k): v for side in ("left", "right") for k, v in LEGS.items()})       # the stance


def body_index(labels, suffix):
    hits = [i for i, lab in enumerate(labels) if lab.startswith(INSTANCE + "/") and lab.split("/")[-1] == suffix]
    if len(hits) != 1:
        raise RuntimeError(f"G1 body {suffix}: {len(hits)} matches")
    return hits[0]


def mount_pose():
    """((x, y, z) of the pelvis at import, yaw rad), world frame."""
    return (STAND["x"], STAND["y"], PELVIS_ABOVE_FLOOR), math.radians(STAND["yaw_deg"])


def _rot(axis, angle):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(axis, float)*angle).as_matrix()
    return T


def freeze_prefix(full, k, q_prefix):
    """A copy of a SerialKinematics chain with its first k movable joints frozen at q_prefix (folded into the base)."""
    sub = copy.copy(full)
    T = full.base.copy()
    i = j = 0
    entries = list(full.chain)
    while i < k:
        name, origin, axis = entries[j]
        T = T @ origin
        if axis is not None:
            T = T @ _rot(axis, q_prefix[i])
            i += 1
        j += 1
    sub.chain = entries[j:]
    sub.names, sub.kinds = list(full.names[k:]), list(full.kinds[k:])
    sub.lower, sub.upper = full.lower[k:].copy(), full.upper[k:].copy()
    sub.base = T
    return sub


class G1Kinematics:
    """Arm kinematics of the G1 from its URDF (robots.controllers.kinematics.SerialKinematics), the floating base read
    from the simulated pelvis at each call. Arms: 'right' (the torch arm: its tool point is the torch work point -- on
    the wire axis, the nominal 19 mm contact-tip-to-work beyond the contact tip, z along the wire, as welding/
    kinematics.py), 'left' (the handling hand: the palm point HAND_TCP_LOCAL_M), each with its seven joints, the waist
    held where it is; 'right+waist' and 'left+waist' add the three waist joints (the other arm moves with the torso).
    IK computes joint targets only."""

    def __init__(self, world):
        from robots.controllers.kinematics import SerialKinematics
        from .kinematics import torch_tcp_in_tool0
        self.world = world
        handle = world.rig.instances[INSTANCE]
        urdf = handle.metadata["urdf"]
        labels = list(world.model.body_label)
        self.pelvis = body_index(labels, "pelvis")
        sc = world.scene
        self._full, self.arms, self.bodies = {}, {}, {}
        for side in ("right", "left"):
            full = SerialKinematics(urdf, f"{side}_wrist_yaw_link", base=np.eye(4))
            expect = [f"{w}_joint" for w in WAIST_JOINTS] + [f"{side}_{a}_joint" for a in ARM_JOINTS]
            if full.names != expect:
                raise RuntimeError(f"G1 {side} chain {full.names} != {expect}")
            T = np.eye(4)
            if side == TORCH_SIDE:
                pt, Rt = sc.tool0_in_body
                T[:3, :3], T[:3, 3] = Rt, pt
                T = T @ torch_tcp_in_tool0(sc.gun_spec)
            else:
                T[:3, 3] = HAND_TCP_LOCAL_M
            full.chain[-1] = ("tool_offset", T, None)
            self._full[side] = full
            self.bodies[side] = body_index(labels, f"{side}_wrist_yaw_link")
            names = [f"{INSTANCE}/{n}" for n in full.names]
            self.arms[side] = (None, names[3:])
            self.arms[side + "+waist"] = (None, names)
        self.check()

    def _base(self):
        q = self.world.state.body_q.numpy()[self.pelvis].astype(float)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(q[3:]).as_matrix()
        T[:3, 3] = q[:3]
        return T

    def chain(self, arm):
        """The live chain of an arm (base = the pelvis now; for the 7-joint arms the waist frozen where it is)."""
        if arm not in self.arms:
            raise ValueError(f"No arm '{arm}' on the G1 (right = torch, left = hand, right+waist, left+waist)")
        side = arm.split("+")[0]
        full = copy.copy(self._full[side])
        full.base = self._base()
        if arm.endswith("+waist"):
            return full
        return freeze_prefix(full, 3, self.joints_now(side + "+waist")[:3])

    def check(self):
        """FK of both chains against the simulated bodies (a mount or frame mismatch fails loudly)."""
        w = self.world
        for side in ("right", "left"):
            fk = self.forward(side)
            if side == TORCH_SIDE:
                _, _, ref = w.torch_pose()
            else:
                p, R = w.body_pose(self.bodies[side])
                ref = p + R @ np.asarray(HAND_TCP_LOCAL_M)
            if np.linalg.norm(fk[:3, 3] - ref) > 1e-3:
                raise RuntimeError(f"G1 {side}: IK tool point {fk[:3, 3]} differs from the simulated one {ref}")

    def joints_now(self, arm):
        q = self.world.state.joint_q.numpy()
        return np.array([q[self.world.joints[n]["q"]] for n in self.arms[arm][1]])

    def forward(self, arm, q=None):
        chain = self.chain(arm)
        return chain.fk(self.joints_now(arm) if q is None else np.asarray(q, float))

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=1e-3, tol_rad=math.radians(1.)):
        chain = self.chain(arm)
        names = self.arms[arm][1]
        seed = self.joints_now(arm) if seed is None else np.asarray(seed, float)
        seed = np.clip(seed, chain.lower + 1e-6, chain.upper - 1e-6)
        if orientation_xyzw is None:
            orientation_xyzw = Rotation.from_matrix(chain.fk(seed)[:3, :3]).as_quat()
        r = chain.solve(position, orientation_xyzw, seed, max_evaluations=400)
        if r["position_error_m"] > tol_m or r["orientation_error_rad"] > tol_rad:
            raise ValueError(f"IK unreachable ({r['position_error_m']*1e3:.2f} mm, "
                             f"{math.degrees(r['orientation_error_rad']):.2f} deg); no action executed")
        return dict(zip(names, map(float, r["joints"])))


def pelvis_state(world):
    """Pelvis height above the floor (m) and tilt from vertical (deg): what the robot's IMU would report."""
    i = body_index(list(world.model.body_label), "pelvis")
    q = world.state.body_q.numpy()[i]
    z = Rotation.from_quat(q[3:]).apply((0., 0., 1.))
    return float(q[2]), math.degrees(math.acos(max(-1., min(1., float(z[2])))))
