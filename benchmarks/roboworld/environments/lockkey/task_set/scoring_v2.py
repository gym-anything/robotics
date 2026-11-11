"""lockkey-tasks-v2 progress for the seven additions; never changes unlock.

Success reuses unlock's geometry/contact predicates, exactly as in lockkey-tasks-v1 (scoring.py). Scoring uses
distances, depth and angles, not a fraction of passed checks or a required action history. The one historical
witness is the lock's irreversible release tick, as in unlock.

Zero points come from the settled start (settle_v2.py), each with an idle margin, so an untouched or idle robot
cannot earn credit from the start's own offsets: the shackle offset (swing-open starts 0.81 mm off its bore
against the unlock chain's 1 mm deadband), the plug angle of a key seated at the start (turn-to-open), the key's
withdrawal (remove-key) and the flat key's tilt from its stand orientation (return-key). Success predicates are
unchanged. The unlock task's own chain reading of the state is reported as `unlock_chain_reading`, not as
top-level `lock_progress`/`stage`.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from lockkey.newton15.evaluation import (
    UnlockProgress, SEAT_X, AXIS_TOL, RELEASE_DEG, OPEN_LATERAL, SWING_DEADBAND,
    STAND_SLIDE, STAND_ACROSS, STAND_HEIGHT, STAND_TILT_DEG, FAIL_SCORE_SCALE,
)
from lockkey.newton15.shackle import geometry_state
from .catalogue import UNLOCK_CONTRACT, validate
from .catalogue_v2 import CONTRACT, task_id
from .scoring import clip, insertion

# Idle margins past the settled start before credit starts. Unsettled, the seated key snapped 0.47 mm and the
# plug 0.37 deg in the first 33 ms. Settled, 2-2.5 s idle moved the seated key and the shackle offset < 0.001 mm
# and the plug 0.02 deg (closing); the flat key on the mat rocks within +-0.6 deg of its start tilt.
SWING_IDLE = .0005         # m of shackle short-leg offset
TURN_IDLE_DEG = .5         # deg of plug turn with the key seated at the start
WITHDRAW_IDLE = .0005      # m of key withdrawal (+x)
TILT_IDLE_DEG = 5.         # deg of the flat key's roll toward its stand orientation


def stand_tilt(pose, target):
    return math.degrees((Rotation.from_quat(target[3:]).inv()*Rotation.from_quat(pose[3:])).magnitude())


def stand_fit(pose, target, tilt_zero=90.):
    """near x upright; upright is zero from `tilt_zero` (90 deg unless re-based on the settled start) down to
    full at the stand's 20 deg tolerance."""
    outside = np.maximum(np.abs(pose[:3]-target[:3])-[STAND_SLIDE, STAND_ACROSS, STAND_HEIGHT], 0.)
    near = clip(1.-float(np.linalg.norm(outside))/.15)
    upright = clip((tilt_zero-stand_tilt(pose, target))/(tilt_zero-STAND_TILT_DEG))
    return near*upright


class TaskProgress(UnlockProgress):
    def __init__(self, env, task, stands):
        if validate(task) == "unlock":
            raise ValueError("unlock must use its own unchanged evaluator")
        super().__init__(env)
        self.task = task
        # Goal poses are the catalogue stands, NOT a mid-cycle or mat start.
        self.start = {name: pose.copy() for name, pose in stands.items()}
        poses = env.state.body_q.numpy()
        key = poses[env.bodies["key"]]
        self.withdrawal_start_x = float(key[0])
        # Zero points from this (settled) start; see the module docstring.
        lateral = geometry_state(poses[env.bodies["shackle"]], self.origin)["shackle_short_leg_lateral_offset_m"]
        self.swing_zero = (SWING_DEADBAND if lateral >= OPEN_LATERAL else
                           min(max(SWING_DEADBAND, lateral+SWING_IDLE), OPEN_LATERAL/2))
        on_axis = abs(key[1]-self.origin[1]) <= AXIS_TOL and abs(key[2]-self.origin[2]) <= AXIS_TOL
        # A key seated at the start (turn-to-open): the plug's zero point is its start angle, read by the first
        # update, which the environment runs on the start state itself.
        self.seated_start = bool(on_axis and key[0] <= SEAT_X and not env.released)
        self.turn_zero = None
        self.withdraw_zero = self.withdrawal_start_x+WITHDRAW_IDLE if task == "remove-key" else SEAT_X
        self.tilt_zero = (max(STAND_TILT_DEG+10., min(90., stand_tilt(key, self.start["key"]))-TILT_IDLE_DEG)
                          if task == "return-key" else 90.)

    def update(self, env, plug_angle):
        if self.turn_zero is None:
            self.turn_zero = min(max(0., plug_angle)+TURN_IDLE_DEG, RELEASE_DEG/2) if self.seated_start else 0.
        base = super().update(env, plug_angle)
        poses = env.state.body_q.numpy()
        key_body = env.bodies["key"]
        key = poses[key_body]
        held = self.gripped(env, poses, key_body)
        key_back = self.on_stand(key, "key") and not held
        other_back = base["checks"]["other_key_on_its_stand"]
        released = bool(env.released)
        unlocked = released and bool(self.seated_at_release)
        lateral = geometry_state(poses[env.bodies["shackle"]], self.origin)["shackle_short_leg_lateral_offset_m"]
        swing = clip((lateral-self.swing_zero)/(OPEN_LATERAL-self.swing_zero))
        opened = unlocked and lateral > OPEN_LATERAL
        seated = base["key_seated_now"]
        insert = insertion(key, self.origin, self.entry_x, base["key_roll_error_deg"])
        back = stand_fit(key, self.start["key"], self.tilt_zero)
        out = clip((key[0]-self.withdraw_zero)/(self.entry_x-self.withdraw_zero))
        stow = .3*out+.7*back
        turn = clip((plug_angle-self.turn_zero)/(RELEASE_DEG-self.turn_zero))
        chain = .8+.2*swing if unlocked else 0. if released else insert*(.5+.3*turn)
        if self.task == "swing-open":
            success, progress = opened and other_back, swing if unlocked else 0.
        elif self.task == "turn-to-open":
            success, progress = opened and other_back, clip((chain-.5)/.5)
        elif self.task == "insert-key":
            # Read the seat NOW; opening and subsequently removing the key is not insertion.
            success, progress = seated and not held and other_back, insert
        elif self.task == "return-key":
            success, progress = key_back and other_back, back
        elif self.task == "remove-key":
            success, progress = key_back and opened and other_back, stow*swing if unlocked else 0.
        elif self.task == "which-key":
            success, progress = opened and other_back, chain
        else:  # unlock-and-stow: returning the key before unlocking earns nothing.
            success = unlocked and opened and key_back and other_back
            progress = .6*chain + (.4*stow if unlocked else 0.)
        key_needed = self.task in ("insert-key", "return-key", "remove-key", "unlock-and-stow") or not unlocked
        lost = key_needed and self.off_bench(env, poses, key_body)
        bad_release = self.task not in ("insert-key", "return-key") and released and not unlocked
        progress = 0. if lost or bad_release else clip(progress)*base["other_key_factor"]
        base = dict(base)
        chain_progress, chain_stage = base.pop("lock_progress"), base.pop("stage")
        return {**base, "task_id": task_id(self.task), "contract": CONTRACT,
                # The unlock contract's own chain reading of this state (take key -> seated -> turned ->
                # released -> open), before its other-key factor. Diagnostic only: not this task's progress.
                "unlock_chain_reading": {"contract": UNLOCK_CONTRACT, "progress": chain_progress,
                                         "stage": chain_stage,
                                         "note": "unlock task's chain on this state; not this task's "
                                                 "progress or score"},
                "score": 1. if success else FAIL_SCORE_SCALE*progress,
                "success": bool(success), "progress": progress,
                "unrecoverable": bool(lost or bad_release or base["off_bench"]["other_key"]),
                "criterion": f"{self.task}: current job predicates",
                "checks": {"key_seated_now": seated, "key_back_on_stand": key_back,
                           "key_released_by_robot": not held, "padlock_released": released,
                           "correct_key_seated_at_release": self.seated_at_release,
                           "shackle_swung_open": bool(opened), "other_key_on_its_stand": other_back},
                "components": {"insertion": insert, "turn": turn, "swing": swing,
                               "return": back, "withdrawal": out, "lock_chain": chain},
                "zero_points": {"shackle_offset_m": self.swing_zero, "plug_deg": self.turn_zero,
                                "key_withdrawal_x_m": self.withdraw_zero, "key_tilt_deg": self.tilt_zero},
                "start_preset": env.start_preset}
