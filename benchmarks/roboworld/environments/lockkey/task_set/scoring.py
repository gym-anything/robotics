"""State-based continuous progress for the seven additions; never changes unlock.

Success reuses unlock's geometry/contact predicates. Scoring uses distances,
depth and angles, not a fraction of passed checks or a required action history.
The one historical witness is the lock's irreversible release tick, as in unlock.

Each task's success and progress are its verifier.py's verify (tasks/<task>/), which TaskProgress.update here and
in scoring_v2.py (lockkey-tasks-v2) calls with its readings of the state; the verifier's KEY_ALWAYS_NEEDED and
NEEDS_UNLOCK say when a lost key or a release without the key zeroes the progress, and its WITHDRAWAL_FROM_START
measures the key's withdrawal from where the key starts.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from lockkey.newton15.evaluation import (
    UnlockProgress, SEAT_X, AXIS_TOL, KEY_ROLL_TOL_DEG, RELEASE_DEG,
    OPEN_LATERAL, SWING_DEADBAND, STAND_SLIDE, STAND_ACROSS, STAND_HEIGHT,
    STAND_TILT_DEG, FAIL_SCORE_SCALE,
)
from lockkey.newton15.shackle import geometry_state
from .catalogue import CONTRACT, task_verifier, validate


def clip(value):
    return float(np.clip(value, 0., 1.))


def insertion(pose, origin, entry_x, roll_error):
    """Zero outside the 4 cm axis corridor / 8 cm approach; one at the seat.

    Full alignment inside unlock's 2 mm / 15 degree tolerances, falling
    continuously to zero at 4 cm / 90 degrees. A key on either stand gets zero.
    """
    axis = math.prod(clip((.04-abs(pose[i]-origin[i]))/(.04-AXIS_TOL)) for i in (1, 2))
    roll = clip((90.-roll_error)/(90.-KEY_ROLL_TOL_DEG))
    approach = clip(1.-max(0., pose[0]-entry_x)/.08)
    depth = clip((entry_x-pose[0])/(entry_x-SEAT_X))
    return axis*roll*approach*(.25+.75*depth)


def stand_fit(pose, target):
    outside = np.maximum(np.abs(pose[:3]-target[:3])-[STAND_SLIDE, STAND_ACROSS, STAND_HEIGHT], 0.)
    near = clip(1.-float(np.linalg.norm(outside))/.15)
    tilt = math.degrees((Rotation.from_quat(target[3:]).inv()*Rotation.from_quat(pose[3:])).magnitude())
    upright = clip((90.-tilt)/(90.-STAND_TILT_DEG))
    return near*upright


class TaskProgress(UnlockProgress):
    def __init__(self, env, task, stands):
        if validate(task) == "unlock":
            raise ValueError("unlock must use its own unchanged evaluator")
        super().__init__(env)
        self.task = task
        # Goal poses are the catalogue stands, NOT a mid-cycle or mat start.
        self.start = {name: pose.copy() for name, pose in stands.items()}
        self.withdrawal_start_x = float(env.state.body_q.numpy()[env.bodies["key"], 0])

    def update(self, env, plug_angle):
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
        swing = clip((lateral-SWING_DEADBAND)/(OPEN_LATERAL-SWING_DEADBAND))
        opened = unlocked and lateral > OPEN_LATERAL
        seated = base["key_seated_now"]
        insert = insertion(key, self.origin, self.entry_x, base["key_roll_error_deg"])
        back = stand_fit(key, self.start["key"])
        own = task_verifier(self.task)
        seat_x = self.withdrawal_start_x if getattr(own, "WITHDRAWAL_FROM_START", False) else SEAT_X
        out = clip((key[0]-seat_x)/(self.entry_x-seat_x))
        stow = .3*out+.7*back
        turn = clip(plug_angle/RELEASE_DEG)
        chain = .8+.2*swing if unlocked else 0. if released else insert*(.5+.3*turn)
        success, progress = own.verify(seated=seated, held=held, key_back=key_back, other_back=other_back,
                                       released=released, unlocked=unlocked, opened=opened, insert=insert,
                                       turn=turn, swing=swing, back=back, out=out, stow=stow, chain=chain)
        key_needed = own.KEY_ALWAYS_NEEDED or not unlocked
        lost = key_needed and self.off_bench(env, poses, key_body)
        bad_release = own.NEEDS_UNLOCK and released and not unlocked
        progress = 0. if lost or bad_release else clip(progress)*base["other_key_factor"]
        return {**base, "task_id": f"lockkey/{self.task}", "contract": CONTRACT,
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
                "start_preset": env.start_preset}
