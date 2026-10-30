"""Capping evaluator and progress reward, from the current pen/cap state only.

Independent of robot strategy, of which part moves, of world orientation and of
how the cap went on: no alignment gate, no insertion history, no surface.
"""

from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation

DESCRIPTION = ("Put the blue cap on the yellow pen's writing tip and press it until it snaps fully into place, "
    "then leave the capped pen still, whether held or set down.")

WINDOW_S = .5          # trailing window: the cap stays snapped on and the capped pen still
STILL_M = .010         # each part's mid-length point stays within this of its final position over the window
FAIL_SCALE = .9        # a non-success scores 0.9 x progress, so every failure stays below every success
PRESENT_SHARE = .10    # share of progress for the cap presented to the writing tip (axes opposed, mouth near it)
PRESENT_RADIUS_M = .025  # tip-to-mouth distance at which that pre-engagement credit starts
CRITERION = ("Writing tip inside the cap; every nub past the bead crest (centre beyond its trailing edge, "
    "not resting on that corner); held for 0.5 s with each part within 10 mm of its final position")
SCORE_RULE = "1 if success else 0.9 x progress"


def nub_layout(c):
    """Nub centres in the cap frame, and the pen-frame axial position of the bead crest's trailing edge."""
    ring = c.bead_radius-c.interference+c.nub_radius
    angles = 2*np.pi*np.arange(c.nub_count)/c.nub_count
    nubs = np.stack([ring*np.cos(angles), np.full(c.nub_count, c.nub_offset), ring*np.sin(angles)], -1)
    return nubs, c.bead_u+c.bead_flat/2


def nubs_snapped(c, axial, radial):
    """Per nub, from its own pen-frame position, whatever the cap's tilt or sideways shift: snapped once its
    centre is past the crest's trailing edge, unless it still rests on that corner, i.e. touches it with a
    contact normal within the friction angle atan(mu) of radial. Beyond that, corner or 50 degree return-face
    contact pushes the cap on, and coming off means dragging the nub back over the corner."""
    _, edge = nub_layout(c)
    dy, dr = axial-edge, radial-c.bead_radius
    on_corner = (np.hypot(dy, dr) < c.nub_radius) & (np.arctan2(dy, dr) <= math.atan(c.plastic_friction))
    return (dy > 0) & ~on_corner


def measure(c, pen, cap):
    """Work-product state of one sample. pen, cap: body poses (x, y, z, qx, qy, qz, qw).
    Pen +Y runs tip-to-tail from the writing tip; cap +Y runs mouth-to-closed-end from the mouth."""
    rp, rc = (Rotation.from_quat(p[3:]).as_matrix() for p in (pen, cap))
    mouth = rp.T@(cap[:3]-pen[:3])
    tip = rc.T@(pen[:3]-cap[:3])
    nubs, _ = nub_layout(c)
    local = (nubs@rc.T+cap[:3]-pen[:3])@rp
    opposed = float(-rp[:, 1]@rc[:, 1])
    # Containment, not alignment: the writing tip is inside the cap's bore and the axes oppose.
    engaged = bool(opposed > 0 and 0 <= tip[1] <= c.cap_length-.002 and math.hypot(tip[0], tip[2]) < c.bore_radius)
    snapped = int(nubs_snapped(c, local[:, 1], np.hypot(local[:, 0], local[:, 2])).sum()) if engaged else 0
    reach = float(np.linalg.norm(tip))
    # Pre-engagement credit from the relative pose only: the axes opposed and the mouth near the tip.
    presented = 1. if engaged else max(opposed, 0.)*max(1.-reach/PRESENT_RADIUS_M, 0.)
    onto = .5*min(max(tip[1]/c.seat_u, 0.), 1.)+.5*snapped/c.nub_count if engaged else 0.
    return {"engaged": engaged, "seated": snapped == c.nub_count, "nubs_snapped": snapped,
        "progress": float(PRESENT_SHARE*presented+(1.-PRESENT_SHARE)*onto),
        "tip_depth_m": float(tip[1]) if engaged else None, "tip_to_mouth_m": reach, "mouth": mouth,
        "mouth_past_collar_m": max(float(mouth[1]-c.shoulder_u), 0.) if engaged else None,
        "angle": float(np.arccos(np.clip(opposed, -1., 1.))),
        "mid": np.r_[pen[:3]+rp[:, 1]*c.pen_length/2, cap[:3]+rc[:, 1]*c.cap_length/2]}


class Evaluator:
    def __init__(self):
        self.seated_samples = 0
        self.recent = None

    def sample(self, env):
        c = env.config
        poses = env.state.body_q.numpy()
        m = measure(c, *(np.asarray(poses[env.bodies[n]], float) for n in ("pen", "cap")))
        need = max(1, round(WINDOW_S/c.dt))
        if self.recent is None:
            self.recent = deque(maxlen=need)
        self.recent.append(m["mid"])
        self.seated_samples = self.seated_samples+1 if m["seated"] else 0
        drift = None
        if len(self.recent) == need:
            window = np.asarray(self.recent).reshape(need, 2, 3)
            drift = float(np.linalg.norm(window-window[-1], axis=-1).max())
        success = self.seated_samples >= need and drift is not None and drift < STILL_M
        return {"success": success, "score": 1. if success else FAIL_SCALE*m["progress"],
            "progress": m["progress"], "seated": m["seated"], "seated_s": self.seated_samples*c.dt,
            "writing_tip_in_cap": m["engaged"], "tip_depth_in_cap_m": m["tip_depth_m"],
            "tip_to_mouth_m": m["tip_to_mouth_m"], "mouth_past_collar_m": m["mouth_past_collar_m"],
            "nubs_snapped": m["nubs_snapped"], "cap_mouth_in_pen_frame_m": m["mouth"].tolist(),
            "axis_angle_rad": m["angle"], "window_max_drift_m": drift,
            "criterion": CRITERION, "score_rule": SCORE_RULE}
