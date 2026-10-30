"""Task scoring from poses, solved contact forces and a trailing time window.

seat-cap's original evaluator is delegated to verbatim. New rewards use continuous
distances, angles, insertion and per-nub axial positions, never check counts.
No command, hand identity, action order or best-so-far milestone affects credit.
"""

from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .newton15.task import Evaluator as LegacyEvaluator, measure, nub_layout, FAIL_SCALE
from .tasks import CONTRACTS, DEFAULT_TASK, validate
from .task_presets import (FIXTURE_XY, FIXTURE_HEIGHT, FIXTURE_HALF, BORE_DEPTH,
                           BORE_HALF, NOTEBOOK_HALF, NOTEBOOK_TOP)

WINDOW_S = .5
REST_M = .002
TOUCH_N = .05


def clip(x):
    return float(np.clip(x, 0., 1.))


def ramp(error, full, zero):
    return clip((zero-error)/(zero-full))


def gain(error, initial):
    return clip(1.-error/max(initial, 1e-12))


def frame(pose):
    return Rotation.from_quat(pose[3:]).as_matrix()


def part_points(c, poses):
    result = {}
    for name, length in (("pen", c.pen_length), ("cap", c.cap_length)):
        p, r = poses[name], frame(poses[name])
        result[name] = {"axis": r[:, 1], "mid": p[:3]+r[:, 1]*length/2,
                        "end": p[:3]+r[:, 1]*length}
    return result


def notebook_state(c, notebook):
    r = frame(notebook)
    corners = np.array([[x, y, 0.] for x in (-.076, .076) for y in (-.107, .107)])@r.T+notebook[:3]
    on_table = bool(np.all(np.abs(corners[:, :2]) <= [.55, .45]))
    valid = on_table and r[2, 2] >= .98 and abs(notebook[2]-(c.table_top+.008)) <= .003
    return r, valid


def on_notebook(c, mid, axis, notebook):
    r, notebook_ok = notebook_state(c, notebook)
    local = r.T@(mid-notebook[:3])
    over = bool(np.all(np.abs(local[:2]) <= NOTEBOOK_HALF))
    above = float(local[2]-NOTEBOOK_TOP)
    lying = abs(axis[2]) <= math.sin(math.radians(15))
    # Lower as well as upper bound: a pen beneath the notebook cannot pass.
    supported = over and c.shaft_radius-.0015 <= above <= c.shaft_radius+.003
    excess = np.maximum(np.abs(local[:2])-NOTEBOOK_HALF, 0.)
    distance = float(np.linalg.norm(np.r_[excess, above-.0065]))
    quality = math.exp(-distance/.010)*ramp(abs(axis[2]), 0., 1.)
    return bool(notebook_ok and supported and lying), distance, quality


def on_bench(c, mid, axis, notebook=None):
    over = abs(mid[0]) <= .55 and abs(mid[1]) <= .45
    height = float(mid[2]-c.table_top)
    lying = abs(axis[2]) <= math.sin(math.radians(15))
    valid = over and .003 <= height <= .020 and lying
    xy_error = float(np.linalg.norm(np.maximum(np.abs(mid[:2])-[.55, .45], 0.)))
    quality = math.exp(-math.hypot(xy_error, height-.0065)/.020)*(1.-abs(axis[2]))
    if notebook is not None:
        book_valid, _, book_quality = on_notebook(c, mid, axis, notebook)
        valid, quality = valid or book_valid, max(quality, book_quality)
    return bool(valid), float(quality)


def fixture_state(c, tail, axis, depth_required=.040):
    top = c.table_top+FIXTURE_HEIGHT
    depth = float(top-tail[2])
    offset = tail[:2]-FIXTURE_XY
    upright = clip(-axis[2])
    # Cross-section at the mouth, not just a tail point that can poke through a wall.
    entry = tail-axis*max(depth, 0.)/max(-axis[2], 1e-9)
    fits = (np.max(np.abs(offset)) <= BORE_HALF-.0042 and
            np.max(np.abs(entry[:2]-FIXTURE_XY)) <= BORE_HALF-c.shaft_radius)
    inside = (fits and depth_required <= depth <= BORE_DEPTH+.001 and
              -axis[2] >= math.cos(math.radians(10)))
    radial = ramp(float(np.max(np.abs(offset))), BORE_HALF-c.shaft_radius, FIXTURE_HALF)
    insertion = clip(depth/.040)*radial*upright
    remaining = clip(depth/BORE_DEPTH)*radial*BORE_DEPTH
    out = depth <= 0. or np.max(np.abs(offset)) >= FIXTURE_HALF+c.shaft_radius
    return dict(inside=bool(inside), out=bool(out), remaining=remaining, insertion=insertion,
                upright=upright, depth_m=depth)


def cap_geometry(c, pen, cap):
    """Smooth assembly credit, including continuous axial progress of each nub.

    The discrete corner test of newton15/task.py is used ONLY for success. For
    shaping, each nub ramps from the beginning of the lead to the seated axial
    position; containment tapers to zero at the bore boundary.
    """
    rp, rc = frame(pen), frame(cap)
    tip = rc.T@(pen[:3]-cap[:3])
    nubs, _ = nub_layout(c)
    local = (nubs@rc.T+cap[:3]-pen[:3])@rp
    lead = c.bead_u-c.bead_flat/2-c.lead_run
    target = c.seat_u-c.nub_offset
    axial = float(np.clip((local[:, 1]-lead)/(target-lead), 0., 1.).mean())
    facing = clip(-rp[:, 1]@rc[:, 1])
    radial = float(np.linalg.norm(tip[[0, 2]]))
    lateral = ramp(radial, c.bore_radius-c.bead_radius, c.bore_radius)
    closed_end = ramp(float(tip[1]), c.cap_length-.002, c.cap_length)
    coverage = facing*lateral*closed_end
    presented = max(facing*ramp(float(np.linalg.norm(tip)), 0., .025), coverage*clip(tip[1]/.002))
    quality = .1*presented+.9*coverage*(.5*clip(tip[1]/c.seat_u)+.5*axial)
    return dict(quality=clip(quality), axial=axial, depth=float(tip[1]), radial=radial,
                coverage=coverage)


def contact_forces(env, solved):
    """Any robot link touching a named part; force from a separate solver buffer.

    Like newton15/trace.py, this reads native solved forces without overwriting
    the collision inputs. A missing force buffer is an error, not 'released'.
    """
    forces = {name: 0. for name in env.bodies}
    if env.tick == 0:
        return forces
    env.solver.update_contacts(solved)
    n = int(solved.rigid_contact_count.numpy()[0])
    a, b = (getattr(solved, "rigid_contact_"+name).numpy()[:n] for name in ("shape0", "shape1"))
    force = solved.force.numpy()[:n, :3]
    names = {i: name for name, i in env.bodies.items()}
    for i, j, f in zip(a, b, force):
        if i < 0 or j < 0:
            continue
        bi, bj = (env.builder.shape_body[int(k)] for k in (i, j))
        for part, other in ((bi, bj), (bj, bi)):
            if part in names and other >= 0 and env.builder.body_label[other].startswith(("left/", "right/")):
                magnitude = float(np.linalg.norm(f))
                if not math.isfinite(magnitude):
                    raise FloatingPointError("Non-finite solved robot contact force")
                forces[names[part]] = max(forces[names[part]], magnitude)
    return forces


class SeatCapEvaluator:
    """Metadata only: the original computation and sample timing are untouched."""
    def __init__(self):
        self.legacy = LegacyEvaluator()

    def sample(self, env):
        return dict(self.legacy.sample(env), task_id="pencap/seat-cap", contract=CONTRACTS[DEFAULT_TASK])


class TaskEvaluator:
    def __init__(self, task):
        self.task = validate(task)
        if task == DEFAULT_TASK:
            raise ValueError("Use SeatCapEvaluator for seat-cap")
        self.initial = None
        self.recent = deque()
        self.legacy = LegacyEvaluator()
        self.solved = None
        self.last_tick, self.last_result = None, None

    def sample(self, env):
        if env.tick == self.last_tick:
            return self.last_result
        if self.last_tick is not None and env.tick != self.last_tick+1:
            raise ValueError("Task evaluator requires every physics tick; recreate it on reset")
        if self.initial is None and (env.tick != 0 or getattr(env, "task_start", None) != self.task):
            raise ValueError("Task evaluator requires its authored preset at tick 0")
        c = env.config
        q = env.state.body_q.numpy()
        poses = {name: np.asarray(q[i], float) for name, i in env.bodies.items()}
        if not all(np.isfinite(p).all() for p in poses.values()):
            raise FloatingPointError("Non-finite task object pose")
        if self.solved is None:
            self.solved = env.pipeline.contacts()
        forces = contact_forces(env, self.solved)
        points = part_points(c, poses)
        pen, cap = points["pen"], points["cap"]
        snap = measure(c, poses["pen"], poses["cap"])
        smooth = cap_geometry(c, poses["pen"], poses["cap"])
        fixture = fixture_state(c, pen["end"], pen["axis"], .030 if self.task == "cap-in-fixture" else .040)
        book, book_distance, book_quality = on_notebook(c, pen["mid"], pen["axis"], poses["notebook"])
        pen_down, pen_quality = on_bench(c, pen["mid"], pen["axis"], poses["notebook"] if self.task == "uncap" else None)
        cap_down, cap_quality = on_bench(c, cap["mid"], cap["axis"], poses["notebook"] if self.task == "uncap" else None)
        target = np.array([*FIXTURE_XY, c.table_top+FIXTURE_HEIGHT-BORE_DEPTH])
        tail_distance = float(np.linalg.norm(pen["end"]-target))
        if self.initial is None:
            self.initial = dict(book_distance=book_distance, tail_distance=tail_distance,
                                remaining=fixture["remaining"], upright=fixture["upright"], cap=smooth)
        initial = self.initial
        release = {name: ramp(force, 0., TOUCH_N) for name, force in forces.items()}
        released = {name: force <= TOUCH_N for name, force in forces.items()}
        collateral = sum(clip((c.table_top-float(p[2]))/.050) for p in
                         (pen["mid"], cap["mid"], poses["notebook"][:3]))*.25
        intact = all(p[2] >= c.table_top-.050 for p in
                     (pen["mid"], cap["mid"], poses["notebook"][:3]))
        task = self.task
        if task in ("pen-on-notebook", "pass-pen"):
            valid, tracked = book and released["pen"], ("pen",)
        elif task == "load-fixture":
            valid, tracked = fixture["inside"] and released["pen"], ("pen",)
        elif task == "unload-fixture":
            valid = snap["seated"] and fixture["out"] and pen_down and cap_down and released["pen"] and released["cap"]
            tracked = ("pen", "cap")
        elif task == "uncap":
            valid = (not snap["engaged"] and snap["tip_to_mouth_m"] >= .010 and pen_down and cap_down
                     and released["pen"] and released["cap"])
            tracked = ("pen", "cap")
        else:
            valid, tracked = snap["seated"], ("pen", "cap")
            if task == "cap-in-fixture":
                valid = valid and fixture["inside"]
        valid = bool(valid and intact)
        now = env.tick*c.dt
        mid = np.array([points[name]["mid"] for name in tracked])
        self.recent.append((now, mid, valid))
        while len(self.recent) > 1 and self.recent[1][0] <= now-WINDOW_S+1e-10:
            self.recent.popleft()
        window = np.array([p for _, p, _ in self.recent])
        drift = float(np.linalg.norm(window-mid, axis=-1).max())
        duration = now-self.recent[0][0]
        still = ramp(drift, 0., REST_M)*clip(duration/WINDOW_S)
        success = duration >= WINDOW_S-1e-10 and drift <= REST_M and all(v for _, _, v in self.recent)
        if task in ("reseat-cap", "cap-in-fixture"):
            # Same snap and 10 mm stillness test as seat-cap.
            legacy = self.legacy.sample(env)
            success = legacy["success"] and intact
            if task == "cap-in-fixture":
                success = success and all(v for _, _, v in self.recent)
        if task in ("pen-on-notebook", "pass-pen"):
            progress = gain(book_distance, initial["book_distance"])*(.7+.3*book_quality*release["pen"]*still)
        elif task == "load-fixture":
            approach = gain(tail_distance, initial["tail_distance"])
            upright = clip((fixture["upright"]-initial["upright"])/max(1.-initial["upright"], 1e-12))
            progress = (.25*approach+.25*upright+.5*fixture["insertion"])*(.9+.1*release["pen"]*still)
        elif task == "unload-fixture":
            out = gain(fixture["remaining"], initial["remaining"])
            down = min(pen_quality, cap_quality)*min(release["pen"], release["cap"])*still
            progress = out*(.4+.3*(1.-abs(pen["axis"][2]))+.3*down)*(.5+.5*smooth["quality"])
        elif task == "reseat-cap":
            progress = clip((smooth["quality"]-initial["cap"]["quality"])/(1.-initial["cap"]["quality"]))
        elif task == "cap-in-fixture":
            progress = smooth["quality"]*(.8+.2*fixture["insertion"])
        else:  # uncap: withdrawal through either open end or sideways, independent of which part moves.
            start = initial["cap"]
            withdrawn = max(clip((start["depth"]-smooth["depth"])/(start["depth"]+.010)),
                            clip((smooth["radial"]-start["radial"])/(c.bore_radius+.010)),
                            clip((smooth["depth"]-start["depth"])/(c.cap_length+.010-start["depth"])))
            # Normalize to the actual stored (float32 in Newton) seated start.
            # Its micrometre-scale rounding must not award idle uncapping credit.
            unsnapped = gain(smooth["axial"]*smooth["coverage"], start["axial"]*start["coverage"])
            down = .5*(pen_quality*release["pen"]+cap_quality*release["cap"])*still
            progress = .4*unsnapped+.4*withdrawn+.2*withdrawn*down
        progress = clip(progress-collateral)
        result = dict(task_id=f"pencap/{task}", contract=CONTRACTS[task], success=bool(success),
                      score=1. if success else FAIL_SCALE*progress, progress=progress,
                      score_rule="1 if success else 0.9 x continuous progress", current_predicates=valid,
                      robot_contact_force_n=forces, window_max_drift_m=drift, window_s=duration,
                      seated=snap["seated"], nubs_snapped=snap["nubs_snapped"],
                      writing_tip_in_cap=snap["engaged"], fixture=fixture, collateral=collateral)
        self.last_tick, self.last_result = env.tick, result
        return result
