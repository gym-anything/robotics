"""pencap-tasks-v2 scoring: task_scoring.py with the render-check fixes.

Scoring from poses, solved contact forces and a trailing time window. seat-cap is
not part of this module: it keeps its released launch and evaluator. New rewards
use continuous distances, angles, insertion and per-nub axial positions, never
check counts. No command, hand identity, action order or best-so-far milestone
affects credit. Changes from task_scoring.py: gain() has the rest-tolerance
deadband and gives no credit for a start without error, load-fixture's upright
term is standing(), uncap progress is uncap_terms() from the snap groove, and
results report contract pencap-tasks-v2 with its task_id. Each task's own
predicates, success verdict and progress are in its verifier.py (tasks/<task>/;
progress_v2 where this contract's progress differs); TaskEvaluator.grade calls them.
"""

from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .newton15.task import Evaluator as LegacyEvaluator, measure, nub_layout, FAIL_SCALE
from .tasks import DEFAULT_TASK, task_verifier, validate
from .task_set_v2 import CONTRACT, task_id
from .task_presets_v2 import (FIXTURE_XY, FIXTURE_HEIGHT, FIXTURE_HALF, BORE_DEPTH,
                              BORE_HALF, NOTEBOOK_HALF, NOTEBOOK_TOP)

WINDOW_S = .5
REST_M = .002
TOUCH_N = .05
LYING_DEG = 15.


def clip(x):
    return float(np.clip(x, 0., 1.))


def ramp(error, full, zero):
    return clip((zero-error)/(zero-full))


def gain(error, initial, deadband=REST_M):
    """Share of the start's error (metres) removed beyond the rest tolerance.

    Settling smaller than REST_M is what the rest test itself calls still, so
    it earns nothing (render-check probe: an idle pen settling
    0.3 mm scored 0.0005 on pen-on-notebook). A start with no error beyond the
    deadband gives no credit (the first version gave full credit): an
    untouched start must score 0.
    """
    span = initial-deadband
    if not span > 1e-9:
        return 0.
    return clip((initial-error-deadband)/span)


def standing(axis):
    """Tail-down elevation credit: 0 while lying (within LYING_DEG of horizontal,
    the set's own lying test), 1 within 10 degrees of vertical (the spec's ramp)."""
    elevation = math.degrees(math.asin(clip(-axis[2])))
    return clip((elevation-LYING_DEG)/(80.-LYING_DEG))


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
    lying = abs(axis[2]) <= math.sin(math.radians(LYING_DEG))
    # Lower as well as upper bound: a pen beneath the notebook cannot pass.
    supported = over and c.shaft_radius-.0015 <= above <= c.shaft_radius+.003
    excess = np.maximum(np.abs(local[:2])-NOTEBOOK_HALF, 0.)
    distance = float(np.linalg.norm(np.r_[excess, above-.0065]))
    quality = math.exp(-distance/.010)*ramp(abs(axis[2]), 0., 1.)
    return bool(notebook_ok and supported and lying), distance, quality


def on_bench(c, mid, axis, notebook=None):
    over = abs(mid[0]) <= .55 and abs(mid[1]) <= .45
    height = float(mid[2]-c.table_top)
    lying = abs(axis[2]) <= math.sin(math.radians(LYING_DEG))
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
                coverage=coverage, nub_axial=local[:, 1].copy())


def uncap_terms(c, snap, smooth):
    """The spec's uncap terms, continuous and measured from the snap groove.

    unsnapped: per nub, how far its centre has been dragged back from the bead
    crest's trailing edge (where the audited test stops calling it snapped) to
    the start of the lead ramp; mean over the nubs. withdrawn: tip depth below
    the depth at which every nub sits on that edge, as a share of it. Both are
    0 for a cap anywhere in its snapped range, whatever its tilt or radial play
    on the pen (render check: the idle seated start settled 1.5
    degrees off-axis and 0.2 mm deeper, still seated, and the first version
    credited that play at 0.085). Both are 1 once the tip is out of the cap.
    """
    if not snap["engaged"]:
        return 1., 1.
    crest = nub_layout(c)[1]
    lead = c.bead_u-c.bead_flat/2-c.lead_run
    unsnapped = float(np.clip((crest-smooth["nub_axial"])/(crest-lead), 0., 1.).mean())
    on_crest = crest+c.nub_offset
    return unsnapped, clip((on_crest-smooth["depth"])/on_crest)


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


class TaskEvaluator:
    def __init__(self, task):
        self.task = validate(task)
        if task == DEFAULT_TASK:
            raise ValueError("seat-cap keeps its released evaluator (task_scoring.SeatCapEvaluator)")
        self.initial = None
        self.recent = deque()
        self.legacy = LegacyEvaluator()
        self.solved = None
        self.last_tick, self.last_result = None, None

    @property
    def folder(self):
        """The task's verifier.py (tasks/<task>/)."""
        return task_verifier(self.task)

    def sample(self, env):
        """One sample, every physics tick: the task's verifier.py judges it (verify, through grade)."""
        return self.folder.verify(self, env)

    def grade(self, env):
        """One sample: the scene measured (poses, solved robot contact forces, the trailing window) and judged by the
        task's verifier.py, which receives the measurements as keyword arguments: predicates(...) -> (the success
        predicates now, the parts the rest test tracks); success(evaluator, env, ...), if defined, gives the success
        verdict; progress_v2(...) -> the progress, else progress(...) as in task_scoring.py. FIXTURE_DEPTH_M (the
        insertion the fixture test requires; else 40 mm) and NOTEBOOK_ON_BENCH (a part may lie on the notebook as on the
        bench) apply if it sets them."""
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
        folder = self.folder
        fixture = fixture_state(c, pen["end"], pen["axis"], getattr(folder, "FIXTURE_DEPTH_M", .040))
        book, book_distance, book_quality = on_notebook(c, pen["mid"], pen["axis"], poses["notebook"])
        bench = poses["notebook"] if getattr(folder, "NOTEBOOK_ON_BENCH", False) else None
        pen_down, pen_quality = on_bench(c, pen["mid"], pen["axis"], bench)
        cap_down, cap_quality = on_bench(c, cap["mid"], cap["axis"], bench)
        target = np.array([*FIXTURE_XY, c.table_top+FIXTURE_HEIGHT-BORE_DEPTH])
        tail_distance = float(np.linalg.norm(pen["end"]-target))
        if self.initial is None:
            self.initial = dict(book_distance=book_distance, tail_distance=tail_distance,
                                remaining=fixture["remaining"], upright=standing(pen["axis"]), cap=smooth,
                                uncap=uncap_terms(c, snap, smooth))
        initial = self.initial
        release = {name: ramp(force, 0., TOUCH_N) for name, force in forces.items()}
        released = {name: force <= TOUCH_N for name, force in forces.items()}
        collateral = sum(clip((c.table_top-float(p[2]))/.050) for p in
                         (pen["mid"], cap["mid"], poses["notebook"][:3]))*.25
        intact = all(p[2] >= c.table_top-.050 for p in
                     (pen["mid"], cap["mid"], poses["notebook"][:3]))
        task = self.task
        m = dict(c=c, pen=pen, snap=snap, smooth=smooth, fixture=fixture, book=book, book_distance=book_distance,
                 book_quality=book_quality, pen_down=pen_down, pen_quality=pen_quality, cap_down=cap_down,
                 cap_quality=cap_quality, tail_distance=tail_distance, initial=initial, release=release,
                 released=released, intact=intact, gain=gain)
        valid, tracked = folder.predicates(**m)
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
        if hasattr(folder, "success"):
            success = folder.success(self, env, **m)
        progress = getattr(folder, "progress_v2", folder.progress)(still=still, **m)
        progress = clip(progress-collateral)
        result = dict(task_id=task_id(task), contract=CONTRACT, success=bool(success),
                      score=1. if success else FAIL_SCALE*progress, progress=progress,
                      score_rule="1 if success else 0.9 x continuous progress", current_predicates=valid,
                      robot_contact_force_n=forces, window_max_drift_m=drift, window_s=duration,
                      seated=snap["seated"], nubs_snapped=snap["nubs_snapped"],
                      writing_tip_in_cap=snap["engaged"], fixture=fixture, collateral=collateral)
        self.last_tick, self.last_result = env.tick, result
        return result
