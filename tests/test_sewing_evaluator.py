"""Self-test of the sewing evaluator (sewing/task.py, class Evaluator) on synthetic worlds, the way a verifier is
validated before it judges anything: known-good inputs must pass, and each known defect must fail the check that names
it (never pass silently).

No GPU, no physics. The fakes carry exactly what the evaluator reads from a SewingEnv: the machine's stitch ledger and
holes (ply, material uv, host triangle and barycentric coordinates, as sewing.machine.Hole), the needle, tether, tails
and threads, the plies' material mesh (the real mesher, sewing.mesh, called as World._build_cloth calls it) and the
vertices' world positions, the hands' cloth forces and the reflex history. The table is the real one (sewing.layout).

Every defect test is a minimal pair with the perfect seam: the same ledger with ONE deviation. It must fail the named
check, and it may fail only the other checks listed with it, each of which the same deviation genuinely violates or
collaterally trips (the reason is written where the check is listed).

Plain seam (TASK_SETUP["plain-seam"]): plies 200 x 420 mm, allowance 10 mm (layout.SEAM_ALLOWANCE), stitch 2.5 mm.
The ledger follows the order the scripted demonstration sews (demos/guided_seam.py): 3 reverse stitches toward the
far end, forward stitches every 2.5 mm on u = W - 10 mm from 4 mm short of the far end to 3.5 mm from the near end,
then 3 reverse stitches back. Curve (TASK_SETUP["curve"]): plies 200 x 300 mm with a 100 mm corner, holes every
2.5 mm along the line 10 mm inside mesh.seam_edge_path.

  python -m pytest tests/test_sewing_evaluator.py
"""

from dataclasses import dataclass
import math
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from sewing import params as P
from sewing.layout import SEAM_ALLOWANCE, TABLE
from sewing.mesh import mesh_panel, rect_panel, seam_edge_path
from sewing.task import TASK_SETUP, Evaluator, _polyline_distance, _triangle_under
from tests.packages import PACKAGES

ROOT = PACKAGES

PLAIN = TASK_SETUP["plain-seam"]
W, L, STITCH = PLAIN["ply_width"], PLAIN["ply_length"], PLAIN["stitch_length"]
A = SEAM_ALLOWANCE
THICK = P.FABRICS["muslin_125"]["thickness_free"]
N_FORWARD = 166                     # (L - 4 mm) - 165 x 2.5 mm = 3.5 mm from the near end

PLAIN_CHECKS = {"stitches_formed", "joined_far_end", "joined_near_end", "seam_line", "seam_mean", "stitch_length",
                "stitch_density", "skips", "backtack_start", "backtack_end", "no_tucks", "ends_meet",
                "needle_intact", "no_reflex_while_sewing", "threads_trimmed", "released", "no_thread_break"}
CURVE_CHECKS = {"stitches_formed", "seam_line", "plies_registered", "no_gaps", "stitch_density", "no_long_threads",
                "joins_secured", "joined_far_end", "reaches_left_edge", "skips", "no_tucks", "needle_intact", "no_reflex_while_sewing",
                "threads_trimmed", "released", "no_thread_break"}


@dataclass(frozen=True)
class Hole:
    """The fields of sewing.machine.Hole the evaluator reads (machine.py imports the cloth solver, hence warp)."""
    ply: int
    uv: tuple
    face: int = -1
    bary: tuple = (1.0, 0.0, 0.0)


class FakeCloth:
    """What the evaluator reads of the cloth solver: the plies' material coordinates and triangles in the solver's
    global numbering (plies stacked in order, each a copy of the panel mesh), the ply registry and world positions."""

    def __init__(self, X, tri, positions):
        nv, nt = len(X), len(tri)
        self.X = np.vstack([X]*len(positions))
        self.tri_host = np.vstack([tri + k*nv for k in range(len(positions))]).astype(np.int32)
        self.plies = [dict(v0=k*nv, nv=nv, t0=k*nt, nt=nt) for k in range(len(positions))]
        self._x = np.vstack(positions)

    def positions(self):
        return self._x.copy()


class FakeWorld:
    def __init__(self, config, machine, mesh=None, cloth=None):
        self.config, self.machine, self.mesh, self.cloth = config, machine, mesh, cloth
        self.ply_ids = [0, 1]
        self.hand_arms = ("left", "right")

    def ply_positions(self, k):                     # as sewing.world.World.ply_positions
        p = self.cloth.plies[self.ply_ids[k]]
        return self.cloth.positions()[p["v0"]:p["v0"] + p["nv"]]

    def finger_force(self, arm):                    # the hands have let go of the work
        return np.zeros(3)


_MESH = {}


def ply_mesh(width, length, allowance):
    key = (width, length, allowance)
    if key not in _MESH:
        _MESH[key] = mesh_panel(rect_panel("ply", width, length, seam_offset=allowance))
    return _MESH[key]


def locate(X, tri, uv, chunk=64):
    """Host triangle and barycentric coordinates of material points uv (k, 2) in the panel mesh (X, tri)."""
    a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    v0, v1 = b - a, c - a
    den = v0[:, 0]*v1[:, 1] - v1[:, 0]*v0[:, 1]
    faces, barys = [], []
    for s in range(0, len(uv), chunk):
        q = np.asarray(uv[s:s + chunk], float)[:, None, :] - a[None]
        l1 = (q[..., 0]*v1[:, 1] - v1[:, 0]*q[..., 1])/den
        l2 = (v0[:, 0]*q[..., 1] - q[..., 0]*v0[:, 1])/den
        l0 = 1.0 - l1 - l2
        t = np.argmax(np.minimum(np.minimum(l0, l1), l2), axis=1)
        i = np.arange(len(t))
        bary = np.column_stack([l0[i, t], l1[i, t], l2[i, t]])
        assert bary.min() >= -1e-9, "a hole outside the panel"
        faces.append(t)
        barys.append(bary)
    return np.concatenate(faces), np.vstack(barys)


def machine(stitch_specs, holes):
    """The machine's ledger after a finished seam: needle intact, threads trimmed (3 mm tails), none broken."""
    stitches = []
    for cyc, (spec, idx) in enumerate(zip(stitch_specs, holes)):
        stitches.append(dict(cycle=cyc, reverse=spec["reverse"], holes=idx, formed=spec["formed"],
                             reason=None if spec["formed"] else "missed_loop",
                             balance="balanced" if spec["formed"] else None))
    n_formed = sum(1 for s in stitches if s["formed"])
    return SimpleNamespace(stitches=stitches, holes=[], needle_state="ok", tether="trimmed",
                           tails=dict(start=P.TRIM_TAIL, end=P.TRIM_TAIL),
                           threads=[dict(kind=k, broken=False) for _ in range(max(0, n_formed - 1))
                                    for k in ("needle", "bobbin")])


def env_of(world):
    return SimpleNamespace(world=world, config=world.config, history=dict(reflex_while_sewing=0))


# ---------------------------------------------------------------------------------------------- plain seam
def stitch(v, reverse=False, formed=True, holes=None):
    """One needle cycle through both plies on the seam line at material v (or through the given (ply, u, v) holes)."""
    return dict(reverse=reverse, formed=formed, holes=holes or [(0, W - A, v), (1, W - A, v)])


def forward_v():
    return (L - 4e-3) - STITCH*np.arange(N_FORWARD)


def plain_ledger(start_backtack=True, end_backtack=True):
    """The perfect seam as sewn: start backtack toward the far end, the forward run, end backtack."""
    fwd = forward_v()
    start = [stitch(fwd[0] - STITCH*k, reverse=True) for k in (3, 2, 1)]     # 11.5, 9, 6.5 mm from the far end
    end = [stitch(fwd[-1] + STITCH*k, reverse=True) for k in (1, 2, 3)]      # 6, 8.5, 11 mm from the near end
    return (start if start_backtack else []) + [stitch(v) for v in fwd] + (end if end_backtack else [])


MID = 3 + N_FORWARD//2              # ledger index of a forward stitch half way along the seam


def flat(X, k, y0=-0.100, squeeze=1.0):
    """The stack lying flat on the table: the seam line over the needle's x, ply k at its height in the stack;
    `squeeze` < 1 shortens the plies' world length along the seam (a gathered seam)."""
    return np.column_stack([X[:, 0] - (W - A), y0 + squeeze*X[:, 1], np.full(len(X), (k + 0.5)*THICK)])


def draped(X, k, radius=3e-3):
    """The near half flat on the table up to its back edge, the far half bent over the edge (concentric bends of
    radius + the ply's height) and hanging straight down: material length preserved along v."""
    y_edge = TABLE["y"][1]
    p = flat(X, k, y0=y_edge - L/2)
    r = radius + p[:, 2]
    s = p[:, 1] - y_edge
    bend = (s > 0) & (s <= 0.5*math.pi*r)
    hang = s > 0.5*math.pi*r
    phi = s[bend]/r[bend]
    p[bend, 1] = y_edge + r[bend]*np.sin(phi)
    p[bend, 2] = -radius + r[bend]*np.cos(phi)
    p[hang, 1] = y_edge + r[hang]
    p[hang, 2] = -radius - (s[hang] - 0.5*math.pi*r[hang])
    return p


def plain_seam_world(ledger, place=flat):
    mesh = ply_mesh(W, L, A)
    X, tri = mesh["X"], mesh["tri"]
    cloth = FakeCloth(X, tri, [place(X, k) for k in (0, 1)])
    pts = [h for s in ledger for h in s["holes"]]
    face, bary = locate(X, tri, [(u, v) for _, u, v in pts])
    holes, idx, i = [], [], 0
    for s in ledger:
        idx.append([])
        for ply, u, v in s["holes"]:
            holes.append(Hole(ply=ply, uv=(float(u), float(v)), face=int(face[i]) + cloth.plies[ply]["t0"],
                              bary=tuple(map(float, bary[i]))))
            idx[-1].append(len(holes) - 1)
            i += 1
    M = machine(ledger, idx)
    M.holes = holes
    config = SimpleNamespace(task="plain-seam", ply_width=W, ply_length=L, seam_allowance=A, corner_radius=0.0,
                             stitch_length=STITCH)
    return FakeWorld(config, M, mesh=mesh, cloth=cloth)


def evaluate(world):
    return Evaluator(env_of(world)).evaluate()


def failed(result):
    return {k for k, c in result["checks"].items() if not c["ok"]}


class EvaluatorTestCase(unittest.TestCase):
    def assertFailsOn(self, result, named, also=()):
        """`named` fail; besides them only the checks in `also` may fail."""
        bad = failed(result)
        self.assertFalse(result["success"])
        self.assertLessEqual(set(named), bad, f"must fail {sorted(named)}; failed {sorted(bad)}")
        self.assertLessEqual(bad, set(named) | set(also), f"unexpected failures {sorted(bad - set(named) - set(also))}")


class ImportTest(unittest.TestCase):
    def test_evaluator_import_closure_is_engine_free(self):
        # in a fresh interpreter: under pytest the gym_anything_robotics package (this test's parent) is imported first
        code = ("import sys; import sewing.task, sewing.layout, sewing.mesh; "
                "print(sorted(m for m in ('warp', 'newton', 'mujoco', 'torch') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")


class PlainSeamTest(EvaluatorTestCase):
    def test_fixture_geometry(self):
        fwd = forward_v()
        self.assertAlmostEqual(fwd[0], L - 4e-3, places=12)
        self.assertAlmostEqual(fwd[-1], 3.5e-3, places=12)
        w = plain_seam_world(plain_ledger())
        # every hole's host triangle and barycentric coordinates reproduce its material point
        cl = w.cloth
        for h in w.machine.holes:
            np.testing.assert_allclose(np.dot(h.bary, cl.X[cl.tri_host[h.face]]), h.uv, atol=1e-12)
        self.assertEqual(len(w.machine.stitches), 3 + N_FORWARD + 3)

    def test_perfect_seam_passes(self):
        r = evaluate(plain_seam_world(plain_ledger()))
        self.assertEqual(set(r["checks"]), PLAIN_CHECKS)
        self.assertEqual(failed(r), set())
        self.assertTrue(r["success"])
        self.assertEqual(r["score"], 1.0)
        self.assertTrue(r["primary_outcome"])
        c = r["checks"]
        self.assertEqual(c["stitches_formed"]["value"], 3 + N_FORWARD + 3)
        self.assertEqual(c["joined_far_end"]["value"], 4.0)
        self.assertEqual(c["joined_near_end"]["value"], 3.5)
        self.assertEqual(c["seam_line"]["value"], 0.0)
        self.assertEqual(c["stitch_length"]["value"], [2.5, 2.5])
        self.assertEqual(c["stitch_density"]["value"], [10, 11])
        self.assertEqual((c["backtack_start"]["value"], c["backtack_end"]["value"]), (3, 3))
        self.assertEqual(c["no_tucks"]["value"], dict(stitches_through_a_fold=0, farthest_hole_mm=10.0))
        self.assertEqual(r["info"]["pucker"]["value"], dict(rms_mm=0.0, contraction=0.0))
        self.assertEqual(c["ends_meet"]["value"], 0.0)

    def test_hole_off_the_line_fails_seam_line_and_no_tucks(self):
        ledger = plain_ledger()
        i = MID
        v = ledger[i]["holes"][0][2]
        ledger[i] = stitch(v, holes=[(0, W - 21.5e-3, v), (1, W - A, v)])
        r = evaluate(plain_seam_world(ledger))
        self.assertFailsOn(r, {"seam_line", "no_tucks"})
        self.assertEqual(r["checks"]["seam_line"]["value"], 11.5)
        self.assertAlmostEqual(r["checks"]["no_tucks"]["value"]["farthest_hole_mm"], 21.5, places=2)
        self.assertEqual(r["checks"]["no_tucks"]["value"]["stitches_through_a_fold"], 0)

    def test_fold_carried_under_the_needle_fails_no_tucks(self):
        # the machine makes at most one hole per ply per stroke (machine.py), so a fold carried under the needle shows
        # as one ply's hole far from the edge: here ply 0 at 35 mm
        ledger = plain_ledger()
        i = MID
        v = ledger[i]["holes"][0][2]
        ledger[i] = stitch(v, holes=[(0, W - 35e-3, v), (1, W - A, v)])
        r = evaluate(plain_seam_world(ledger))
        self.assertFailsOn(r, {"seam_line", "no_tucks"})
        self.assertAlmostEqual(r["checks"]["no_tucks"]["value"]["farthest_hole_mm"], 35.0, places=2)

    def test_ply_pierced_twice_fails_no_tucks(self):
        ledger = plain_ledger()
        i = MID
        v = ledger[i]["holes"][0][2]
        # a pleat across the seam: two material points of ply 0 (5 mm apart along v) under the needle, plies [0, 0, 1]
        ledger[i] = stitch(v, holes=[(0, W - A, v), (0, W - A, v + 5e-3), (1, W - A, v)])
        w = plain_seam_world(ledger)
        self.assertEqual(Evaluator(env_of(w)).stitches()[i]["plies"], [0, 0, 1])
        r = evaluate(w)
        # also stitch_length: the tuck stitch is not in the evaluator's list of stitches binding plies [0, 1], so its
        # neighbours read as one 5 mm stitch although the interlock between them exists (a misattribution; the verdict
        # is right)
        self.assertFailsOn(r, {"no_tucks"}, also={"stitch_length"})
        self.assertEqual(r["checks"]["no_tucks"]["value"], dict(stitches_through_a_fold=1, farthest_hole_mm=10.0))

    def test_drape_over_the_table_edge_is_not_contraction(self):
        w = plain_seam_world(plain_ledger(), place=draped)
        top = w.ply_positions(1)
        X = w.mesh["X"]
        self.assertLess(top[X[:, 1] > 0.9*L, 2].max(), -0.1)        # the far end hangs > 10 cm below the table top
        # a chord between the first and last forward holes would read the drape as contraction (here > 10 %)
        M, cl = w.machine, w.cloth
        fwd = [M.holes[j] for s in M.stitches if s["formed"] and not s["reverse"] for j in s["holes"]
               if M.holes[j].ply == 1]
        a, b = min(fwd, key=lambda h: h.uv[1]), max(fwd, key=lambda h: h.uv[1])
        xs = cl.positions()
        chord = np.linalg.norm(np.dot(b.bary, xs[cl.tri_host[b.face]]) - np.dot(a.bary, xs[cl.tri_host[a.face]]))
        self.assertGreater(1.0 - chord/math.dist(a.uv, b.uv), 0.10)
        r = evaluate(w)
        self.assertEqual(failed(r), set())
        self.assertTrue(r["success"])
        self.assertEqual(r["info"]["pucker"]["value"]["rms_mm"], 0.0)
        self.assertLess(abs(r["info"]["pucker"]["value"]["contraction"]), 2e-3)

    def test_gathered_seam_fails_pucker_on_contraction(self):
        # flat but 5 % shorter in the world than in the material along the seam: the band's height stays uniform, so
        # only the contraction can fail. Pucker is recorded, not scored, while identical inputs sew two seam shapes:
        # the record says it failed and the score does not count it
        r = evaluate(plain_seam_world(plain_ledger(), place=lambda X, k: flat(X, k, squeeze=0.95)))
        self.assertNotIn("pucker", r["checks"])
        self.assertTrue(r["success"])
        self.assertFalse(r["info"]["pucker"]["ok"])
        self.assertEqual(r["info"]["pucker"]["value"]["rms_mm"], 0.0)
        self.assertAlmostEqual(r["info"]["pucker"]["value"]["contraction"], 0.05, places=4)

    def test_damaged_needle_fails_needle_intact(self):
        # a needle that struck the plate or the hook is burred: it fails as a bent one does
        for state in ("damaged", "bent", "broken"):
            w = plain_seam_world(plain_ledger())
            w.machine.needle_state = state
            r = evaluate(w)
            self.assertFailsOn(r, {"needle_intact"})
            self.assertEqual(r["checks"]["needle_intact"]["value"], state)

    def test_balance_is_recorded_not_scored(self):
        # the tensions that set balance are fixed by the task: a check of it could not fail
        r = evaluate(plain_seam_world(plain_ledger()))
        self.assertNotIn("balance", r["checks"])
        self.assertEqual(r["info"]["balance"]["value"], 1.0)

    def test_raw_edge_lift_is_not_pucker_but_a_wave_in_the_seam_band_is(self):
        # the cut edge of the allowance lifted 2 mm along the whole seam (the feed's tick artifact): not pucker
        def lifted(X, k):
            p = flat(X, k)
            p[X[:, 0] > W - 3e-3, 2] += 2e-3*(k == 1)
            return p
        r = evaluate(plain_seam_world(plain_ledger(), place=lifted))
        self.assertEqual(failed(r), set())
        self.assertTrue(r["info"]["pucker"]["ok"])
        self.assertEqual(r["info"]["pucker"]["value"]["rms_mm"], 0.0)

        # a 1.5 mm wave of the work along the seam band, 30 mm long: pucker
        def waved(X, k):
            p = flat(X, k)
            band = X[:, 0] > W - 20e-3
            p[band, 2] += 1.5e-3*np.sin(2*math.pi*X[band, 1]/30e-3)
            return p
        r = evaluate(plain_seam_world(plain_ledger(), place=waved))
        self.assertFalse(r["info"]["pucker"]["ok"])
        self.assertGreater(r["info"]["pucker"]["value"]["rms_mm"], 0.5)

    def test_missing_end_backtack_fails_backtack_end(self):
        r = evaluate(plain_seam_world(plain_ledger(end_backtack=False)))
        self.assertFailsOn(r, {"backtack_end"})
        self.assertEqual((r["checks"]["backtack_start"]["value"], r["checks"]["backtack_end"]["value"]), (3, 0))

    def test_missing_start_backtack_fails_backtack_start(self):
        r = evaluate(plain_seam_world(plain_ledger(start_backtack=False)))
        self.assertFailsOn(r, {"backtack_start"})
        self.assertEqual((r["checks"]["backtack_start"]["value"], r["checks"]["backtack_end"]["value"]), (0, 3))

    def test_skipped_run_fails_skips(self):
        ledger = plain_ledger()
        i = MID
        for j in (i, i + 1):        # the needle pierced both plies and fed on, the hook missed the loop twice
            ledger[j] = dict(ledger[j], formed=False)
        r = evaluate(plain_seam_world(ledger))
        # also stitch_density, true of this seam: an inch window over the run holds 8 interlocks. stitch_length judges
        # the 7.5 mm span against the three stitches it spans (2.5 mm each); the run itself is the skips check's
        self.assertFailsOn(r, {"skips"}, also={"stitch_density"})
        self.assertEqual(r["checks"]["skips"]["value"], dict(isolated=2, longest_run=2))
        self.assertEqual(r["checks"]["stitch_length"]["value"], [2.5, 2.5])

    # One forward stitch half way along unformed (holes in both plies on the line, reason missed_loop, the feed going
    # on as the machine's does): within the tolerance of one isolated skip (sewing/REPORT.pdf §10, FED-STD-751a).
    # stitch_length judges each span against the stitches it spans (task.py), so the skip's 5 mm span is not read as
    # one stitch.
    def test_one_isolated_skip_is_within_the_documented_tolerance(self):
        ledger = plain_ledger()
        i = MID
        ledger[i] = dict(ledger[i], formed=False)
        r = evaluate(plain_seam_world(ledger))
        self.assertEqual(r["checks"]["skips"]["value"], dict(isolated=1, longest_run=1))
        self.assertEqual(failed(r), set())

    # Only the start backtack (3 reverse stitches toward the far end), then trim and let go, as a policy that stops
    # after the first backtack leaves it: a verdict, not a crash. The backtack split takes the middle of all stitches
    # when none was sewn forward.
    def test_start_backtack_only_is_judged_not_crashed(self):
        r = evaluate(plain_seam_world(plain_ledger()[:3]))
        self.assertFalse(r["success"])
        self.assertFalse(r["checks"]["joined_near_end"]["ok"])


# --------------------------------------------------------------------------------------------------- curve
CURVE = TASK_SETUP["curve"]
CW, CL, CR = CURVE["ply_width"], CURVE["ply_length"], CURVE["corner_radius"]


def curve_line(start=4e-3, stop_short=3.5e-3):
    """Material points every 2.5 mm of arc length along the line A inside the curve's edge (down the right edge, around
    the corner, along the near edge to the left edge), from `start` after the far end to `stop_short` of the left
    edge."""
    r = CR - A
    legs = (CL - CR, 0.5*math.pi*r, CW - CR)
    out = []
    for s in np.arange(start, sum(legs) - stop_short + 1e-12, STITCH):
        if s <= legs[0]:
            out.append((CW - A, CL - s))
        elif s <= legs[0] + legs[1]:
            a = -(s - legs[0])/r
            out.append((CW - CR + r*math.cos(a), CR + r*math.sin(a)))
        else:
            out.append((CW - CR - (s - legs[0] - legs[1]), A))
    return np.array(out)


def curve_world(points):
    """The curve evaluation reads only the stitch ledger and the common state: no mesh, no cloth (a read of either
    would raise here)."""
    ledger = [dict(reverse=False, formed=True, holes=[(0, u, v), (1, u, v)]) for u, v in points]
    holes = [Hole(ply=p, uv=(float(u), float(v))) for s in ledger for p, u, v in s["holes"]]
    M = machine(ledger, [[2*i, 2*i + 1] for i in range(len(ledger))])
    M.holes = holes
    config = SimpleNamespace(task="curve", ply_width=CW, ply_length=CL, seam_allowance=A, corner_radius=CR,
                             stitch_length=STITCH)
    return FakeWorld(config, M)


# Curve ledgers written stitch by stitch: each stitch at arc length s along the line A inside the edge, measured from
# the far end of the right edge; `inward` moves both holes further from the edge; `s_bottom` puts the bottom ply's
# hole at another place along its own line (ply creep).
CURVE_PATH = (CL - CR) + 0.5*math.pi*(CR - A) + (CW - CR)       # the line's length, 441.4 mm at seed 0
CURVE_END = CURVE_PATH - 3.5e-3                                  # as curve_line: 3.5 mm short of the left edge


def curve_at(s, inward=0.0, R=CR):
    legs = (CL - R, 0.5*math.pi*(R - A), CW - R)
    r = R - A
    if s <= legs[0]:
        return (CW - A - inward, CL - s)
    if s <= legs[0] + legs[1]:
        a = -(s - legs[0])/r
        return (CW - R + (r - inward)*math.cos(a), R + (r - inward)*math.sin(a))
    return (CW - R - (s - legs[0] - legs[1]), A + inward)


def cst(s, reverse=False, inward=0.0, s_bottom=None, R=CR, formed=True, reason=None):
    """One needle cycle through both plies at arc length s; formed, or skipped for `reason` with its holes."""
    u1, v1 = curve_at(s, inward, R)
    u0, v0 = curve_at(s if s_bottom is None else s_bottom, inward, R)
    return dict(reverse=reverse, formed=formed, reason=reason, holes=[(0, u0, v0), (1, u1, v1)])


def cfwd(s0, s1, **kw):
    return [cst(s, **kw) for s in np.arange(s0, s1 + 1e-9, STITCH)]


def cycle_without_holes(reason):
    """A needle cycle that put no hole in the work: no fabric under the needle, or the needle bent or broken."""
    return dict(reverse=False, formed=False, reason=reason, holes=[])


def one_ply(s, ply, inward=0.0, R=CR):
    """A formed stitch through one ply only at arc length s (the other ply missed)."""
    u, v = curve_at(s, inward, R)
    return dict(reverse=False, formed=True, reason=None, holes=[(ply, u, v)])


def curve_ledger_world(ledger, R=CR, needle="ok", tether="trimmed", broken_cycles=(), reflex=0, pads=None):
    """The curve's world after the ledger: the needle's state, the tether, the thread elements between consecutive
    formed stitches (those whose stitch cycle is in broken_cycles broken), the reflex count while sewing, and the
    hands' pads as World.pads and the cloth's reactions on them (`pads`: (arm, force) per pad; default no pad
    records, the hands off the work as FakeWorld reports). A "trim" entry ends a seam piece: the stitches after it
    carry the next seam number, as the machine's interlocks do after a trim (machine.py _form_stitch)."""
    holes, stitches, seam = [], [], 0
    for spec in ledger:
        if spec == "trim":
            seam += 1
            continue
        idx = []
        for ply, u, v in spec["holes"]:
            holes.append(Hole(ply=ply, uv=(float(u), float(v))))
            idx.append(len(holes) - 1)
        stitches.append(dict(cycle=len(stitches), reverse=spec["reverse"], holes=idx, formed=spec["formed"],
                             reason=None if spec["formed"] else spec["reason"], seam=seam,
                             balance="balanced" if spec["formed"] else None))
    formed_cycles = [s["cycle"] for s in stitches if s["formed"]]
    threads = [dict(kind=k, cycle=c, broken=c in broken_cycles) for c in formed_cycles[1:] for k in ("needle", "bobbin")]
    M = SimpleNamespace(stitches=stitches, holes=holes, needle_state=needle, tether=tether, threads=threads,
                        tails=dict(start=P.TRIM_TAIL, end=P.TRIM_TAIL if tether == "trimmed" else None))
    config = SimpleNamespace(task="curve", ply_width=CW, ply_length=CL, seam_allowance=A, corner_radius=R,
                             stitch_length=STITCH)
    w = FakeWorld(config, M)
    if pads is not None:
        forces = [np.asarray(f, float) for _, f in pads]
        w.pads = [(i, None, None, None) for i in range(len(pads))]
        w.cloth = SimpleNamespace(shapes=[dict(name=f"{arm}_finger") for arm, _ in pads],
                                  shape_reaction=lambda i: (forces[i], np.zeros(3)))
    env = SimpleNamespace(world=w, config=config, history=dict(reflex_while_sewing=reflex))
    return Evaluator(env).evaluate()


PERFECT = cfwd(4e-3, CURVE_END)                     # the fixture's seam: forward from 4 mm to 3.5 mm short of the left
LAST = 4e-3 + STITCH*(len(PERFECT) - 1)


class CurveTest(EvaluatorTestCase):
    """The curve's seam is judged on the finished work: where its holes are in both plies, not the order, direction
    or kind (forward, reverse) of the stitches that made them, nor how the work was turned."""

    def test_fixture_follows_the_seam_line(self):
        d = _polyline_distance(curve_line(), seam_edge_path(CW, CL, CR, n_arc=96))[0]
        self.assertLess(np.abs(d - A).max(), 0.01e-3)
        self.assertAlmostEqual(CURVE_PATH, float(np.linalg.norm(np.diff(seam_edge_path(CW, CL, CR, A, n_arc=96),
                                                                         axis=0), axis=1).sum()), delta=0.01e-3)

    def test_curve_seam_passes(self):
        r = evaluate(curve_world(curve_line()))
        self.assertEqual(set(r["checks"]), CURVE_CHECKS)
        self.assertEqual(failed(r), set())
        self.assertTrue(r["success"])
        self.assertEqual(r["score"], 1.0)
        c = r["checks"]
        self.assertLess(c["seam_line"]["value"], 0.01)
        self.assertLessEqual(c["no_gaps"]["value"], 2.5)
        self.assertEqual(c["plies_registered"]["value"], 0.0)
        self.assertEqual(c["joined_far_end"]["value"], 4.0)
        self.assertAlmostEqual(c["reaches_left_edge"]["value"], curve_line()[-1][0]*1e3, places=2)    # 4.87 mm
        self.assertEqual(c["released"]["value"], {"left": 0.0, "right": 0.0})
        self.assertEqual(curve_ledger_world(PERFECT)["checks"], c)

    def test_curve_sewn_from_the_left_edge_passes(self):
        # the same holes sewn from the left edge to the far end: the same seam
        r = evaluate(curve_world(curve_line()[::-1]))
        self.assertEqual(failed(r), set())
        self.assertEqual(r["score"], 1.0)

    def test_reverse_first_backtack_start_passes(self):
        # the touch-back held three stitches from the start position toward the far end, then forward: every saved
        # demonstration's start. The reverse stitches join the plies from 5 mm (the forward ones restart 7.5 mm from
        # the far end)
        r = curve_ledger_world([cst(s, reverse=True) for s in (10e-3, 7.5e-3, 5e-3)] + cfwd(7.5e-3, CURVE_END))
        self.assertEqual(failed(r), set())
        self.assertEqual(r["checks"]["joined_far_end"]["value"], 5.0)

    def test_locks_and_a_restart_over_the_seam_pass(self):
        # stitches sewn forward over earlier ones: a forward stitch is not compared with the one sewn before it
        for name, ledger in (
                ("start lock: 4 forward, 3 reverse to 4 mm, forward over them",
                 cfwd(4e-3, 11.5e-3) + [cst(s, reverse=True) for s in (9e-3, 6.5e-3, 4e-3)] + cfwd(6.5e-3, CURVE_END)),
                ("end lock: 3 reverse, 3 forward back to the edge",
                 PERFECT + [cst(LAST - k*STITCH, reverse=True) for k in (1, 2, 3)]
                 + [cst(LAST - k*STITCH) for k in (2, 1, 0)]),
                ("end backtack in reverse only", PERFECT + [cst(LAST - k*STITCH, reverse=True) for k in (1, 2, 3)]),
                ("stopped at the middle, restarted three stitches back over the seam",
                 cfwd(4e-3, 220e-3) + cfwd(220e-3 - 3*STITCH, CURVE_END))):
            with self.subTest(case=name):
                r = curve_ledger_world(ledger)
                self.assertEqual(failed(r), set())
                self.assertEqual(r["score"], 1.0)

    def test_cycles_with_no_fabric_under_the_needle_are_not_skips(self):
        # the machine run two cycles past the left edge; a skipped stitch is a missed loop in
        # the seam, so a cycle with no fabric under the needle is not one
        r = curve_ledger_world(PERFECT + [cycle_without_holes("no_fabric")]*2)
        self.assertEqual(failed(r), set())
        self.assertEqual(r["checks"]["skips"]["value"], dict(isolated=0, longest_run=0))
        # with a bent needle the cycles after it are not skipped stitches either: needle_intact judges the needle
        r = curve_ledger_world(PERFECT + [cycle_without_holes("needle_bent")]*3, needle="bent")
        self.assertFailsOn(r, {"needle_intact"})

    def test_one_isolated_skip_is_within_the_tolerance_and_a_run_of_two_fails_skips(self):
        # the hook missed the loop: the needle's holes are in both plies, no interlock; the stitches either side are
        # 5 mm apart, one stitch setting plus the skipped one (FED-STD-751a: one isolated skip)
        i = len(PERFECT)//2
        one = PERFECT[:i] + [dict(PERFECT[i], formed=False, reason="missed_loop")] + PERFECT[i + 1:]
        r = curve_ledger_world(one)
        self.assertEqual(failed(r), set())
        self.assertEqual(r["checks"]["no_gaps"]["value"], 2.5)
        two = one[:i + 1] + [dict(PERFECT[i + 1], formed=False, reason="missed_loop")] + PERFECT[i + 2:]
        # also stitch_density, true of this seam: an inch over the run holds 8 interlocks (as the plain seam's test)
        self.assertFailsOn(curve_ledger_world(two), {"skips"}, also={"stitch_density"})

    def test_an_unsewn_stretch_fails_no_gaps(self):
        # three stitches missing on the arc: 10 mm between neighbours, no skipped stitch there (the work moved on)
        i = int((CL - CR + 50e-3 - 4e-3)/STITCH)
        r = curve_ledger_world(PERFECT[:i] + PERFECT[i + 3:])
        # also stitch_density (sequences have no step window), true of this seam: the inch over the 10 mm
        # unsewn stretch holds 7 stitches
        self.assertFailsOn(r, {"no_gaps"}, also={"stitch_density"})
        self.assertAlmostEqual(r["checks"]["no_gaps"]["value"], 10.0, places=2)

    def test_seam_started_or_ended_short_fails_its_end(self):
        # forward only from the start position, 9.9 mm from the far end: a real unsewn end
        self.assertFailsOn(curve_ledger_world(cfwd(10e-3, CURVE_END)), {"joined_far_end"})
        self.assertFailsOn(curve_ledger_world(cfwd(4e-3, CURVE_PATH - 10e-3)), {"reaches_left_edge"})

    def test_plies_joined_apart_fail_plies_registered(self):
        # each stitch pierces the bottom ply further along its own line than the top (ply creep); every hole on its
        # own ply's line
        top = np.arange(CURVE_PATH - 5.9e-3, 0.4e-3, -STITCH)[::-1]
        for shift, bad in ((2.0e-3, set()), (2.9e-3, set()), (5.0e-3, {"plies_registered"}),
                           (5.5e-3, {"plies_registered"})):
            with self.subTest(shift_mm=shift*1e3):
                r = curve_ledger_world([cst(s, s_bottom=s + shift) for s in top])
                self.assertEqual(failed(r), bad)
                self.assertAlmostEqual(r["checks"]["plies_registered"]["value"], shift*1e3, places=2)

    def test_each_ply_must_reach_the_ends(self):
        # the plies 2.5 mm apart (registered) and the top ply's seam from 4 mm: the bottom ply's starts 6.5 mm from its
        # far end
        r = curve_ledger_world([cst(s, s_bottom=s + 2.5e-3) for s in np.arange(4e-3, CURVE_END + 1e-9, STITCH)])
        self.assertFailsOn(r, {"joined_far_end"})
        self.assertEqual(r["checks"]["joined_far_end"]["value"], 6.5)

    def test_a_hand_squeezing_the_work_has_not_let_go(self):
        # a Franka Hand closed on slack work, its two fingers pressing the cloth from opposite sides. Their vector sum
        # (what finger_force reports and the net-force rule read) nearly cancels; the curve adds each pad's contact as
        # a magnitude
        for label, (a, b), let_go in (("position mode", (1.152, 1.145), False), ("2 N force mode", (2.011, 1.989), False),
                                      ("fingers open", (0.0, 0.0), True)):
            with self.subTest(grip=label):
                pads = [("left", (a, 0.0, 0.0)), ("left", (-b, 0.0, 0.0)), ("left", (0.0, 0.0, 0.0)),
                        ("right", (0.0, 0.0, 0.0))]
                self.assertLess(abs(a - b), 0.05)                       # the net force passes the 0.05 N gate
                r = curve_ledger_world(PERFECT, pads=pads)
                self.assertEqual(r["checks"]["released"]["ok"], let_go)
                self.assertAlmostEqual(r["checks"]["released"]["value"]["left"], a + b, places=3)
                self.assertEqual(r["success"], let_go)
        # a hand resting on the work (one pad pressing down 0.5 N) fails both rules
        r = curve_ledger_world(PERFECT, pads=[("left", (0.0, 0.0, -0.5)), ("right", (0.0, 0.0, 0.0))])
        self.assertFailsOn(r, {"released"})

    def test_reverse_dominated_ledgers_get_a_verdict(self):
        # only reverse stitches, or one forward stitch then reverse ones: a verdict in every state of the episode
        for ledger in ([cst(s, reverse=True) for s in np.arange(20e-3, 7e-3, -STITCH)],
                       [cst(20e-3)] + [cst(s, reverse=True) for s in np.arange(17.5e-3, 7e-3, -STITCH)]):
            r = curve_ledger_world(ledger)
            self.assertFalse(r["success"])
            self.assertIn("reaches_left_edge", failed(r))
        self.assertEqual(curve_ledger_world([])["score"], 0.0)

    def test_pivots_pass_and_a_square_corner_fails(self):
        # the arc as 12.5 mm chords (stop needle down, turn, five stitches): the method is not judged. Straight down
        # to (10, 10) mm and a square turn cuts outside the rounded corner
        l0, l1 = CL - CR, 0.5*math.pi*(CR - A)
        pivot5 = []
        for s in np.arange(4e-3, CURVE_END + 1e-9, STITCH):
            if l0 < s < l0 + l1:
                k = math.floor((s - l0)/(5*STITCH))
                sa, sb = l0 + k*5*STITCH, min(l0 + (k + 1)*5*STITCH, l0 + l1)
                u, v = np.array(curve_at(sa)) + (s - sa)/(sb - sa)*(np.array(curve_at(sb)) - np.array(curve_at(sa)))
                pivot5.append(dict(reverse=False, formed=True, reason=None, holes=[(0, u, v), (1, u, v)]))
            else:
                pivot5.append(cst(s))
        r = curve_ledger_world(pivot5)
        self.assertEqual(failed(r), set())
        self.assertLess(r["checks"]["seam_line"]["value"], 0.25)
        square = [dict(reverse=False, formed=True, reason=None, holes=[(0, CW - A, v), (1, CW - A, v)])
                  for v in np.arange(CL - 4e-3, A - 1e-9, -STITCH)]
        square += [dict(reverse=False, formed=True, reason=None, holes=[(0, u, A), (1, u, A)])
                   for u in np.arange(CW - A - STITCH, 3.5e-3, -STITCH)]
        self.assertIn("seam_line", failed(curve_ledger_world(square)))

    def test_the_episodes_radius_is_judged(self):
        from sewing.variants import resolve
        R3 = resolve("sample-room", "curve", 3)["corner_radius"]
        path3 = (CL - R3) + 0.5*math.pi*(R3 - A) + (CW - R3)
        r = curve_ledger_world([cst(s, R=R3) for s in np.arange(4e-3, path3 - 3.5e-3 + 1e-9, STITCH)], R=R3)
        self.assertEqual(failed(r), set())
        self.assertIn("seam_line", failed(curve_ledger_world(PERFECT, R=R3)))


def curve_with_step(step):
    """A complete seam on the line whose stitches are `step` apart (the work held back against the feed or pushed)."""
    return [cst(s) for s in np.arange(4e-3, CURVE_PATH - 1e-3, step)]


class CurveMinimalPairTest(EvaluatorTestCase):
    """Minimal pairs with the perfect seam: the authored mean allowance is not judged; the stitches an inch along
    continuous sewing are; thread strung across the work between consecutive stitches of one seam piece is; a formed
    stitch through one ply is judged by seam_line."""

    def test_a_seam_inside_the_tolerance_everywhere_passes_whatever_its_mean(self):
        for inward in (0.85e-3, -0.85e-3, 1.2e-3, -1.5e-3):
            with self.subTest(inward_mm=inward*1e3):
                r = curve_ledger_world(cfwd(4e-3, CURVE_END, inward=inward))
                self.assertEqual(failed(r), set())
                self.assertEqual(r["score"], 1.0)
                self.assertAlmostEqual(r["info"]["seam_mean"]["value"], 10.0 + inward*1e3, places=1)

    def test_stitches_an_inch(self):
        # the plain seam's band about the set 10 SPI: 9-11, 12 allowed (Army 11 +- 1); 1.0 / 1.25 mm and 3.4 mm fail
        self.assertEqual(curve_ledger_world(PERFECT)["checks"]["stitch_density"]["value"], [10, 11])
        for step, bad in ((1.0e-3, True), (1.25e-3, True), (2.5e-3, False), (3.4e-3, True)):
            with self.subTest(step_mm=step*1e3):
                r = curve_ledger_world(curve_with_step(step))
                if bad:
                    self.assertFailsOn(r, {"stitch_density"})
                else:
                    self.assertEqual(failed(r), set())

    def test_overlaps_and_piles_are_not_density(self):
        # a restart over the seam whose holes fall between the old ones, with or without a trim, and 30 stitches
        # piled in one place (a bar tack): each run of continuous sewing is counted on its own
        first = cfwd(4e-3, 219e-3)
        for name, ledger in (("restart offset, trimmed", first + ["trim"] + cfwd(212.75e-3, CURVE_END)),
                             ("restart offset, no trim", first + cfwd(212.75e-3, CURVE_END)),
                             ("30 stitches in one place", PERFECT[:101] + [PERFECT[100]]*30 + PERFECT[101:])):
            with self.subTest(case=name):
                self.assertEqual(failed(curve_ledger_world(ledger)), set())

    def test_thread_strung_across_the_work_fails_no_long_threads(self):
        # sewn out of order without a trim leaves 100 and 231 mm of thread across the work, which
        # the trimmer cannot cut; the same with a trim before each move, a restart three stitches back over the seam
        # without a trim (7.5 mm over the sewn seam) and a trimmed piece butted 3.4 mm on pass
        j1 = [cfwd(4e-3, 149e-3), cfwd(251.5e-3, CURVE_END), cfwd(151.5e-3, 249e-3)]
        r = curve_ledger_world(j1[0] + j1[1] + j1[2])
        self.assertFailsOn(r, {"no_long_threads"})
        self.assertGreater(r["checks"]["no_long_threads"]["value"], 200.0)
        self.assertAlmostEqual(r["score"], 0.95*(1 - 2/((CURVE_PATH - 12e-3)/STITCH)), places=4)
        first = cfwd(4e-3, 219e-3)
        self.assertEqual(failed(curve_ledger_world(first + cfwd(211.5e-3, CURVE_END))), set())   # restart, no trim
        # trimmed before each move: no thread across the work, but the pieces butt with their ends unsecured
        self.assertFailsOn(curve_ledger_world(j1[0] + ["trim"] + j1[1] + ["trim"] + j1[2]), {"joins_secured"})

    def test_a_stitch_through_one_ply_is_judged_by_seam_line(self):
        i = len(PERFECT)//2
        s_i = 4e-3 + STITCH*i
        # 25 mm inside the line through the top ply (its thread to the seam is also 25 mm), 2 mm from the edge through
        # the bottom ply: both fail
        r = curve_ledger_world(PERFECT[:i] + [one_ply(s_i, 1, inward=25e-3)] + PERFECT[i:])
        self.assertFailsOn(r, {"seam_line"}, also={"no_long_threads"})
        self.assertFailsOn(curve_ledger_world(PERFECT[:i] + [one_ply(s_i, 0, inward=-8e-3)] + PERFECT[i:]),
                           {"seam_line"})
        # on the line, at the end of the seam (the plies' edges a little apart): passes
        self.assertEqual(failed(curve_ledger_world(PERFECT + [one_ply(LAST + STITCH, 1)])), set())


class CurveStitchLengthTest(EvaluatorTestCase):
    """Stitch length is measured on places in sequences with no window, so crowded and long stitches cannot escape
    it; a skipped stitch's place counts for density; each end of a trimmed piece inside the seam is secured; stitches
    the best nowhere count as rework in proportion."""

    def test_crowded_stitches_fail_however_crowded(self):
        for step in (0.3e-3, 0.45e-3, 0.49e-3, 0.6e-3):
            with self.subTest(step_mm=step*1e3):
                r = curve_ledger_world(curve_with_step(step))
                self.assertFailsOn(r, {"stitch_density"})
                self.assertEqual(r["score"], 0.0)
        # a 20 mm stretch of the right edge at 0.3 mm a stitch, the rest at 2.5 mm
        i, j = int((99e-3 - 4e-3)/STITCH), int((119e-3 - 4e-3)/STITCH)
        creep = [cst(s) for s in np.arange(4e-3 + STITCH*i, 4e-3 + STITCH*j, 0.3e-3)]
        self.assertFailsOn(curve_ledger_world(PERFECT[:i] + creep + PERFECT[j:]), {"stitch_density"})
        # piles, locks and a creep of a few stitches are places, not density
        self.assertEqual(failed(curve_ledger_world(PERFECT[:101] + [PERFECT[100]]*30 + PERFECT[101:])), set())

    def test_longer_stitches_never_score_more(self):
        base = self.score_of(curve_with_step(3.4e-3))
        for step in (3.51e-3, 3.6e-3, 4e-3, 5e-3):
            with self.subTest(step_mm=step*1e3):
                self.assertLessEqual(self.score_of(curve_with_step(step)), base)

    def score_of(self, ledger):
        return curve_ledger_world(ledger)["score"]

    def test_one_skip_at_a_long_realised_stitch_is_within_the_tolerance(self):
        for step in (2.55e-3, 2.7e-3):
            led = curve_with_step(step)
            k = len(led)//2
            led[k] = dict(led[k], formed=False, reason="missed_loop")
            with self.subTest(step_mm=step*1e3):
                self.assertEqual(failed(curve_ledger_world(led)), set())

    def test_ends_of_trimmed_pieces_inside_the_seam_are_secured(self):
        first = cfwd(4e-3, 219e-3)
        end_lock = [cst(219e-3 - k*STITCH, reverse=True) for k in (1, 2, 3)]
        start_lock = (cfwd(221.5e-3, 221.5e-3 + 2*STITCH) + [cst(221.5e-3 + k*STITCH, reverse=True) for k in (1, 0)]
                      + cfwd(221.5e-3 + STITCH, CURVE_END))
        for name, ledger, ok in (
                ("butted 3.4 mm on", first + ["trim"] + cfwd(222.4e-3, CURVE_END), False),
                ("only the first piece's end backtacked", first + end_lock + ["trim"] + cfwd(221.5e-3, CURVE_END), False),
                ("both ends backtacked", first + end_lock + ["trim"] + start_lock, True),
                ("the next piece overlapping 6.25 mm", first + ["trim"] + cfwd(212.75e-3, CURVE_END), True),
                ("the next piece overlapping 1/2 in", first + ["trim"] + cfwd(206.3e-3, CURVE_END), True),
                ("one continuous seam", PERFECT, True)):
            with self.subTest(case=name):
                r = curve_ledger_world(ledger)
                if ok:
                    self.assertEqual(failed(r), set())
                else:
                    self.assertFailsOn(r, {"joins_secured"})

    def test_rework_is_continuous_in_how_far_off_it_was(self):
        # the correct seam sewn over a seam sewn d off the line: the score falls smoothly with d
        scores = [self.score_of(cfwd(4e-3, CURVE_END, inward=d) + [dict(x, reverse=True) for x in PERFECT[::-1]])
                  for d in (0.0, 1.7e-3, 2.5e-3, 3.1e-3, 3.3e-3)]
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[1:], sorted(scores[1:], reverse=True))
        self.assertLess(scores[3] - scores[4], 0.07)


class CurveScoreTest(unittest.TestCase):
    """The curve's score: 1.0 only on success; short of it, continuous in the share of the
    seam sewn correctly, finishing actions in proportion to it, damage scaling it down; nothing for the order,
    direction or method."""

    def score(self, ledger, **kw):
        return curve_ledger_world(ledger, **kw)["score"]

    def test_nothing_sewn_scores_nothing_and_six_stitches_are_their_share(self):
        self.assertEqual(self.score([]), 0.0)
        self.assertEqual(self.score([], tether="attached"), 0.0)
        six = self.score(cfwd(4e-3, 4e-3 + 5*STITCH))
        self.assertGreater(six, 0.0)
        self.assertLess(six, 15e-3/CURVE_PATH + 0.01)

    def test_more_seam_scores_more(self):
        scores = []
        for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 0.97):
            led = cfwd(4e-3, 4e-3 + frac*(CURVE_END - 4e-3))
            s = self.score(led, tether="attached")
            self.assertAlmostEqual(s, 0.95*0.90*frac, delta=0.02)
            scores.append(s)
        self.assertEqual(scores, sorted(scores))
        self.assertLess(scores[-1], self.score(PERFECT, tether="attached"))

    def test_finishing_counts_once_and_in_proportion(self):
        self.assertAlmostEqual(self.score(PERFECT, tether="attached"), 0.95*0.90, places=3)       # not trimmed
        held = [("left", (1.15, 0.0, 0.0)), ("left", (-1.15, 0.0, 0.0)), ("right", (0.0, 0.0, 0.0))]
        self.assertAlmostEqual(self.score(PERFECT, pads=held), 0.95*0.95, places=3)               # trimmed, held
        self.assertEqual(self.score(PERFECT), 1.0)
        half = cfwd(4e-3, 220e-3)
        attached, trimmed = self.score(half, tether="attached"), self.score(half)
        coverage = curve_ledger_world(half)["info"]["partial_score"]["coverage"]
        self.assertAlmostEqual(trimmed - attached, 0.1*0.95*coverage, places=4)     # a trim mid-seam: in proportion
        # a hand on the work while sewing (steering by contact) costs nothing before the trim: letting go counts once
        # the threads are trimmed
        self.assertEqual(self.score(half, tether="attached", pads=held), attached)

    def test_further_off_the_line_never_scores_more(self):
        scores = [self.score(cfwd(4e-3, CURVE_END, inward=d)) for d in (0.0, 1.5e-3, 1.7e-3, 2.0e-3, 2.5e-3, 3.3e-3,
                                                                          5e-3, 20e-3)]
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(scores[1], 1.0)        # every hole within 1/16 in, the mean 11.5 mm: success
        self.assertEqual(scores[5:], [0.0, 0.0, 0.0])
        # stitches off the line elsewhere on the work lower the score; a complete seam with ten stray stitches still
        # ranks above a seam sewn 20 mm off the line
        stray = PERFECT[:len(PERFECT)//2] + cfwd(100e-3, 100e-3 + 9*STITCH, inward=25e-3)
        self.assertLess(self.score(stray), self.score(PERFECT[:len(PERFECT)//2]))
        self.assertGreater(self.score(PERFECT + cfwd(100e-3, 100e-3 + 9*STITCH, inward=25e-3)), scores[-1])

    def test_plies_further_apart_never_score_more(self):
        top = np.arange(CURVE_PATH - 5.9e-3, 0.4e-3, -STITCH)[::-1]
        scores = [self.score([cst(s, s_bottom=s + d) for s in top]) for d in (0.0, 2.9e-3, 3.1e-3, 4.5e-3, 6.1e-3)]
        self.assertEqual(scores[:2], [1.0, 1.0])
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[2], 0.9)                      # just past the tolerance: no cliff
        self.assertLess(scores[-1], 0.05)

    def test_damage_scales_the_seam_down_without_erasing_it(self):
        half = cfwd(4e-3, 220e-3)
        clean = self.score(half)
        damaged = self.score(half, needle="damaged")
        bent = self.score(half + [cycle_without_holes("needle_bent")]*319, needle="bent")
        self.assertLess(damaged, clean)
        self.assertLess(bent, damaged)
        self.assertGreater(bent, 0.0)
        self.assertEqual(bent, self.score(half + [cycle_without_holes("needle_bent")], needle="bent"))
        self.assertLess(self.score(PERFECT, broken_cycles=(100,)), 0.95*0.75 + 1e-9)
        # a recovered protective stop fails success (no_reflex_while_sewing) and costs the work nothing
        self.assertAlmostEqual(self.score(PERFECT, reflex=1), 0.95, places=6)

    def test_sewing_over_the_seam_again_never_changes_the_score(self):
        # stray stitches are rework counted against the stitches the job takes, so passes over places already sewn
        # neither add credit nor dilute the strays
        def again(ledger):
            return [dict(x, reverse=True) for x in ledger[::-1]]
        half = cfwd(4e-3, 219e-3)
        strays = cfwd(100e-3, 100e-3 + 9*STITCH, inward=25e-3)
        r1 = self.score(half + strays)
        self.assertEqual(self.score(half + again(half) + strays), r1)
        self.assertEqual(self.score(half + again(half) + half + again(half) + strays), r1)
        self.assertEqual(self.score(PERFECT + again(PERFECT) + strays), self.score(PERFECT + strays))
        self.assertLess(r1, self.score(half))
        # formed one-ply strays count too
        one = [one_ply(100e-3 + k*STITCH, 1, inward=25e-3) for k in range(10)]
        self.assertLess(self.score(PERFECT + one), 0.95)
        self.assertGreater(self.score(PERFECT + one), 0.85)

    def test_short_and_long_stitches_score_less(self):
        crowded, long_ = self.score(curve_with_step(1.25e-3)), self.score(curve_with_step(3.4e-3))
        self.assertEqual(crowded, 0.0)
        self.assertGreater(long_, 0.0)
        self.assertLess(long_, 0.95*0.2)
        self.assertGreater(self.score(curve_with_step(2.0e-3)), long_)     # 12.7 an inch: just outside the band

    def test_seeds_score_their_own_radius(self):
        from sewing.variants import resolve
        R3 = resolve("sample-room", "curve", 3)["corner_radius"]
        path3 = (CL - R3) + 0.5*math.pi*(R3 - A) + (CW - R3)
        half3 = [cst(s, R=R3) for s in np.arange(4e-3, 0.5*path3, STITCH)]
        self.assertAlmostEqual(self.score(half3, R=R3, tether="attached"), 0.95*0.90*(0.5*path3 - 6e-3)/(path3 - 12e-3),
                               delta=0.01)



# ---------------------------------------------------------------------------------------------------- load
def load_world(top_shift=(0.0, 0.0), hang_apart=3.6e-3):
    """Two plies stacked on the table up to its front edge, their near 200 mm hanging straight down over it (the load
    task's end state). The top ply's on-table part is shifted by top_shift (m, plan); its hanging part hangs
    hang_apart further out than the bottom ply's, as two draped plies do."""
    mesh = ply_mesh(W, L, A)
    X = mesh["X"]
    y_edge = TABLE["y"][0]
    hang = 0.200
    pos = []
    for k, (dz, dx, dy, out) in enumerate(((0.3e-3, 0.0, 0.0, 0.0),
                                           (0.9e-3, top_shift[0], top_shift[1], hang_apart))):
        x = np.zeros((len(X), 3))
        on = X[:, 1] >= hang
        x[:, 0] = X[:, 0] - (W - A) + (dx if k else 0.0)
        x[on, 1] = y_edge + (X[on, 1] - hang) + (dy if k else 0.0)
        x[on, 2] = dz
        x[~on, 1] = y_edge - out
        x[~on, 2] = -(hang - X[~on, 1])
        pos.append(x)
    cloth = FakeCloth(X, mesh["tri"], pos)
    config = SimpleNamespace(task="load", ply_width=W, ply_length=L, seam_allowance=A, stitch_length=STITCH)
    return FakeWorld(config, SimpleNamespace(stitches=[], holes=[]), mesh=mesh, cloth=cloth)


class LoadRegistrationTest(unittest.TestCase):
    """The plies' registration is judged in material coordinates where they lie stacked, not by where draped
    corners hang."""

    def registration(self, world):
        return Evaluator(env_of(world)).ply_registration()

    def test_registered_plies_pass_although_their_hanging_corners_lie_apart(self):
        off, n = self.registration(load_world())
        self.assertGreater(n, 100)
        self.assertLess(off, 1e-6)

    def test_top_ply_shifted_along_the_seam_is_misregistered(self):
        off, n = self.registration(load_world(top_shift=(0.0, 3e-3)))
        self.assertAlmostEqual(off, 3e-3, delta=0.1e-3)

    def test_top_ply_shifted_across_the_seam_is_misregistered(self):
        off, n = self.registration(load_world(top_shift=(2e-3, 0.0)))
        self.assertAlmostEqual(off, 2e-3, delta=0.1e-3)


# ------------------------------------------------------------------ the load evaluation's searches, shortened
# The load task's evaluation matches every edge vertex of each ply against the resting triangles of the other, and
# finds each ply's material point under the needle, with shortened searches. The functions below are the plain
# searches (every edge vertex against every resting triangle, and a Python loop over every triangle under the
# needle), the reference the shortened ones are held to.
def _every_triangle(q, p0, p1, p2):
    """The plain search: for each point the triangle with the largest smallest barycentric coordinate among
    all of them (the first of equals), and the point's coordinates in it."""
    v0, v1 = p1 - p0, p2 - p0
    den = v0[:, 0]*v1[:, 1] - v1[:, 0]*v0[:, 1]
    den = np.where(np.abs(den) < 1e-18, 1e-18, den)
    T, B = [], []
    for s0 in range(0, len(q), 64):
        d = q[s0:s0 + 64][:, None, :] - p0[None]
        l1 = (d[..., 0]*v1[:, 1] - v1[:, 0]*d[..., 1])/den
        l2 = (v0[:, 0]*d[..., 1] - d[..., 0]*v0[:, 1])/den
        l0 = 1.0 - l1 - l2
        t = np.argmax(np.minimum(np.minimum(l0, l1), l2), axis=1)
        i = np.arange(len(t))
        T.append(t)
        B.append(np.column_stack([l0[i, t], l1[i, t], l2[i, t]]))
    return np.concatenate(T), np.vstack(B)


def _registration_inputs(world, band=3e-3, support=2e-3):
    """What ply_registration searches, for each ply against the other: the resting edge vertices' indices, the other
    ply's resting triangles and its plan positions."""
    X = np.asarray(world.mesh["X"], float)
    tri = np.asarray(world.mesh["tri"], int)
    Wd, Ln = world.config.ply_width, world.config.ply_length
    xs = [np.asarray(world.ply_positions(k), float) for k in (0, 1)]

    def resting(x):
        return ((x[:, 2] >= -1e-3) & (x[:, 2] <= support) & (x[:, 0] > TABLE["x"][0]) & (x[:, 0] < TABLE["x"][1])
                & (x[:, 1] > TABLE["y"][0]) & (x[:, 1] < TABLE["y"][1]))
    edge = np.minimum(np.minimum(X[:, 0], Wd - X[:, 0]), np.minimum(X[:, 1], Ln - X[:, 1])) <= band
    for a, b in ((0, 1), (1, 0)):
        q_idx = np.nonzero(edge & resting(xs[a]))[0]
        ok_b = resting(xs[b])
        yield X, q_idx, xs[a][q_idx, :2], tri[ok_b[tri].all(axis=1)], xs[b][:, :2]


def _registration_reference(world):
    """Evaluator.ply_registration by the plain search."""
    worst, n = 0.0, 0
    for X, q_idx, q, tb, P in _registration_inputs(world):
        if not len(q_idx) or not len(tb):
            continue
        t, bary = _every_triangle(q, P[tb[:, 0]], P[tb[:, 1]], P[tb[:, 2]])
        Xb = bary[:, 0, None]*X[tb[t, 0]] + bary[:, 1, None]*X[tb[t, 1]] + bary[:, 2, None]*X[tb[t, 2]]
        worst = max(worst, float(np.linalg.norm(Xb - X[q_idx], axis=1).max()))
        n += len(q_idx)
    return worst, n


def _under_needle_reference(world, k):
    """Evaluator.under_needle as a loop over the ply's triangles."""
    cl = world.cloth
    p = cl.plies[world.ply_ids[k]]
    x = cl.positions()
    T = cl.tri_host[p["t0"]:p["t0"] + p["nt"]]
    X = cl.X
    hits = []
    for a, b, c, t in zip(x[T[:, 0]], x[T[:, 1]], x[T[:, 2]], T):
        v0, v1 = b[:2] - a[:2], c[:2] - a[:2]
        den = v0[0]*v1[1] - v1[0]*v0[1]
        if abs(den) < 1e-18:
            continue
        q = -a[:2]
        l1 = (q[0]*v1[1] - v1[0]*q[1])/den
        l2 = (v0[0]*q[1] - q[0]*v0[1])/den
        l0 = 1 - l1 - l2
        if min(l0, l1, l2) >= -1e-9:
            hits.append(l0*X[t[0]] + l1*X[t[1]] + l2*X[t[2]])
    return hits


def stacked_world(top, bottom=None):
    """Two plies of the load task's mesh at the given world positions (default bottom: flat on the table)."""
    mesh = ply_mesh(W, L, A)
    X = mesh["X"]
    cloth = FakeCloth(X, mesh["tri"], [flat(X, 0) if bottom is None else bottom, top])
    config = SimpleNamespace(task="load", ply_width=W, ply_length=L, seam_allowance=A, stitch_length=STITCH)
    return FakeWorld(config, SimpleNamespace(stitches=[], holes=[]), mesh=mesh, cloth=cloth)


def _moved(x, shift=(0.0, 0.0), degrees=0.0):
    th = math.radians(degrees)
    c = x[:, :2].mean(axis=0)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    y = x.copy()
    y[:, :2] = (x[:, :2] - c) @ R.T + c + np.asarray(shift)
    return y


def random_stack(seed):
    """A randomised stacked-ply state: the top ply shifted (0.03 to 30 mm) and turned on the bottom one, its vertices
    jittered, and on some seeds a patch of either ply lifted off the table and the stack pushed over the table's
    front edge."""
    rng = np.random.default_rng(seed)
    X = ply_mesh(W, L, A)["X"]
    bottom = flat(X, 0)
    top = _moved(flat(X, 1), shift=rng.uniform(-1, 1, 2)*10.0**rng.uniform(-4.5, -1.5), degrees=rng.normal(0.0, 1.5))
    top[:, :2] += rng.normal(0.0, 10.0**rng.uniform(-5.0, -3.5), (len(X), 2))
    for x in (bottom, top):
        if rng.random() < 0.5:
            x[np.linalg.norm(x[:, :2] - x[rng.integers(len(x)), :2], axis=1) < rng.uniform(0.01, 0.06), 2] += 0.01
    if rng.random() < 0.5:
        d = rng.uniform(0.0, 0.25)
        bottom[:, 1] -= d
        top[:, 1] -= d
    return stacked_world(top, bottom)


class LoadSearchEqualityTest(unittest.TestCase):
    """The shortened searches return what the plain ones do: the same triangle for every edge point (the same
    index, so the same barycentric coordinates to the last bit), the same misregistration to 1e-9, the same count."""

    def assertSameRegistration(self, world, name):
        for X, q_idx, q, tb, P in _registration_inputs(world):
            if not len(q_idx) or not len(tb):
                continue
            p = P[tb[:, 0]], P[tb[:, 1]], P[tb[:, 2]]
            t_ref, bary_ref = _every_triangle(q, *p)
            t, l0, l1, l2 = _triangle_under(q, *p)
            np.testing.assert_array_equal(t, t_ref, err_msg=f"{name}: another triangle")
            np.testing.assert_array_equal(np.column_stack([l0, l1, l2]), bary_ref, err_msg=name)
        off_ref, n_ref = _registration_reference(world)
        off, n = Evaluator(env_of(world)).ply_registration()
        self.assertEqual(n, n_ref, name)
        self.assertAlmostEqual(off, off_ref, delta=1e-9, msg=name)
        return off, n

    def test_the_evaluators_fixtures(self):
        for name, world in (("registered, hanging corners apart", load_world()),
                            ("shifted along the seam", load_world(top_shift=(0.0, 3e-3))),
                            ("shifted across the seam", load_world(top_shift=(2e-3, 0.0))),
                            ("shifted both ways, hanging together", load_world(top_shift=(-1e-3, 0.4e-3),
                                                                             hang_apart=0.0))):
            with self.subTest(fixture=name):
                off, n = self.assertSameRegistration(world, name)
                self.assertGreater(n, 100)

    def test_randomised_stacked_plies(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                off, n = self.assertSameRegistration(random_stack(seed), f"seed {seed}")
                self.assertGreater(n, 0)

    def test_plies_apart_folded_and_edge_on(self):
        X = ply_mesh(W, L, A)["X"]
        top = flat(X, 1)
        apart = top.copy()
        apart[:, 0] += 0.26                      # beside the bottom ply: every edge point takes its nearest triangle
        fold = top.copy()                        # the seam edge folded back 10 mm onto the ply
        back = X[:, 0] > W - 0.010
        fold[back, 0] = 2*fold[back, 0].min() - fold[back, 0]
        fold[back, 2] += 2*THICK
        thin = top.copy()                        # a strip of triangles squeezed edge-on in plan
        strip = (X[:, 1] > 0.200) & (X[:, 1] < 0.210)
        thin[strip, 1] = thin[strip, 1].min() + (thin[strip, 1] - thin[strip, 1].min())*1e-7
        for name, x in (("apart", apart), ("folded", fold), ("edge-on strip", thin)):
            with self.subTest(case=name):
                self.assertSameRegistration(stacked_world(x), name)

    def test_the_search_in_small_pieces(self):
        # the pairs of a search are taken a bounded number at a time: with a tiny bound every boundary is crossed
        from unittest.mock import patch
        X = ply_mesh(W, L, A)["X"]
        with patch("sewing.task._PAIRS_AT_ONCE", 700):
            self.assertSameRegistration(stacked_world(_moved(flat(X, 1), shift=(4e-3, -2.5e-3), degrees=1.0)),
                                        "shifted and turned, 700 pairs at a time")

    def test_few_triangles_and_none(self):
        # a handful of triangles (the short path) and no resting triangle at all
        rng = np.random.default_rng(3)
        p = rng.uniform(0.0, 0.05, (40, 3, 2))
        q = rng.uniform(-0.02, 0.07, (300, 2))
        t_ref, bary_ref = _every_triangle(q, p[:, 0], p[:, 1], p[:, 2])
        t, l0, l1, l2 = _triangle_under(q, p[:, 0], p[:, 1], p[:, 2])
        np.testing.assert_array_equal(t, t_ref)
        np.testing.assert_array_equal(np.column_stack([l0, l1, l2]), bary_ref)
        # a coordinate that is not a number (never a resting vertex's; the search must still end as the plain one does)
        big = rng.uniform(0.0, 0.05, (700, 3, 2))
        q[5] = np.nan
        with np.errstate(invalid="ignore"):
            t_ref, bary_ref = _every_triangle(q, big[:, 0], big[:, 1], big[:, 2])
            t, l0, l1, l2 = _triangle_under(q, big[:, 0], big[:, 1], big[:, 2])
        np.testing.assert_array_equal(t, t_ref)
        np.testing.assert_array_equal(np.column_stack([l0, l1, l2]), bary_ref)
        X = ply_mesh(W, L, A)["X"]
        lifted = flat(X, 1)
        lifted[:, 2] += 0.05
        world = stacked_world(lifted)
        self.assertEqual(Evaluator(env_of(world)).ply_registration(), _registration_reference(world))
        self.assertEqual(Evaluator(env_of(world)).ply_registration(), (0.0, 0))

    def test_under_the_needle(self):
        X = ply_mesh(W, L, A)["X"]
        zfold = flat(X, 1)                       # the top ply crosses the needle axis three times (a Z fold)
        u = X[:, 0]
        zfold[:, 0] = np.interp(u, [0.0, W - 0.030, W - 0.020, W - 0.010, W], [-0.150, 0.004, -0.004, 0.004, 0.012])
        zfold[:, 2] += np.interp(u, [0.0, W - 0.030, W - 0.020, W - 0.010, W], [0.0, 0.0, 1e-3, 2e-3, 2e-3])
        away = flat(X, 1)
        away[:, 0] -= 0.25                       # the needle axis misses the ply
        for name, top, crossings in (("flat", flat(X, 1), 1), ("Z fold", zfold, 3), ("beside the needle", away, 0)):
            with self.subTest(case=name):
                world = stacked_world(top)
                ev = Evaluator(env_of(world))
                for k in (0, 1):
                    ref = _under_needle_reference(world, k)
                    new = ev.under_needle(k)
                    self.assertEqual(len(new), len(ref))
                    for a, b in zip(new, ref):
                        np.testing.assert_array_equal(a, b)
                self.assertEqual(len(ev.under_needle(1)), crossings)
                self.assertEqual(len(ev.under_needle(0)), 1)


if __name__ == "__main__":
    unittest.main()
