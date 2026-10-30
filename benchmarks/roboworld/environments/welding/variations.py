"""Frozen catalogue of variations of the robotic GMAW cell (welding-steel-cell-v1): 10,000 initial conditions of the
steel-cell preset, each with a hash-derived train / validation / test split. A variation is a cell state, not a task:
every variation serves every task (welding/task.py TASKS: eleven), and all share its split.

What varies is what changes from job to job in a real cell and what the model takes without any change to its physics,
procedure, dimensions or masses (PARAMETERS gives each basis):
  coupon pose   where the fitter clamped the coupon on the fixture table: dx, dy within +-3 mm, yaw within +-2 deg
                [authored: the play of a part placed by hand under the toggle clamps; the same range as
                the legacy seeds 1-9], uniform, redrawn where a clamp pad would sit within 2 mm of the plate's edge
                (fixture_check; 45 of 9,999 first draws, all with dy > 2.3 mm and yaw > 1.7 deg); reach of both arms is
                checked with the environment's IK (python -m welding.variations --reach).
  wire stick-out  the wire left in the gun by the last weld: 6.35-12.7 mm (1/4-1/2 in) [derived: Lincoln's electrical
                stick-out = contact tip to work - 1/4 in (L-56, read) over the robotic contact-tip-to-work range 1/2-3/4
                in (Bernard/Tregaskiss, Lincoln C4.200, read)]. Applied to the start state before tick 0.
  draft         air moving across the joint, below the AWS D1.1 5.12.1 limit for GMAW without a shelter (2.2 m/s):
                0-2.0 m/s [range authored under the read limit]. Air reaches the shield from 1.4 m/s (Kobelco's chart
                at the cell's 19 L/min, arc.air_fraction): at most 0.6 % at 2 m/s, under the porosity onset on its own,
                adding to a longer stand-off or a steeper torch; convection does not depend on air speed.
Not varied: ambient / initial coupon temperature (20 C is fixed in the thermal kernels), root gap and
bracket fit-up (the thermal grid is built at the authored fit-up), the robots' start poses (a cell starts its program
from its taught home pose), the ground clamp's place (layout constant), the source's loaded job.

Variation 0 is the authored layout (the coupon at its authored pose, the start state's own stick-out, still air):
identical to legacy seed 0, and in the train split, because every published demonstration and probe used it. The
start state of a variation is built by the environment as for any seed; a tacked start state (every task that does
not start from the loose bracket) is made by its ASSISTED fixture once per physics version (157 s to 45 min, cached ~194 MB) and placed on each
entry's own world in about 10 s.

  python -m welding.variations --export     # write catalogue.jsonl + catalogue.manifest.json
  python -m welding.variations --check      # re-derive every entry and compare with the frozen file
  python -m welding.variations --reach      # IK reach of the seam (and the bracket for the hand) at the box's corners
"""

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

VERSION = "welding-steel-cell-v1"
COUNT = 10000
SPLITS = ("train", "validation", "test")
CONFIG_SEED_BASE = 100000      # Config.seed of variation N >= 1 (the start-state cache key; legacy seeds are 0-9)
FROZEN = Path(__file__).resolve().parent/"data"/"variations"/"catalogue.jsonl"

PARAMETERS = {
    "coupon_dx_m": dict(low=-3e-3, high=3e-3, unit="m", decimals=6, label="authored",
                        basis="the play of a part a fitter places by hand under the toggle clamps "
                              "(+-3 mm); world x, along the table and the joint", physical_effect="where the joint is"),
    "coupon_dy_m": dict(low=-3e-3, high=3e-3, unit="m", decimals=6, label="authored",
                        basis="+-3 mm; world y, toward / away from the robots",
                        physical_effect="where the joint is"),
    "coupon_yaw_deg": dict(low=-2., high=2., unit="deg", decimals=4, label="authored",
                           basis="+-2 deg about the vertical through the root's start",
                           physical_effect="the joint's direction"),
    "wire_stickout_m": dict(low=6.35e-3, high=12.7e-3, unit="m", decimals=5, label="derived",
                            basis="the wire a previous weld leaves: electrical stick-out = contact tip to work - 1/4 in "
                                  "(Lincoln L-56, read) over the robotic contact-tip-to-work 1/2-3/4 in (Bernard/"
                                  "Tregaskiss, Lincoln C4.200, read) = 1/4-1/2 in",
                            physical_effect="how far the wire stands out of the contact tip at tick 0 (run-in to the "
                                            "strike, touch sensing, the position-torch wire-on-root check)"),
    "draft_m_s": dict(low=0., high=2.0, unit="m/s", decimals=3, label="authored range under a read limit",
                      basis="AWS D1.1 s5.12.1: GMAW only where the wind at the weld is <= 5 mph (2.2 m/s) unless "
                            "sheltered (read); 0-2.0 m/s keeps every entry under the limit",
                      physical_effect="air reaches the shield from 1.4 m/s (Kobelco's chart at the cell's 19 L/min, "
                                      "arc.air_fraction): at most 0.6 % air at 2 m/s, under the porosity onset alone, "
                                      "adding to a longer stand-off or a steeper torch; convection does not depend on "
                                      "air speed"),
}


def _integer(index, field):
    return int.from_bytes(hashlib.sha256(f"{VERSION}/{index}/{field}".encode()).digest()[:8], "big")


def _check_index(index):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < COUNT:
        raise ValueError(f"variation_id must be an integer in [0, {COUNT - 1}]")
    return index


PAD_MARGIN_M = 2e-3            # a clamp pad lies on the plate at least this far from every edge [authored; chosen before
                               # the catalogue was drawn: a fitter sets the pad on the plate, not on its edge]
MAX_DRAWS = 64
PLATE_FOOTPRINT = (-.025, .175, -.079, .071)   # base plate x0, x1, y0, y1 in the weld frame (welding.process.tee_coupon:
                                               # 200 x 150 mm plate, bracket 8 mm thick; checked at export and in tests)


def _draw(index, name, k):
    p = PARAMETERS[name]
    u = _integer(index, name if k == 0 else f"{name}/{k}")/(2**64 - 1)
    return round(p["low"] + (p["high"] - p["low"])*u, p["decimals"])


def _coupon(index):
    """(coupon pose, draws): uniform in the +-3 mm / +-2 deg box, redrawn (deterministically) while a clamp pad would
    come within PAD_MARGIN_M of the plate's edge -- the box's corner where dy and yaw are both near their maxima."""
    for k in range(MAX_DRAWS):
        c = dict(dx_m=_draw(index, "coupon_dx_m", k), dy_m=_draw(index, "coupon_dy_m", k),
                 yaw_deg=_draw(index, "coupon_yaw_deg", k))
        if _holding(c)["ok"]:
            return c, k + 1
    raise RuntimeError(f"variation {index}: no coupon pose inside the clamps' holding region in {MAX_DRAWS} draws")


@lru_cache(maxsize=None)
def _spec(index):
    def value(name):
        return _draw(index, name, 0)

    if index == 0:
        split = "train"            # the authored layout of every published demonstration and probe
        coupon = dict(dx_m=0., dy_m=0., yaw_deg=0.)
        stickout, draft, config_seed = None, 0., 0
    else:
        n = _integer(index, "split") % 10
        split = "train" if n < 8 else "validation" if n == 8 else "test"
        coupon, _ = _coupon(index)
        stickout, draft, config_seed = value("wire_stickout_m"), value("draft_m_s"), CONFIG_SEED_BASE + index
    data = dict(catalogue=VERSION, variation_id=index, split=split, coupon=coupon, wire_stickout_m=stickout,
                draft_m_s=draft, config_seed=config_seed)
    data["sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return json.dumps(data, sort_keys=True)


def specification(index):
    """The catalogue entry of a variation id (a fresh dict; deterministic)."""
    return json.loads(_spec(_check_index(index)))


def select(*, variation_id=None, split=None, selection_seed=0):
    """An entry by id (its split must match a given split) or, without an id, the selection_seed-th pick among the
    entries of the split (or of all). Constraints are never relaxed."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise ValueError("selection_seed must be an integer")
    if variation_id is not None:
        item = specification(variation_id)
        if split is not None and item["split"] != split:
            raise ValueError(f"variation {variation_id} is in split {item['split']!r}, not {split!r}; "
                             "constraints are not relaxed")
        return item
    candidates = [i for i in range(COUNT) if split is None or specification(i)["split"] == split]
    if not candidates:
        raise ValueError("No matching variation; constraints were not relaxed")
    return specification(candidates[_integer(selection_seed, "selection") % len(candidates)])


def config_values(entry):
    """welding.world.Config fields of an entry (the coupon pose, the draft and the start-state cache key)."""
    c = entry["coupon"]
    return dict(seed=int(entry["config_seed"]), coupon_offset=(float(c["dx_m"]), float(c["dy_m"]), float(c["yaw_deg"])),
                draft_m_s=float(entry["draft_m_s"]))


def apply_initial(env, entry):
    """Initial conditions set on a built start state before tick 0 (disclosed): the wire stick-out left in the gun.
    The env's initial checkpoint is taken again, so a reset returns to this state."""
    s = entry["wire_stickout_m"]
    if s is None:
        return None
    w = env.world
    if w.process is None:
        raise ValueError("this world has no weld process (no wire)")
    st = w.process.src.st
    if st.arc or st.on_cmd or st.inch:
        raise RuntimeError("the stick-out is an initial condition: the source must be idle")
    if math.isfinite(st.d_tw) and s >= st.d_tw:
        raise ValueError(f"stick-out {s*1e3:.2f} mm would reach the metal {st.d_tw*1e3:.2f} mm from the contact tip "
                         "at the start pose")
    before = float(st.l_s)
    st.l_s = st.l_s0 = float(s)                        # an initial condition: the wire balance starts here
    env.assisted_log.append(dict(initial_condition="wire stick-out", variation_id=entry["variation_id"],
                                 before_m=before, set_m=float(s)))
    env.initial = env.checkpoint()
    return dict(wire_stickout_m=float(s), start_state_stickout_m=before)


def fixture_check(entry):
    """Derived geometry (no physics): each toggle clamp's pad must lie on the base plate with >= PAD_MARGIN_M to every
    edge, and the coupon must stay on the table. Returns the smallest pad-to-edge margin (m) and the verdict."""
    return _holding(entry["coupon"])


def _holding(c):
    import numpy as np
    from . import layout as LY
    dx, dy, yaw = c["dx_m"], c["dy_m"], math.radians(c["yaw_deg"])
    x0, x1, y0, y1 = PLATE_FOOTPRINT
    o = LY.WELD_ORIGIN[:2] + np.array([dx, dy])
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    r = LY.CLAMP_PAD["radius"]
    margins = []
    for C in LY.CLAMPS:
        px, py = R.T @ (np.asarray(C["pad_xy"], float) - o)
        margins.append(min(px - x0, x1 - px, py - y0, y1 - py) - r)
    corners = [o + R @ np.array([x, y]) for x in (x0, x1) for y in (y0, y1)]
    on_table = all(abs(p[0]) < LY.TABLE["size"][0]/2 and abs(p[1]) < LY.TABLE["size"][1]/2 for p in corners)
    m = float(min(margins))
    return dict(pad_margin_m=m, on_table=on_table, ok=bool(m >= PAD_MARGIN_M and on_table))


def _check_plate_constants():
    """The footprint fixture_check uses is the coupon the process builds (imports the process module)."""
    from .process import tee_coupon
    c = tee_coupon()
    return all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(
        (c.base_x0, c.base_x0 + c.base[0], c.base_y[0], c.base_y[1]), PLATE_FOOTPRINT))


def contract(current=None):
    return dict(supported=True, catalogue=VERSION, count=COUNT, integer_id_range=[0, COUNT - 1],
                selectors=["variation_id", "split", "selection_seed"], splits=list(SPLITS),
                parameters={k: {kk: vv for kk, vv in v.items() if kk != "decimals"} for k, v in PARAMETERS.items()},
                current=current, preview_mutates=False, start_replaces_episode=True,
                note="A variation is the cell's initial state (coupon pose, wire stick-out, draft); the session's task "
                     "and robot are kept. Variation 0 is the authored layout (= seed 0). Starting a variation builds "
                     "its world; a tacked start state (every task that does not start from the loose bracket) is made "
                     "by its ASSISTED tack fixture once per physics version (157 s on a quiet GPU, up to 45 min on a saturated one) and placed on "
                     "each entry's coupon; a switch through the API takes about 30 s (its response carries the new "
                     "world's first observation).",
                qualification="Defined reproducible catalogue; entries are not each physically qualified "
                              "(E22 samples them).")


def select_request(request):
    """GAR /v1/variations/select request -> entry (exact keys; never relaxed)."""
    if not isinstance(request, dict) or set(request) - {"variation_id", "split", "selection_seed"}:
        raise ValueError("Choose variation_id and/or split and selection_seed; unknown fields are rejected")
    return select(**request)


def rows():
    return [specification(i) for i in range(COUNT)]


def export(folder=FROZEN.parent):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rs = rows()
    bad = [r["variation_id"] for r in rs if not fixture_check(r)["ok"]]
    if bad:
        raise RuntimeError(f"entries outside the fixture's holding region: {bad[:10]}")
    if not _check_plate_constants():
        raise RuntimeError("fixture_check's plate footprint differs from welding.process.tee_coupon")
    payload = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rs).encode()
    (folder/"catalogue.jsonl").write_bytes(payload)
    margins = [fixture_check(r)["pad_margin_m"] for r in rs]
    redrawn = sum(_coupon(r["variation_id"])[1] > 1 for r in rs if r["variation_id"])
    manifest = dict(catalogue=VERSION, count=COUNT, sha256=hashlib.sha256(payload).hexdigest(),
                    splits={s: sum(r["split"] == s for r in rs) for s in SPLITS},
                    parameters={k: {kk: vv for kk, vv in v.items() if kk != "decimals"} for k, v in PARAMETERS.items()},
                    variation_0="authored layout (= legacy seed 0), split train",
                    config_seed="Config.seed = %d + variation_id (0 for variation 0): the entry's identity in the world's "
                                "configuration (the tack weld is cached per physics hash, not per entry)"
                                % CONFIG_SEED_BASE,
                    fixture_check=dict(min_pad_margin_m=min(margins), all_ok=True, redrawn_coupon_poses=redrawn,
                                       rule="both clamp pads on the plate with >= 2 mm to every edge (authored); plate "
                                            "on the table; a pose outside is redrawn (the box's corner with dy > 2.3 mm "
                                            "and yaw > 1.7 deg)"),
                    dimensions_changed=False, procedure_changed=False, masses_changed=False,
                    physics_parameters_changed=False, solver_included=False,
                    generator="python -m welding.variations --export")
    (folder/"catalogue.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check(path=FROZEN):
    payload = Path(path).read_bytes()
    manifest = json.loads(Path(path).with_suffix(".manifest.json").read_text())
    derived = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows()).encode()
    return dict(file_sha256=hashlib.sha256(payload).hexdigest(), manifest_sha256=manifest["sha256"],
                derived_sha256=hashlib.sha256(derived).hexdigest(), identical=payload == derived)


# ---------------------------------------------------------------------------------------------- reach (IK)
RIGS = ("ur10e-fr3", "ur10e", "dual-fr3", "single-fr3", "dual-panda", "g1")


def reach(samples=64, out=FROZEN.parent/"reach.json", rigs=RIGS):
    """IK reach for every rig on rigid-only worlds (no weld process, CPU): the torch work point on the root at the seam's
    start, middle and end at the procedure's angles (45 deg work, 7.5 deg push, 19 mm contact tip to work) and 30 mm
    back along the wire over the start; for rigs with a hand, the Franka Hand pointing down over the bracket's top edge
    at 30 and 120 mm along it. Poses: the 8 corners of the coupon-pose box, the authored pose and `samples` catalogue
    entries (the first of a fixed permutation). IK from the start pose; a pose counts as reached within the IK's 1 mm /
    1 deg tolerance and the joint limits (collision-free paths are not planned or checked). The G1's "hand" is its left
    Dex3 palm, asked for the same pointing-down pose from the standing waist. With a subset of `rigs`, the other rigs'
    results already in `out` (same catalogue and method) are kept."""
    import itertools
    import time
    import numpy as np
    from scipy.spatial.transform import Rotation
    from . import layout as LY
    from .kinematics import torch_orientation
    from .world import Config, World
    P = PARAMETERS
    poses = [("authored", (0., 0., 0.))]
    for sx, sy, sz in itertools.product((-1, 1), repeat=3):
        poses.append((f"corner{'+' if sx > 0 else '-'}{'+' if sy > 0 else '-'}{'+' if sz > 0 else '-'}",
                      (P["coupon_dx_m"]["high"]*sx, P["coupon_dy_m"]["high"]*sy, P["coupon_yaw_deg"]["high"]*sz)))
    order = sorted(range(1, COUNT), key=lambda i: _integer(i, "reach"))[:samples]
    for i in order:
        c = specification(i)["coupon"]
        poses.append((f"variation-{i}", (c["dx_m"], c["dy_m"], c["yaw_deg"])))
    report = dict(catalogue=VERSION, method=reach.__doc__, rigs={})
    if set(rigs) != set(RIGS) and Path(out).is_file():
        old = json.loads(Path(out).read_text())
        if old.get("catalogue") == VERSION:
            report["rigs"] = {k: v for k, v in old["rigs"].items() if k not in rigs}
    for robot in rigs:
        t0 = time.perf_counter()
        w = World(Config(robot=robot, process=False, device="cpu", task="tack"))
        o0, R0 = w.weld_frame()
        frame_err = float(np.linalg.norm(o0 - LY.WELD_ORIGIN))
        rows_ = []
        for name, (dx, dy, yaw) in poses:
            o = LY.WELD_ORIGIN + np.array([dx, dy, 0.])
            a = math.radians(yaw)
            R = np.array([[math.cos(a), -math.sin(a), 0.], [math.sin(a), math.cos(a), 0.], [0., 0., 1.]])
            q = torch_orientation(45., 7.5, R)
            z = Rotation.from_quat(q).as_matrix()[:, 2]
            res = dict(pose=name, coupon=[dx, dy, yaw])
            for label, x, back in (("torch_start", .001, 0.), ("torch_mid", .075, 0.), ("torch_end", .149, 0.),
                                   ("torch_approach", .001, .03)):
                p = o + R @ np.array([x, 0., 0.]) - z*back
                try:
                    w.kin.solve(w.torch_arm, p, q)
                    res[label] = True
                except ValueError as exc:
                    res[label] = str(exc)
            for hand in w.hand_arms:
                qh = (Rotation.from_matrix(R)*Rotation.from_euler("x", math.pi)).as_quat()
                for label, x in (("hand_bracket_30", .030), ("hand_bracket_120", .120)):
                    p = o + R @ np.array([x, -.004, .075])
                    try:
                        w.kin.solve(hand, p, qh)
                        res[label] = True
                    except ValueError as exc:
                        res[label] = str(exc)
            rows_.append(res)
        failures = [r for r in rows_ if any(v is not True for k, v in r.items() if k not in ("pose", "coupon"))]
        report["rigs"][robot] = dict(poses=len(rows_), failures=failures, all_reached=not failures,
                                     weld_frame_vs_layout_m=frame_err, wall_s=time.perf_counter() - t0, rows=rows_)
        del w
    report["all_reached"] = all(r["all_reached"] for r in report["rigs"].values())
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reach", action="store_true")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--rigs", default=",".join(RIGS), help="rigs for --reach (comma separated; others kept from the file)")
    a = ap.parse_args()
    if a.export:
        print(json.dumps(export(), indent=1))
    if a.check:
        print(json.dumps(check(), indent=1))
    if a.reach:
        r = reach(a.samples, rigs=tuple(a.rigs.split(",")))
        print(json.dumps({k: dict(all_reached=v["all_reached"], failures=len(v["failures"]), wall_s=round(v["wall_s"], 1))
                          for k, v in r["rigs"].items()}, indent=1))


if __name__ == "__main__":
    main()
