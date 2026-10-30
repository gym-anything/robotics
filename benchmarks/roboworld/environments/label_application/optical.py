"""Optical verification: scan the applied label the way it is scanned at work and decode with zxing-cpp.

Success uses DECLARED reader poses: a handheld imager (40 deg HFOV, 16:9) held at poses fixed by the
item's pose and the task's target zone, never by where the label actually ended up:
  (carton tasks)   the carton with its label in a reader station clear of the bench, the printer and the robots,
                   its turn about the vertical removed, so the read does not depend on where or how it was left:
  carrier-top,     over each flap half, three positions along the carton, 0.22 m above the top
  press-down
  cover-old-label  over the old label (its seeded position), three positions, 0.22 m
  gs1-case-side    in front of EACH long side face at two heights and three positions, 0.30 m
  tube-wrap        rotating tube reader: the tube (with its label) is placed in the reader station, clear of
                   the bench and of the robots, and turned in 30 deg steps (12 stops) in front of a fixed imager
                   0.09 m off the wall at the label band, as an accessioning tube reader spins the tube. The
                   tube's resting roll in the cradle therefore cannot decide the verdict.
  part-id-hdpe     the canister (with its label) lifted into a reader station clear of the bench and the
                   printer stand; around its side at three heights (-25, 0, +25 mm), 12 stops of 30 deg, 0.19 m
  part-id-match    over the target plate on a 4 x 3 grid of positions, the imager turned 0-170 deg in 10 deg
                   steps (a person turns the scanner to the code), 0.19 m
Neither the robot nor a tool is ever in a verification frame (their bodies are moved out of view): the scan happens
after the robot lets go and puts its tool down, and the hands-and-tools check already fails a robot or a tool
touching the label or the item.
The poses are tried in order and scanning stops at the first good read (a person stops at the beep).
A read counts only if the decoded text equals the payload AND the whole barcode symbol including its
quiet zones (its printed edges, interpolated through the label mesh) projects inside that image on the side
facing the reader (a clipped symbol that happens to decode is not a verified scan). A pose at which the
symbol is not framed cannot count, so it is reported and not rendered (same verdict, fewer renders).

A render that fails at a framing pose is retried once, then the evaluation raises VerificationUnavailable (no
verdict); a frame that renders and shows no symbol is "no read". A renderer that cannot run also raises
VerificationUnavailable. The verification render has its own fixed resolution and sample count (VERIFY_RENDER),
independent of the session's policy-camera options. The reader envelopes cover every placement the rules allow (plate label at any angle, canister label off
mid-height, GS1 on the robot-facing side).

A privileged auto-aim reader (placed on the label's own centroid and mean normal) is kept as
`diagnostic_auto_aim`: reported, never used for success.
Shared by the GAR backend (evaluate / gar verify) and the demos.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HFOV_DEG = 40.
ASPECT = 20.25/36.          # reader sensor aperture (render.py)
FILL = .75
# The verification render: fixed, so the verdict cannot depend on the policy cameras' resolution or samples
# (at 640 x 360 the same passing labels fail, with 0.375 mm modules ~1.5 px wide).
VERIFY_RENDER = dict(width=1280, height=720, samples=24)
TUBE_STATION_LIFT = .30     # m above the cradle: the rotating reader station, clear of the bench and the robots
CANISTER_STATION_LIFT = .50 # m: the canister reader station, above the printer (top ~1.36 m) and the bench clutter
CARTON_STATION_LIFT = .50   # m: the carton reader station (GS1 side readers at z ~1.5 m, above the printer top)
CARTON_STATION_XY = (.12, -.09)   # m: over the carton's start area, away from the printer stand and the robots
TABLE_Z = .91               # bench top (scene.TABLE)
PLATE_IMAGER_ROLLS = tuple(range(0, 180, 10))   # deg: a person turns the scanner to the code (a 1-D symbol reads both ways);
                            # 10 deg steps leave a label at most 5 deg off a roll; the decode holds to ~8.5-9 deg
HIDDEN_Z = -50.             # bodies kept out of a verification frame go here (robots always; all but the item in a station)
RENDER_TRIES = 2            # a declared read whose render fails is retried once, then the evaluation has no verdict


class VerificationUnavailable(RuntimeError):
    """The verification renderer could not run: there is no verdict (not a task failure)."""


def read_distance(world):
    lab = world.label
    return max(lab.length, lab.width)/(FILL*2*np.tan(np.radians(HFOV_DEG/2)))


def _pose(eye, forward, right):
    """Camera body pose: camera -Z along `forward`, +X along `right`."""
    z = -np.asarray(forward, float); z /= np.linalg.norm(z)
    x = np.asarray(right, float); x = x - (x @ z)*z; x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.r_[eye, Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()]


def _tube_station(world, k):
    """Relocation of the tube (and everything on it) into the rotating reader at stop k: (body, R_rel, p_body, p_station).
    A point x on the tube maps to p_station + R_rel (x - p_body)."""
    g = world.scene.task_geometry["tube"]
    p, R = world.body_pose("tube")
    cx, cy, cz = g.get("cradle", (p[0], p[1], p[2]))
    R_base = Rotation.from_rotvec([0., -np.pi/2, 0.]).as_matrix()      # tube axis along -x (cap toward -x), as in the cradle
    R_k = R_base @ Rotation.from_rotvec([0., 0., np.radians(30*k)]).as_matrix()
    return dict(body="tube", R_rel=R_k @ R.T, p_body=p, p_station=np.array([cx, cy, cz + TUBE_STATION_LIFT]), R_station=R_k)


def _carton_station(world):
    """Carton tasks: the carton (with its label) moved to a reader station CARTON_STATION_LIFT above a fixed point of the
    bench, its turn about the vertical removed, every other body out of view. The read then depends only on the
    label's place on the carton, not on where or how the carton was left: "you may steady it or turn it" allows a
    carton turned toward the printer, whose GS1 readers would otherwise sit inside the printer stand."""
    p, R = world.body_pose("carton")
    g = world.scene.task_geometry["carton"]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    R_station = Rotation.from_rotvec([0., 0., -yaw]).as_matrix() @ R
    p_station = np.array([CARTON_STATION_XY[0], CARTON_STATION_XY[1], TABLE_Z + g["H"]/2 + CARTON_STATION_LIFT])
    return dict(body="carton", R_rel=R_station @ R.T, p_body=p, p_station=p_station, R_station=R_station)


def _canister_station(world):
    """part-id-hdpe: the canister (and its label) lifted CANISTER_STATION_LIFT straight up, clear of the bench, the
    printer stand and the robots, for the handheld reader to circle it. Without this, a reader stop on the printer
    side sits inside the printer stand's 20 mm side plate for canister offsets dy of -10 to -20 mm."""
    p, R = world.body_pose("part")
    return dict(body="part", R_rel=np.eye(3), p_body=p, p_station=p + np.array([0., 0., CANISTER_STATION_LIFT]), R_station=R)


def relocate(points, reloc, vectors=False):
    if reloc is None:
        return points
    if vectors:
        return points @ reloc["R_rel"].T
    return reloc["p_station"] + (points - reloc["p_body"]) @ reloc["R_rel"].T


def declared_reads(world, task):
    """[(name, pose7, relocation or None)] fixed by the item pose and the task's target zone. The ladder tasks read
    as the task whose bench they share (smooth-corner, finish-securing: press-down; tube-label: tube-wrap)."""
    from .config import base_task
    task = base_task(task)
    sc = world.scene
    out = []
    if task in ("carrier-top", "press-down", "cover-old-label", "gs1-case-side"):
        reloc = _carton_station(world)
        p, R = reloc["p_station"], reloc["R_station"]
        g = sc.task_geometry["carton"]
        W = lambda v: p + R @ np.asarray(v, float)
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
        # 0.22 m over a carton top: the 0.375 mm carrier-code module spans ~3 px of the 1280 x 720 imager
        # (at 0.30 m it is 2.2 px, below reliable decoding; the whole label still fits the field); the
        # GS1 side face keeps 0.30 m (0.5 mm modules)
        d = .22 if task != "gs1-case-side" else .30
        if task in ("carrier-top", "press-down"):
            for side, yh in (("rear", 1), ("front", -1)):
                yc = yh*(g["tape_half"] + g["W"]/2)/2
                for xo in (0., -.10, .10):
                    out.append((f"{side}_half_x{xo*1e3:+.0f}mm", _pose(W((xo, yc, g["H"]/2 + d)), -ez, ex), reloc))
        elif task == "cover-old-label":
            o = sc.task_geometry["old_label"]
            uc, vc = np.mean(o["u"]), np.mean(o["v"])
            for xo in (0., -.05, .05):
                out.append((f"old_label_x{xo*1e3:+.0f}mm", _pose(W((uc + xo, vc, g["H"]/2 + d)), -ez, ex), reloc))
        else:
            for face, s in (("front", -1), ("back", 1)):           # front = -y (away from the robots), back = +y
                for zc in (.07, .12):
                    for xo in (0., -.10, .10):
                        out.append((f"{face}_side_z{zc*1e3:.0f}mm_x{xo*1e3:+.0f}mm",
                                    _pose(W((xo, s*(g["W"]/2 + d), -g["H"]/2 + zc)), -s*ey, -s*ex), reloc))
    elif task == "tube-wrap":
        g = sc.task_geometry["tube"]
        lab = world.label
        zc = g["L"]/2 - g["mfr_label_top_from_cap_end"] - lab.width/2     # centre of the label band (tube axis)
        for k in range(12):
            reloc = _tube_station(world, k)
            band = reloc["p_station"] + reloc["R_station"] @ np.array([0., 0., zc])
            eye = band + np.array([0., -(g["r"] + .09), 0.])
            out.append((f"tube_reader_stop_{30*k:03d}deg", _pose(eye, np.array([0., 1., 0.]), reloc["R_station"][:, 2]), reloc))
    elif task == "part-id-match":
        name = sc.target
        p, R = world.body_pose(name)
        it = sc.items[name]
        for roll in PLATE_IMAGER_ROLLS:
            a = np.radians(roll)
            right = np.cos(a)*R[:, 0] + np.sin(a)*R[:, 1]
            for yo in (0., -.03, .03):
                for xo in (-.045, -.015, .015, .045):
                    eye = p + R @ np.array([xo, yo, it["size"][2]/2 + .19])
                    out.append((f"plate_x{xo*1e3:+.0f}mm_y{yo*1e3:+.0f}mm_roll{roll:03d}", _pose(eye, -R[:, 2], right), None))
    elif task == "part-id-hdpe":
        reloc = _canister_station(world)
        p, R = reloc["p_station"], reloc["R_station"]
        it = sc.items["part"]
        for zh in (0., -.025, .025):
            for k in range(12):
                a = np.radians(30*k)
                radial = np.array([np.cos(a), np.sin(a), 0.])
                tang = np.array([-np.sin(a), np.cos(a), 0.])
                eye = p + R @ (radial*(it["radius"] + .19) + np.array([0., 0., zh]))
                out.append((f"canister_z{zh*1e3:+.0f}mm_turned_{30*k:03d}deg", _pose(eye, -(R @ radial), R @ tang), reloc))
    return out


def declared_poses(world, task):
    """[(name, pose7)] (compatibility; tube-wrap and part-id-hdpe poses are in a reader station, see declared_reads)."""
    return [(n, p) for n, p, _ in declared_reads(world, task)]


def barcode_mask(world, art):
    """Label vertices inside the barcode symbol including its quiet zones (artwork barcode_px)."""
    lab = world.label
    W, H = art["image_px"]
    x0, y0, x1, y1 = art["barcode_px"]
    v0, v1 = x0/W*lab.width, x1/W*lab.width
    u0, u1 = (1 - y1/H)*lab.length, (1 - y0/H)*lab.length
    X = lab.X
    return (X[:, 1] >= v0 - 1e-9) & (X[:, 1] <= v1 + 1e-9) & (X[:, 0] >= u0 - 1e-9) & (X[:, 0] <= u1 + 1e-9)


def symbol_points(world, art, relocation=None):
    """Points over the printed symbol with quiet zones (its edges included), interpolated through the label mesh,
    with the facestock normal at each; relocated into the reader station if given."""
    from .task import material_points, symbol_samples
    lab = world.label
    pts, n = material_points(lab, symbol_samples(lab, art))
    return relocate(pts, relocation), relocate(n, relocation, vectors=True)


def symbol_in_view(world, art, pose, margin=.01, relocation=None):
    """Does the whole symbol (with quiet zones) project inside the image, facing the reader?"""
    pts, n = symbol_points(world, art, relocation)
    eye = pose[:3]
    Rc = Rotation.from_quat(pose[3:]).as_matrix()
    q = (pts - eye) @ Rc                         # camera frame: looks along -z
    if (q[:, 2] >= 0).any():
        return False, 0.
    tx = np.tan(np.radians(HFOV_DEG/2))
    u = q[:, 0]/(-q[:, 2])/tx                    # -1..1 across the image width
    v = q[:, 1]/(-q[:, 2])/(tx*ASPECT)
    inside = (np.abs(u) <= 1 - margin) & (np.abs(v) <= 1 - margin)
    facing = ((eye - pts)*n).sum(1) > 0          # printed face (+n) toward the reader
    frac = float((inside & facing).mean())
    return bool(frac >= 1.), frac


def framed_reads(world, art, task):
    """The declared reads at which the symbol is framed (only these can produce a good read)."""
    out = []
    for name, pose, reloc in declared_reads(world, task):
        ok, frac = symbol_in_view(world, art, pose, relocation=reloc)
        out.append((name, pose, reloc, ok, frac))
    return out


def render_state(world, relocation):
    """Render state for a read. Neither the robot nor a tool is ever in a verification frame: the scan happens after
    the robot has let go and put its tool down. An arm resting above a finished label or a tool still held
    clear of it would otherwise block the only framing reader. A tool touching the label or the item fails the
    hands-and-tools check, so hiding tools cannot turn a failure into a pass. Other items stay. In a reader station the
    item and its label move into the station and every other body moves out of view."""
    from .render import label_sheets
    front, back, n = label_sheets(world)
    body_q = world.state.body_q.numpy().copy()
    if relocation is None:
        tools = [c.body for c in world.label.colliders if c.role == "tool" and c.body >= 0]
        body_q[list(world.scene.robot_bodies) + tools, :3] = (0., 0., HIDDEN_Z)
        return dict(tick=world.tick, body_q=body_q, front=front, back=back, n=n)
    b = world.scene.bodies[relocation["body"]]
    body_q[:, :3] = (0., 0., HIDDEN_Z)
    body_q[b, :3] = relocation["p_station"]
    body_q[b, 3:] = Rotation.from_matrix(relocation["R_station"]).as_quat()
    return dict(tick=world.tick, body_q=body_q, front=relocate(front, relocation), back=relocate(back, relocation),
                n=relocate(n, relocation, vectors=True))


def auto_aim_pose(world):
    """DIAGNOSTIC ONLY (privileged): reader on the label's own centroid and mean normal."""
    from .render import label_sheets
    front, _, n = label_sheets(world)
    c = front.mean(0)
    nm = n.mean(0); nm /= np.linalg.norm(nm)
    up_ref = np.array([0., 0., 1.]) if abs(nm[2]) < .9 else np.array([0., 1., 0.])
    return _pose(c + nm*read_distance(world), -nm, np.cross(up_ref, nm))


reader_pose = auto_aim_pose      # backwards-compatible name (diagnostic)


def _decode(renderer, art, pose, path, state=None):
    from . import artwork
    img = renderer.observe(("reader",), state=state, reader_pose=pose)["reader"]
    path.write_bytes(img)
    decoded = [dict(text=t, format=f) for t, f in artwork.decode(path)]
    clean = lambda t: t.replace("(", "").replace(")", "").replace(" ", "")
    return decoded, any(clean(d["text"]) == art["payload"] for d in decoded)


def _renderer_gone(renderer):
    proc = getattr(renderer, "process", None)
    return bool(getattr(renderer, "closed", False)) or (proc is not None and proc.poll() is not None)


def read(renderer, art, task, out_dir, tag, world=None, diagnostic=True):
    world = world or renderer.world
    out_dir = Path(out_dir)
    attempts, success = [], None
    reads = framed_reads(world, art, task)
    for name, pose, reloc, framed, frac in reads:
        rec = dict(pose=name, symbol_with_quiet_zones_in_view=framed, symbol_fraction_in_view=frac,
                   reader_station=reloc is not None)
        if not framed:
            rec.update(rendered=False, decoded=[], payload_match=False,
                       note="symbol not framed at this pose: a read here cannot count (not rendered)")
            attempts.append(rec)
            continue
        if renderer is None:
            raise VerificationUnavailable("a declared reader frames the symbol but no verification renderer is available")
        path = out_dir/f"reader_{tag}_{name}.png"
        state = render_state(world, reloc)
        errors = []
        for _ in range(RENDER_TRIES):
            try:
                decoded, match = _decode(renderer, art, pose, path, state)
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}"[:300])
                if _renderer_gone(renderer):
                    raise VerificationUnavailable(f"verification renderer stopped at pose {name}: {exc}") from exc
        else:
            # a frame that renders and shows no symbol is "no read"; a frame that cannot be rendered (timeout, a worker
            # error) says nothing about the label: no verdict
            raise VerificationUnavailable(f"verification render failed {RENDER_TRIES} times at pose {name}: {errors}")
        if errors:
            rec["render_errors_retried"] = errors
        rec.update(rendered=True, image=str(path), decoded=decoded, payload_match=bool(match))
        attempts.append(rec)
        if match:
            success = rec
            break
    res = dict(available=True, mode="declared reader envelope (item pose + task target zone)",
               expected=art["payload"], match=success is not None, good_read=success, attempts=attempts,
               reader=dict(hfov_deg=HFOV_DEG, poses_declared=len(reads),
                           poses_framing_symbol=sum(1 for r in reads if r[3]), render=dict(VERIFY_RENDER)))
    if diagnostic and renderer is not None:
        try:
            pose = auto_aim_pose(world)
            path = out_dir/f"reader_{tag}_DIAGNOSTIC_auto_aim.png"
            decoded, match = _decode(renderer, art, pose, path, render_state(world, None))
            framed, frac = symbol_in_view(world, art, pose)
            res["diagnostic_auto_aim"] = dict(note="privileged pose from the label's own state; not used for success",
                                              image=str(path), decoded=decoded, payload_match=bool(match),
                                              symbol_with_quiet_zones_in_view=framed, symbol_fraction_in_view=frac)
        except Exception as exc:
            res["diagnostic_auto_aim"] = dict(note="diagnostic render failed (not used for success)",
                                              error=f"{type(exc).__name__}: {exc}"[:300])
    # compatibility with earlier consumers
    res["decoded"] = (success or {}).get("decoded", [])
    return res
