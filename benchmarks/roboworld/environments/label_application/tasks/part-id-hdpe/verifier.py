"""Outcome of part-id-hdpe: the printed polyester ID label taken from the printer and applied to the round HDPE
canister's side, running level round it (within 5 deg) about half-way up (its middle within 30 mm of mid-height, 20 mm
clear of the wall's top and bottom); wetted out to the standard over its area and along its edges, without trapped air
or a crease in the barcode, untorn, bonded only to the canister and settled; the canister upright on the bench and
nothing touching the label or the canister; the barcode decodes when the optical reader runs.

verify(e, optical) gets the episode's evaluator (task.Evaluator) and the optical reader's verdict (dict with "match", or
None) and returns the evaluation (task.Evaluator.evaluate_task: success, score, checks, metrics, progress, reward). The
evaluator reads this task's item (TARGET), placement rules (placement), progress zone (zone: the canister's side wall)
and registration factors (registration) from here; reader_views(world) gives its optical reader stops
(optical.declared_reads calls it)."""
import numpy as np

TARGET = "part"
CANISTER_STATION_LIFT = .50 # m: the canister reader station, above the printer (top ~1.36 m) and the bench clutter
CANISTER_MID_TOL = .030             # m, middle_within_30mm_of_mid_height; 0 where the label reaches the wall's end
CANISTER_END_MARGIN = .020          # m, the same check's 20 mm from the top and the bottom; 0 at the wall's end


def canister_registration_factors(mean_z, top_margin, bottom_margin, tilt_deg):
    """part-id-hdpe registration factors for a label whose nodes' mean height is mean_z m from the canister's
    mid-height, leaving top_margin / bottom_margin m of side wall above / below it, its long side tilt_deg off level.
    Each is 1 where the placement check passes: mid-height falls linearly to 0 where the label would reach the end of
    the wall by moving the same way, the margin to 0 at the wall's end, level to 0 at 45 deg."""
    from label_application.task import _level_factor
    r_margin = float(np.clip(min(top_margin, bottom_margin)/CANISTER_END_MARGIN, 0., 1.))
    off = abs(mean_z) - CANISTER_MID_TOL
    room = top_margin if mean_z > 0 else bottom_margin
    r_mid = 1. if off <= 0 else (float(np.clip(room/(room + off), 0., 1.)) if room > 0 else 0.)
    return r_mid, r_margin, _level_factor(tilt_deg)


def placement(e, on_target):
    g = e.world.scene.items["part"]
    lab = e.world.label
    q = e._item_local("part")
    side = bool((np.hypot(q[:, 0], q[:, 1]) > g["radius"] - 1e-3).all())
    z = q[:, 2]
    centred = bool(abs(z.mean()) < .03 and z.max() < g["height"]/2 - .02 and z.min() > -g["height"]/2 + .02)
    cd = q[lab.X[:, 1] > lab.width - lab.h*.5].mean(0) - q[lab.X[:, 1] < lab.h*.5].mean(0)
    tilt = float(np.degrees(np.arcsin(min(1., abs(cd[2])/max(np.linalg.norm(cd), 1e-12)))))
    return dict(checks={"on_side": side, "middle_within_30mm_of_mid_height": centred, "horizontal_within_5deg": tilt <= 5.},
                metrics=dict(mean_height_m=float(z.mean()), tilt_deg=tilt))


def zone(e, on_target):
    it = e.world.scene.items["part"]
    q = e._item_local("part")
    # the on_side rule node by node (radius > r - 1 mm), between the lid and the base
    return (np.hypot(q[:, 0], q[:, 1]) > it["radius"] - 1e-3) & (np.abs(q[:, 2]) < it["height"]/2)


def registration(e, placed, on_target=None):
    """part-id-hdpe: the label part on the side wall, unrolled round the canister (_unrolled_along_label) and fitted
    over the label, gives its mean height, its margins to the wall's top and bottom and its long side's tilt from
    level (as placement measures them); canister_registration_factors turns them into factors."""
    it = e.world.scene.items["part"]
    lab = e.world.label
    q = e._item_local("part")
    if placed.sum() < 3:
        return dict(factor=1., note="fewer than 3 label nodes on the side wall: nothing to register")
    sz = e._unrolled_along_label(q, placed, it["radius"] + .5*lab.stock.thickness)
    coef = e._fit(sz, placed)
    if coef is None:
        return dict(factor=1., note="label nodes on the side wall are collinear: no registration fit")
    z = np.c_[lab.X, np.ones(lab.n)] @ coef[:, 1]
    top, bottom = float(it["height"]/2 - z.max()), float(z.min() + it["height"]/2)
    d = coef[1]                                                      # the label's CD (long side) round the wall
    tilt = float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))
    r_mid, r_margin, r_level = canister_registration_factors(float(z.mean()), top, bottom, tilt)
    return dict(factor=r_mid*r_margin*r_level, mid_height=r_mid, end_margin=r_margin, level=r_level,
                predicted_mean_height_m=float(z.mean()), predicted_margins_m=(top, bottom), fitted_tilt_deg=tilt)



def _canister_station(world):
    """part-id-hdpe: the canister (and its label) lifted CANISTER_STATION_LIFT straight up, clear of the bench, the
    printer stand and the robots, for the handheld reader to circle it. Without this, a reader stop on the printer
    side sits inside the printer stand's 20 mm side plate for canister offsets dy of -10 to -20 mm."""
    p, R = world.body_pose("part")
    return dict(body="part", R_rel=np.eye(3), p_body=p, p_station=p + np.array([0., 0., CANISTER_STATION_LIFT]), R_station=R)


def reader_views(world):
    """The optical reader's declared stops for optical.declared_reads, [(name, pose7, relocation)]: the canister with
    its label lifted into its reader station (_canister_station), the imager around its side at three heights (-25, 0,
    +25 mm), 12 stops of 30 deg, 0.19 m off the wall."""
    from label_application.optical import _pose
    sc = world.scene
    out = []
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


def verify(e, optical=None):
    return e.evaluate_task(optical)
