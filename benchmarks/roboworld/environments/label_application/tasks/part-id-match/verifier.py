"""Outcome of part-id-match: the printed polyester ID label taken from the printer and applied to the stainless plate
whose laser-marked part number matches the one printed on the label (the scene's target plate), on its marked face and
without covering the marking; wetted out to the standard over its area and along its edges, without trapped air or a
crease in the barcode, untorn, bonded only to that plate and settled; the plate lying flat on the bench and nothing
touching the label or the plate; the barcode decodes when the optical reader runs.

verify(e, optical) gets the episode's evaluator (task.Evaluator) and the optical reader's verdict (dict with "match", or
None) and returns the evaluation (task.Evaluator.evaluate_task: success, score, checks, metrics, progress, reward). The
evaluator reads this task's item (TARGET: None, the scene names the target plate), placement rules (placement), progress
zone (zone: the target plate's upper face) and registration factor (registration: the marking left clear) from here;
reader_views(world) gives its optical reader stops (optical.declared_reads calls it)."""
import numpy as np

TARGET = None
PLATE_IMAGER_ROLLS = tuple(range(0, 180, 10))   # deg: a person turns the scanner to the code (a 1-D symbol reads both ways);
                            # 10 deg steps leave a label at most 5 deg off a roll; the decode holds to ~8.5-9 deg


def placement(e, on_target):
    name = e.target
    it = e.world.scene.items[name]
    sx, sy, sz = it["size"]
    q = e._item_local(name)
    on_face = bool((q[:, 2] > sz/2 - 1e-3).all() and (np.abs(q[:, 0]) <= sx/2).all() and (np.abs(q[:, 1]) <= sy/2).all())
    mk = it["marking"]
    uu, vv = np.meshgrid(np.linspace(*mk["u"], 13), np.linspace(*mk["v"], 5))
    covered = e._covered(q[:, :2], e._alive_tris(), np.stack([uu.ravel(), vv.ravel()], 1))
    lab = e.world.label
    bonded = {}
    if hasattr(lab, "colliders"):                    # (absent in the geometry harness)
        cols = [c.name for c in lab.colliders]
        for b in lab.bond[lab.bond >= 0]:
            bonded[cols[b]] = bonded.get(cols[b], 0) + 1
    return dict(checks={"on_target_plate_upper_face": on_face, "marking_not_covered": covered == 0.},
                metrics=dict(target_plate=name, target_part_number=it["part_number"], marking_covered_fraction=covered,
                             nodes_bonded_per_surface=bonded))


def zone(e, on_target):
    sx, sy, sz = e.world.scene.items[e.target]["size"]
    q = e._item_local(e.target)
    # the on_target_plate_upper_face rule node by node
    return (q[:, 2] > sz/2 - 1e-3) & (np.abs(q[:, 0]) <= sx/2) & (np.abs(q[:, 1]) <= sy/2)


def registration(e, placed, on_target=None):
    """part-id-match: 1 - the share of the target plate's marking (the marking_not_covered check's 13 x 5 points)
    that the label lying on its face covers, extrapolated over the label (_fit, _fitted_cover)."""
    it = e.world.scene.items[e.target]
    q = e._item_local(e.target)
    coef = e._fit(q[placed][:, :2], placed)
    if coef is None:
        return dict(factor=1., note="fewer than 3 label nodes (or collinear ones) on the plate face: nothing to register")
    mk = it["marking"]
    uu, vv = np.meshgrid(np.linspace(*mk["u"], 13), np.linspace(*mk["v"], 5))
    c = e._fitted_cover(coef, np.stack([uu.ravel(), vv.ravel()], 1))
    return dict(factor=1. - c, marking_clear=1. - c, predicted_marking_covered_fraction=c)



def reader_views(world):
    """The optical reader's declared stops for optical.declared_reads, [(name, pose7, relocation)]: over the target
    plate on a 4 x 3 grid of positions 0.19 m above its face, the imager turned 0-170 deg in 10 deg steps
    (PLATE_IMAGER_ROLLS)."""
    from label_application.optical import _pose
    sc = world.scene
    out = []
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
    return out


def verify(e, optical=None):
    return e.evaluate_task(optical)
