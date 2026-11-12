"""Outcome of gs1-case-side: the printed GS1 logistic label taken from the printer and applied to one of the carton's
long sides, its bars vertical within 5 deg, the bottom of the SSCC barcode 32 +- 6 mm above the carton's base and the
barcode with its quiet zones at least 19 mm from either vertical edge of that side; the label wetted out to the standard
over its area and along its edges, without trapped air or a crease in the barcode, untorn, bonded only to the carton and
settled; the carton resting on the bench and nothing touching the label or the carton; the barcode decodes when the
optical reader runs.

verify(e, optical) gets the episode's evaluator (task.Evaluator) and the optical reader's verdict (dict with "match", or
None) and returns the evaluation (task.Evaluator.evaluate_task: success, score, checks, metrics, progress, reward). The
evaluator reads this task's item (TARGET), placement rules (placement), progress zone (zone: the long side face holding
the label) and registration factors (registration) from here; reader_views(world) gives its optical reader stops
(optical.declared_reads calls it)."""
import numpy as np

TARGET = "carton"
GS1_BAR_BOTTOM_BAND = (.026, .038)  # m, barcode_bottom_32+-6mm_above_base; 0 where the label reaches the base / top
GS1_EDGE_CLEARANCE = .019           # m, barcode_>=19mm_from_vertical_edges; 0 with the symbol at the vertical edge


def gs1_registration_factors(bottom, clearance, tilt_deg, label_low, label_high, H):
    """gs1-case-side registration factors for a label whose SSCC symbol's printed bottom edge is `bottom` m above the
    carton's base, the symbol with its quiet zones `clearance` m from the nearer vertical edge, its bars tilt_deg off
    vertical, and the label itself spanning label_low..label_high m above the base on a side H m tall. Each is 1
    where the placement check passes: the barcode height falls linearly to 0 where the label would reach the base (or
    the top edge) by moving the same way, the edge clearance to 0 with the symbol at the edge, the bars to 0 at 45 deg."""
    from label_application.task import _level_factor
    lo, hi = GS1_BAR_BOTTOM_BAND
    if lo <= bottom <= hi:
        r_h = 1.
    elif bottom < lo:
        r_h = float(np.clip(label_low/(lo - bottom + label_low), 0., 1.)) if label_low > 0 else 0.
    else:
        room = H - label_high
        r_h = float(np.clip(room/(room + bottom - hi), 0., 1.)) if room > 0 else 0.
    r_e = float(np.clip(clearance/GS1_EDGE_CLEARANCE, 0., 1.))
    return r_h, r_e, _level_factor(tilt_deg)


def side_face(e, q=None, on_target=None):
    """gs1-case-side: the long side face (-1: front, -y; +1: back, +y) holding the label, i.e. the one with more
    label area bonded to the carton within 2 mm of its plane; with nothing bonded there, the front (nothing is
    credited then). Either long side is accepted (on_a_long_side_face)."""
    lab = e.world.label
    g = e.world.scene.task_geometry["carton"]
    q = e._item_local("carton") if q is None else q
    if on_target is None:
        on_target = e._contact_state()[0]
    a = {s: float((lab.vertex_area*on_target*(s*q[:, 1] > g["W"]/2 - 2e-3)).sum()) for s in (-1., 1.)}
    return 1. if a[1.] > a[-1.] else -1.


def placement(e, on_target):
    from label_application.task import material_points
    g = e.world.scene.task_geometry["carton"]
    lab = e.world.label
    q = e._item_local("carton")
    front = bool((q[:, 1] < -g["W"]/2 + 2e-3).all())
    back = bool((q[:, 1] > g["W"]/2 - 2e-3).all())
    X = lab.X
    # the printed symbol's own edges, interpolated through the mesh (outermost vertices inside the symbol sit up to
    # one mesh cell inside its printed edge)
    (u0, u1), (v0, v1) = e.bar_u, e.bar_v
    n = 25
    bottom_uv = np.stack([np.full(n, u0), np.linspace(v0, v1, n)], 1)
    edge_uv = np.concatenate([bottom_uv, np.stack([np.full(n, u1), np.linspace(v0, v1, n)], 1),
                              np.stack([np.linspace(u0, u1, n), np.full(n, v0)], 1),
                              np.stack([np.linspace(u0, u1, n), np.full(n, v1)], 1)])
    pts, _ = material_points(lab, np.concatenate([bottom_uv, edge_uv]))
    ql = e._local("carton", pts)
    base_height = float((ql[:n, 2] + g["H"]/2).mean())
    edge = float(g["L"]/2 - np.abs(ql[n:, 0]).max())
    # bars vertical: label MD axis vs carton vertical
    md = q[X[:, 0] > lab.length - lab.h*.5].mean(0) - q[X[:, 0] < lab.h*.5].mean(0)
    tilt = float(np.degrees(np.arccos(abs(md[2])/max(np.linalg.norm(md), 1e-12))))
    return dict(checks={"on_a_long_side_face": front or back, "barcode_bottom_32+-6mm_above_base": abs(base_height - .032) <= .006,
                        "bars_vertical_within_5deg": tilt <= 5., "barcode_>=19mm_from_vertical_edges": edge >= .019},
                metrics=dict(face="front(-y)" if front else "back(+y)" if back else None,
                             barcode_bottom_above_base_m=base_height, md_tilt_from_vertical_deg=tilt,
                             barcode_edge_clearance_m=edge))


def zone(e, on_target):
    g = e.world.scene.task_geometry["carton"]
    q = e._item_local("carton")
    side = side_face(e, q, on_target)
    # the on_a_long_side_face rule node by node: beyond that face's plane less 2 mm, inside the face
    return ((side*q[:, 1] > g["W"]/2 - 2e-3) & (np.abs(q[:, 0]) <= g["L"]/2)
            & (np.abs(q[:, 2]) <= g["H"]/2))


def registration(e, placed, on_target=None):
    """gs1-case-side: the label part on the long side face, fitted over the label (_fit on the face's along / up
    coordinates), gives the SSCC symbol's printed bottom-edge height above the base, its clearance (quiet zones
    included) from the nearer vertical edge and the bars' tilt from vertical, measured as placement
    measures them, and the label's own height span; gs1_registration_factors turns them into factors."""
    g = e.world.scene.task_geometry["carton"]
    lab = e.world.label
    q = e._item_local("carton")
    coef = e._fit(q[placed][:, [0, 2]], placed)
    if coef is None:
        return dict(factor=1., note="fewer than 3 label nodes (or collinear ones) on the side face: nothing to register")
    (u0, u1), (v0, v1) = e.bar_u, e.bar_v
    n = 25
    bottom_uv = np.stack([np.full(n, u0), np.linspace(v0, v1, n)], 1)
    edge_uv = np.concatenate([bottom_uv, np.stack([np.full(n, u1), np.linspace(v0, v1, n)], 1),
                              np.stack([np.linspace(u0, u1, n), np.full(n, v0)], 1),
                              np.stack([np.linspace(u0, u1, n), np.full(n, v1)], 1)])
    P = lambda uv: np.c_[uv, np.ones(len(uv))] @ coef                  # (along, up) in the carton frame
    bottom = float((P(bottom_uv)[:, 1] + g["H"]/2).mean())
    clearance = float(g["L"]/2 - np.abs(P(edge_uv)[:, 0]).max())
    d = coef[0]                                                      # the label's MD (bars) direction on the face
    tilt = float(np.degrees(np.arccos(min(1., abs(d[1])/max(np.linalg.norm(d), 1e-12)))))
    span = P(np.array([[0., 0.], [0., lab.width], [lab.length, 0.], [lab.length, lab.width]]))[:, 1] + g["H"]/2
    r_h, r_e, r_t = gs1_registration_factors(bottom, clearance, tilt, float(span.min()), float(span.max()), g["H"])
    return dict(factor=r_h*r_e*r_t, barcode_height=r_h, edge_clearance=r_e, bars_vertical=r_t,
                face="front(-y)" if side_face(e, q, on_target) < 0 else "back(+y)",
                predicted_barcode_bottom_above_base_m=bottom, predicted_barcode_edge_clearance_m=clearance,
                fitted_bar_tilt_deg=tilt, predicted_label_span_above_base_m=(float(span.min()), float(span.max())))



def reader_views(world):
    """The optical reader's declared stops for optical.declared_reads, [(name, pose7, relocation)]: the carton with its
    label in the carton reader station (optical._carton_station), the imager in front of EACH long side face at two
    heights and three positions along it."""
    from label_application.optical import _carton_station, _pose
    sc = world.scene
    out = []
    reloc = _carton_station(world)
    p, R = reloc["p_station"], reloc["R_station"]
    g = sc.task_geometry["carton"]
    W = lambda v: p + R @ np.asarray(v, float)
    ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
    # the GS1 side face keeps 0.30 m (0.5 mm modules)
    d = .30
    for face, s in (("front", -1), ("back", 1)):           # front = -y (away from the robots), back = +y
        for zc in (.07, .12):
            for xo in (0., -.10, .10):
                out.append((f"{face}_side_z{zc*1e3:.0f}mm_x{xo*1e3:+.0f}mm",
                            _pose(W((xo, s*(g["W"]/2 + d), -g["H"]/2 + zc)), -s*ey, -s*ex), reloc))
    return out


def verify(e, optical=None):
    return e.evaluate_task(optical)
