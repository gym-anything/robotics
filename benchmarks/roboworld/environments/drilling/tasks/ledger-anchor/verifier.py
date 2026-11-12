"""Outcome of ledger-anchor: the ledger fastened to the wall where it was set, with a KB-TZ2 anchor through each of
its four marks: each bore through the ledger at its mark and 108 to 125 mm into the concrete, square to the wall, the
anchor driven to its nominal embedment with its washer on the ledger, its nut torqued to the installation torque and
its expansion engaged; no other full-diameter hole, no bar cut, the ledger still level.

verify(ev) receives the episode's evaluator (task.Evaluator) and returns its evaluation dict
(Evaluator.evaluate_checks): checks(ev) below, which also returns the continuous progress the score is read from.
track_tick(ev) records each anchor's largest installation torque every tick (Evaluator.tick); payload_diagnostic(ev)
reports the rated payload instead of judging it (Evaluator.common, Evaluator.evaluate_checks), and
NO_SCORE_WHEN_UNEVALUABLE makes a state that cannot be evaluated carry no score (env.DrillingEnv.get_state)."""
import numpy as np

from drilling import params as P
from drilling.layout import WALL_AXIS_TOL_DEG, WALL_MARK_TOL_M
from drilling.task import SCORE_OVER_TORQUE, _angle_deg, _check, _in_plane, _ramp

# The rated-payload seconds decide no success here: the task is judged on its work product, not on how the tools were
# carried (one hand or two), and the policy is never told the rating. They are reported under `diagnostics`
# (payload_diagnostic below).

# the search radius of a mark: the openings at a mark are the records within this distance (the wall tasks' radius),
# and the mark's bore is the opening that carries the most work (ledger_openings below; the record nearest the
# mark, of any depth, can be an abandoned start instead of the bore carrying the anchor). The radius
# is not a licence for other holes: every other record must be a pit shallower than the bit's point (a running bit's
# touch, no full-diameter hole), a re-entry (its entry within one bit radius of the judged record in the same part: the
# same opening, cut again off axis; the hole module opens a new record for a bit more than CAPTURE_TOL off a hole's
# axis), a start in the ledger that the anchor's washer covers (its whole width within WASHER_R_M of the bore's axis:
# at most 10.55 mm off for this bit), or a blind start in the ledger elsewhere (reported and charged in progress, not
# failed: surface damage in the timber). Any other full-diameter hole fails the job: through the ledger
# beside or away from a mark's opening, or in the concrete. A start that has
# reached the concrete has gone through the ledger beside the bore, and an open hole that close to an anchor's hole
# breaks the environment's own aborted-hole rule (ABORTED_HOLE_DISTANCE_FACTOR, ETAG 001 Part 1 7.3: at least twice
# its depth away unless filled; the scene has no mortar). Justified as "under the washer" alone, the radius would let
# a second full bore 15-19 mm beside an anchor pass; with no depth floor a 0.82 mm pit 30 mm from a mark would fail a
# finished job.
MARK_RADIUS_M = 0.020
# "with its washer on the ledger": the washer's bearing face stands at most this far off the ledger's front face
WASHER_ON_FACE_M = 0.001
# "set level on temporary brackets ... Fasten it": the ledger is fastened where it was set. Its centre may end at most
# this far from its start pose in the wall's plane (a ledger slid 60 mm along its brackets and fastened there does
# not pass). Authored: twice the 5 mm nudge the task accepts, which also covers the ledger's give under
# the anchor blows (0.03-1.04 mm per anchor driven); half the mark search radius.
LEDGER_SHIFT_MAX_M = 0.010
LEDGER_SCORING = ("ledger-anchor progress v3: score = 1 on success, else 0.9 x progress; progress "
                  "= mean over the four marks of the opening carrying the most work, placement x (bore + level x "
                  "position x (embedment + torque))/3, less a quarter per bar cut off a mark's opening and each other "
                  "defect hole's share of a bore; embedment and torque count only at the embedment reached once torqued")
# a state that cannot be evaluated (a world fault, a non-finite state, an evaluator error) carries no score in the
# per-step record instead of 0, which reads as "nothing done". The backend's step reward is then None
# (BackendStep.reward may be None); its evaluate() already refuses a faulted state.
NO_SCORE_WHEN_UNEVALUABLE = True


def verify(ev):
    return ev.evaluate_checks(checks)


def track_tick(ev):
    """The all-time largest installation-torque record of each anchor, every tick.
    parts.WedgeAnchor resets its record to 0 when the cone loses its grip (clip_left_hole, cone_pulled_through,
    cone_driven_out), so a nut taken past 1.7 T_inst, then struck loose and tightened again, would leave no trace of
    the over-torque in the record alone."""
    rec = ev.s.setdefault("anchor_max_turn_nm", {})
    for n, it in ev.w.hm.items.items():
        if it.kind == "anchor":
            t = float(it.state.get("setting", {}).get("max_turn_torque_nm", 0.))
            if t > rec.get(n, 0.):
                rec[n] = t


def payload_diagnostic(ev):
    """The rated-payload seconds as a report, deciding nothing (Evaluator.common leaves out the payload check)."""
    secs = {a: round(float(t), 3) for a, t in getattr(ev.w, "over_payload_s", {}).items()}
    rated = P.FR3_PAYLOAD_KG - P.FRANKA_HAND_KG
    return dict(name=f"seconds each arm carried more than its rated payload ({rated:g} kg beyond its hand)",
                seconds_over_rating=secs, allowance_s=P.PAYLOAD_OVER_S,
                within_rating=not any(t > P.PAYLOAD_OVER_S for t in secs.values()),
                note="reported, not scored: this task is judged on its work product, not on how the tools were "
                     "carried")


def ledger_marks(ev):
    """Each mark where render.py draws it, on the ledger's front face, in the ledger's own frame (from its start
    pose, so a mark moves with the ledger), and that mark's present place in the wall's frame."""
    hm = ev.w.hm
    lg, wall = hm.drillables["ledger"], hm.drillables["wall"]
    p0, R0 = ev.s["start_pose"]["ledger"]
    marks_l, marks_w = [], []
    for mk in ev.L["marks"]:
        m = np.asarray(R0).T @ (np.array([mk["x"], P.WALL_FACE_Y, mk["z"]]) - np.asarray(p0))
        m[1] = float(lg.center[1] - lg.half[1])
        marks_l.append(m)
        marks_w.append(hm.local(wall, hm.world_point(lg, m)))
    return marks_l, marks_w


def washer_gap(ev, a, lg):
    """How far an anchor's washer bearing face stands off the ledger's front face (m; negative: pressed into it).
    The bearing face's centre as parts.WedgeAnchor finds it: the washer link plus half its thickness along the
    stud towards the tip."""
    w = ev.w
    part = w.scene.parts[a.name]
    _, Rs = w.body_pose(part["body"])
    pw, _ = w.body_pose(part["washer"])
    c = pw + (Rs @ np.array([1., 0., 0.]))*0.5*P.WASHER_T_M
    pl, Rl = w.body_pose(lg.body)
    n = Rl @ np.array([0., -1., 0.])
    face = pl + Rl @ np.array([0., float(lg.center[1] - lg.half[1]), 0.])
    return float((c - face) @ n)


def ledger_anchor_in(ev, k, h, anchors, lg):
    """The anchor checks of mark k's hole h in the concrete, and that anchor's progress terms."""
    w = ev.w
    key = f"wall/{h.id}"
    T = P.KBTZ2["torque"]
    t_lo, t_hi = (1 - P.T_INST_TOL)*T, 1.7*T
    h_lo = P.KBTZ2["h_nom"] - 0.002
    band = f"{t_lo:.0f}-{t_hi:.0f} N m" if SCORE_OVER_TORQUE else f"at least {t_lo:.0f} N m"
    names = (f"hole {k}: anchor driven to h_nom (at least {h_lo*1e3:.0f} mm) with its washer on the ledger",
             f"hole {k}: anchor straight (within 3 deg)", f"hole {k}: nut torqued to T_inst ({band})",
             f"hole {k}: expansion engaged (cone drawn into a gripping clip)",
             f"hole {k}: anchor did not spin (<= {P.ANCHOR_SPIN_MAX_DEG:.0f} deg)")
    clean_note = f"vacuum passes {len(h.vacuum_passes)}, brush strokes {len(h.brush_strokes)}, " \
                 f"dust column {h.dust_column_m()*1e3:.1f} mm"
    here = [a for a in anchors if tuple(a.slots.get("wall") or (None, None, None))[1:] == ("hole", h.id)]
    if not here:
        return [_check(names[0], False, None, f"at least {h_lo*1e3:.0f} mm",
                       note=f"no anchor in this hole (a record left by an anchor that has gone counts for "
                            f"nothing); {clean_note}"),
                _check(names[2], False, None, "no anchor in the hole"),
                _check(names[3], False, None, "no anchor in the hole")], dict(E=0., Q=0.)
    a = max(here, key=lambda x: x.state["insertion_m"].get(key, -1.))
    z = float(a.state["insertion_m"].get(key, 0.))
    z_drive = max(float(a.state.get("insertion_max_m", {}).get(key, z)), z)
    back_out = z < z_drive - 0.003
    gap = washer_gap(ev, a, lg)
    out = [_check(names[0], z_drive >= h_lo and not back_out and gap <= WASHER_ON_FACE_M, round(z_drive*1e3, 2),
                  f"at least {h_lo*1e3:.0f} mm, washer at most {WASHER_ON_FACE_M*1e3:g} mm off the ledger",
                  note=f"{a.name}: driven to {z_drive*1e3:.2f} mm, now at {z*1e3:.2f} mm; washer bearing face "
                       f"{gap*1e3:+.2f} mm off the ledger's face; {clean_note}")]
    _, R = w.body_pose(a.body)
    phi = _angle_deg(R @ np.array([1., 0., 0.]), w.hm.world_dir(w.hm.drillables["wall"], h.a))
    out.append(_check(names[1], phi <= 3., round(phi, 2), "<= 3 deg"))
    st = a.state.get("setting", {})
    here_clip = (a.state.get("clip") or {}).get("key") == key
    tq = float(st.get("max_turn_torque_nm", 0.)) if here_clip else 0.
    tq_ever = max(float(ev.s.get("anchor_max_turn_nm", {}).get(a.name, 0.)),
                  float(st.get("max_turn_torque_nm", 0.)))
    over = tq_ever > t_hi
    out.append(_check(names[2], t_lo <= tq and (not over or not SCORE_OVER_TORQUE), round(tq, 2), band,
                      note=f"largest friction torque the nut was turned on against with the cone gripping; the "
                           f"anchor's largest ever {tq_ever:.1f} N m, over-torque level {t_hi:.0f} N m "
                           f"{'EXCEEDED' if over else 'not reached'}"
                           f"{'' if SCORE_OVER_TORQUE else ' (reported, not scored)'}; stud tension: largest "
                           f"{st.get('max_set_n', 0.):.0f} N, now {st.get('thread_n', 0.):.0f} N"
                           f"{'; ANCHOR FAILED (' + str(st.get('failed')) + ')' if st.get('failed') else ''}"))
    engaged = bool(here_clip and st.get("grip") and st.get("cone_n", 0.) > 0.)
    out.append(_check(names[3], engaged, round(float(st.get("s_m", 0.))*1e3, 3) if here_clip else None,
                      "cone drawn in, clip gripping", note=f"cone load {st.get('cone_n', 0.):.0f} N"))
    spin = float(st.get("spin_max_deg", 0.))
    out.append(_check(names[4], spin <= P.ANCHOR_SPIN_MAX_DEG, round(spin, 2), f"<= {P.ANCHOR_SPIN_MAX_DEG:.0f} deg",
                      kind="constraint"))
    # progress terms: the embedment reached (its present depth if it has been drawn back out); the installation
    # torque counted only while the cone grips in this hole (hammering alone turns no nut: no credit) and only for
    # the embedment reached, falling to nothing 2 mm short of h_lo (a torque record means the
    # washer bears on the ledger and the ev-locking nut has run down the stud, so a nut tightened short cannot be
    # driven further, and a further drive would reset the record: torque recorded short is never part of a finished
    # job); nothing for an anchor that can no longer be set (stud failed, taken past 1.7 T_inst, or spun: no spare
    # anchor, no reverse on the wrench, and the spin record only grows)
    short = _ramp(h_lo - z_drive, 0., 0.002)        # 1 at h_lo and beyond, 0 two millimetres short
    E = min((z if back_out else z_drive)/h_lo, 1.)*_ramp(phi, 3., 6.)*(short if tq > 0. else 1.)
    Q = min(tq/t_lo, 1.)*short if engaged else 0.
    spun = spin > P.ANCHOR_SPIN_MAX_DEG
    spoiled = bool(st.get("failed")) or over or spun
    if spoiled:
        E = Q = 0.
    return out, dict(anchor=a.name, driven_mm=round(z_drive*1e3, 2), now_mm=round(z*1e3, 2),
                     washer_gap_mm=round(gap*1e3, 3), torque_nm=round(tq, 2), largest_torque_ever_nm=round(tq_ever, 2),
                     spoiled=spoiled, spun=spun, set_short=bool(tq > 0. and z_drive < h_lo), E=E, Q=Q)


def ledger_openings(ev, k, marks_l, marks_w, anchors, lg):
    """The openings at mark k: each wall record within MARK_RADIUS_M of the mark's
    place on the wall, with the ledger record in line with it (its axis passing within one bit radius of the wall
    record's entry; a through record first, then the nearest the mark), or alone; and each ledger record within the
    radius that has no wall record in line, alone. The hole module can split one physical opening into several
    records (a bit coming back more than CAPTURE_TOL off a started hole's axis seats a new one), so the mark's bore
    is the opening that carries the most work, not the record nearest the mark. Returns [(ledger rec, offset) or
    None, (wall rec, offset) or None]."""
    w, hm = ev.w, ev.w.hm
    wall = hm.drillables["wall"]
    led = [(h, _in_plane(h, marks_l[k])) for h in ev.holes("ledger")]
    led = [x for x in led if x[1] <= MARK_RADIUS_M]
    wal = [(h, _in_plane(h, marks_w[k])) for h in ev.holes("wall")]
    wal = [x for x in wal if x[1] <= MARK_RADIUS_M]
    out, paired = [], set()
    for hw in wal:
        q = hm.world_point(wall, hw[0].p0)
        inline = []
        for hl in led:
            p, a = hm.world_point(lg, hl[0].p0), hm.world_dir(lg, hl[0].a)
            r = q - p
            if float(np.linalg.norm(r - a*float(r @ a))) <= 0.5e-3*hw[0].d_mm:
                inline.append(hl)
        hl = min(inline, key=lambda x: (not x[0].through, x[1])) if inline else None
        if hl is not None:
            paired.add(hl[0].id)
        out.append((hl, hw))
    out += [(hl, None) for hl in led if hl[0].id not in paired]
    return out


def checks(ev):
    """The job: fasten the ledger, set level on its temporary brackets, to the wall with the four KB-TZ2 anchors
    through it at its marks. All read from physical state; for each mark k the
    verifier judges, and the score credits, the same opening: of the openings at the mark (ledger_openings), the
    one that carries the most work (largest M_k below; ties: through the ledger, then deeper, then nearer):
      - through the ledger, its entry within WALL_MARK_TOL_M of the mark, which is where it is drawn: on the
        ledger's face in the ledger's frame;
      - in the concrete behind it, its axis within WALL_AXIS_TOL_DEG of the wall's normal, 108-125 mm deep to its
        DEEPEST point ('depth of drill hole to deepest point', fischer ETA-19/0520 Table B2.1). 108 mm
        is the anchor's h0; 125 mm is the site's limit, which keeps the hole in front of the far mat's bars, whose
        near surface is at 149.2 mm in this 200 mm wall;
      - the anchor whose clip is in that concrete hole now (a record left by an anchor that has gone counts for
        nothing): driven to at least h_nom - 2 mm (the deepest its tip stayed, params.DRIVEN_DWELL),
        not drawn back out by more than 3 mm, its washer on the ledger (bearing face at most WASHER_ON_FACE_M off
        the ledger's face; no upper bound: driving further needs the timber to crush, which is not modelled);
        straight; the nut turned on against at least 0.9 T_inst while the
        cone gripped and never past 1.7 T_inst over the episode (track_tick); expansion
        engaged; not spun.
    Every other drilled record is one of (MARK_RADIUS_M): a pit shallower than the bit's point; a re-entry (entry
    within one bit radius of the judged opening's record in the same part: the same opening); a
    start in the ledger the anchor's washer covers; a blind start in the ledger elsewhere (reported and charged in
    progress, not failed: surface damage a carpenter fills); or a hole that fails the job (through the
    ledger beside or away from a mark's opening, or in the concrete). The ledger fastened where it was set (centre
    within LEDGER_SHIFT_MAX_M of its start pose in the wall's plane) and still level; no bar cut; the bit
    not chipped. The rated payload is a diagnostic (payload_diagnostic). No cleaning sequence, tool, order or
    hand is judged.

    Progress v3: for each mark M_k = p_k (B_k + L S (E_k + Q_k))/3, each mark
    worth a quarter whatever the others' state; progress = max(0, mean M_k - n/4 - sum c), where
      p_k = ramp(entry offset; 3, 6 mm) x ramp(axis; 3, 6 deg): placement
      B_k = min((ledger bored + concrete depth)/(38 + 108 mm), 1) x ramp(concrete depth; 125 mm, the far mat's
            near surface): the bore, to its deepest point
      E_k = min(driven depth/93 mm, 1) x ramp(anchor tilt; 3, 6 deg), and x ramp(93 mm - driven depth; 0, 2 mm) once
            a torque is recorded (an anchor tightened short can no longer be driven)
      Q_k = min(installation torque/61.2 N m, 1) x ramp(93 mm - driven depth; 0, 2 mm) while the cone grips here
      L   = ramp(ledger drop over its length; 3, 6 mm), S = ramp(ledger offset in the wall's plane; 10, 20 mm): the
            fastening holds the ledger as it was set; bores keep their value
      M_k = 0 if the opening cut a bar; E_k = Q_k = 0 for a spoiled anchor (stud failed, over-torqued or spun)
      n   = bars cut by records that are not a mark's opening; c = each other defect record's share of a bore:
            min(depth, 38 mm)/146 mm/12 in the ledger, min(depth, 108 mm)/146 mm/12 in the concrete (a whole stray
            bore costs one operation's 1/12)
    ramp(x; a, b) is 1 up to the acceptance limit a and falls linearly to 0 at b. Every factor is 1 inside
    acceptance and no cost applies to a success, so a success has progress 1. `damage` lists every record's class
    and cost."""
    w, hm = ev.w, ev.w.hm
    lg = hm.drillables["ledger"]
    anchors = [hm.items[f"anchor{k}"] for k in range(4)]
    marks_l, marks_w = ledger_marks(ev)
    tol, ax_tol, R_mark = WALL_MARK_TOL_M, WALL_AXIS_TOL_DEG, MARK_RADIUS_M
    d_lo, d_hi, t_led = P.KBTZ2["h0"], 0.125, P.LEDGER_THICK_M
    far_mat = P.WALL_THICK_M - P.FAR_MAT_COVER_M - P.REBAR_D_M          # near surface of the far mat's bars
    depth_name = f"depth {d_lo*1e3:.0f}-{d_hi*1e3:.0f} mm into the concrete (to its deepest point)"
    p_lg, R_lg = w.body_pose(w.scene.parts["ledger"]["body"])
    p0_lg = np.asarray(ev.s["start_pose"]["ledger"][0])
    drop = abs(float((R_lg @ np.array([1., 0., 0.]))[2]))*1.2
    shift = float(np.hypot(*(p_lg - p0_lg)[[0, 2]]))                    # in the wall's plane (x along, z up)
    L, S = _ramp(drop, 0.003, 0.006), _ramp(shift, LEDGER_SHIFT_MAX_M, 2*LEDGER_SHIFT_MAX_M)
    out, rows, judged = [], [], {}
    damage = dict(stray_holes=[], extra_holes_at_marks=[], blind_starts=[], misplaced_bores=[], rebar_cut=[],
                  spoiled_anchors=[], spun_anchors=[], set_short=[], pits=[], covered_starts=[], re_entries=[])

    def opening(k, hL, hW):
        """(checks, progress row, M) of one opening at mark k."""
        lab, chk = f"hole {k}", []
        if hL is None:
            chk.append(_check(f"{lab}: through the ledger at the mark", False, None,
                              note=f"no hole in the ledger within {R_mark*1e3:.0f} mm of the mark"))
        else:
            chk.append(_check(f"{lab}: through the ledger at the mark", hL[0].through,
                              "through" if hL[0].through else f"{hL[0].depth*1e3:.1f} mm deep"))
            chk.append(_check(f"{lab}: within {tol*1e3:.0f} mm of the mark on the ledger", hL[1] <= tol,
                              round(hL[1]*1e3, 2), f"<= {tol*1e3:.0f} mm"))
        terms = dict(E=0., Q=0.)
        if hW is None:
            chk.append(_check(f"{lab}: {depth_name}", False, None, f"{d_lo*1e3:.0f}-{d_hi*1e3:.0f} mm",
                              note="no hole in the concrete behind the mark"))
            chk.append(_check(f"{lab}: anchor driven to h_nom (at least {(P.KBTZ2['h_nom'] - 0.002)*1e3:.0f} mm) "
                              f"with its washer on the ledger", False, None, note="no hole"))
        else:
            h = hW[0]
            ang = _angle_deg(h.a, -h.face_normal)
            chk.append(_check(f"{lab}: axis within {ax_tol:.0f} deg of the wall normal", ang <= ax_tol, round(ang, 2),
                              f"<= {ax_tol:.0f} deg"))
            chk.append(_check(f"{lab}: {depth_name}", d_lo <= h.depth <= d_hi, round(h.depth*1e3, 1),
                              f"{d_lo*1e3:.0f}-{d_hi*1e3:.0f} mm", note=f"full-diameter depth {h.full_depth()*1e3:.1f} mm"))
            o, terms = ledger_anchor_in(ev, k, h, anchors, lg)
            chk += o
        row, M, key = dict(mark=k, ledger_hole=None if hL is None else hL[0].id,
                           wall_hole=None if hW is None else hW[0].id), 0., (0., False, 0., 0.)
        if hL is not None or hW is not None:
            e = hL[1] if hL is not None else hW[1]
            hx = hW[0] if hW is not None else hL[0]
            th = _angle_deg(hx.a, -hx.face_normal)
            p = _ramp(e, tol, 2*tol)*_ramp(th, ax_tol, 2*ax_tol)
            ell = 0. if hL is None else (t_led if hL[0].through else min(hL[0].depth, t_led))
            z = 0. if hW is None else hW[0].depth
            B = min((ell + z)/(t_led + d_lo), 1.)*_ramp(z, d_hi, far_mat)
            cut = hW is not None and hW[0].rebar_damage_m > 0.0005
            M = 0. if cut else p*(B + L*S*(terms["E"] + terms["Q"]))/3.
            row.update(entry_offset_mm=round(e*1e3, 2), axis_deg=round(th, 2), placement=round(p, 4),
                       ledger_bored_mm=round(ell*1e3, 1), concrete_depth_mm=round(z*1e3, 1), bore=round(B, 4),
                       cut=cut, misplaced=p == 0.)
            key = (round(M, 9), hL is not None and hL[0].through, ell + z, -e)
        row.update({x: (round(v, 4) if isinstance(v, float) else v) for x, v in terms.items()}, M=M)
        return chk, row, key

    for k in range(len(marks_l)):
        cands = [opening(k, hL, hW) for hL, hW in ledger_openings(ev, k, marks_l, marks_w, anchors, lg)]
        chk, row, _ = max(cands, key=lambda c: c[2]) if cands else opening(k, None, None)
        out += chk
        judged[k] = (None if row["ledger_hole"] is None else next(h for h in ev.holes("ledger") if h.id == row["ledger_hole"]),
                     None if row["wall_hole"] is None else next(h for h in ev.holes("wall") if h.id == row["wall_hole"]))
        if row.pop("misplaced", False):
            damage["misplaced_bores"].append(k)
        if row.pop("cut", False):
            damage["rebar_cut"].append(dict(mark=k, hole=row["wall_hole"]))
        for flag, dk in (("spoiled", "spoiled_anchors"), ("spun", "spun_anchors"), ("set_short", "set_short")):
            if row.get(flag):
                damage[dk].append(row["anchor"])
        rows.append(row)
    walls = [h.rebar_damage_m for h in ev.holes("wall")]
    out.append(_check("no reinforcement cut (damage at most 0.5 mm in every hole)", max(walls, default=0.) <= 0.0005,
                      round(max(walls, default=0.)*1e3, 3), "<= 0.5 mm", kind="constraint"))
    # every other drilled record
    n_cut, cost, cut_at = 0, 0., []          # wall entries (world) of bars cut off a mark's opening
    for name, pts, side, full in (("wall", marks_w, 1, d_lo), ("ledger", marks_l, 0, t_led)):
        for h in ev.holes(name):
            if any(j[side] is h for j in judged.values()):
                continue
            dists = [_in_plane(h, m) for m in pts]
            k = int(np.argmin(dists))
            rec = dict(workpiece=name, hole=h.id, from_nearest_mark_mm=round(dists[k]*1e3, 1), mark=k,
                       depth_mm=round(h.depth*1e3, 2), through=h.through)
            if h.full_depth() <= 0.:
                damage["pits"].append(rec)
                continue
            bore = judged[k][side] if dists[k] <= R_mark else None
            if bore is not None:
                gap = _in_plane(h, bore.p0)
                rec["from_bore_mm"] = round(gap*1e3, 1)
                if gap <= 0.5e-3*h.d_mm:
                    damage["re_entries"].append(rec)
                    continue
                lbore = judged[k][0]
                if side == 0 and lbore is not None and _in_plane(h, lbore.p0) + 0.5e-3*h.d_mm <= P.WASHER_R_M:
                    damage["covered_starts"].append(rec)
                    continue
            inline_cut = side == 0 and any(
                float(np.linalg.norm((q - hm.world_point(lg, h.p0)) - hm.world_dir(lg, h.a)*float(
                    (q - hm.world_point(lg, h.p0)) @ hm.world_dir(lg, h.a)))) <= 0.5e-3*h.d_mm for q in cut_at)
            if side == 1 and h.rebar_damage_m > 0.0005:
                n_cut += 1
                rec["cost"] = 0.25
                cut_at.append(hm.world_point(hm.drillables["wall"], h.p0))
            elif inline_cut:
                rec["cost"] = 0.                     # the bore that cut a bar costs its quarter once
            else:
                rec["cost"] = round(min(h.depth, full)/(t_led + d_lo)/12., 5)
                cost += rec["cost"]
            if side == 0 and not h.through:
                damage["blind_starts"].append(rec)
            else:
                damage["extra_holes_at_marks" if bore is not None else "stray_holes"].append(rec)
    bad = len(damage["stray_holes"]) + len(damage["extra_holes_at_marks"])
    out.append(_check("no hole drilled away from the marks or beside a mark's bore", not bad, bad, "0",
                      kind="constraint", note=f"{len(damage['stray_holes'])} away from every mark, "
                                              f"{len(damage['extra_holes_at_marks'])} beside a mark's bore; not "
                                              f"failing: {len(damage['blind_starts'])} blind starts in the ledger, "
                                              f"{len(damage['pits'])} pits, {len(damage['covered_starts'])} starts "
                                              f"under a washer, {len(damage['re_entries'])} re-entries"))
    out.append(_check(f"ledger fastened where it was set (moved at most {LEDGER_SHIFT_MAX_M*1e3:.0f} mm in the "
                      f"wall's plane)", shift <= LEDGER_SHIFT_MAX_M, round(shift*1e3, 2),
                      f"<= {LEDGER_SHIFT_MAX_M*1e3:.0f} mm", kind="constraint"))
    out.append(_check("ledger still level within 3 mm over its length", drop <= 0.003, round(drop*1e3, 2), "<= 3 mm",
                      kind="constraint"))
    out.append(ev._bit_not_chipped())
    damage.update(ledger_drop_mm=round(drop*1e3, 2), ledger_offset_mm=round(shift*1e3, 2), bars_cut_off_bore=n_cut,
                  defect_cost=round(cost, 5),
                  bit_chipped=[n for n, it in hm.items.items() if it.kind == "bit" and it.state["chipped"]],
                  bit_broken=[n for n, it in hm.items.items() if it.kind == "bit" and it.state["broken"]])
    progress = max(0., sum(r["M"] for r in rows)/len(rows) - 0.25*n_cut - cost)
    for r in rows:
        r["M"] = round(r["M"], 4)
    return out, dict(progress=round(progress, 4), scoring=LEDGER_SCORING,
                     progress_terms=dict(level_factor=round(L, 4), position_factor=round(S, 4), marks=rows),
                     damage=damage)
