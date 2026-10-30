"""Dock leveler, trailer and their support forces.

Trailer: a 53 ft dry van (Wabash DuraPlate HD inside dimensions, read) as one free rigid body — oak floor (a drivable
surface for the tyre model and a native collider for loads), side walls, nose wall, roof, rear frame — carried by four
support elements computed here every tick: the tandem axle group (left/right: air spring and damper, braked tyres with
presliding stick and Coulomb slip, chocks as one-sided stops) and the landing gear (left/right: stiff legs, sand shoes
with friction). An optional rotating-hook restraint is a one-sided stop at the rear impact guard. Trailer creep is not
written anywhere: it emerges when the forklift's braking pushes the trailer harder than its tyres and shoes hold.
Trailer frame: x from the rear sill toward the nose, y left, z up, origin on the floor top at the rear sill centre.

Leveler: a hinged deck (revolute about the dock-face-parallel axis at its rear hinge) and a hinged lip resting on the
trailer floor; deployed before the episode and then floating with the trailer (a velocity-fuse stop limits the deck's
free fall), as hydraulic levelers do. Deck and lip are drivable surfaces (tyre model) and collide natively with the
trailer floor and objects (group 'deck').
"""

import math

import numpy as np

from . import params as P
from .mathx import quat_to_R, cross3


def _q_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def add_trailer(scene, name, rear, yaw, bed_z, ground_z, restraint=False, chocks=True, stands=False):
    """A trailer whose rear sill centre sits at `rear` (x, y), floor top at bed_z, nose toward `yaw`, its tyres and
    landing gear on the ground at ground_z; stands: the two stabilizing jacks under its nose (P.TRAILER_STAND)."""
    wp, b = scene.wp, scene.b
    Ti = P.TRAILER_INSIDE
    Tp = P.TRAILER
    L, W, H, fl = Ti["L"], Ti["W"], Ti["H"], Ti["floor"]
    wall, sub = Tp["wall"], Tp["subframe"]
    Wo = Tp["outside_W"]
    m = Tp["mass"]
    # mass distribution: floor+subframe 45 %, walls 35 %, roof 8 %, nose 4 %, rear frame/doors 8 %
    com = np.array([L * P.TRAILER_BODY["cg_x_fraction"], 0.0, P.TRAILER_BODY["cg_z"]])
    h = P.TRAILER_BODY["inertia_height"]
    I = np.diag([m * (Wo ** 2 + h ** 2) / 12, m * (L ** 2 + h ** 2) / 12, m * (L ** 2 + Wo ** 2) / 12])
    body = b.add_link(xform=wp.transform((rear[0], rear[1], bed_z), wp.quat(*_q_yaw(yaw))), mass=m, com=tuple(com),
                      inertia=wp.mat33(*I.ravel()), lock_inertia=True, label=f"trailer/{name}")
    j = b.add_joint_free(body, label=f"trailer/{name}")
    b.add_articulation([j], label=f"trailer/{name}")
    cfg = scene.cfg(mu=P.MU_STRUCTURE)

    def box(tag, c, h, group="obj", material="trailer_wall"):
        from .scene import box_shape
        s = box_shape(b, scene.newton, body, wp.transform(tuple(map(float, c)), wp.quat_identity()), h, cfg,
                      f"{name}:{tag}")
        scene.shape_group[s] = group
        scene.materials[s] = material
        # the floor and the rear sill (flush with it) are driven on; walls, nose, roof, posts and header are struck
        scene.statics.setdefault("trailer_floor" if tag in ("floor", "rear_sill") else "trailer_wall", []).append(s)
        return s
    box("floor", (L / 2, 0.0, -sub / 2), (L / 2, W / 2, sub / 2), group="drivable", material="oak")
    box("wall_l", (L / 2, W / 2 + wall / 2, H / 2), (L / 2, wall / 2, H / 2))
    box("wall_r", (L / 2, -W / 2 - wall / 2, H / 2), (L / 2, wall / 2, H / 2))
    box("nose", (L + wall / 2, 0.0, H / 2), (wall / 2, Wo / 2, H / 2))
    box("roof", (L / 2, 0.0, H + 0.03), (L / 2, Wo / 2, 0.03))
    box("rear_post_l", (-0.05, W / 2 + 0.04, H / 2), (0.05, 0.05, H / 2))
    box("rear_post_r", (-0.05, -W / 2 - 0.04, H / 2), (0.05, 0.05, H / 2))
    box("rear_header", (-0.05, 0.0, H + 0.08), (0.05, Wo / 2, 0.08))
    # the rear sill is flush with the floor: driven over like it (the tyre model carries the wheels on the floor and the
    # leveler lip; as an obj-group solid it met the wheel cylinders as a 50 kN bump, E15 unload-trailer)
    box("rear_sill", (-0.05, 0.0, -0.10), (0.05, Wo / 2, 0.10), group="drivable")
    box("icc_bar", (0.25, 0.0, -0.55), (0.04, 1.1, 0.05))           # rear impact guard (restraint hooks it)
    # floor for the tyre model (oak): the whole inside floor
    scene.surfaces.append(_surface(f"{name}:floor", body, (L / 2, 0.0, 0.0), (L / 2, W / 2), "oak", 2))
    tr = dict(body=body, name=name, rear=tuple(rear), yaw=yaw, bed_z=bed_z, ground_z=ground_z, restraint=restraint,
              chocks=chocks, stands=stands, L=L, W=W, H=H)
    scene.trailers[name] = tr
    return tr


def _surface(name, body, origin, half, material, priority):
    from .tyres import Surface
    return Surface(name, body, np.asarray(origin, float), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
                   tuple(half), material, priority)


LIP_T = P.LEVELER_LIP_PLATE["thickness"]
LIP_EXTENDED = math.radians(P.LEVELER_LIP_PLATE["extended_stop_deg"])  # the extended lip's stop relative to the deck (lip slightly below in-line; estimate)


def add_leveler(scene, name, x_centre, dock_y, bed_z, deployed=True):
    """Leveler at a door: deck from the rear hinge (dock_y - length) to the dock face, lip beyond; dock floor z = 0.
    deployed: the lip extended onto the trailer floor at bed_z (joint values set by initial_q, then settled by
    physics); else stowed: deck level with the floor, lip hanging in front of the dock face."""
    wp, b = scene.wp, scene.b
    Lv = P.LEVELER
    Wd, Ld, Ll = Lv["width"], Lv["length"], Lv["lip"]
    md, ml = P.LEVELER_MASS["deck"], P.LEVELER_MASS["lip"]
    t = 0.10                                                          # deck plate + stiffeners (collider thickness)
    hinge = np.array([x_centre, dock_y - Ld, 0.0])
    if deployed:
        # the lip at its extended stop (the lip cylinder holds it; LIP_EXTENDED to the deck), the deck's grade such
        # that the lip's underside is 3 mm above the bed at its tip
        rel = LIP_EXTENDED
        grade = 0.0
        for _ in range(20):
            h_edge = bed_z + LIP_T + 0.003 - Ll * math.sin(grade + rel)
            grade = math.asin(max(-0.2, min(0.2, h_edge / Ld)))
        q_deck, q_lip = grade, rel
    else:
        q_deck, q_lip = 0.0, math.radians(-85.0)
    deck = b.add_link(xform=wp.transform(tuple(hinge), wp.quat_identity()), mass=md, com=(0.0, Ld / 2, -t / 2),
                      inertia=wp.mat33(*np.diag([md * (Ld ** 2) / 12, md * (Wd ** 2) / 12, md * (Wd ** 2 + Ld ** 2) / 12]).ravel()),
                      lock_inertia=True, label=f"leveler/{name}_deck")
    # a stowed deck rests level on its cross-traffic support legs; free below 0 it hangs 5.7 deg nose-down from its
    # hinge with its rear-bottom edge wedged ~99 kN against the dock floor slab (E0)
    jd = b.add_joint_revolute(-1, deck, parent_xform=wp.transform(tuple(hinge), wp.quat_identity()),
                              child_xform=wp.transform_identity(), axis=(1.0, 0.0, 0.0),
                              limit_lower=math.radians(-P.LEVELER_TRAVEL["deck_deg"]) if deployed else 0.0,
                              limit_upper=math.radians(P.LEVELER_TRAVEL["deck_deg"]),
                              label=f"leveler/{name}_deck",
                              friction=P.LEVELER_HINGE_FRICTION["deck"])
    lip = b.add_link(xform=wp.transform(tuple(hinge + np.array([0.0, Ld, 0.0])), wp.quat_identity()), mass=ml,
                     com=(0.0, Ll / 2, -0.02),
                     inertia=wp.mat33(*np.diag([ml * Ll ** 2 / 12, ml * Wd ** 2 / 12, ml * Wd ** 2 / 12]).ravel()),
                     lock_inertia=True, label=f"leveler/{name}_lip")
    # the lip pivots down to pendant (-95 deg) but not past its extended stop: free to 15 deg, it lies on the bed as a
    # pinned strut 2 deg from horizontal (a vertical load there needs ~29x as much horizontal force), creeps 13 mm into
    # the trailer in 26 s at rest and shoves the trailer and its pallets (E0)
    jl = b.add_joint_revolute(deck, lip, parent_xform=wp.transform((0.0, Ld, 0.0), wp.quat_identity()),
                              child_xform=wp.transform_identity(), axis=(1.0, 0.0, 0.0),
                              limit_lower=math.radians(P.LEVELER_TRAVEL["lip_pendant_deg"]),
                              limit_upper=LIP_EXTENDED if deployed else math.radians(P.LEVELER_TRAVEL["lip_stowed_up_deg"]),
                              label=f"leveler/{name}_lip",
                              friction=P.LEVELER_HINGE_FRICTION["lip"])
    b.add_articulation([jd, jl], label=f"leveler/{name}")
    cfg = scene.cfg(mu=P.MU_STRUCTURE)
    from .tyres import Surface
    for body, tag, c, h in ((deck, "deck", (0.0, Ld / 2, -t / 2), (Wd / 2, Ld / 2, t / 2)),
                            (lip, "lip", (0.0, Ll / 2, -LIP_T / 2), (Wd / 2 - 0.02, Ll / 2, LIP_T / 2))):
        from .scene import box_shape
        s = box_shape(b, scene.newton, body, wp.transform(c, wp.quat_identity()), h, cfg, f"{name}:{tag}")
        scene.shape_group[s] = "deck"
        scene.materials[s] = "steel"
        scene.surfaces.append(Surface(f"{name}:{tag}", body, np.array([0.0, c[1], 0.0]), np.array([0.0, 0.0, 1.0]),
                                      np.array([0.0, 1.0, 0.0]), (h[1], h[0]), "steel", 3))
    scene.levelers[name] = dict(deck=deck, lip=lip, joints=(jd, jl), q=(q_deck, q_lip), x=x_centre, dock_y=dock_y,
                                deployed=deployed, grade=q_deck, width=Wd, length=Ld)
    return scene.levelers[name]


def add_ramp_lip(scene, name, hinge, width, yaw, bed_z, rests_on):
    """The yard ramp's multi-piece hinged lip (P.YARD_RAMP_LIP): plates pinned along the level-off's front edge, the pin
    axis at `hinge` (x, y, z of the deck top), each plate extending along the frame's +y turned by `yaw` and free about
    its own pin between its down stop (the tip 3 in below in-line) and upright. Lowered onto the trailer floor (bed_z)
    before the episode, the plates then rest on it by contact as the trailer sinks and rises; their tips' weight is
    part of the trailer's static load (TrailerSupports._rest)."""
    wp, b = scene.wp, scene.b
    Lp = P.YARD_RAMP_LIP
    Ll, t, n = Lp["length"], Lp["plate"], Lp["pieces"]
    wpc = (width - (n - 1) * Lp["piece_gap"]) / n
    m = P.STEEL_DENSITY * Ll * wpc * t
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    down = -math.asin(Lp["travel"] / Ll)
    # resting: the plate's underside at its tip on the floor (the plate's top is flush with the deck at the pin)
    q0 = max(down, math.asin(max(-1.0, min(1.0, (bed_z - hinge[2] + t + 0.002) / Ll))))
    cfg = scene.cfg(mu=P.MU_STRUCTURE)
    from .tyres import Surface
    from .scene import box_shape
    out = dict(bodies=[], joints=[], q=[], mass=m, length=Ll, trailer=rests_on, hinge=tuple(hinge), yaw=yaw, width=width)
    for k in range(n):
        x_loc = -width / 2 + wpc / 2 + k * (wpc + Lp["piece_gap"])
        p = np.asarray(hinge, float) + Rz @ np.array([x_loc, 0.0, 0.0])
        body = b.add_link(xform=wp.transform(tuple(p), wp.quat(*_q_yaw(yaw))), mass=m, com=(0.0, Ll / 2, -t / 2),
                          inertia=wp.mat33(*np.diag([m * (Ll ** 2 + t ** 2) / 12, m * (wpc ** 2 + t ** 2) / 12,
                                                     m * (wpc ** 2 + Ll ** 2) / 12]).ravel()),
                          lock_inertia=True, label=f"ramp_lip/{name}_{k}")
        j = b.add_joint_revolute(-1, body, parent_xform=wp.transform(tuple(p), wp.quat(*_q_yaw(yaw))),
                                 child_xform=wp.transform_identity(), axis=(1.0, 0.0, 0.0), limit_lower=down,
                                 limit_upper=math.pi / 2, label=f"ramp_lip/{name}_{k}",
                                 friction=Lp["hinge_friction"])
        b.add_articulation([j], label=f"ramp_lip/{name}_{k}")
        sh = box_shape(b, scene.newton, body, wp.transform((0.0, Ll / 2, -t / 2), wp.quat_identity()), (wpc / 2, Ll / 2, t / 2),
                       cfg, f"{name}:lip{k}")
        scene.shape_group[sh] = "deck"
        scene.materials[sh] = "steel"
        scene.surfaces.append(Surface(f"{name}:lip{k}", body, np.array([0.0, Ll / 2, 0.0]), np.array([0.0, 0.0, 1.0]),
                                      np.array([0.0, 1.0, 0.0]), (Ll / 2, wpc / 2), "steel", 3))
        out["bodies"].append(body)
        out["joints"].append(f"ramp_lip/{name}_{k}")
        out["q"].append(q0)
    scene.ramp_lips[name] = out
    return out


def _rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


class TrailerSupports:
    """Per-tick support forces of each trailer: tandem air springs + braked tyres (+ chocks), landing gear legs +
    sand shoes, optional restraint hook. Each contact point keeps a presliding anchor (stick spring capped at mu N);
    when the cap is exceeded the anchor slides with the point — the trailer walks."""

    def __init__(self, world):
        self.w = world
        self.items = []
        Tp = P.TRAILER
        for tr in world.scene.trailers.values():
            pts = []
            for side, y in (("l", Tp["track"] / 2), ("r", -Tp["track"] / 2)):
                pts.append(dict(kind="tandem", local=np.array([Tp["tandem_from_rear"], y, -1.0]), side=side))
                pts.append(dict(kind="gear", local=np.array([Tp["gear_from_rear"], y * P.GEAR_TRACK_RATIO, -1.0]), side=side))
                if tr.get("stands"):
                    S = P.TRAILER_STAND
                    pts.append(dict(kind="stand", local=np.array([tr["L"] - S["from_nose"], (S["y"] if side == "l" else -S["y"]), -1.0]),
                                    side=side))
            self.items.append(dict(tr=tr, pts=pts, anchors=[None] * len(pts), rest=None, hook_anchor=None,
                                   chock_ref=[None] * len(pts)))
        self.energy = 0.0
        self.tele = {}

    def step(self, world, wrench):
        bq = world.state.body_q.numpy().astype(float)
        bqd = world.state.body_qd.numpy().astype(float)
        com = world._com_world(bq)
        dt = world.config.dt
        for it in self.items:
            tr = it["tr"]
            b = tr["body"]
            R = quat_to_R(bq[b:b + 1, 3:7])[0]
            o = bq[b, :3]
            v, om = bqd[b, :3], bqd[b, 3:]
            gz = tr["ground_z"]
            if it["rest"] is None:
                # rest lengths: the spotted bed height under the empty trailer's weight (static share per support)
                it["rest"] = self._rest(it, R, o, com[b])
            F_tot = np.zeros(3); M_tot = np.zeros(3)
            fx_dir = R[:, 0]
            for k, pt in enumerate(it["pts"]):
                p_attach = o + R @ np.array([pt["local"][0], pt["local"][1], 0.0])     # under the floor at the axle line
                p_ground = np.array([p_attach[0], p_attach[1], gz])
                vp = v + cross3(om, p_attach - com[b])
                h = p_attach[2] - gz                                                 # support height
                if pt["kind"] == "tandem":
                    k_v, c_v, mu = P.TRAILER_K, P.TRAILER_C, P.MU_TYRE_SPRINGBRAKE
                elif pt["kind"] == "gear":
                    k_v, c_v, mu = P.GEAR_K, P.GEAR_C, P.MU_SAND_SHOE
                else:                                   # stabilizing jack: touching at the start, pushes only when pressed
                    k_v, c_v, mu = P.TRAILER_STAND["k"], P.GEAR_C, P.MU_SAND_SHOE
                comp = it["rest"][k] - h
                if pt["kind"] == "tandem" and P.TRAILER_ARM_SLOPE:
                    # trailing-arm air suspension: as the air spring compresses beyond its static set the arm rotates
                    # about its front hanger and the axle swings rearward relative to the body (dx = -slope dcomp);
                    # with the wheels braked or chocked the body is pushed toward the nose: dock walk
                    dcomp = comp - it["comp0"][k]
                    shift = R @ np.array([-P.TRAILER_ARM_SLOPE * dcomp, 0.0, 0.0])
                    p_ground = p_ground + np.array([shift[0], shift[1], 0.0])
                    vp = vp + R @ np.array([P.TRAILER_ARM_SLOPE * vp[2], 0.0, 0.0])     # d(dx)/dt = slope dh/dt
                # every support pushes only while it touches the ground (compressed), so no damper pushes the trailer
                # up through the air as it falls
                Fz = max(0.0, k_v * comp - c_v * vp[2]) if comp > 0.0 else 0.0
                if pt["kind"] == "stand" and Fz == 0.0:
                    it["anchors"][k] = None                 # lifted off its pad: no friction memory
                    continue
                # presliding stick at the contact, capped at mu Fz (Coulomb)
                if it["anchors"][k] is None:
                    it["anchors"][k] = p_ground[:2].copy()
                d = p_ground[:2] - it["anchors"][k]
                Ft = -P.STICK_K * d - 2 * P.STICK_DAMPING["zeta"] * math.sqrt(P.STICK_K * P.STICK_DAMPING["mass"]) * vp[:2]
                cap = mu * Fz
                nt = math.hypot(*Ft)
                if nt > cap and nt > 0:
                    Ft *= cap / nt
                    it["anchors"][k] = p_ground[:2] + Ft / P.STICK_K       # slipping: the anchor is dragged along
                    self.energy += cap * math.hypot(*vp[:2]) * dt
                F = np.array([Ft[0], Ft[1], Fz])
                # chocks on the tandem: one-sided stop against motion toward the nose (away from the dock). A chock is
                # fixed to the ground: the tyre's travel is measured from where it stood, along the axis the trailer
                # had when it was chocked (measuring the absolute position along the trailer's current axis made a
                # yaw of 0.02 deg read as 3 mm, 20 m from the origin); an overloaded chock slides
                if pt["kind"] == "tandem" and tr["chocks"]:
                    if it["chock_ref"][k] is None:
                        a0 = fx_dir[:2] / max(np.linalg.norm(fx_dir[:2]), 1e-9)
                        it["chock_ref"][k] = np.array([p_ground[0], p_ground[1], a0[0], a0[1]])
                    ref = it["chock_ref"][k]
                    s = float((p_ground[:2] - ref[:2]) @ ref[2:])
                    if s > 0:
                        Fc = min(P.CHOCK["k"] * s, P.CHOCK["capacity"])
                        if P.CHOCK["k"] * s > P.CHOCK["capacity"]:
                            ref[:2] += (s - P.CHOCK["capacity"] / P.CHOCK["k"]) * ref[2:]
                        F[:2] -= Fc * ref[2:]
                F_tot += F
                M_tot += cross3(p_ground - com[b], F)
            if tr["restraint"]:
                # the hook is fixed to the dock wall: the ICC bar's travel from where it was hooked, along the dock's
                # normal (the trailer's axis when hooked)
                p_icc = o + R @ np.array([0.25, 0.0, -0.55])
                if it["hook_anchor"] is None:
                    a0 = fx_dir[:2] / max(np.linalg.norm(fx_dir[:2]), 1e-9)
                    it["hook_anchor"] = np.array([p_icc[0], p_icc[1], a0[0], a0[1]])
                ha = it["hook_anchor"]
                s = float((p_icc[:2] - ha[:2]) @ ha[2:])
                if s > 0:
                    Fh = min(P.RESTRAINT["k"] * s, P.RESTRAINT["capacity"])
                    Fv = np.array([-Fh * ha[2], -Fh * ha[3], 0.0])
                    F_tot += Fv
                    M_tot += cross3(p_icc - com[b], Fv)
            wrench[b, :3] += F_tot
            wrench[b, 3:] += M_tot
            self.tele[tr["name"]] = dict(F=F_tot.tolist(), x_from_spot=float((o[:2] - np.asarray(tr["rear"])) @ fx_dir[:2]))

    def _rest(self, it, R, o, com):
        """Rest lengths so the trailer stands at its spotted floor height under its static load: its own weight, every
        load standing in it, and the deployed leveler's share (deck and lip as one beam pinned at the deck's rear hinge,
        resting on the bed at the lip's tip). Built for the empty trailer only, a trailer at a deployed leveler sank
        about 10 mm in 2-3 mm stick-slip steps over 20 s after the construction settle had declared it at rest."""
        tr = it["tr"]
        w = self.w
        xt, xg = P.TRAILER["tandem_from_rear"], P.TRAILER["gear_from_rear"]
        loads = [(P.TRAILER["mass"] * P.G, float((com - o) @ R[:, 0]))]
        bq = w.state.body_q.numpy().astype(float)
        bcom = w._com_world(bq)
        L_, W_ = P.TRAILER_INSIDE["L"], P.TRAILER_INSIDE["W"]
        for name, info in w.scene.loads.items():
            if name == "carried" or name.startswith("carried_"):
                continue                        # on the truck's forks: carried by the truck's tyres, whose load the trailer
                                                # feels through them (a pallet counted here would lift the trailer 5 mm
                                                # once the truck drives off with it)
            loc = R.T @ (bcom[info["body"]] - o)
            if 0.0 < loc[0] < L_ and abs(loc[1]) < W_ / 2 and -0.1 < loc[2] < 3.0:
                loads.append((float(w.body_mass[info["body"]]) * P.G, float(loc[0])))
        lv = w.scene.levelers.get(tr["name"])
        if lv is not None and lv["deployed"]:
            Lv, Ld = P.LEVELER, lv["length"]
            md, ml = P.LEVELER_MASS["deck"], P.LEVELER_MASS["lip"]
            span = Ld + Lv["lip"]
            R_tip = P.G * (md * Ld / 2 + ml * (Ld + Lv["lip"] / 2)) / span
            tip = np.array([lv["x"], lv["dock_y"] + Lv["lip"], o[2]])
            loads.append((R_tip, float((tip - o) @ R[:, 0])))
        for lip in w.scene.ramp_lips.values():
            if lip["trailer"] == tr["name"]:
                # each plate pinned at the ramp and resting at its tip: half its weight on the floor
                c_, s_ = math.cos(lip["yaw"]), math.sin(lip["yaw"])
                tip = np.asarray(lip["hinge"], float) + np.array([-s_, c_, 0.0]) * lip["length"]
                loads.append((len(lip["bodies"]) * lip["mass"] * P.G / 2, float((tip - o) @ R[:, 0])))
        m = sum(f for f, _ in loads)
        Wg = sum(f * (x - xt) for f, x in loads) / (xg - xt)
        Wt = m - Wg
        it["static"] = dict(loads=[(round(f), round(x, 3)) for f, x in loads], tandem_N=Wt, gear_N=Wg)
        rest = []
        it["comp0"] = [(Wt / 2) / P.TRAILER_K if pt["kind"] == "tandem" else 0.0 for pt in it["pts"]]
        for pt in it["pts"]:
            p_attach = o + R @ np.array([pt["local"][0], pt["local"][1], 0.0])
            h = p_attach[2] - tr["ground_z"]
            if pt["kind"] == "tandem":
                rest.append(h + (Wt / 2) / P.TRAILER_K)
            elif pt["kind"] == "gear":
                rest.append(h + (Wg / 2) / P.GEAR_K)
            else:
                rest.append(h)                              # cranked up to touch, no preload
        return rest

    def checkpoint(self):
        import copy
        return copy.deepcopy([dict(anchors=it["anchors"], rest=it["rest"], hook=it["hook_anchor"], chock=it["chock_ref"],
                                   comp0=it.get("comp0")) for it in self.items])

    def restore(self, c):
        for it, s in zip(self.items, c):
            import copy                                  # the chock references are updated in place: never share them
            s = copy.deepcopy(s)
            it["anchors"], it["rest"], it["hook_anchor"], it["chock_ref"] = s["anchors"], s["rest"], s["hook"], s["chock"]
            if s.get("comp0") is not None:
                it["comp0"] = s["comp0"]
