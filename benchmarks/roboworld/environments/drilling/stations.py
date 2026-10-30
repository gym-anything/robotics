"""Station builders: the concrete wall bay (preset `wall`), the drill-press bench (preset `bench`) and probe benches
(preset `probe`, ASSISTED experiments only). Positions come from the task layout (layout.py)."""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import tools as T


def _pose(spec):
    p = np.asarray(spec["xyz"], float)
    R = Rotation.from_euler("xyz", spec.get("rpy", (0., 0., 0.)))
    return p, R


def build_probe(sc):
    """Probe bench: static boxes, drillable workpieces and ASSISTED phantom rigs listed in the layout."""
    L = sc.layout
    for s in L.get("statics", []):
        sc.box(s["name"], s["xyz"], s["half"], s.get("material", "steel_dark"),
               rot=T.wq(Rotation.from_euler("xyz", s.get("rpy", (0., 0., 0.)))), mu=s.get("mu", P.MU["default"]))
    for d in L.get("drillables", []):
        fields, fmap = None, None
        if d.get("concrete_fields") is not None:
            # a concrete block whose face is its local -y face (as the wall): fields frame x along, y depth, z up
            from .concrete import ConcreteFields
            cf = d["concrete_fields"]
            c, h = np.asarray(d["xyz"], float), np.asarray(d["half"], float)
            fields = ConcreteFields(cf["seed"], (c[0] - h[0], c[0] + h[0]), (c[2] - h[2], c[2] + h[2]), thickness=2*h[1],
                                    rebar_offset=tuple(cf.get("rebar_offset", (0., 0.))))
            if cf.get("no_rebar"):
                fields.bars = []
            fmap = (np.array([-c[0], -h[1], -c[2]]), np.eye(3))
        sc.add_drillable(d["name"], d["material"], _pose(d), d["half"], static=d.get("static", True),
                         n_slots=d.get("slots", 3), fields=fields, fields_map=fmap)
    for ph in L.get("phantoms", []):
        sc.add_tool(T.PhantomSpindle(sc, ph["name"], ph["tip_start"], ph["axis"], ph["d_mm"],
                                     bit_len_m=ph.get("bit_len", 0.06), carriage_kg=ph.get("carriage_kg", 2.0),
                                     bit_type=ph.get("bit_type", "hss"), lateral_free=ph.get("lateral_free", False)))
    for t in L.get("tools", []):
        build_tool(sc, t)
    if L.get("parts"):
        from . import parts as PT
        for p_ in L["parts"]:
            PT.build_part(sc, p_)


def _point_cradle(sc, name, x_axis, y_mid, half_l, r_part, z_axis, z_base, gap=0.010, fracs=(-0.7, 0.7)):
    """Two hardwood rails `gap` apart, running along y, under a round part of radius r_part whose axis lies along y at
    (x_axis, z_axis). The part bears on rounded points (spheres standing 0.5 mm proud of each rail's inner top corner),
    not on the rails' edges: a cylinder on a sphere is an analytic contact, a cylinder on a box edge goes through
    MuJoCo's convex collision routine, whose contact points move from tick to tick (params.SOLVER_CCD_TOLERANCE)."""
    r_b = 0.003
    d_b = (r_b - 0.0005)/math.sqrt(2.)
    zc = z_axis - math.sqrt((r_part + r_b)**2 - (0.5*gap + d_b)**2)            # the points' centres
    h_rail = zc + d_b - z_base
    assert h_rail > 0.001, (name, h_rail)
    for k_, sgn_ in enumerate((-1, 1)):
        sc.box(f"{name}{k_}", (x_axis + sgn_*(0.5*gap + 0.004), y_mid, z_base + 0.5*h_rail), (0.004, half_l, 0.5*h_rail),
               "hardwood", mu=P.MU["douglas_fir"])
        for j_, fy_ in enumerate(fracs):
            T.Parts(sc).sphere(-1, f"{name}{k_}_point{j_}", (x_axis + sgn_*(0.5*gap + d_b), y_mid + fy_*half_l, zc), r_b,
                               "collision_hidden", mu=P.MU["douglas_fir"])


def build_tool(sc, t):
    kind = t["kind"]
    if kind == "drill":
        return sc.add_tool(T.CordlessDrill(sc, t["name"], _pose(t), t["bit_d_mm"], t["bit_len"],
                                           bit_type=t.get("bit_type", "hss"), gear=t.get("gear", 1),
                                           clutch=t.get("clutch", 16), wear=t.get("wear", 1.), free=t.get("free", True)))
    if kind == "hammer":
        return sc.add_tool(T.RotaryHammer(sc, t["name"], _pose(t), t["bit_d_mm"], t["bit_len"],
                                          mode=t.get("mode", 1), wear=t.get("wear", 1.), free=t.get("free", True)))
    if kind == "press":
        return sc.add_tool(T.DrillPress(sc, t["name"], t["xy"], t.get("yaw", 0.), t["belt_rpm"], t["bit_d_mm"],
                                        t["bit_len"], depth_stop_m=t.get("depth_stop")))
    if kind == "vise":
        return sc.add_tool(T.Vise(sc, t["name"], _pose(t), t["opening"], bolted=t.get("bolted", True)))
    if kind == "punch":
        return sc.add_tool(T.CentrePunch(sc, t["name"], _pose(t), release_n=t.get("release_n")))
    if kind == "impact_wrench":
        return sc.add_tool(T.ImpactWrench(sc, t["name"], _pose(t), free=t.get("free", True), target_nm=t.get("target_nm")))
    raise ValueError(f"unknown tool kind {kind}")


def _cart(sc, L):
    """The robot cart (static while docked) with a 12 mm tool tray plate and lips between the arms."""
    c, h = L["cart"]["center"], L["cart"]["half"]
    sc.box("cart_body", c, h, "cart_grey", mu=P.MU["steel"])
    top = c[2] + h[2]
    (tx, ty), (hx, hy) = L["cart"]["tray"]
    # rubber-lined tool tray between the arm mounts (the arm bases stand on the bare deck)
    sc.box("cart_tray", (tx, ty, top + 0.006), (hx, hy, 0.006), "tray_black", mu=P.MU["rubber"])


def build_wall(sc):
    """North-wall bay: a 2.4 x 1.6 m drillable region of the 200 mm C25/30 wall with its rebar and
    aggregate fields, the rest of the wall static; the cart docked 0.6 m from the face; the tool kit trolley."""
    from .concrete import ConcreteFields
    from . import parts as PT
    L = sc.layout
    R = P.ROOM
    t = P.WALL_THICK_M
    yf = P.WALL_FACE_Y
    x0, x1 = -1.2, 1.2
    z0, z1 = 0.6, 2.2
    for name, (xa, xb, za, zb) in dict(wall_north_w=(R["x0"], x0, 0., R["h"]), wall_north_e=(x1, R["x1"], 0., R["h"]),
                                       wall_north_low=(x0, x1, 0., z0), wall_north_high=(x0, x1, z1, R["h"])).items():
        sc.box(name, (0.5*(xa + xb), yf + 0.5*t, 0.5*(za + zb)), (0.5*(xb - xa), 0.5*t, 0.5*(zb - za)), "concrete_wall",
               mu=P.MU["concrete"])
    fields = ConcreteFields(L["field_seed"], (x0, x1), (z0, z1), thickness=t, rebar_offset=L["rebar_offset"])
    half = (0.5*(x1 - x0), 0.5*t, 0.5*(z1 - z0))
    centre = (0.5*(x0 + x1), yf + 0.5*t, 0.5*(z0 + z1))
    fmap = (np.array([-centre[0], -0.5*t, -centre[2]]), np.eye(3))      # body frame -> (x along, depth, z up)
    sc.add_drillable("wall", "concrete", (np.array(centre), Rotation.identity()), half, static=True, n_slots=10,
                     fields=fields, fields_map=fmap, render_material="concrete_wall")
    sc.wall_fields = fields
    _cart(sc, L)
    if L["task"] in ("clean-and-set", "ledger-anchor"):
        x_c = L["x_c"]
        tx = x_c + 0.66
        sc.box("kit_trolley", (tx, 2.20, 0.39), (0.20, 0.24, 0.39), "cart_grey")
        sc.box("kit_tray", (tx, 2.20, 0.786), (0.19, 0.23, 0.006), "tray_black", mu=P.MU["rubber"])
    if "ledger" in L:
        # the ledger stands flush on the wall (its back face on the concrete, as it is fastened), on two temporary
        # shelf brackets: an angle whose back leg is screwed to the wall BELOW the ledger, its seat under the ledger and
        # a lip in front of it. (Back legs behind the ledger would hold it 3.5 mm off the wall, so a seated washer
        # would leave the anchor's tip 3.5 mm short of h_nom.)
        lg = L["ledger"]
        t, dep = lg["thickness"], lg["depth"]
        hz = 0.5*dep
        sc.add_drillable("ledger", "douglas_fir", (np.array([lg["x_c"], yf - 0.5*t, lg["top_z"] - hz]), Rotation.identity()),
                         (0.60, 0.5*t, hz), static=False, n_slots=6, render_material="douglas_fir")
        for k, sgn in enumerate((-1, 1)):
            xb = lg["x_c"] + sgn*lg["bracket_dx"]
            zb = lg["top_z"] - dep
            sc.box(f"ledger_bracket{k}_seat", (xb, yf - 0.030, zb - 0.003), (0.025, 0.030, 0.003), "steel_galv",
                   mu=P.MU["steel"])
            sc.box(f"ledger_bracket{k}_lip", (xb, yf - t - 0.003, zb + 0.020), (0.025, 0.0025, 0.020), "steel_galv",
                   mu=P.MU["steel"])
            sc.box(f"ledger_bracket{k}_back", (xb, yf - 0.002, zb - 0.006 - 0.046), (0.025, 0.002, 0.046), "steel_galv")
    for t_ in L["tools"]:
        build_tool(sc, t_)
    for p_ in L["parts"]:
        if p_["kind"] == "anchor":
            # a hardwood rest on the kit tray: two rails 10 mm apart cradle the anchor's shank so the washer clears the tray
            # (an anchor lying on its washer rim and shank tip rolls like a cone; loose on the tray one rolled off in 2 s)
            # rails along nearly the whole shank: the assembly's centre of mass (towards the nut and washer) must lie well
            # inside them (with 60 mm rails a 146 mm ledger anchor teetered onto its washer rim and chattered: E0)
            ax_, ay_, az_ = p_["xyz"]
            L_, seat_ = p_["length"], p_["seat"]
            y_mid = ay_ - ((L_ - seat_) + 0.5*seat_)
            # The shank bears on three rounded points along each rail's inner top corner (spheres, 0.5 mm proud of the
            # corner), not on the rail's edge: a cylinder meets a box through MuJoCo's convex collision routine, whose
            # contact points move by about its 1e-6 m tolerance from tick to tick, and on the bare edges the four anchors
            # rocked at 27 Hz, 0.116 mm/s, for good (E0; params.SOLVER_CCD_TOLERANCE). A cylinder on a sphere
            # is an analytic contact. The rails are lower by what the points raise the shank (0.77 mm at 1/2 in).
            r_b = 0.003
            d_b = (r_b - 0.0005)/math.sqrt(2.)
            r_s = 0.5*P.KBTZ2["d"]
            lift_ = math.sqrt((r_s + r_b)**2 - (0.005 + d_b)**2) - d_b - math.sqrt(r_s**2 - 0.005**2)
            h_rail = 0.014 - lift_
            half_l = 0.5*seat_ - 0.005
            for k_, sgn_ in enumerate((-1, 1)):
                sc.box(f"{p_['name']}_rest{k_}", (ax_ + sgn_*0.009, y_mid, 0.792 + 0.5*h_rail), (0.004, half_l, 0.5*h_rail),
                       "hardwood", mu=P.MU["douglas_fir"])
                for j_, fy_ in enumerate((-0.8, 0., 0.8)):
                    T.Parts(sc).sphere(-1, f"{p_['name']}_rest{k_}_point{j_}",
                                       (ax_ + sgn_*(0.005 + d_b), y_mid + fy_*half_l, 0.792 + h_rail - d_b), r_b,
                                       "collision_hidden", mu=P.MU["douglas_fir"])
        if p_["kind"] == "brush":
            # a rest of its own: two pairs of rails, under the grip and under the bristle head, so that the brush lies
            # level and still wherever the hammer beside it is (it lay on the tray propped by the hammer's head, and
            # rocked for good once the hammer was taken). Its centre of mass is 61 mm
            # in front of the grip, so a rest under the grip alone would tip it onto its bristles
            ax_, ay_, az_ = p_["xyz"]
            _point_cradle(sc, f"{p_['name']}_rest_grip", ax_, ay_ + 0.05, 0.035, 0.012, az_, 0.792, gap=0.016)
            _point_cradle(sc, f"{p_['name']}_rest_head", ax_, ay_ - 0.20, 0.035, 0.0062, az_, 0.792, gap=0.008)
        if p_["kind"] == "nozzle":
            # a clip-cradle under the tube near its tip: two rails 6 mm apart hold the 10 mm tube with the wand level
            # on its 19 mm hose stub (lying on the stub and the tube's tip rim it rocked at 0.1-0.6 mm/s indefinitely)
            ax_, ay_, az_ = p_["xyz"]
            for k_, sgn_ in enumerate((-1, 1)):
                sc.box(f"{p_['name']}_cradle{k_}", (ax_ + sgn_*0.005, ay_ - 0.15, 0.792 + 0.0075), (0.002, 0.02, 0.0075),
                       "hardwood", mu=P.MU["douglas_fir"])
        PT.build_part(sc, p_)
    if L.get("vacuum"):
        from .vacuum import build as build_vacuum
        build_vacuum(sc, L["vacuum"])
    for ph in L.get("predrilled", []):
        mk = L["marks"][ph["mark"]]
        sc.initial_holes.append(dict(drillable="wall", face_point=(mk["x"], yf, mk["z"]), axis=(0., 1., 0.),
                                     normal=(0., -1., 0.), d_mm=L.get("sds_bit_d_mm", 13.3), depth=ph["depth"],
                                     dust_mm=ph["dust_mm"]))


def build_bench(sc):
    """East-wall drill-press bench: steel bench, the drill press on its stand with its vise, the cart
    docked facing +x, parts and hand tools per task. Pre-clamped workpieces (the sheet panel on its backer, the two
    boards on theirs) are free bodies lying on the bench under two strap clamps (clamps.StrapClamp: a free strap on a
    stud, a heel block and a tightened flange nut, all colliders). Nothing holds them but contact: the stud tension
    presses the strap's toe on the top part and friction does the rest, so a push or a torque beyond what the faces
    hold makes the part slip (E21 measures the capacity)."""
    from . import parts as PT
    from . import clamps as C
    L = sc.layout
    bx, by, top = (3.24, 4.0), (-1.18, 0.65), P.BENCH_TOP_Z
    cx, cy = 0.5*sum(bx), 0.5*sum(by)
    sc.box("bench_top", (cx, cy, top - 0.02), (0.5*(bx[1] - bx[0]), 0.5*(by[1] - by[0]), 0.02), "bench_steel",
           mu=P.MU["steel"])
    for k, (lx, ly) in enumerate(((bx[0] + 0.05, by[0] + 0.05), (bx[0] + 0.05, by[1] - 0.05), (bx[1] - 0.05, by[0] + 0.05),
                                  (bx[1] - 0.05, by[1] - 0.05))):
        sc.box(f"bench_leg{k}", (lx, ly, 0.5*(top - 0.04)), (0.025, 0.025, 0.5*(top - 0.04)), "bench_steel")
    sc.box("bench_shelf", (cx, cy, 0.20), (0.5*(bx[1] - bx[0]) - 0.03, 0.5*(by[1] - by[0]) - 0.03, 0.01), "bench_steel")
    # the north wall (the wall bay's drillable region is only built by preset `wall`)
    R = P.ROOM
    sc.box("wall_north", (0., P.WALL_FACE_Y + 0.5*P.WALL_THICK_M, 0.5*R["h"]),
           (0.5*(R["x1"] - R["x0"]), 0.5*P.WALL_THICK_M, 0.5*R["h"]), "concrete_wall", mu=P.MU["concrete"])
    _cart(sc, L)
    for t_ in L["tools"]:
        if t_.get("holder"):
            # a hardwood rest with a groove: two rails 10 mm apart cradle the 16 mm punch barrel
            x, y, z = t_["xyz"]
            yaw = t_["rpy"][2]
            ax = np.array([math.cos(yaw), math.sin(yaw), 0.])
            nrm = np.array([-ax[1], ax[0], 0.])
            for k, sgn in enumerate((-1, 1)):
                c = np.array([x, y, P.BENCH_TOP_Z + 0.004]) + ax*T.CentrePunch.BARREL_X + nrm*sgn*0.009
                sc.box(f"{t_['name']}_rest{k}", tuple(c), (0.04, 0.004, 0.004), "hardwood",
                       rot=T.wq(Rotation.from_euler("z", yaw)), mu=P.MU["douglas_fir"])
        build_tool(sc, t_)
    for p_ in L["parts"]:
        if p_["kind"] == "carriage_bolt":
            # a hardwood rest under the shank: two rails 10 mm apart hold the 1/2 in shank level with the dome on the bench
            # (lying on its dome and shank tip it rolled like a cone at 26 mm/s and never stopped: E0)
            bx, by, _ = p_["xyz"]
            for k_, sgn_ in enumerate((-1, 1)):
                sc.box(f"{p_['name']}_rest{k_}", (bx + sgn_*0.009, by + 0.055, P.BENCH_TOP_Z + 0.0051),
                       (0.004, 0.03, 0.0051), "hardwood", mu=P.MU["douglas_fir"])
        PT.build_part(sc, p_)
        if p_.get("clamped"):
            # the stack under this part is held down by two strap clamps at the part's two short ends (clamps.py)
            C.hold_down(sc, p_)
    for p_ in L["parts"]:
        if p_.get("clamped"):
            C.stiffen_stack(sc, p_)          # the contact of the stack's wooden faces (params.HOLD_DOWN_STACK_STIFF)
