"""Physics scene (Newton 1.5 ModelBuilder): a bay of a clinical skills laboratory, the robot cart, the equipment.

World frame: origin on the floor under the counter's front edge, midway between the two arm bases;
x along the counter to the robot's right; y from the front edge toward the wall; z up. The counter top is at
z = 0.914 m. The cart is docked against the casework; the arms face the wall (+y).

What a task puts on the counter is in `task_layout`; everything loose is a free body at rest after the settle.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import parts as T
from . import devices as D
from . import hydraulics as H

CZ = P.COUNTER_Z
RZ180 = Rotation.from_euler("z", math.pi)

# task -> what lies on the tray and stands at the working position
#   syringes: (name, kind, liquid, liquid mL, air mL, fitting, slot)   fitting: needle kind, 'cap' or None
TASK_SETUP = {
    "sharps-away": dict(trainer="pad", syringes=(("syringe_a", "s3", "medication", 0.05, 0.0, "g25", -4),
                                                  ("syringe_b", "s10", "saline", 0.07, 0.0, "g21", 0),
                                                  ("syringe_c", "s1", "medication", 0.04, 0.0, "g27", 4)),
                        bare=True, loose_sheaths=("syringe_a", "syringe_b")),
    "set-dose": dict(trainer="pad", syringes=(("syringe", "s3", "medication", 2.8, 0.3, "g25", 0),), dose_ml=2.5),
    "draw-dose": dict(trainer="pad", syringes=(("syringe", "s3", None, 0.0, 0.0, "g18b", 0),), vial_ml=10.0, dose_ml=2.5),
    "inject": dict(trainer="pad", syringes=(("syringe", "s3", "medication", 0.5, 0.0, "g23", 0),), dose_ml=0.5),
    "vial-to-pad": dict(trainer="pad", syringes=(("syringe", "s3", None, 0.0, 0.0, "g18b", -2),), vial_ml=10.0, dose_ml=1.0,
                        spare_needles=(("needle_im", "g25", 3),)),
    "flush-lock": dict(trainer="lock", syringes=(("flush", "flush", "saline", 10.0, 0.3, "cap", 0),), flush_ml=5.0),
    "slow-push": dict(trainer="lock", syringes=(("dose", "s10", "medication", 4.0, 0.0, "cap", -3),
                                                ("flush", "flush", "saline", 10.0, 0.3, "cap", 3)),
                      dose_ml=4.0, dose_s=120.0, flush_ml=5.0),
    "fast-bolus": dict(trainer="lock", syringes=(("dose", "s10", "contrast", 10.0, 0.0, "cap", -3),
                                                 ("flush", "flush", "saline", 10.0, 0.3, "cap", 3)),
                       dose_ml=10.0, bolus_s=8.0, flush_ml=5.0),
}


def task_layout(config):
    """Seeded placement and contents: loose objects move by up to 15 mm and 10 degrees with the seed,
    volumes by a few per cent; seed 0 is the nominal bay."""
    L = dict(TASK_SETUP[config.task])
    L["task"], L["variant"], L["seed"] = config.task, config.variant, config.seed
    rng = np.random.default_rng([config.seed, sum(map(ord, config.task))])
    jit = (lambda s: float(rng.uniform(-s, s))) if config.seed else (lambda s: 0.0)
    # the tray, the basin and the trainer stand 25 to 35 mm apart: they move less than the small things
    for key, (s, yaw) in dict(tray=(0.006, 2.0), basin=(0.006, 4.0), work=(0.006, 3.0), vial=(0.015, 180.0), wipes=(0.015, 10.0)).items():
        x, y = P.LAYOUT[key]
        L[key] = (x + jit(s), y + jit(s), jit(math.radians(yaw)))
    L["sharps"], L["stand"] = P.LAYOUT["sharps"], P.LAYOUT["stand"]
    L["cannula"] = "g20"
    L["clamp_closed"], L["blocked"] = True, False
    L["vial_gauge_kpa"], L["adapter"], L["upright"] = 0.0, False, False
    v = config.variant
    if v == "upright":
        L["upright"] = True
    elif v == "adapter":
        L["adapter"] = True
        L["syringes"] = (("syringe", "s3", None, 0.0, 0.0, None, 0),)
    elif v == "half-vial":
        L["vial_ml"] = 5.0
    elif v == "vacuum-vial":
        L["vial_ml"], L["vial_gauge_kpa"] = 7.0, -25.0
    elif v == "subcut":
        L["syringes"] = (("syringe", "s1", "medication", 0.3, 0.0, "g27", 0),)
        L["dose_ml"], L["layer"] = 0.3, "fat"
    elif v == "blocked":
        L["blocked"] = True
    elif v == "clamp-open":
        L["clamp_closed"] = False
    elif v in ("g22", "g18"):
        L["cannula"] = v
    elif v == "short":
        L["syringes"] = (("dose", "s3", "medication", 1.0, 0.0, "cap", -3), L["syringes"][1])
        L["dose_ml"], L["dose_s"] = 1.0, 60.0
    if config.seed:
        # contents vary a little: what a previous user left, how much air came with the draw
        out = []
        for (name, kind, liq, ml, air, fit, slot) in L["syringes"]:
            if liq and ml > 0.2 and config.task == "set-dose":
                ml, air = round(ml + jit(0.1), 3), round(max(air + jit(0.1), 0.1), 3)       # at most 3.3 mL in all: the barrel holds 3.36
            out.append((name, kind, liq, ml, air, fit, slot))
        L["syringes"] = tuple(out)
        if "vial_ml" in L and not v:
            L["vial_ml"] = round(L["vial_ml"] - float(rng.uniform(0.0, 3.0)), 2)
    # A filled syringe's volume is what its scale reads: the liquid above the dead space of the nozzle, which the
    # stopper cannot expel. (Filled as "0.5 mL" in all, at most 0.45 mL could be given and the injection's criterion
    # of 90 % could not be met.) A used syringe holds its dead space only.
    L["syringes"] = tuple((name, kind, liq, round(ml + P.SYRINGES[kind]["dead_ml"], 4) if liq and ml > 0.2 else ml, air, fit, slot)
                          for (name, kind, liq, ml, air, fit, slot) in L["syringes"])
    L.setdefault("layer", "muscle")
    L["ready_pose"] = dict(P.READY_POSE_ARM)
    return L


def _frame_x(x_axis, roll=0.0):
    """A rotation whose x axis is `x_axis` (its y as level as it can be), turned by `roll` about that axis."""
    x = np.asarray(x_axis, float)/np.linalg.norm(x_axis)
    up = np.array([0., 0., 1.]) if abs(x[2]) < 0.95 else np.array([0., 1., 0.])
    y = np.cross(up, x)
    y /= np.linalg.norm(y)
    return Rotation.from_matrix(np.column_stack([x, y, np.cross(x, y)]))*Rotation.from_euler("x", roll)


class Scene:
    FRANKA_FINGER_BOXES = (((0., 18.5e-3, 11e-3), (0., 0., 0.), (22e-3, 15e-3, 20e-3)),
                           ((0., 6.8e-3, 2.2e-3), (0., 0., 0.), (22e-3, 8.8e-3, 3.8e-3)),
                           ((0., 15.9e-3, 28.35e-3), (math.pi/6, 0., 0.), (17.5e-3, 7e-3, 23.5e-3)),
                           ((0., 7.58e-3, 45.25e-3), (0., 0., 0.), (17.5e-3, 15.2e-3, 18.5e-3)))

    def __init__(self, config, newton, wp):
        self.c, self.newton, self.wp = config, newton, wp
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=(0., 0., -P.GRAVITY))
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.visual = {}, set()
        self.bodies, self.cameras = {}, {}
        self.static_boxes, self.static_caps = [], []
        self.places, self.shape_place, self.loose = {}, {}, {}
        self._filters = []
        self.robot_bodies, self.pad_shapes = [], {}
        self.arm_hw, self.joints, self.arms = {}, {}, {}
        self.props = {}
        self.sx = 0.0
        self.layout = L = config.layout if getattr(config, "layout", None) else task_layout(config)
        self.fluid = D.Fluidics(self)
        self.room()
        if L.get("bench"):
            self.bench(L["bench"])
        else:
            self.station()
        self.fluid.finish()
        if L.get("robot", True) and config.robot != "none":
            self.robots()
        self.apply_filters()
        self.views()

    # ------------------------------------------------------------------ helpers
    def cfg(self, **kw):
        v = dict(mu=P.MU["default"], ke=2e4, kd=100., density=1000., gap=P.CONTACT_GAP_M, margin=0., mu_torsional=0.,
                 mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def visual_cfg(self):
        return self.newton.ModelBuilder.ShapeConfig(density=0., has_shape_collision=False, has_particle_collision=False)

    def box(self, name, pos, half, material, body=-1, rot=None, collide=True, place=None, **kw):
        cfg = self.cfg(**kw) if collide else self.visual_cfg()
        i = self.b.add_shape_box(body, xform=self.wp.transform(tuple(map(float, pos)), rot or self.wp.quat_identity()),
                                 hx=float(half[0]), hy=float(half[1]), hz=float(half[2]), cfg=cfg, label=name)
        self.materials[i] = material
        if not collide:
            self.visual.add(i)
        if body < 0:
            self.static_boxes.append((name, tuple(map(float, pos)), tuple(map(float, half)), material,
                                      tuple(rot) if rot is not None else (0., 0., 0., 1.)))
            self.shape_place[i] = place or name.split(":")[0]
        return i

    def span(self, name, x0, x1, y0, y1, z0, z1, material, **kw):
        return self.box(name, (0.5*(x0 + x1), 0.5*(y0 + y1), 0.5*(z0 + z1)), (0.5*(x1 - x0), 0.5*(y1 - y0), 0.5*(z1 - z0)),
                        material, **kw)

    def free(self, name, body, place=None):
        self.b.add_articulation([self.b.add_joint_free(body, label=f"{name}_free")], label=f"{name}_assembly")
        self.bodies[name] = body
        self.places[body] = place or name
        return body

    def mutual_filter(self, bodies):
        bodies = list(bodies)
        for i, a in enumerate(bodies):
            for c in bodies[i + 1:]:
                self._filters.append((int(a), int(c)))

    def filter_bodies(self, a, c):
        self._filters.append((int(a), int(c)))

    def apply_filters(self):
        """Whole body pairs (they become MuJoCo excludes; shape-level filters are not reliable in a large model)."""
        sb = np.asarray(self.b.shape_body)
        seen = set()
        for a, c in self._filters:
            if (a, c) in seen or (c, a) in seen or a == c:
                continue
            seen.add((a, c))
            for sa in np.nonzero(sb == a)[0]:
                for sc_ in np.nonzero(sb == c)[0]:
                    self.b.add_shape_collision_filter_pair(int(sa), int(sc_))

    def place_of(self, body, shape=-1):
        """The name of the place a point of contact or a landed drop is on."""
        if body is None or body < 0:
            return self.shape_place.get(int(shape), "room")
        if body in self.places:
            return self.places[body]
        if body in self._robot_set:
            return "robot"
        return str(self.b.body_label[body])

    # ------------------------------------------------------------------ the bay
    def room(self):
        b = self.b
        b.add_ground_plane(cfg=self.cfg(mu=0.6), label="floor")
        self.materials[b.shape_count - 1] = "floor"
        self.shape_place[b.shape_count - 1] = "floor"
        R = P.ROOM
        cl, cd, ct = P.COUNTER_LENGTH, P.COUNTER_DEPTH, P.COUNTER_TOP_THICK
        x0 = P.COUNTER_X0
        x1 = x0 + cl
        self.counter_x = (x0, x1)
        self.span("counter_top", x0, x1, 0.0, cd, CZ - ct, CZ, "solid_surface", mu=P.MU["laminate"], place="counter")
        self.span("counter_upstand", x0, x1, cd - 0.02, cd, CZ, CZ + 0.10, "solid_surface", place="counter_upstand")
        self.span("casework", x0, x1, 0.025, cd, 0.10, CZ - ct, "casework", place="casework")
        self.span("toe_kick", x0, x1, 0.075, cd, 0.0, 0.10, "toe_kick", place="casework")
        xm, hx = 0.5*(R["x0"] + R["x1"]), 0.5*(R["x1"] - R["x0"])
        ym, hy = 0.5*(R["y0"] + R["y1"]), 0.5*(R["y1"] - R["y0"])
        self.box("back_wall", (xm, R["y1"] + 0.05, 0.5*R["z1"]), (hx, 0.05, 0.5*R["z1"]), "wall_panel", place="wall")
        self.box("wall_left", (R["x0"] - 0.05, ym, 0.5*R["z1"]), (0.05, hy, 0.5*R["z1"]), "wall_paint", place="wall")
        self.box("wall_right", (R["x1"] + 0.05, ym, 0.5*R["z1"]), (0.05, hy, 0.5*R["z1"]), "wall_paint", place="wall")
        self.box("wall_ceiling", (xm, ym, R["z1"] + 0.05), (hx, hy, 0.05), "ceiling", place="ceiling")
        # the bed-head services trunking over the counter, and what hangs on the wall within an arm's reach
        tz = P.TRUNKING["z"]
        self.span("trunking", x0 + 0.05, x1 - 0.05, R["y1"] - P.TRUNKING["depth"], R["y1"], tz, tz + P.TRUNKING["height"],
                  "trunking", place="trunking")
        for name, (cx, w, h, d, z) in P.WALL_FITTINGS.items():
            self.span(f"fit_{name}", cx - 0.5*w, cx + 0.5*w, R["y1"] - d, R["y1"], z, z + h, "fitting_" + name, place="wall_fitting")
        # the hand-wash basin at the left end of the run, and its tap
        s = P.SINK
        self.span("sink_block", s["x"][0], s["x"][1], s["y"][0], s["y"][1], CZ, CZ + s["rim"], "ceramic", place="sink")
        self.span("sink_tap", 0.5*sum(s["x"]) - 0.02, 0.5*sum(s["x"]) + 0.02, s["y"][1] - 0.05, s["y"][1], CZ, CZ + s["tap_z"],
                  "steel_brushed", place="sink")
        # the docked cart: deck plate, four posts, lower shelf with the two controllers (the teaching kitchen's cart)
        sx, y0 = (1.15 if self.c.robot == "g1" else self.sx), -P.ARM_SETBACK
        deck_t = 0.012
        self.box("cart_deck", (sx, y0 - 0.06, P.ARM_BASE_Z - 0.5*deck_t), (0.44, 0.24, 0.5*deck_t), "cart_plate", place="cart")
        for ix in (-1, 1):
            for iy in (-1, 1):
                self.box(f"cart_post_{ix}_{iy}", (sx + ix*0.41, y0 - 0.06 + iy*0.21, 0.5*(P.ARM_BASE_Z - deck_t)),
                         (0.02, 0.02, 0.5*(P.ARM_BASE_Z - deck_t)), "cart_frame", place="cart")
        self.box("cart_shelf", (sx, y0 - 0.06, 0.20), (0.42, 0.22, 0.008), "cart_plate", place="cart")
        for ix in (-1, 1):
            self.box(f"cart_controller_{ix}", (sx + ix*0.20, y0 - 0.06, 0.208 + 0.09), (0.16, 0.18, 0.09), "controller", place="cart")
        # the drip stand (pole and base as boxes) and the bed beyond it (outside every arm's reach)
        dx, dy = self.layout["stand"]
        self.span("stand_pole", dx - 0.0125, dx + 0.0125, dy - 0.0125, dy + 0.0125, 0.06, 1.90, "steel_brushed", place="drip_stand")
        self.span("stand_base", dx - 0.28, dx + 0.28, dy - 0.28, dy + 0.28, 0.0, 0.06, "stand_base", place="drip_stand")
        bd = P.BED
        self.span("bed", bd["x"][0], bd["x"][1], bd["y"][0], bd["y"][1], 0.0, bd["z"], "bed", place="bed")
        self.counter_plane = dict(z=CZ, lo=(x0, 0.0), hi=(x1, cd))

    # ------------------------------------------------------------------ what the task works with
    def station(self):
        L, fl = self.layout, self.fluid
        self.tray()
        self.basin()
        self.small_props()
        self.sharps()
        if L["trainer"] == "pad":
            self.pad()
            if "vial_ml" in L or L["task"] in ("inject", "set-dose", "sharps-away"):
                self.vial(L.get("vial_ml", 10.0))
        else:
            self.lock()
        sheath_slots = iter((2, -2, 5))
        for (name, kind, liquid, ml, air, fit, slot) in L["syringes"]:
            spec = P.SYRINGES[kind]
            front = [(0.5*spec["od"], 0.5*spec["od"])]          # the tube's collider is a capsule: its front sphere's centre
            lu = P.LUER
            seat = -lu["nozzle_len"] + lu["d_seat"] + 0.00055
            bare = L.get("bare", False)
            if fit == "cap":
                front.append((seat - 0.006, 0.0065))
            elif fit:
                n = P.NEEDLES[fit]
                tipx = seat - P.NEEDLE_HUB["length"] - n["length"]
                front.append((tipx, 0.5*n["od"]) if bare else (tipx - 0.004, 0.5*P.NEEDLE_SHEATH["od"]*math.cos(math.pi/8)))
                front.append((seat - 0.5*P.NEEDLE_HUB["length"], 0.5*P.NEEDLE_HUB["od"]))
            pose, x0 = self.tray_pose(spec, front, slot, liquid, ml, air)
            total = None if liquid else 0.0
            syr = D.Syringe(self, name, kind, pose, liquid=liquid, liquid_ml=ml, gas_ml=air if liquid else None, plunger_ml=total)
            fl.syringes[name] = syr
            if fit == "cap":
                cap = D.TipCap(self, f"{name}_cap", syr.mated_pose(lu["d_seat"] + 0.00055))
                fl.caps[cap.name] = cap
                fl.add_female(cap.female)
                syr.start_on(cap.female)
            elif fit:
                has_sheath = (False if name in L.get("loose_sheaths", ()) else None) if bare else True
                if has_sheath is False:
                    j = next(sheath_slots)
                    tx, ty, _ = L["tray"]
                    self.loose[f"{name}_needle_sheath"] = (np.array([tx - 0.09, ty + j*P.TRAY["rib_pitch"], self.tray_floor + 0.0036]),
                                                           Rotation.from_euler("z", 0.2*j))
                ndl = D.Needle(self, f"{name}_needle", fit, syr.mated_pose(lu["d_seat"] + 0.00055), sheathed=has_sheath)
                fl.needles[ndl.name] = ndl
                fl.add_female(ndl.female)
                syr.start_on(ndl.female)
        for (name, kind, slot) in L.get("spare_needles", ()):
            tx, ty, _ = L["tray"]
            pose = (np.array([tx + 0.02, ty + slot*P.TRAY["rib_pitch"], self.tray_floor + 0.0036]), RZ180)
            ndl = D.Needle(self, name, kind, pose, sheathed=True)
            fl.needles[name] = ndl
            fl.add_female(ndl.female)

    def bench(self, items):
        """A bench set-up for the probes (owner only): the listed devices at given poses, some held in a clamp.
        Each item: dict(type='syringe'|'vial'|'pad'|'lock'|'needle', ...)."""
        fl = self.fluid
        lu = P.LUER
        for it in items:
            kind = it["type"]
            if kind == "syringe":
                if it.get("on") == "lock":          # mated on the lock's connector, as deep as a joint made with 27.5 N
                    pc, Rc = self.lock_geometry["connector"]
                    pose = (pc - Rc.apply((-lu["nozzle_len"] + lu["d_seat"] + 0.00055, 0., 0.)), Rc)
                else:
                    nozzle = np.asarray(it.get("nozzle", (1., 0., 0.)), float)
                    nozzle /= np.linalg.norm(nozzle)
                    pose = (np.asarray(it["pos"], float), _frame_x(-nozzle, it.get("roll", 0.0)))
                liquid = it.get("liquid")
                syr = D.Syringe(self, it["name"], it["kind"], pose, liquid=liquid, liquid_ml=it.get("ml", 0.0),
                                gas_ml=it.get("air", 0.0) if (liquid or it.get("air")) else None, plunger_ml=it.get("plunger_ml"),
                                mount=it.get("mount", "free"))
                fl.syringes[syr.name] = syr
                if it.get("mount", "free") != "free" and it.get("on") != "lock":
                    self.stand(syr.name, pose[0] + pose[1].apply((0.5*syr.s["barrel_len"], 0., 0.)), 0.5*syr.s["od"], body=syr.barrel,
                               local=(0.5*syr.s["barrel_len"], 0., 0.), sliding=it.get("mount") == "slide")
                fit = it.get("fit")
                if it.get("on") == "lock":
                    syr.start_on(fl.locks["lock"].female)
                elif fit == "cap":
                    cap = D.TipCap(self, f"{syr.name}_cap", syr.mated_pose())
                    fl.caps[cap.name] = cap
                    fl.add_female(cap.female)
                    syr.start_on(cap.female)
                elif fit:
                    ndl = D.Needle(self, f"{syr.name}_needle", fit, syr.mated_pose(), sheathed=it.get("sheathed"), tight=it.get("tight", False))
                    fl.needles[ndl.name] = ndl
                    fl.add_female(ndl.female)
                    syr.start_on(ndl.female)
            elif kind == "needle":
                ndl = D.Needle(self, it["name"], it["kind"], (np.asarray(it["pos"], float), _frame_x(-np.asarray(it.get("point", (1., 0., 0.)), float))),
                               sheathed=it.get("sheathed", True))
                fl.needles[ndl.name] = ndl
                fl.add_female(ndl.female)
            elif kind == "vial":
                up = np.asarray(it.get("up", (0., 0., 1.)), float)
                Rv = T.rot_z_to(up/np.linalg.norm(up))
                v = D.Vial(self, it.get("name", "vial"), (np.asarray(it["pos"], float), Rv), liquid=it.get("liquid", "medication"),
                           liquid_ml=it.get("ml", 10.0), gauge_kpa=it.get("gauge_kpa", 0.0), adapter=it.get("adapter", False),
                           mount=it.get("mount", "free"))
                fl.vials[v.name] = v
                if it.get("mount", "free") != "free":
                    self.stand(v.name, np.asarray(it["pos"], float) + Rv.apply((0., 0., 0.5*P.VIAL["height"])), 0.5*P.VIAL["od"], body=v.body,
                               local=(0., 0., 0.4*P.VIAL["height"]))
                fl.add_penetrable(v.site)
                if v.female is not None:
                    fl.add_female(v.female)
            elif kind == "pad":
                self.pad()
            elif kind == "lock":
                self.lock()
            elif kind == "tray":
                self.tray()
            elif kind == "basin":
                self.basin()
            elif kind == "sharps":
                self.sharps()
            else:
                raise ValueError(f"bench item {kind!r}")

    def stand(self, name, at, radius, body, local, sliding=False):
        """A bench clamp as it looks (drawn only; the clamp itself is the fixed or driven joint): a base on the counter,
        a post behind the held thing, an arm to it and a jaw on the held body."""
        y_post = at[1] + 0.09
        self.box(f"bench_{name}_base", (at[0], y_post, CZ + 0.006), (0.06, 0.045, 0.006), "stand_base", collide=False)
        self.box(f"bench_{name}_post", (at[0], y_post, 0.5*(CZ + at[2] + 0.03)), (0.006, 0.006, 0.5*(at[2] + 0.03 - CZ)), "steel_brushed", collide=False)
        if not sliding:
            self.box(f"bench_{name}_arm", (at[0], 0.5*(at[1] + y_post) + 0.5*radius, at[2]), (0.005, 0.5*(y_post - at[1]) - 0.5*radius, 0.005), "steel_brushed",
                     collide=False)
        self.box(f"bench_{name}_jaw", local, (0.007, radius + 0.004, radius + 0.004), "clamp", body=body, collide=False, density=0.)

    def tray_pose(self, spec, front, slot, liquid, ml, air):
        """Pose of a syringe laid in a channel of the tray, nozzle to the right: its flange stands on two ribs and its
        front end on the floor (the pitch that makes both true), 0.4 mm above where it will rest."""
        tx, ty, yaw = self.layout["tray"]
        total = (ml + air) if liquid else 0.0
        x0 = max((total*H.ML - spec["dead_ml"]*H.ML)/(math.pi*spec["bore"]**2/4), 0.0)
        Lb = spec["barrel_len"]
        rear = Lb + spec["flange"][2] + spec["proj"] + x0 + spec["thumb_t"]
        ahead = -min(x for x, _ in front)
        z_r = P.TRAY["rib_h"] + 0.5*spec["flange"][1]
        sin_p = min((z_r - r)/(Lb - x) for x, r in front)
        z_o = z_r - Lb*sin_p
        Rt = Rotation.from_euler("z", yaw)
        local = np.array([0.5*(rear - ahead), slot*P.TRAY["rib_pitch"], 0.0])
        p = np.array([tx, ty, 0.0]) + Rt.apply(local) + np.array([0., 0., self.tray_floor + z_o + 0.0004])
        return (p, Rt*RZ180*Rotation.from_euler("y", -math.asin(sin_p))), x0

    def tray(self):
        t = P.TRAY
        tx, ty, yaw = self.layout["tray"]
        (sx, sy, sz), wl = t["size"], t["wall"]
        pose = (np.array([tx, ty, CZ + 0.0003]), Rotation.from_euler("z", yaw))
        n_rib = int((sy - 2*wl)//t["rib_pitch"])
        vol = sx*sy*wl + 2*(sx + sy)*wl*(sz - wl) + n_rib*(sx - 2*wl)*0.002*t["rib_h"]
        rho = t["mass"]/vol
        B = T.link(self, "tray", pose)
        kw = dict(body=B, density=rho, mu=P.MU["pp"])
        self.box("tray_floor", (0., 0., 0.5*wl), (0.5*sx, 0.5*sy, 0.5*wl), "tray", **kw)
        for s_, nm in ((-1, "front"), (1, "back")):
            self.box(f"tray_wall_{nm}", (0., s_*(0.5*sy - 0.5*wl), 0.5*(sz + wl)), (0.5*sx, 0.5*wl, 0.5*(sz - wl)), "tray", **kw)
        for s_, nm in ((-1, "left"), (1, "right")):
            self.box(f"tray_wall_{nm}", (s_*(0.5*sx - 0.5*wl), 0., 0.5*(sz + wl)), (0.5*wl, 0.5*sy - wl, 0.5*(sz - wl)), "tray", **kw)
        self.rib_y = [(k - 0.5*(n_rib - 1))*t["rib_pitch"] for k in range(n_rib)]
        for k, ry in enumerate(self.rib_y):
            self.box(f"tray_rib{k}", (0., ry, wl + 0.5*t["rib_h"]), (0.5*sx - wl, 0.001, 0.5*t["rib_h"]), "tray", **kw)
        self.free("tray", B)
        self.tray_floor = CZ + 0.0003 + wl

    def basin(self):
        s = P.BASIN
        bx, by, yaw = self.layout["basin"]
        (sx, sy, sz), wl = s["size"], s["wall"]
        rho = s["mass"]/(sx*sy*wl + 2*(sx + sy)*wl*(sz - wl))
        B = T.link(self, "basin", (np.array([bx, by, CZ + 0.0003]), Rotation.from_euler("z", yaw)))
        kw = dict(body=B, density=rho, mu=P.MU["pp"])
        self.box("basin_floor", (0., 0., 0.5*wl), (0.5*sx, 0.5*sy, 0.5*wl), "basin", **kw)
        for s_, nm in ((-1, "front"), (1, "back")):
            self.box(f"basin_wall_{nm}", (0., s_*(0.5*sy - 0.5*wl), 0.5*(sz + wl)), (0.5*sx, 0.5*wl, 0.5*(sz - wl)), "basin", **kw)
        for s_, nm in ((-1, "left"), (1, "right")):
            self.box(f"basin_wall_{nm}", (s_*(0.5*sx - 0.5*wl), 0., 0.5*(sz + wl)), (0.5*wl, 0.5*sy - wl, 0.5*(sz - wl)), "basin", **kw)
        self.free("basin", B)

    def small_props(self):
        """The box of alcohol prep pads and a stack of gauze squares: free rigid bodies with their packs' masses."""
        wx, wy, yaw = self.layout["wipes"]
        for name, (dx, dy, half, mass, material) in P.PROPS.items():
            hz = half[2]
            B = T.link(self, name, (np.array([wx + dx, wy + dy, CZ + 0.0003]), Rotation.from_euler("z", yaw)), mass=mass,
                       com=(0., 0., hz), inertia=(mass*(half[1]**2 + hz**2)/3, mass*(half[0]**2 + hz**2)/3,
                                                  mass*(half[0]**2 + half[1]**2)/3))
            self.box(f"{name}_shell", (0., 0., hz), half, material, body=B, density=0., mu=P.MU["default"])
            self.free(name, B)
            self.props[name] = dict(body=int(B), half=half, material=material)
        if self.layout.get("bare"):
            # a used gauze square on the tray (sharps-away: something that is not a sharp)
            tx, ty, _ = self.layout["tray"]
            B = T.link(self, "gauze_used", (np.array([tx - 0.10, ty - 0.05, self.tray_floor + P.TRAY["rib_h"] + 0.0004]), Rotation.from_euler("z", 0.4)),
                       mass=0.0012, com=(0., 0., 0.002), inertia=(3e-7, 3e-7, 5e-7))
            self.box("gauze_used_shell", (0., 0., 0.002), (0.025, 0.025, 0.002), "gauze", body=B, density=0., mu=0.6)
            self.free("gauze_used", B)
            self.props["gauze_used"] = dict(body=int(B), half=(0.025, 0.025, 0.002), material="gauze")

    def sharps(self):
        """Counter-standing sharps container in its wire bracket (static), with a counterbalanced flap in its top slot."""
        s = P.SHARPS
        cx, cy = self.layout["sharps"]
        (sx, sy, sz), wl = s["size"], s["wall"]
        slx, sly = s["slot"]
        z0, z1 = CZ, CZ + sz
        nm = "sharps"
        self.span(f"{nm}_floor", cx - 0.5*sx, cx + 0.5*sx, cy - 0.5*sy, cy + 0.5*sy, z0, z0 + wl, "sharps_red", place="sharps_outside")
        for tag, ya in (("front", cy - 0.5*sy), ("back", cy + 0.5*sy - wl)):
            self.span(f"{nm}_wall_{tag}", cx - 0.5*sx, cx + 0.5*sx, ya, ya + wl, z0 + wl, z1 - wl, "sharps_red", place="sharps_outside")
        for s_, tag in ((-1, "left"), (1, "right")):
            xa = cx + s_*0.5*sx - (wl if s_ > 0 else 0.0)
            self.span(f"{nm}_wall_{tag}", xa, xa + wl, cy - 0.5*sy + wl, cy + 0.5*sy - wl, z0 + wl, z1 - wl, "sharps_red", place="sharps_outside")
        # the lid around the slot (four pieces); the slot runs along x, nearer the front
        sy0 = cy - 0.5*sy + 0.020
        sy1 = sy0 + sly
        self.span(f"{nm}_lid_front", cx - 0.5*sx, cx + 0.5*sx, cy - 0.5*sy, sy0, z1 - wl, z1, "sharps_lid", place="sharps_outside")
        self.span(f"{nm}_lid_back", cx - 0.5*sx, cx + 0.5*sx, sy1, cy + 0.5*sy, z1 - wl, z1, "sharps_lid", place="sharps_outside")
        self.span(f"{nm}_lid_left", cx - 0.5*sx, cx - 0.5*slx, sy0, sy1, z1 - wl, z1, "sharps_lid", place="sharps_outside")
        self.span(f"{nm}_lid_right", cx + 0.5*slx, cx + 0.5*sx, sy0, sy1, z1 - wl, z1, "sharps_lid", place="sharps_outside")
        # the bracket on the upstand
        self.span(f"{nm}_bracket", cx - 0.5*sx - 0.004, cx + 0.5*sx + 0.004, cy + 0.5*sy, P.COUNTER_DEPTH - 0.02, z0 + 0.06, z0 + 0.075,
                  "steel_brushed", place="sharps_outside")
        # the flap: hinged along the slot's back edge, its free edge swings down and in; a counterweight behind the
        # hinge holds it level with FLAP_CLOSING_NM and brings it back
        fw, ft = sly - 0.004, 0.002
        hinge = np.array([cx, sy1 - 0.001, z1 - 0.5*wl])
        m = s["flap_mass"]
        F = T.link(self, "sharps_flap", (hinge, Rotation.identity()), mass=m, com=(0., s["flap_closing_nm"]/(m*P.GRAVITY), 0.),
                   inertia=(m*fw*fw/3, m*slx*slx/12, m*(fw*fw/3 + slx*slx/12)))
        self.box("sharps_flap_plate", (0., -0.5*fw - 0.001, 0.), (0.5*slx - 0.002, 0.5*fw, 0.5*ft), "sharps_flap", body=F, density=0.,
                 mu=P.MU["pp"])
        j = self.b.add_joint_revolute(-1, F, parent_xform=self.wp.transform(tuple(map(float, hinge)), self.wp.quat_identity()),
                                      child_xform=self.wp.transform_identity(), axis=(1., 0., 0.), limit_lower=0.0, limit_upper=1.45,
                                      armature=2e-6, friction=1e-5, label="sharps_flap_hinge")
        self.b.add_articulation([j], label="sharps_flap_assembly")
        self.bodies["sharps_flap"] = F
        self.places[F] = "sharps_flap"
        self.flap_joint = T.JointRef(j, "sharps_flap")
        self.sharps_cavity = dict(lo=(cx - 0.5*sx + wl, cy - 0.5*sy + wl, z0 + wl), hi=(cx + 0.5*sx - wl, cy + 0.5*sy - wl, z1 - wl - 0.004))
        self.sharps_slot = dict(centre=(cx, 0.5*(sy0 + sy1), z1), size=(slx, sly))
        self.sharps_box = dict(lo=(cx - 0.5*sx, cy - 0.5*sy, z0 - 0.02), hi=(cx + 0.5*sx, cy + 0.5*sy, z1 - wl))

    def pad(self):
        """The injection pad: skin, fat and muscle on a board. Flat on a non-slip mat, or strapped to an upright
        upper-arm model clamped to the counter's edge (variant)."""
        s = P.PAD
        wx, wy, yaw = self.layout["work"]
        thick = sum(P.LAYERS[k]["thick"] for k in ("skin", "fat", "muscle")) + s["board"]
        hx, hy = 0.5*s["size"][0], 0.5*s["size"][1]
        inertia = (s["mass"]*(hy*hy + thick*thick/4)/3, s["mass"]*(hx*hx + thick*thick/4)/3, s["mass"]*(hx*hx + hy*hy)/3)
        if not self.layout.get("upright"):
            self.span("pad_mat", wx - hx - 0.02, wx + hx + 0.02, wy - hy - 0.02, wy + hy + 0.02, CZ, CZ + 0.002, "mat", mu=P.MU["skin"],
                      place="mat")
            pose = (np.array([wx, wy, CZ + 0.002 + 0.0003]), Rotation.from_euler("z", yaw))
            B = T.link(self, "pad", pose, mass=s["mass"], com=(0., 0., 0.5*thick), inertia=inertia)
            self.box("pad_block", (0., 0., 0.5*thick), (hx, hy, 0.5*thick), "pad_skin", body=B, density=0., mu=P.MU["skin"])
            self.free("pad", B)
        else:
            # the upper-arm model: a post clamped to the counter's front edge carrying the pad with its skin facing -y
            z_c = P.PAD_UPRIGHT_Z
            # (the foot's front edge 25 mm behind the tray's back wall, whatever the seed moves: at py = 0.20 the post and
            # the foot stood 35 mm inside the tray, which was wedged a degree off level with kilonewtons in its contacts,
            # and the syringes on it slid: found by the evaluator's rest criterion in the do-nothing check)
            px, py = wx - 0.10, 0.26
            self.span("arm_model_post", px - 0.03, px + 0.03, py + thick, py + thick + 0.05, CZ, z_c + 0.10, "arm_model", place="arm_model")
            self.span("arm_model_foot", px - 0.08, px + 0.08, py + thick - 0.02, py + thick + 0.12, CZ, CZ + 0.012, "steel_brushed", place="arm_model")
            # pad frame: z out of the skin (toward -y), x along world x
            R = Rotation.from_matrix(np.column_stack([(1., 0., 0.), (0., 0., 1.), (0., -1., 0.)]))
            pose = (np.array([px, py + thick, z_c]), R)
            B = T.link(self, "pad", pose, mass=s["mass"], com=(0., 0., 0.5*thick), inertia=inertia)
            self.box("pad_block", (0., 0., 0.5*thick), (hx, hy, 0.5*thick), "pad_skin", body=B, density=0., mu=P.MU["skin"])
            j = self.b.add_joint_fixed(-1, B, parent_xform=self.wp.transform(tuple(map(float, pose[0])), T.wq(R)),
                                       child_xform=self.wp.transform_identity(), label="pad_strap")
            self.b.add_articulation([j], label="pad_assembly")
            self.bodies["pad"] = B
            self.places[B] = "pad"
        self.pad_thick = thick
        site = D.PadSite("pad", B, (0., 0., thick), (0., 0., 1.), ["skin", "fat", "muscle"], face_half=(hx, hy),
                         target_radius=0.5*s["ring_d"])
        self.fluid.add_penetrable(site)
        self.pad_site = site

    def vial(self, ml):
        vx, vy, yaw = self.layout["vial"]
        v = D.Vial(self, "vial", (np.array([vx, vy, CZ + 0.0003]), Rotation.from_euler("z", yaw)), liquid="medication", liquid_ml=ml,
                   gauge_kpa=self.layout.get("vial_gauge_kpa", 0.0), adapter=self.layout.get("adapter", False))
        self.fluid.vials["vial"] = v
        self.fluid.add_penetrable(v.site)
        if v.female is not None:
            self.fluid.add_female(v.female)
        self.vial_start = np.array([vx, vy])

    def lock(self):
        """The intravenous training arm on its board (static), with the saline lock: connector under a strip of tape,
        extension tube with a slide clamp, cannula in the vein."""
        a = P.TRAINING_ARM
        wx, wy, _ = self.layout["work"]
        y = wy - 0.03
        zb = CZ + a["board"]
        self.span("arm_board", wx - 0.20, wx + 0.34, y - 0.05, y + 0.05, CZ, zb, "arm_board", mu=P.MU["skin"], place="arm_board")
        x_h0, x_w, x_e = wx - a["hand"], wx, wx + a["forearm"]
        self.span("arm_hand", x_h0, x_w, y - 0.045, y + 0.045, zb, zb + a["hand_thick"], "arm_skin", mu=P.MU["skin"], place="training_arm")
        r0, r1 = a["wrist_r"], a["elbow_r"]
        radius = lambda k: r0 + (r1 - r0)*(k + 0.5)/4
        seg = (x_e - x_w)/4
        for k in range(4):
            xa, xb, r = x_w + seg*k, x_w + seg*(k + 1), radius(k)
            p0, p1 = (xa + 0.3*r, y, zb + r), (xb - 0.3*r, y, zb + r)
            i = T.capsule(self, -1, f"arm_forearm{k}", p0, p1, r, "arm_skin", mu=P.MU["skin"])
            self.shape_place[i] = "training_arm"
            self.static_caps.append((f"arm_forearm{k}", p0, p1, r, "arm_skin"))
        top = lambda x: zb + 2*radius(min(max(int((x - x_w)/seg), 0), 3))
        # the connector lies on the back of the wrist, its mouth up and to the left, its low end on the skin
        ang = math.radians(a["connector_tilt_deg"])
        c = P.CONNECTOR
        Rc = Rotation.from_matrix(np.column_stack([(-math.cos(ang), 0., math.sin(ang)), (0., -1., 0.), (math.sin(ang), 0., math.cos(ang))]))
        xc = wx + 0.030
        pc = np.array([xc, y, top(xc) + 0.5*c["od"]*math.cos(ang) + c["length"]*math.sin(ang) + 0.0012])
        xk = wx + 0.110
        clamp_p = np.array([xk, y - 0.5*P.EXTENSION["clamp_travel"], top(xk) + 0.0062])
        lk = D.IVLock(self, "lock", (pc, Rc), (clamp_p, Rotation.from_euler("z", math.pi/2)), cannula=self.layout["cannula"],
                      clamp_closed=self.layout["clamp_closed"], blocked=self.layout["blocked"])
        self.fluid.locks["lock"] = lk
        self.fluid.add_female(lk.female)
        # the extension tube's run (drawn; its liquid is drawn along it): from the connector's back end over the clamp to
        # the cannula's hub under its dressing
        back = pc - Rc.apply((c["length"], 0., 0.))
        xa = wx + 0.20
        tube = [back, back + np.array([0.012, 0., -0.002]), np.array([xk - 0.03, y, top(xk) + 0.004]), np.array([xk, y, top(xk) + 0.0045]),
                np.array([xk + 0.03, y, top(xk + 0.03) + 0.004]), np.array([xa - 0.03, y + 0.012, top(xa) + 0.004]),
                np.array([xa - 0.008, y + 0.004, top(xa) + 0.0035])]
        self.lock_geometry = dict(connector=(pc, Rc), clamp=clamp_p, cannula=np.array([xa, y, top(xa) + 0.003]), y=y, wrist_x=wx, board_z=zb,
                                  radii=[radius(k) for k in range(4)], seg=seg, tube=tube, hand=(x_h0, x_w, a["hand_thick"]))

    # ------------------------------------------------------------------ robots
    def mounts(self):
        sx = self.sx
        return {"left_arm_mount": (sx - 0.5*P.ARM_SPACING, -P.ARM_SETBACK, P.ARM_BASE_Z),
                "right_arm_mount": (sx + 0.5*P.ARM_SPACING, -P.ARM_SETBACK, P.ARM_BASE_Z)}

    def robots(self):
        from robots.catalog import for_environment
        from robots.backends.newton import mount
        import robots
        setup = for_environment("syringe-handling", self.c.robot)
        self.robot_setup = setup
        if self.c.robot == "g1":
            from .g1 import setup_scene
            setup_scene(self, setup)
            return
        self._hardware(setup)
        mounts = {k: mount(v, P.ARM_YAW) for k, v in self.mounts().items()}
        first, first_shape = self.b.body_count, self.b.shape_count
        self.rig_pending = robots.add_to(self.b, setup, mounts=mounts)
        self.robot_bodies = list(range(first, self.b.body_count))
        for i in range(first_shape, self.b.shape_count):
            box = self.b.shape_type[i] in (self.newton.GeoType.BOX, self.newton.GeoType.PLANE)
            self.b.shape_gap[i] = P.CONTACT_GAP_M if box else P.MESH_CONTACT_GAP_M
        for name, handle in self.rig_pending.instances.items():
            for j, entry in handle.joints.items():
                self.joints[j] = entry
            self.arms[name] = handle
            for i in handle.shape_ids:
                self.materials[i] = "robot"
            self._fingertips(name, handle)
            self._wrist_camera(name, handle)

    @property
    def _robot_set(self):
        if not hasattr(self, "_rs") or len(self._rs) != len(self.robot_bodies):
            self._rs = set(self.robot_bodies)
        return self._rs

    def _hardware(self, setup):
        from robots.catalog import model
        data = setup.to_dict()
        for name, inst in data["instances"].items():
            spec = model(inst["model"]).to_dict()
            self.arm_hw[name] = dict(model=inst["model"], mount_name=inst["mount"],
                                     wrist_link=spec.get("frames", {}).get("wrist", {}).get("parent"),
                                     arm_joints=spec["groups"]["arm"]["joints"],
                                     gripper_joints=spec["groups"].get("gripper", {}).get("joints", []))
        for att in data.get("attachments", {}).values():
            owner = att["parent"].split("/")[0]
            pads = model(att["model"]).to_dict().get("contact", {}).get("fingertip_pads")
            self.arm_hw[owner].update(hand=att["model"], pads=pads)

    def _fingertips(self, arm, handle):
        """Franka Hand finger colliders (URDF boxes; Panda meshes get the same boxes); the rubber tip box is the pad
        (barcode_scanning / wine_pouring / drilling / ingredient_mixing lineage)."""
        b = self.b
        fingers = sorted({b.shape_body[i] for i in handle.shape_ids
                          if b.shape_body[i] >= 0 and "finger" in b.body_label[b.shape_body[i]]})
        for body in fingers:
            shapes = [i for i in handle.shape_ids if b.shape_body[i] == body]
            boxes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.BOX]
            pad_z = self.arm_hw[arm]["pads"]["finger_frame_offset_z_m"]
            if boxes:
                for i in boxes:
                    if abs(float(np.asarray(b.shape_transform[i])[2]) - pad_z) < 1e-4:
                        self.pad_shapes[i] = b.body_label[body]
                        b.shape_material_mu[i] = P.MU["pad"]
                continue
            meshes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.MESH]
            if not meshes:
                continue
            tf = np.asarray(b.shape_transform[meshes[0]], float)
            Rm, pm = Rotation.from_quat(tf[3:]), tf[:3]
            for k, (xyz, rpy, size) in enumerate(self.FRANKA_FINGER_BOXES):
                R = Rm*Rotation.from_euler("xyz", rpy)
                p = pm + Rm.apply(xyz)
                i = self.box(f"{b.body_label[body].split('/')[-1]}_box{k}", tuple(p), tuple(np.asarray(size)/2),
                             "robot_dark", body=body, rot=tuple(R.as_quat()), density=0., mu=P.MU["pad"])
                if abs(xyz[2] - pad_z) < 1e-6:
                    self.pad_shapes[i] = b.body_label[body]

    def _wrist_camera(self, arm, handle):
        """D405-envelope wrist RGB on link7 (barcode_scanning / wine_pouring mount geometry)."""
        labels = self.b.body_label
        wl = self.arm_hw[arm]["wrist_link"]
        link7 = [i for i in handle.body_ids if wl and labels[i].endswith(wl)]
        if not link7:
            return
        body = link7[0]
        self.box(f"{arm}_camera_case", (.075, 0, .07), (.021, .021, .0115), "camera_dark", body=body,
                 density=.060/(.042*.042*.023))
        self.box(f"{arm}_camera_bracket", (.05, 0, .046), (.025, .013, .003), "aluminum", body=body,
                 density=.010/(.050*.026*.006))
        self.cameras[f"{arm}_wrist_rgb"] = dict(body=body, eye=[.075, -.012, .083], target=[.075, -.012, .283],
                                                up=[0, 1, 0], fov=87., role="policy", width=640, height=480)

    # ------------------------------------------------------------------ cameras
    def views(self):
        R = P.ROOM
        wx, wy, _ = self.layout["work"]
        tx, ty, _ = self.layout["tray"]
        c = P.STATION_CAMERA
        yw = R["y1"] - c["off_wall"]
        mx = 0.5*(wx + tx)
        # the policy's station camera: on the trunking, looking back over the tray and the trainer at the robot
        self.box("fit_station_camera", (mx, yw + 0.012, c["z"]), c["half"], "camera_dark", place="wall_fitting")
        self.box("fit_station_camera_arm", (mx, 0.5*(yw + 0.037 + R["y1"]), c["z"]), (0.008, 0.5*(R["y1"] - yw - 0.037), 0.008), "camera_dark",
                 place="wall_fitting")
        self.cameras["station_rgb"] = dict(body=-1, eye=[mx, yw - 0.03, c["z"]], target=[mx, 0.20, CZ + 0.02], up=[0, 0, 1],
                                           fov=c["fov"], role="policy", width=960, height=720)
        self.cameras["overhead_rgb"] = dict(body=-1, eye=[mx, 0.27, CZ + 0.95], target=[mx, 0.265, CZ], up=[0, 1, 0], fov=44.,
                                            role="policy", width=720, height=720)
        bc = P.BAY_CAMERA                              # on a ceiling bracket behind the operator (drawn in dressing.py)
        self.cameras["bay_rgb"] = dict(body=-1, eye=list(bc["eye"]), target=list(bc["target"]), up=[0, 0, 1], fov=bc["fov"],
                                       role="policy", width=1280, height=720)
        # the room and the bay (diagnostic)
        self.cameras["hero"] = dict(body=-1, eye=[1.55, -1.75, 1.66], target=[-0.10, 0.20, 1.02], up=[0, 0, 1], fov=56.,
                                    role="diagnostic", width=1280, height=720)
        self.cameras["hero_left"] = dict(body=-1, eye=[-1.65, -1.80, 1.66], target=[0.20, 0.15, 1.05], up=[0, 0, 1], fov=56.,
                                         role="diagnostic", width=1280, height=720)
        self.cameras["bay"] = dict(body=-1, eye=[R["x0"] + 0.30, R["y0"] + 0.25, 2.05], target=[0.35, -0.10, 0.95], up=[0, 0, 1],
                                   fov=68., role="diagnostic", width=1280, height=720)
        self.cameras["bay_right"] = dict(body=-1, eye=[R["x1"] - 0.30, R["y0"] + 0.30, 1.95], target=[-0.35, -0.10, 0.95],
                                         up=[0, 0, 1], fov=66., role="diagnostic", width=1280, height=720)
        self.cameras["station"] = dict(body=-1, eye=[1.30, -1.05, 1.60], target=[0.00, 0.20, 1.02], up=[0, 0, 1], fov=46.,
                                       role="diagnostic", width=1280, height=720)
        self.cameras["front"] = dict(body=-1, eye=[0.95, -0.50, 1.62], target=[mx, 0.25, CZ + 0.06], up=[0, 0, 1], fov=40.,
                                     role="diagnostic", width=1280, height=720)
        self.cameras["front_left"] = dict(body=-1, eye=[-0.95, -0.45, 1.62], target=[mx, 0.25, CZ + 0.06], up=[0, 0, 1], fov=40.,
                                          role="diagnostic", width=1280, height=720)
        self.cameras["tray_close"] = dict(body=-1, eye=[tx + 0.22, ty - 0.33, CZ + 0.26], target=[tx, ty, CZ + 0.02], up=[0, 0, 1],
                                          fov=44., role="diagnostic", width=1280, height=720)
        self.cameras["work_close"] = dict(body=-1, eye=[wx - 0.26, wy - 0.36, CZ + 0.30], target=[wx, wy - 0.02, CZ + 0.07],
                                          up=[0, 0, 1], fov=44., role="diagnostic", width=1280, height=720)
        self.cameras["tray_macro"] = dict(body=-1, eye=[tx + 0.02, ty - 0.17, CZ + 0.15], target=[tx + 0.0, ty, CZ + 0.012], up=[0, 0, 1],
                                          fov=38., role="diagnostic", width=1280, height=720)
        vx, vy = self.layout.get("vial", P.LAYOUT["vial"])[:2]
        self.cameras["vial_macro"] = dict(body=-1, eye=[vx + 0.05, vy - 0.16, CZ + 0.09], target=[vx, vy, CZ + 0.022], up=[0, 0, 1],
                                          fov=30., role="diagnostic", width=1280, height=720)
        # close-ups that ride a point the film follows (video.Film): they keep their direction
        self.cameras["follow"] = dict(body=-2, eye=[0.07, -0.33, 0.13], target=[0., 0., 0.005], up=[0, 0, 1], fov=36., role="diagnostic",
                                      width=1280, height=720)
        self.cameras["follow_right"] = dict(body=-2, eye=[0.30, -0.16, 0.12], target=[0., 0., 0.005], up=[0, 0, 1], fov=36., role="diagnostic",
                                            width=1280, height=720)
        self.cameras["side"] = dict(body=-1, eye=[mx + 0.05, P.COUNTER_DEPTH - 0.04, CZ + 0.34], target=[mx, 0.22, CZ + 0.05],
                                    up=[0, 0, 1], fov=58., role="diagnostic", width=1280, height=720)
