"""Physics scene (Newton 1.5 ModelBuilder): the teaching-kitchen station, the robot cart, vessels, tools, appliances.

World frame: origin on the floor under the counter's front edge at the prep station; x along the
counter to the robot's right; y from the front edge toward the wall; z up. The counter top is at z = 0.914 m. The cart
is docked against the cabinet fronts, centred on the preset's work zone (P.STATION_X); the arms face the wall (+y).
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import geometry as G
from . import tools as T

CZ = P.COUNTER_Z
# what each task puts on the counter
TASK_SETUP = {
    "dry-whisk": dict(vessel="bowl5", tools=("whisk", "spatula", "spoon")),
    "muffin-fold": dict(vessel="bowl5", tools=("spatula", "whisk", "spoon")),
    "pancake-whisk": dict(vessel="bowl5", tools=("whisk", "spatula", "spoon")),
    "pudding-whisk": dict(vessel="bowl5", tools=("whisk", "spatula", "spoon")),
    "cake-mixer": dict(vessel="bowl5", tools=("spatula", "whisk", "spoon"), mixer=True),
    "sauce-blend": dict(vessel="bowl2", tools=("spoon", "spatula", "whisk")),
    # hob: the knob's position at the start (params.HOB levels_w: 12 is the manual's level 6.0, 1000 W; 7 is 3.5, 580 W)
    "stir-sauce": dict(vessel="pan", tools=("stirrer", "whisk", "thermometer"), hob=12),
    "reheat-porridge": dict(vessel="pan", tools=("stirrer", "spoon", "thermometer"), hob=7),
}
SLOT_ANGLES = (math.radians(270.), math.radians(30.), math.radians(150.))      # crock compartments (the robot's side first)


def task_layout(config):
    """Seeded placement of the station: vessel position, cart offset, mat, hob level, tool order."""
    setup = dict(TASK_SETUP[config.task])
    rng = np.random.default_rng([config.seed, sum(map(ord, config.task))])
    sx = P.STATION_X[config.preset]
    jit = (lambda s: float(rng.uniform(-s, s))) if config.seed else (lambda s: 0.0)
    L = dict(setup, preset=config.preset, task=config.task, station_x=sx + jit(0.02))
    vx, vy = (P.HOB_ZONE if config.preset == "hob" else (sx, P.VESSEL_Y))
    hob_slack = 0.012 if config.preset == "hob" else 0.03
    L["vessel_xy"] = (vx + jit(hob_slack), vy + jit(hob_slack))
    L["vessel_yaw"] = jit(0.3)
    L["mat"] = (config.variant == "mat") or (config.preset == "bake" and config.seed % 4 == 3)
    if "hob" in setup and config.seed:
        L["hob"] = int(setup["hob"] + rng.integers(-1, 2))
    order = list(setup["tools"])
    if config.seed:
        rng.shuffle(order)
    L["tools"] = tuple(order)
    L["tool_lean"] = [(0.035 + jit(0.02), jit(0.02)) for _ in order]    # a slight outward lean: they come to rest on the rim
    # the crock stands to the right of the vessel in both presets: the right arm takes the tool, the left hand is on
    # the rim or the pan's handle. (On the left, 3 cm from where the left hand takes the pan's handle on the hob, it
    # would have the right arm reach across the left one and set off both arms' collision reflexes.)
    L["pot_xy"] = (L["station_x"] + 0.37, 0.25)
    L["mixer_xy"] = (L["station_x"] - 0.34, 0.38)
    L["knob_xy"] = (P.HOB_ZONE[0], 0.085)
    L["scale"] = 1.0 + (jit(0.10) if config.seed else 0.0)
    L["ready_pose"] = dict(P.READY_POSE_ARM)
    L["milk_c"] = 5.0 if (config.seed % 2 == 1) else 20.0
    L["stand_min"] = float(rng.uniform(1.0, 4.0)) if config.seed else 2.0
    return L


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
        self.static_boxes, self.static_cyls = [], []
        self.tools, self.vessels = {}, {}
        self.robot_bodies, self.pad_shapes = [], {}
        self.sleeves = config.variant == "sleeves"     # squared grip sleeves on the hand tools (tools.handle)
        self.arm_hw, self.joints, self.arms = {}, {}, {}
        self.layout = L = config.layout if getattr(config, "layout", None) else task_layout(config)
        self.sx = L["station_x"]
        self.room()
        self.station()
        if L.get("robot", True) and config.robot != "none":
            self.robots()
        self.views()

    # ------------------------------------------------------------------ helpers
    def cfg(self, **kw):
        v = dict(mu=P.MU["default"], ke=2e4, kd=100., density=1000., gap=P.CONTACT_GAP_M, margin=0., mu_torsional=0.,
                 mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def visual_cfg(self):
        return self.newton.ModelBuilder.ShapeConfig(density=0., has_shape_collision=False, has_particle_collision=False)

    def box(self, name, pos, half, material, body=-1, rot=None, collide=True, **kw):
        cfg = self.cfg(**kw) if collide else self.visual_cfg()
        i = self.b.add_shape_box(body, xform=self.wp.transform(tuple(map(float, pos)), rot or self.wp.quat_identity()),
                                 hx=float(half[0]), hy=float(half[1]), hz=float(half[2]), cfg=cfg, label=name)
        self.materials[i] = material
        if not collide:
            self.visual.add(i)
        if body < 0:
            self.static_boxes.append((name, tuple(map(float, pos)), tuple(map(float, half)), material,
                                      tuple(rot) if rot is not None else (0., 0., 0., 1.)))
        return i

    # ------------------------------------------------------------------ room, counter, cart
    def room(self):
        b = self.b
        b.add_ground_plane(cfg=self.cfg(mu=0.6), label="floor")
        self.materials[b.shape_count - 1] = "floor"
        R = P.ROOM
        cl, cd, ct = P.COUNTER_LENGTH, P.COUNTER_DEPTH, P.COUNTER_TOP_THICK
        cx = 0.25                                           # the counter run spans x = -0.95 .. 1.45
        self.counter_x = (cx - 0.5*cl, cx + 0.5*cl)
        self.box("counter_top", (cx, 0.5*cd, CZ - 0.5*ct), (0.5*cl, 0.5*cd, 0.5*ct), "quartz", mu=P.MU["steel_quartz"])
        self.box("cabinet_carcass", (cx, 0.5*cd + 0.0125, 0.5*(CZ - ct) + 0.05), (0.5*cl, 0.5*cd - 0.0125, 0.5*(CZ - ct) - 0.05),
                 "cabinet")
        self.box("toe_kick", (cx, 0.5*cd + 0.05, 0.05), (0.5*cl, 0.5*cd - 0.05, 0.05), "toe_kick")
        xm, hx = 0.5*(R["x0"] + R["x1"]), 0.5*(R["x1"] - R["x0"])
        ym, hy = 0.5*(R["y0"] + R["y1"]), 0.5*(R["y1"] - R["y0"])
        self.box("back_wall", (xm, R["y1"] + 0.05, 0.5*R["z1"]), (hx, 0.05, 0.5*R["z1"]), "wall_tile")
        # the room's other walls and its ceiling (the window's glass and the door's leaf are in their planes)
        self.box("wall_front", (xm, R["y0"] - 0.05, 0.5*R["z1"]), (hx, 0.05, 0.5*R["z1"]), "wall_paint")
        self.box("wall_left", (R["x0"] - 0.05, ym, 0.5*R["z1"]), (0.05, hy, 0.5*R["z1"]), "wall_paint")
        self.box("wall_right", (R["x1"] + 0.05, ym, 0.5*R["z1"]), (0.05, hy, 0.5*R["z1"]), "wall_paint")
        self.box("wall_ceiling", (xm, ym, R["z1"] + 0.05), (hx, hy, 0.05), "ceiling")
        # an open wall shelf, not wall cabinets, over the robot's bay: with 305 mm cabinets from 1.37 m up the hand
        # cannot come down on a tool standing in the crock (the wrist camera met the cabinet at 1259 N in a first reach)
        self.shelf_z = 1.74
        self.box("wall_shelf", (cx, R["y1"] - 0.125, self.shelf_z + 0.015), (0.5*cl, 0.125, 0.015), "shelf_wood")
        # the docked cart: deck plate, four posts, lower shelf with the two controllers
        sx, y0 = self.sx, -P.ARM_SETBACK
        deck_t = 0.012
        self.box("cart_deck", (sx, y0 - 0.06, P.ARM_BASE_Z - 0.5*deck_t), (0.44, 0.24, 0.5*deck_t), "cart_plate")
        for ix in (-1, 1):
            for iy in (-1, 1):
                self.box(f"cart_post_{ix}_{iy}", (sx + ix*0.41, y0 - 0.06 + iy*0.21, 0.5*(P.ARM_BASE_Z - deck_t)),
                         (0.02, 0.02, 0.5*(P.ARM_BASE_Z - deck_t)), "cart_frame")
        self.box("cart_shelf", (sx, y0 - 0.06, 0.20), (0.42, 0.22, 0.008), "cart_plate")
        for ix in (-1, 1):
            self.box(f"cart_controller_{ix}", (sx + ix*0.20, y0 - 0.06, 0.208 + 0.09), (0.16, 0.18, 0.09), "controller")
        self.counter_plane = dict(z=CZ, lo=(self.counter_x[0], 0.0), hi=(self.counter_x[1], P.COUNTER_DEPTH), floor=0.0)
        self.fittings()
        self.furniture()
        self.props()

    def furniture(self):
        """Static collision proxies of the room's furniture (params: WORK_TABLE, STOOLS, SHELVING, BASIN, DISPENSER,
        BIN, FRIDGE, PLANT_POT): boxes of the drawn outlines, upright cylinders for the plant's pot. Nothing here moves:
        a stool is fixed where it stands."""
        def span(name, x0, x1, ya, yb, z0, z1, material):
            self.box(f"fur_{name}", (0.5*(x0 + x1), 0.5*(ya + yb), 0.5*(z0 + z1)), (0.5*(x1 - x0), 0.5*(yb - ya), 0.5*(z1 - z0)),
                     material)
        t = P.WORK_TABLE
        (x0, x1), (ya, yb), zt = t["x"], t["y"], t["z"]
        span("table_top", x0, x1, ya, yb, zt - t["top"], zt, "steel_brushed")
        span("table_shelf", x0 + 0.03, x1 - 0.03, ya + 0.03, yb - 0.03, t["shelf_z"] - 0.02, t["shelf_z"], "steel_brushed")
        for i, lx in enumerate((x0 + 0.5*t["leg"], x1 - 0.5*t["leg"])):
            for j, ly in enumerate((ya + 0.5*t["leg"], yb - 0.5*t["leg"])):
                span(f"table_leg{i}{j}", lx - 0.5*t["leg"], lx + 0.5*t["leg"], ly - 0.5*t["leg"], ly + 0.5*t["leg"], 0.0, zt - t["top"],
                     "steel_brushed")
        r, zs = P.STOOL["seat_r"], P.STOOL["seat_z"]
        for k, (sx_, sy_) in enumerate(P.STOOLS):
            span(f"stool{k}", sx_ - r, sx_ + r, sy_ - r, sy_ + r, 0.0, zs, "wood")
        sh = P.SHELVING
        span("shelving", sh["x"][0], sh["x"][1], sh["y"][0], sh["y"][1], 0.0, sh["z"], "steel_brushed")
        bs = P.BASIN
        span("basin", bs["x"][0], bs["x"][1], bs["y"][0], bs["y"][1], bs["z"][0], bs["z"][1], "ceramic")
        span("basin_tap", 0.5*sum(bs["x"]) - 0.02, 0.5*sum(bs["x"]) + 0.02, bs["y"][1] - 0.16, bs["y"][1], bs["z"][1], bs["tap_z"],
             "steel_brushed")
        ds = P.DISPENSER
        span("dispenser", ds["x"][0], ds["x"][1], ds["y"][0], ds["y"][1], ds["z"][0], ds["z"][1], "plastic")
        bn = P.BIN
        span("bin", bn["xy"][0] - bn["r"], bn["xy"][0] + bn["r"], bn["xy"][1] - bn["r"], bn["xy"][1] + bn["r"], 0.0, bn["h"],
             "steel_brushed")
        # the fridge and the plant's pot
        f = P.FRIDGE
        (x0, x1), (ya, yb), (z0, z1), hd, ft = f["x"], f["y"], f["z"], f["handle"], f["foot"]
        span("fridge", x0, x1, ya, yb, z0, z1, "fridge_steel")
        for k, (za, zb) in enumerate(f["doors"]):
            span(f"fridge_door{k}", x0, x1, f["door_y"][0], ya, za, zb, "fridge_steel")
        for k, (za, zb) in enumerate(hd["z"]):
            span(f"fridge_handle{k}", x1 - hd["from_right"] - hd["r"], x1 - hd["from_right"] + hd["r"],
                 f["door_y"][0] - hd["stand"] - hd["r"], f["door_y"][0], za - hd["over"], zb + hd["over"], "steel_brushed")
        for k, (fx, fy) in enumerate((x, y) for y in (ya + ft["inset"], yb - ft["inset"]) for x in (x0 + ft["inset"], x1 - ft["inset"])):
            span(f"fridge_foot{k}", fx - ft["r"], fx + ft["r"], fy - ft["r"], fy + ft["r"], 0.0, z0, "steel_brushed")
        for k, (za, zb, r) in enumerate(P.PLANT_POT):
            pos = (P.PLANT_XY[0], P.PLANT_XY[1], 0.5*(za + zb))
            T.cylinder(self, -1, f"fur_plant_pot{k}", pos, r, 0.5*(zb - za), "z", "ceramic", gap=P.MESH_CONTACT_GAP_M)
            self.static_cyls.append((f"fur_plant_pot{k}", pos, r, 0.5*(zb - za)))

    def props(self):
        """What stands on the counter beside the work (params.PROPS for the task), each a free body with its pack's mass:
        a box of its outline, or for a jar, tin or bottle a twelve-sided prism of its outline; the body's frame is at
        the middle of its foot. dressing.py draws each on its body. (As static boxes, an arm that
        brushed a milk carton would be stopped by its reflex.)"""
        self.prop_list = []
        sx = P.STATION_X[self.c.preset]                      # the packets stand where the cook left them, not with the cart
        for k, (kind, dx, y, yaw) in enumerate(P.PROPS.get(self.c.task, ())):
            d = P.PROP[kind]
            a = math.radians(yaw)
            (hx, hy), h, m = d["half"], d["h"], d["mass"]
            name = f"prop_{k}_{kind}"
            mu = d.get("mu", P.MU["default"])
            pose = (np.array([sx + dx, y, CZ + 5e-4]), Rotation.from_euler("z", a))
            if d.get("round"):
                inertia = (m*(3*hx*hx + h*h)/12, m*(3*hx*hx + h*h)/12, 0.5*m*hx*hx)
            else:
                inertia = (m*(4*hy*hy + h*h)/12, m*(4*hx*hx + h*h)/12, m*(4*hx*hx + 4*hy*hy)/12)
            body = T.link(self, name, pose, mass=m, com=(0., 0., 0.45*h), inertia=inertia)
            if d.get("round"):
                T.disc(self, body, f"{name}_shell", (0., 0., 0.5*h), hx, 0.5*h, "plastic", density=0., mu=mu)
            else:
                i = self.box(f"{name}_shell", (0., 0., 0.5*h), (hx, hy, 0.5*h), "plastic", body=body, density=0., mu=mu)
                self.materials[i] = "collision_hidden"
                self.visual.add(i)
            jf = self.b.add_joint_free(body, label=f"{name}_free")
            self.b.add_articulation([jf], label=f"{name}_assembly")
            self.bodies[name] = body
            self.prop_list.append(dict(kind=kind, x=sx + dx, y=y, yaw=a, name=name, body=int(body)))
        # the policy camera's housing on its wall bracket
        c = P.STATION_CAMERA
        vx = self.layout["vessel_xy"][0]
        self.box("fit_station_camera", (vx, P.ROOM["y1"] - c["off_wall"] + 0.01, c["z"]), c["half"], "camera_dark")

    def fittings(self):
        """Static collision proxies of the room fittings an arm or a held tool can reach (params: SINK, TOWEL_RAIL,
        SHELF_ITEMS, SOCKETS). Each is the drawn outline as boxes; dressing.py draws them from the same numbers. They do
        not move: a canister on the shelf is fixed where it stands, the towel is a rigid slab."""
        cd, ct, y1 = P.COUNTER_DEPTH, P.COUNTER_TOP_THICK, P.ROOM["y1"]

        def span(name, x0, x1, ya, yb, z0, z1, material, **kw):
            self.box(f"fit_{name}", (0.5*(x0 + x1), 0.5*(ya + yb), 0.5*(z0 + z1)), (0.5*(x1 - x0), 0.5*(yb - ya), 0.5*(z1 - z0)),
                     material, **kw)
        # the sink unit: the top around the bowl, the carcass around and under the bowl, the toe kick
        sx0, sx1 = P.SINK["x"]
        bx0, bx1 = sx0 + P.SINK["bowl_inset"][0], sx1 - P.SINK["bowl_inset"][1]
        by0, by1 = P.SINK["bowl_y"]
        zb = CZ - P.SINK["bowl_depth"]
        for name, (ax0, ax1, ay0, ay1) in dict(left=(sx0, bx0, 0., cd), right=(bx1, sx1, 0., cd), front=(bx0, bx1, 0., by0),
                                               back=(bx0, bx1, by1, cd)).items():
            span(f"sink_top_{name}", ax0, ax1, ay0, ay1, CZ - ct, CZ, "quartz", mu=P.MU["steel_quartz"])
            span(f"sink_carcass_{name}", ax0, ax1, max(ay0, 0.025), ay1, zb, CZ - ct, "cabinet")
        span("sink_base", sx0, sx1, 0.025, cd, 0.10, zb, "cabinet")
        span("sink_toe_kick", sx0, sx1, 0.10, cd, 0.0, 0.10, "toe_kick")
        # the towel rail and the towel over it
        r = P.TOWEL_RAIL
        ry = y1 - r["off_wall"]
        span("rail_rod", r["x"][0], r["x"][1], ry - r["rod_r"], ry + r["rod_r"], r["z"] - r["rod_r"], r["z"] + r["rod_r"], "steel_brushed")
        for k, px in enumerate(r["posts"]):
            span(f"rail_post{k}", px - 0.005, px + 0.005, ry, y1, r["z"] - 0.005, r["z"] + 0.005, "steel_brushed")
        span("rail_towel", r["towel_x"][0], r["towel_x"][1], ry - r["rod_r"] - 0.004, y1, r["z"] - r["towel_drop"][1],
             r["z"] + r["rod_r"] + 0.004, "towel")
        # what stands on the shelf
        z_on = self.shelf_z + 0.030
        it = P.SHELF_ITEMS
        for k, (x, rad, h) in enumerate(it["canisters"]):
            span(f"canister{k}", x - rad, x + rad, y1 - it["canister_off_wall"] - rad, y1 - it["canister_off_wall"] + rad, z_on,
                 z_on + h + 0.033, "ceramic")
        for k, (x, rad, h) in enumerate(it["jars"]):
            span(f"jar{k}", x - rad, x + rad, y1 - it["jar_off_wall"] - rad, y1 - it["jar_off_wall"] + rad, z_on, z_on + h + 0.019,
                 "glass")
        x, rad, h = it["prep_bowls"]
        span("prep_bowls", x - rad, x + rad, y1 - it["prep_off_wall"] - rad, y1 - it["prep_off_wall"] + rad, z_on, z_on + h + 0.030,
             "steel_brushed")
        # under the shelf: the brackets (wall plate, arm, diagonal brace) and the light strip's profile
        z_under = self.shelf_z
        for k, bx in enumerate(P.SHELF_BRACKET_X):
            span(f"bracket{k}_wall", bx - 0.015, bx + 0.015, y1 - 0.004, y1, z_under - 0.20, z_under, "cart_frame")
            span(f"bracket{k}_arm", bx - 0.015, bx + 0.015, y1 - 0.22, y1, z_under - 0.004, z_under, "cart_frame")
            a, b = np.array([bx, y1 - 0.006, z_under - 0.175]), np.array([bx, y1 - 0.205, z_under - 0.006])
            d = (b - a)/np.linalg.norm(b - a)
            th = math.atan2(d[2], d[1])                       # the box's long (y) axis turned about x onto the brace
            self.box(f"fit_bracket{k}_brace", 0.5*(a + b), (0.005, 0.5*float(np.linalg.norm(b - a)), 0.005), "cart_frame",
                     rot=self.wp.quat(math.sin(0.5*th), 0., 0., math.cos(0.5*th)))
        sw, sh_ = P.SHELF_STRIP["size"]
        span("strip", self.counter_x[0] + 0.05, self.counter_x[1] - 0.05, y1 - P.SHELF_STRIP["off_wall"] - 0.5*sw,
             y1 - P.SHELF_STRIP["off_wall"] + 0.5*sw, z_under - sh_, z_under, "aluminum")
        # on the sink unit: the tap (column and spout) and the bottle beside the bowl
        t = P.SINK_TAP
        tx, ty = 0.5*(bx0 + bx1), by1 + t["behind_bowl"]
        span("tap_column", tx - t["column"][0], tx + t["column"][0] + 0.03, ty - t["column"][0], ty + t["column"][0], CZ,
             CZ + t["column"][1] + 0.012, "steel_brushed")
        span("tap_spout", tx - t["spout_r"], tx + t["spout_r"], ty - t["spout"][0] - t["spout_r"], ty, CZ + t["spout"][1] - 0.006,
             CZ + t["column"][1] + 0.012, "steel_brushed")
        bt = P.SINK_BOTTLE
        span("sink_bottle", sx1 - bt["from_right"] - bt["half"][0], sx1 - bt["from_right"] + bt["half"][0], bt["y"] - bt["half"][1],
             bt["y"] + bt["half"][1], CZ, CZ + bt["height"], "plastic")
        # socket plates on the tiles
        w, t, hgt = P.SOCKETS["size"]
        for k, x in enumerate(P.SOCKETS["x"]):
            span(f"socket{k}", x - 0.5*w, x + 0.5*w, y1 - t, y1, P.SOCKETS["z"] - 0.5*hgt, P.SOCKETS["z"] + 0.5*hgt, "plastic")

    # ------------------------------------------------------------------ vessels, tools, appliances
    def station(self):
        L, c = self.layout, self.c
        vx, vy = L["vessel_xy"]
        z0 = CZ
        if L.get("mat"):
            mx, my, mz = P.MAT["size"]
            self.box("mat", (vx, vy, CZ + 0.5*mz), (0.5*mx, 0.5*my, 0.5*mz), "silicone_mat", mu=P.MU["silicone"])
            z0 = CZ + mz
        kind = L["vessel"]
        if kind == "bowl5":
            s = P.BOWL5
            prof = G.bowl_profile(s["rim_r"], s["depth"], s["base_r"])
            v = T.Vessel(self, "bowl", prof, wall_rigid=0.003, mass=s["mass"], pos=(vx, vy, z0 + 0.003 + 1e-4),
                         material="steel_brushed", mu=P.MU["steel_steel"], bead=0.005, render_wall=0.0012)
        elif kind == "bowl2":
            s = P.BOWL2
            prof = G.bowl_profile(s["rim_r"], s["depth"], s["base_r"])
            v = T.Vessel(self, "bowl", prof, wall_rigid=s["wall"], mass=s["mass"], pos=(vx, vy, z0 + s["wall"] + 1e-4),
                         material="glass", mu=P.MU["glass"], sectors=20, bands=4)
        elif kind == "pan":
            s = P.PAN
            prof = G.cylinder_profile(s["radius"], s["depth"], corner=s["corner"])
            a = math.radians(215.) + L.get("vessel_yaw", 0.)           # handle toward the left arm
            r0 = s["radius"] + s["wall"] + s["handle_r"]            # the capsule's end cap stays outside the wall
            p0 = (r0*math.cos(a), r0*math.sin(a), s["depth"] - 0.012)
            p1 = ((r0 + s["handle_len"])*math.cos(a), (r0 + s["handle_len"])*math.sin(a), s["depth"] + 0.030)
            v = T.Vessel(self, "pan", prof, wall_rigid=s["wall"], mass=s["mass"] - 0.12, pos=(vx, vy, z0 + s["wall"] + 1e-4),
                         material="steel_brushed", mu=P.MU["steel_steel"], sectors=20, bands=3,
                         handle=dict(p0=p0, p1=p1, r=s["handle_r"], mass=0.12, material="steel_brushed", mu=P.MU["default"]))
        elif kind == "pitcher":
            s = P.PITCHER
            prof = G.tapered_profile(s["r_bottom"], s["r_top"], s["depth"])
            v = T.Vessel(self, "pitcher", prof, wall_rigid=s["wall"], mass=s["mass"], pos=(vx, vy, z0 + s["wall"] + 1e-4),
                         material="glass", mu=P.MU["glass"], sectors=16, bands=3)
        else:
            raise ValueError(kind)
        v.role = kind
        self.vessel = v
        self.vessels[v.name] = v
        self.vessel_start = np.array([vx, vy, float(self.b.body_q[v.body][2])])
        # utensil crock with a three-way divider; tools stand head-down, one per compartment
        px, py = L["pot_xy"]
        s = P.POT
        prof = G.cylinder_profile(s["radius"], s["depth"], corner=0.006)
        pot = T.Vessel(self, "crock", prof, wall_rigid=s["wall"], mass=s["mass"], pos=(px, py, CZ + s["wall"] + 1e-4),
                       material="ceramic", mu=P.MU["default"], sectors=16, bands=2, divider=3)
        self.pot = pot
        self.vessels["crock"] = pot
        r_slot = s["radius"]/(1.0 + math.sin(math.radians(60.)))      # centre of a compartment's inscribed circle
        for k, name in enumerate(L["tools"]):
            a = SLOT_ANGLES[k]
            lean = L["tool_lean"][k] if k < len(L.get("tool_lean", ())) else (0., 0.)
            length = dict(whisk=P.WHISK, spatula=P.SPATULA, spoon=P.SPOON, stirrer=P.STIRRER, rod=P.ROD,
                          thermometer=P.THERMOMETER)[name]["length"]
            butt = np.array([px + r_slot*math.cos(a), py + r_slot*math.sin(a), CZ + s["wall"] + length + 0.004])
            if name in ("spoon", "stirrer", "rod", "thermometer"):
                lean = (lean[0] + 0.16, lean[1])          # a flat-ended tool stands on its end unless it is leant
            R = Rotation.from_euler("z", a + math.pi/2)*Rotation.from_euler("x", math.pi)*Rotation.from_euler("xy", lean)
            # lean about the head (which stands on the crock floor), not the butt
            head = butt + np.array([0., 0., -length])
            pose = (head - R.apply((0., 0., length)), R)
            if name == "whisk":
                t = T.Whisk(self, name, pose)
            elif name == "spatula":
                t = T.Spatula(self, name, pose)
            elif name == "spoon":
                t = T.Paddle(self, name, pose, P.SPOON, "spoon", "wood", P.MU["wood_steel"])
            elif name == "stirrer":
                t = T.Paddle(self, name, pose, P.STIRRER, "stirrer", "wood", P.MU["wood_steel"])
            elif name == "thermometer":
                t = T.Thermometer(self, name, pose)
            elif name == "rod":
                t = T.Paddle(self, name, pose, dict(P.ROD, handle_r=P.ROD["r"], blade=(2*P.ROD["end_r"], 0.030), blade_t=0.004),
                             "rod", "steel_brushed", P.MU["steel_steel"])
            t.slot = k
            self.tools[name] = t
            self.bodies[name] = t.body
        if L.get("mixer"):
            # the mixer stands on its heel, as it is sold and as a cook sets it down: nose up, the handle a vertical bar
            # on the side toward the robot, the beaters level and pointing at the wall. (Standing, its handle is 50 mm
            # above the arm's base height and the FR3 reaches it only near its wrist limits. That is the arm's affair.)
            mx, my = L["mixer_xy"]
            hx, hy, hz = (0.5*v_ for v_ in P.MIXER["body"])
            R = Rotation.from_matrix(np.column_stack([(0., 0., 1.), (-1., 0., 0.), (0., -1., 0.)]))
            self.tools["mixer"] = T.Mixer(self, "mixer", (np.array([mx, my, CZ + hx + 5e-4]), R))
            self.bodies["mixer"] = self.tools["mixer"].body
            self.mixer_rest_z = CZ + hx                     # the housing centre's height when the mixer stands on its heel
        if "hob" in L:
            hx, hy = P.HOB_ZONE
            self.box("hob_glass", (hx, hy, CZ - 0.002 + 1e-4), (0.16, 0.26, 0.002), "hob_glass", collide=False)
            kx, ky = L["knob_xy"]
            self.tools["knob"] = T.HobKnob(self, "hob_knob", (kx, ky, CZ), level=int(L["hob"]))
            self.bodies["hob_knob"] = self.tools["knob"].body

    # ------------------------------------------------------------------ robots
    def mounts(self):
        sx = self.sx
        return {"left_arm_mount": (sx - 0.5*P.ARM_SPACING, -P.ARM_SETBACK, P.ARM_BASE_Z),
                "right_arm_mount": (sx + 0.5*P.ARM_SPACING, -P.ARM_SETBACK, P.ARM_BASE_Z)}

    def robots(self):
        from robots.catalog import for_environment
        from robots.backends.newton import mount
        import robots
        setup = for_environment("ingredient-mixing", self.c.robot)
        self.robot_setup = setup
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
        (barcode_scanning / wine_pouring / drilling lineage)."""
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
            # the finger's own collision mesh is not left beside the boxes: the two overlapped (the mesh 0.09 mm
            # behind the pad's face), the mesh took half of every pinch and the two surfaces' frictions pulled against
            # each other with 10 to 11 N while a 0.09 kg tool hung still in the hand; the FR3's fingers have the
            # boxes alone (found with a per-tick record of the hand)
            off = ~(int(self.newton.ShapeFlags.COLLIDE_SHAPES) | int(self.newton.ShapeFlags.COLLIDE_PARTICLES))
            for i in meshes:
                b.shape_flags[i] = int(b.shape_flags[i]) & off

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
        sx = self.sx
        vx, vy = self.layout["vessel_xy"]
        top = CZ + self.vessel.depth
        # The policy's station camera: on a bracket on the wall above the backsplash, looking back over the vessel at
        # the robot: the vessel's contents, both hands, the crock and the mixer or the knob are in the picture and the
        # arms come into it from the far side. (It first hung from a mast on the cart, looking straight down between the
        # arms: half the picture was robot and the vessel was small. The mast went with it.)
        c = P.STATION_CAMERA
        yw = P.ROOM["y1"] - c["off_wall"]
        self.cameras["station_rgb"] = dict(body=-1, eye=[vx, yw - 0.035, c["z"]], target=[vx, vy - 0.20, CZ + 0.02],
                                           up=[0, 0, 1], fov=c["fov"], role="policy", width=960, height=720)
        self.cameras["overhead_rgb"] = dict(body=-1, eye=[vx, vy + 0.085, self.shelf_z - 0.03], target=[vx, vy, CZ],
                                            up=[0, 1, 0], fov=40., role="policy", width=720, height=720)
        # the room: the robot at its station from the right and from the left, and the whole kitchen from its two
        # far corners (the door's and the dry store's)
        self.cameras["hero"] = dict(body=-1, eye=[sx + 1.95, -1.75, 1.66], target=[sx - 0.20, 0.20, 1.02], up=[0, 0, 1],
                                    fov=56., role="diagnostic", width=1280, height=720)
        self.cameras["hero_left"] = dict(body=-1, eye=[sx - 2.0, -1.95, 1.66], target=[sx + 0.35, 0.15, 1.05], up=[0, 0, 1],
                                         fov=56., role="diagnostic", width=1280, height=720)
        R = P.ROOM
        self.cameras["kitchen"] = dict(body=-1, eye=[R["x0"] + 1.05, R["y0"] + 0.22, 2.05], target=[0.5*sx + 0.75, -0.25, 0.95],
                                       up=[0, 0, 1], fov=66., role="diagnostic", width=1280, height=720)
        self.cameras["kitchen_door"] = dict(body=-1, eye=[R["x1"] - 0.30, R["y0"] + 0.30, 1.95], target=[0.5*sx - 0.35, -0.25, 0.95],
                                            up=[0, 0, 1], fov=64., role="diagnostic", width=1280, height=720)
        self.cameras["back_corner"] = dict(body=-1, eye=[R["x1"] - 0.45, R["y1"] - 0.40, 1.90], target=[0.5*sx - 0.75, -2.55, 0.90],
                                           up=[0, 0, 1], fov=66., role="diagnostic", width=1280, height=720)
        self.cameras["station"] = dict(body=-1, eye=[sx + 1.50, -1.15, 1.62], target=[sx - 0.03, 0.18, 1.04], up=[0, 0, 1],
                                       fov=48., role="diagnostic", width=1280, height=720)
        self.cameras["station_left"] = dict(body=-1, eye=[sx - 1.50, -1.15, 1.62], target=[sx + 0.03, 0.18, 1.04], up=[0, 0, 1],
                                            fov=48., role="diagnostic", width=1280, height=720)
        self.cameras["room"] = dict(body=-1, eye=[sx + 1.55, -1.75, 1.80], target=[sx + 0.05, 0.20, 1.02], up=[0, 0, 1],
                                    fov=50., role="diagnostic", width=1280, height=720)
        # the two close overviews: from the right front (the right hand, its tool, the vessel and the crock) and
        # from the left front (the left hand on the rim, the handle or the mixer). A view from straight ahead of the
        # cart has an arm or the mast in front of the vessel whatever its height.
        self.cameras["front"] = dict(body=-1, eye=[sx + 1.15, -0.55, 1.75], target=[vx + 0.12, vy, CZ + 0.10], up=[0, 0, 1],
                                     fov=40., role="diagnostic", width=1280, height=720)
        self.cameras["front_left"] = dict(body=-1, eye=[sx - 1.05, -0.35, 1.80], target=[vx + 0.12, vy, CZ + 0.10],
                                          up=[0, 0, 1], fov=40., role="diagnostic", width=1280, height=720)
        self.cameras["closeup"] = dict(body=-1, eye=[vx + 0.36, vy - 0.40, top + 0.26], target=[vx, vy, CZ + 0.04],
                                       up=[0, 0, 1], fov=46., role="diagnostic", width=1280, height=720)
        # from the backsplash looking out over the vessel: the crock stands at +x in both presets and the idle or
        # holding hand at -x, so a view from either side has something in front of the vessel
        self.cameras["side"] = dict(body=-1, eye=[vx + 0.30, P.COUNTER_DEPTH - 0.035, CZ + 0.36], target=[vx, vy, CZ + 0.07],
                                    up=[0, 0, 1], fov=52., role="diagnostic", width=1280, height=720)
