"""Newton model of a forklift session: the drivable surfaces (for the tyre model and native contact), the static
structures, the truck (truck.py), the loads (loads.py), the dock and trailer (dock.py) and the robot (robots catalog).

World frame: z up, metres. Each preset's layout (facility.py) fixes its origin; the lab preset is an empty dry concrete
floor with optional probe fixtures (ramps, a tilt table, an obstacle) given by config.extras.
"""

import math

import numpy as np

from . import params as P
from .tyres import Surface


CARRIED_BUILD_Z = 30.0          # build height of a carried load (clear of the scene until initial_q seats it)


def quat_axis_angle(axis, angle):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    s = math.sin(angle / 2)
    return (float(a[0] * s), float(a[1] * s), float(a[2] * s), float(math.cos(angle / 2)))


class Scene:
    def __init__(self, config, newton, wp):
        self.c, self.newton, self.wp = config, newton, wp
        self.b = newton.ModelBuilder(gravity=(0.0, 0.0, -P.G))
        newton.solvers.SolverMuJoCo.register_custom_attributes(self.b)
        self.shape_group = {}           # newton shape -> collision group name (world.GROUPS)
        self.materials = {}             # newton shape -> material name
        self.surfaces = []              # tyre-model drivable surfaces
        self.bodies = {}
        self.statics = {}               # name -> list of shapes (structures the evaluator watches)
        self.props = []                 # render-only geometry (dressing): dicts consumed by usd.py
        self.zones = {}                 # evaluator geometry (polygons, lines), fixed by task and seed
        self.presence_override = False
        self.robot_mass = 0.0
        self.robot_bodies, self.robot_shapes = [], []
        self.rng = np.random.default_rng(config.seed)
        self.layout = None
        self.task_start = None
        self.load_specs = []
        self.trailers, self.levelers, self.machines = {}, {}, {}
        self.ramp_lips = {}          # the yard ramp's hinged lip plates (dock.add_ramp_lip)
        self.loads = {}              # every free load body (loads.build; stored rack units from facility.build_bodies)
        if config.preset == "lab":
            self.lab()
            ex = config.extras
            self.task_start = dict(task="free", seed=config.seed, engine=ex.get("engine", "off"), key=ex.get("key", "off"),
                                   park=ex.get("park", True), direction="N", lift=ex.get("lift", 0.10),
                                   tilt_deg=ex.get("tilt_deg", 0.0), loads=[], load_on_forks=ex.get("fork_load"), floor={},
                                   belt=True)
            self.load_specs = [dict(l) for l in ex.get("loads", [])]
        else:
            from . import facility, task as TK
            facility.build(self)
            self.task_start = st = TK.resolve(config, self.layout)
            if "truck_pose" in config.extras:                 # probes only (extras are refused by the backend)
                st["truck"] = tuple(config.extras["truck_pose"])
            for k in ("lift", "tilt_deg", "engine", "key", "park", "load_on_forks"):
                if k in config.extras:
                    st[k] = config.extras[k]
            if "trailer_override" in config.extras:
                for t in st.get("trailers", {}).values():
                    t.update(config.extras["trailer_override"])
            self.truck_pose = st["truck"]
            facility.build_bodies(self, st)
            for name, wp_ in st["floor"].items():
                self.floor_rect(name, wp_["centre"], wp_["half"], wp_.get("material", "concrete_wet"), priority=1,
                                thickness=0.02, collider=False)
                wet = "wet" in wp_.get("material", "concrete_wet")
                if config.preset in ("warehouse", "dock") and wet:     # a wet-floor sign beside a wet patch only (kit S_WetFloorSign)
                    c, h = wp_["centre"], wp_["half"]
                    self.static_box(f"wet_sign_{name}", (c[0] - h[0] - 0.4, c[1] + h[1] - 0.1, 0.32), (0.16, 0.13, 0.32),
                                    "sign", watch="structure")
            self.load_specs = [dict(l) for l in st["loads"]]
        self.add_truck()
        from . import loads
        loads.build(self)
        if self.task_start and self.task_start.get("load_on_forks"):
            self.add_fork_load(self.task_start["load_on_forks"])
        self.add_robot()

    # ------------------------------------------------------------------ helpers
    def cfg(self, mu=P.MU_TRUCK, **kw):
        # gap: ignored by the engine here (every geom's MuJoCo margin is 0; see truck.build). Unset, Newton would use 0.1
        v = dict(mu=mu, density=0.0, gap=0.004, margin=0.0)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def static_body(self, name, origin):
        """A fixed body (welded to the world, no dof) grouping static shapes: MuJoCo's body-level broadphase then culls
        whole structures instead of testing every static geom against every moving one."""
        wp = self.wp
        b = self.b.add_link(xform=wp.transform(tuple(map(float, origin)), wp.quat_identity()), mass=0.0, label=f"static/{name}")
        j = self.b.add_joint_fixed(-1, b, parent_xform=wp.transform(tuple(map(float, origin)), wp.quat_identity()),
                                   label=f"static/{name}")
        self.b.add_articulation([j], label=f"static/{name}")
        self.bodies[f"static/{name}"] = b
        return b

    def static_box(self, name, centre, half, material, group="obj", rot=None, mu=None, watch=None, body=-1, origin=None,
                   as_mesh=False):
        c = np.asarray(centre, float) - (np.asarray(origin, float) if origin is not None else 0.0)
        tf = self.wp.transform(tuple(map(float, c)), rot or self.wp.quat_identity())
        mu = mu if mu is not None else P.MU_STRUCTURE                # P.MU is the tyres' table, not contact friction
        i = box_shape(self.b, self.newton, body, tf, half, self.cfg(mu=mu), name, force_mesh=as_mesh)
        self.shape_group[i] = group
        self.materials[i] = material
        if watch:
            self.statics.setdefault(watch, []).append(i)
        return i

    def floor_rect(self, name, centre, half, material, z_top=0.0, grade=0.0, axis=(1.0, 0.0), body=-1, priority=0,
                   thickness=0.20, collider=True):
        """A drivable rectangle: a static (or body-attached) box for native contact plus the tyre-model surface.
        grade: rise/run along `axis` (the surface rises in +axis); centre: (x, y) of the surface centre at z_top."""
        ax = np.array([axis[0], axis[1], 0.0]); ax /= np.linalg.norm(ax)
        ang = math.atan(grade)
        side = np.cross([0, 0, 1.0], ax)
        n = np.array([-ax[0] * math.sin(ang), -ax[1] * math.sin(ang), math.cos(ang)])
        u = ax * math.cos(ang) + np.array([0, 0, math.sin(ang)])
        o = np.array([centre[0], centre[1], z_top], float)
        self.surfaces.append(Surface(name, body, o, n, u, (half[0] / math.cos(ang) if grade else half[0], half[1]),
                                     material, priority))
        if collider:
            yaw = math.atan2(ax[1], ax[0])
            q_yaw = quat_axis_angle((0, 0, 1), yaw)
            q_pitch = quat_axis_angle((0, 1, 0), -ang)
            from scipy.spatial.transform import Rotation
            q = (Rotation.from_quat(q_yaw) * Rotation.from_quat(q_pitch)).as_quat()
            c = o - n * thickness / 2
            hx = half[0] / math.cos(ang) if grade else half[0]
            return self.static_box(name, c, (hx, half[1], thickness / 2), material, group="drivable",
                                   rot=self.wp.quat(*q), as_mesh=True)
        return None

    # ------------------------------------------------------------------ presets
    def lab(self):
        """Probe floor: 120 x 60 m of dry sealed concrete, plus optional fixtures from config.extras."""
        ex = self.c.extras
        mat = ex.get("floor_material", "concrete_dry")
        fc, fh = ex.get("floor", ((20.0, 0.0), (60.0, 30.0)))
        self.floor_rect("floor", tuple(fc), tuple(fh), mat, thickness=0.5)
        for k, r in enumerate(ex.get("ramps", [])):
            # ramp: dict(x0, length, grade, width, y=0, material): rises from z=0 at x0 along +x, then a level top
            L, g, w = r["length"], r["grade"], r.get("width", 3.0)
            y = r.get("y", 0.0)
            run = L * math.cos(math.atan(g)); rise = L * math.sin(math.atan(g))
            self.floor_rect(f"ramp{k}", (r["x0"] + run / 2, y), (L / 2, w / 2), r.get("material", "concrete_dry"),
                            z_top=rise / 2, grade=g, priority=1, thickness=0.3)
            top = r.get("top", 6.0)
            self.floor_rect(f"ramp{k}_top", (r["x0"] + run + top / 2, y), (top / 2, w / 2),
                            r.get("material", "concrete_dry"), z_top=rise, priority=1, thickness=0.3)
        self.truck_pose = ex.get("truck_pose", (0.0, 0.0, 0.0))

    # ------------------------------------------------------------------ truck
    def add_truck(self):
        from . import truck as TR
        x, y, yaw = self.truck_pose
        # built 16 mm into the ground (the asset's wheel centres then sit 14 / 16 mm low); the world then lowers and
        # pitches it onto its static tyre deflections (World._spawn_on_tyres)
        self.truck_ground = (self.task_start or {}).get("truck_z", self.ground_height(x, y))
        z0 = self.truck_ground - 0.016
        self.truck = TR.build(self.b, origin=(x, y, z0), yaw=yaw)
        self.truck_origin = (x, y, z0)
        for s, (body, tag, group) in self.truck.shapes.items():
            self.shape_group[s] = group
            self.materials[s] = "steel" if group == "truck" else "rubber"
        self.bodies.update({f"truck/{k}": v for k, v in self.truck.bodies.items()})
        ballast = self.c.extras.get("seat_ballast_kg", P.OPERATOR_MASS if self.c.robot == "none" else 0.0)
        if ballast:
            # ASSISTED rig for robot-less probes: a ballast strapped to the seat stands in for the operator's weight
            sb = self.truck.bodies["seat"]
            self.b.body_mass[sb] += float(ballast)
            self.truck.masses["seat"] += float(ballast)
            self.seat_ballast = float(ballast)
        else:
            self.seat_ballast = 0.0

    def ground_height(self, x, y):
        best = 0.0
        for s in self.surfaces:
            if s.body >= 0:
                continue
            d = np.array([x, y, 0.0]) - s.origin
            v = np.cross(s.normal / np.linalg.norm(s.normal), s.u)
            if abs(d @ s.u) <= s.half[0] and abs(d @ v) <= s.half[1]:
                n = s.normal / np.linalg.norm(s.normal)
                z = s.origin[2] - ((x - s.origin[0]) * n[0] + (y - s.origin[1]) * n[1]) / n[2]
                best = max(best, z)
        return best

    # ------------------------------------------------------------------ robot (robots catalog)
    def add_robot(self):
        self.robot_pending = None
        if self.c.robot == "none":
            return
        from . import operator as OP
        OP.add(self)

    def bind_robot(self, model):
        if self.robot_pending is None:
            return None
        return self.robot_pending.bind(model)

    def robot_start(self, world):
        from . import operator as OP
        OP.start(world)

    def robot_write(self, world, jf):
        world.operator.write(world, jf)

    def start_systems(self, world):
        """Truck systems at the task's start condition (engine idling or off)."""
        st = self.task_start
        if st is not None and st["engine"] == "idle" and st["key"] == "on":
            world.drivetrain.set_running()

    def mujoco_hook(self, world, m):
        from . import loads
        loads.mujoco_hook(world, m)
        if self.c.robot != "none":
            from . import operator as OP
            OP.mujoco_hook(world, m)

    def add_fork_load(self, spec):
        """The load carried at the start, built clear of everything (above the scene); initial_q seats it in the fork
        openings before the solver's first contact pass sees it."""
        from . import loads as LD
        x, y, yaw = self.truck_pose
        x, y = x + 0.0, y + 0.0
        if spec["kind"] == "wrapped":
            g = P.GMA_PALLET
            # depth: the load's length along the forks when it is shorter than the pallet; it stands at the pallet's
            # mast end, against the load backrest (the truck's forward axis is the pallet's -x)
            depth = float(spec.get("depth", g["L"] - 0.04))
            LD.add_pallet(self, "carried", (x, y, yaw), z0=CARRIED_BUILD_Z,
                          load=dict(kind="wrapped", mass=spec["mass"], size=(depth, g["W"] - 0.04, spec["height"]),
                                    offset_x=spec.get("offset_sign", -1.0) * (g["L"] - 0.04 - depth) / 2))
        elif spec["kind"] == "coil":
            LD.add_pallet(self, "carried", (x, y, yaw), load=dict(kind="coil", mass=spec["mass"], size=(P.COIL_LOAD["length"], P.COIL_LOAD["length"],
                                                                     P.COIL_LOAD["diameter"])),
                          z0=CARRIED_BUILD_Z)
        elif spec["kind"] == "stack":
            LD.add_carton_stack(self, "carried", (x, y, yaw), z0=CARRIED_BUILD_Z, layers=spec.get("layers", 5), rng=self.rng)
        self.fork_load = spec

    def initial_q(self, world, q):
        # the seat at its loaded ride height (the robot settles onto it there)
        if self.c.robot != "none" or self.seat_ballast:
            q[world.q_of["seat"]] = P.SEAT_RIDE
        st = self.task_start
        if st is None:
            return
        qs = world.model.joint_q_start.numpy()
        labels = list(world.model.joint_label)
        for name, lv in self.levelers.items():
            for jname, val in zip((f"leveler/{name}_deck", f"leveler/{name}_lip"), lv["q"]):
                q[int(qs[labels.index(jname)])] = val
        for lip in self.ramp_lips.values():
            for jname, val in zip(lip["joints"], lip["q"]):
                q[int(qs[labels.index(jname)])] = val
        from .task import blade_bottom_for
        car, mi = blade_bottom_for(st["lift"])
        q[world.q_of["carriage"]] = car
        q[world.q_of["mast_int"]] = mi
        q[world.q_of["mast_inner"]] = mi
        q[world.q_of["mast_outer"]] = math.radians(st["tilt_deg"])
        q[world.q_of["key"]] = P.KEY["on"] if st["key"] == "on" else 0.0
        q[world.q_of["park_lever"]] = -P.PARK_LEVER["travel"] if st["park"] else 0.0
        q[world.q_of["dir_lever"]] = {"F": -1, "N": 0, "R": 1}[st["direction"]] * P.DIR_LEVER["travel"]
        if getattr(self, "fork_load", None):
            self._seat_fork_load(world, q)

    def _seat_fork_load(self, world, q):
        """Place the carried load in the fork openings: the pallet's top deck resting on the blades, pushed against
        the load backrest, levelled to the blades' pitch (FK of the truck at its start joint values)."""
        from .mathx import quat_to_R
        from scipy.spatial.transform import Rotation
        q_build = np.array(q, copy=True)               # q may be a view of model.joint_q: keep the build positions
        world.state.joint_q.assign(q)
        world.newton.eval_fk(world.model, world.state.joint_q, world.state.joint_qd, world.state)
        bq = world.state.body_q.numpy().astype(float)
        ss = self.truck.bodies["sideshift"]
        R = quat_to_R(bq[ss:ss + 1, 3:7])[0]
        o = bq[ss, :3]
        anchor = np.asarray(self.truck.anchors["sideshift"])
        g = P.GMA_PALLET
        # in the sideshift frame (truck-frame axes at the build pose): blade top z 0.1402 (its collision hull, measured;
        # 0.159 is the heel radius's top), fork face x 0.544
        local = np.array([0.544 + 0.012 + g["L"] / 2, 0.0, 0.1402 + 0.002 - (g["H"] - g["deck"])]) - anchor
        p_pal = o + R @ local
        qs = world.model.joint_q_start.numpy()
        labels = list(world.model.joint_label)
        quat = Rotation.from_matrix(R).as_quat()
        for name, info in self.loads.items():
            if not name.startswith("carried"):
                continue
            j = labels.index(f"load/{name}")
            a = int(qs[j])
            if name == "carried":
                q[a:a + 3] = p_pal
            else:                                       # a carton keeps its build offset from the pallet, in the fork frame
                pb = np.asarray(q_build[a:a + 3], float)
                jp = labels.index("load/carried")
                pp = np.asarray(q_build[int(qs[jp]):int(qs[jp]) + 3], float)
                Ryaw = Rotation.from_euler("z", self.truck_pose[2]).as_matrix()
                q[a:a + 3] = p_pal + R @ (Ryaw.T @ (pb - pp))
            q[a + 3:a + 7] = quat



BOX_MESH_HALF = 0.0          # every box is built as an 8-vertex convex mesh (the fork blades' pieces are the only
                             # box primitives, and never meet another box)


def box_shape(b, newton, body, tf, half, cfg, label, force_mesh=False):
    """A box collision shape, built as a box-shaped convex mesh so MuJoCo collides it through its general convex
    (GJK/EPA, multi-contact) path instead of the dedicated box-box routine. That routine generated two 20 m deep
    contacts between the 52 m floor box and an upside-down carton at rest on it (carry-loose-stack demo, tick 10193;
    the same state recomputed with a plane floor gave the 4 correct contacts), threw a set-down pallet and the
    carriage 1.5 m up off the 16 m trailer floor box (load-trailer, t = 48.0 s), gave two pallets one contact 1.22 m
    deep, and gave two identical 0.5 x 0.4 x 0.25 m cartons in a column stack contacts 0.40-0.56 m deep at 100-260 kN
    (carry-loose-stack set-down): the routine is not safe at any size here, so no box meets a box."""
    hx, hy, hz = (float(v) for v in half)
    if force_mesh or max(hx, hy, hz) >= BOX_MESH_HALF:
        V = np.array([(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
        F = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4,
                      1, 5, 7, 1, 7, 3], np.int32)
        return b.add_shape_mesh(body, xform=tf, mesh=newton.Mesh(V, F, compute_inertia=False), cfg=cfg, label=label)
    return b.add_shape_box(body, xform=tf, hx=hx, hy=hy, hz=hz, cfg=cfg, label=label)
