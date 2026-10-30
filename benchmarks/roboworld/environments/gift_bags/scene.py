"""Scene specifications: furniture, robot mounts, packaging items, sheets and their contact-world setup.

Geometry that a robot or a sheet can touch is physical (furniture boxes in both simulators, items and sheets
in the contact world). Room shells and props beyond reach are render-only dressing (dressing.py).
"""

from dataclasses import dataclass, field
import math
import numpy as np

from . import ipcworld as W
from . import params as Pm


@dataclass
class SceneSpec:
    preset: str
    surface_z: float                     # work-surface height (m)
    mounts: dict                         # mount name -> ((x, y), yaw)
    mount_z: float
    furniture: list                      # dicts: name, center, half, material, mu, (robot_collides, ipc)
    sheets: list = field(default_factory=list)
    rods: list = field(default_factory=list)
    bodies: list = field(default_factory=list)          # contact-world items + furniture near the work area
    ipc_setup: list = field(default_factory=list)       # callables(ipcworld, world): pins, hinges, bonds, films
    pair_mu: dict = field(default_factory=dict)
    ready_pose: dict = None
    cameras: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)       # task geometry for evaluators/renders
    contact_stock: object = None                       # caliper stock when there are no sheets (contact_caliper)
    g1: dict = None                                    # Unitree G1 configuration: where it stands, its arm posture (G1_LAYOUT)


def _rz(yaw):
    c, s_ = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s_, 0.], [s_, c, 0.], [0., 0., 1.]])


def place_rigid(move, centre, *, sheets=(), rods=(), bodies=()):
    """Rigid move (dx, dy, yaw) about the vertical axis through `centre` of settled start-state items: sheet and rod
    vertices, body poses. A settled state on a flat support stays settled (variants.placement)."""
    dx, dy, yaw = move
    if not (dx or dy or yaw):
        return
    R = _rz(yaw)
    c = np.array([centre[0], centre[1], 0.])
    t = np.array([dx, dy, 0.])
    for sh in sheets:
        sh.x0 = (np.asarray(sh.x0, float) - c) @ R.T + c + t
    for rd in rods:
        rd.x0 = (np.asarray(rd.x0, float) - c) @ R.T + c + t
    for b in bodies:
        b.p = R @ (np.asarray(b.p, float) - c) + c + t
        b.R = R @ np.asarray(b.R, float)


def furniture_body(f):
    V, F = W.box_mesh(f["half"])
    return W.Body(f["name"], V, F, kind="static", p=np.asarray(f["center"], float), R=_rz(f["yaw"]) if f.get("yaw") else np.eye(3),
                  mu=f.get("mu", .4), fmat=f.get("fmat"), render=dict(material=f.get("material", "furniture")))


def add_props(spec, bodies, top):
    """Collision boxes of the drawn props (dressing.physical_props) in both simulators: furniture for the robots, static
    bodies for the paper and the items. They take the work surface's friction class and are not drawn (the dressing
    draws the props themselves)."""
    from . import dressing
    props = [dict(p, mu=top.get("mu", .4), fmat=top.get("fmat"), visual=False)
             for p in dressing.physical_props(spec.preset, spec.surface_z, top)]
    spec.furniture += props
    bodies += [furniture_body(p) for p in props]
    spec.geometry["props"] = [dict(name=p["name"], center=list(p["center"]), half=list(p["half"]), yaw=float(p.get("yaw", 0.)))
                              for p in props]
    return props


def box_body(name, half, p, mass, *, R=None, mu=.4, n=(2, 2, 2), material="box", fmat=None):
    V, F = W.box_mesh(half, n=n)
    hx, hy, hz = half
    S = mass/3.*np.diag([hx*hx, hy*hy, hz*hz])            # second moment of a uniform box about its COM
    return W.Body(name, V, F, mass=mass, S=S, kind="dynamic", p=np.asarray(p, float),
                  R=np.eye(3) if R is None else R, mu=mu, fmat=fmat, render=dict(material=material, half=half))


# Ready pose of the FR3 arms at the counter and the table: hands 0.20 m above the work surface, 0.49 m in front of
# each base and 0.24 m to its side of the centre line, fingers down (IK seeded from an elbow-up pose, solved once;
# the base stands at the work height in both presets, so one pose serves both). The catalog's generic pose folded
# the arms with the hands high above the work. Panda joint names do not match these keys and keep the catalog pose.
HOVER_READY = {"left": {"fr3_joint1": .011, "fr3_joint2": .030, "fr3_joint3": .111, "fr3_joint4": -2.398,
                        "fr3_joint5": -.005, "fr3_joint6": 2.427, "fr3_joint7": -2.231},
               "right": {"fr3_joint1": -.114, "fr3_joint2": .029, "fr3_joint3": -.008, "fr3_joint4": -2.398,
                         "fr3_joint5": 0., "fr3_joint6": 2.427, "fr3_joint7": -2.478}}


def prism_mesh(outline, thickness):
    """Closed prism over a convex CCW outline (n x 2, centred near 0): fan-triangulated caps, quad sides."""
    P = np.asarray(outline, float)
    n = len(P)
    c = P.mean(0)
    hz = thickness/2
    V = np.vstack([np.c_[P, np.full(n, -hz)], np.c_[P, np.full(n, hz)], [[c[0], c[1], -hz], [c[0], c[1], hz]]])
    cb, ct = 2*n, 2*n + 1
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [(cb, j, i), (ct, n + i, n + j), (i, j, n + j), (i, n + j, n + i)]
    return V, np.asarray(F, np.int64)


def prism_body(name, outline, thickness, p, mass, *, mu=.4, material="box", R=None, fmat=None):
    """Thin rigid plate with a polygonal outline; second moment from a 1 mm area sampling."""
    P = np.asarray(outline, float)
    lo, hi = P.min(0), P.max(0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], 1e-3) + 5e-4, np.arange(lo[1], hi[1], 1e-3) + 5e-4, indexing="ij")
    pts = np.c_[gx.ravel(), gy.ravel()]
    inside = np.ones(len(pts), bool)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        inside &= (b[0] - a[0])*(pts[:, 1] - a[1]) - (b[1] - a[1])*(pts[:, 0] - a[0]) >= 0
    q = pts[inside]
    com = q.mean(0)
    d = q - com
    S = mass*np.array([[np.mean(d[:, 0]**2), np.mean(d[:, 0]*d[:, 1]), 0.],
                       [np.mean(d[:, 0]*d[:, 1]), np.mean(d[:, 1]**2), 0.],
                       [0., 0., thickness**2/12]])
    V, F = prism_mesh(P - com, thickness)
    return W.Body(name, V, F, mass=mass, S=S, kind="dynamic", p=np.asarray(p, float) + np.r_[com, 0.],
                  R=np.eye(3) if R is None else R, mu=mu, fmat=fmat, render=dict(material=material, outline=(P - com).tolist(),
                                                                       thickness=thickness))


def grid_sheet(name, L, Wd, h, mat, transform, md_angle=0., jitter=5e-6, seed=0):
    """Rectangular sheet: rest coordinates (u along L, v along Wd), placed by `transform(X2) -> x0`."""
    nu, nv = int(round(L/h)) + 1, int(round(Wd/h)) + 1
    u, v = np.linspace(0, L, nu), np.linspace(0, Wd, nv)
    U, V = np.meshgrid(u, v, indexing="ij")
    X2 = np.stack([U.ravel(), V.ravel()], 1)
    idx = np.arange(nu*nv).reshape(nu, nv)
    T = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b, c, d = idx[i, j], idx[i+1, j], idx[i+1, j+1], idx[i, j+1]
            T += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    rng = np.random.default_rng(seed)
    x0 = transform(X2) + rng.uniform(-jitter, jitter, (len(X2), 3))
    return W.Sheet(name, X2, np.asarray(T), x0, [mat], np.zeros(len(T), int), md_angle=md_angle)


# ----------------------------------------------------------------------------- development scene
def dev(config):
    z = .75
    mounts = {"left_arm_mount": ((-.30, .55), -math.pi/2), "right_arm_mount": ((.30, .55), -math.pi/2)}
    table = dict(name="table_top", center=(0., 0., z - .02), half=(.70, .40, .02), material="wood", mu=.4, fmat="oak")
    block = dict(name="block", center=(.25, .15, z + .025), half=(.06, .06, .025), material="wood", mu=.4, fmat="oak")
    spec = SceneSpec("dev", z, mounts, z, [table, block])
    spec.bodies = [furniture_body(table), furniture_body(block)]
    top = z + .05
    # a sheet of gift wrap on the block, overhanging 40 mm toward the robots (+y)
    def place(X2):
        return np.c_[.25 - .05 + X2[:, 1], .15 - .06 + X2[:, 0], np.full(len(X2), top + Pm.WRAP.t/2 + 2e-5)]
    spec.sheets = [grid_sheet("wrap", .16, .10, .008, Pm.WRAP, place)]
    spec.sheets[0].fmat = "wrap"
    spec.ready_pose = {"right": {"fr3_joint1": 0., "fr3_joint2": .1, "fr3_joint4": -2.0, "fr3_joint6": 2.1}}
    return spec


# ----------------------------------------------------------------------------- shared cell
MOUNTS = {"left_arm_mount": ((-.30, .55), -math.pi/2), "right_arm_mount": ((.30, .55), -math.pi/2)}
CLEAR = 5e-5                     # m, authored start clearance between resting items (IPC needs d > dmin at tick 0)


def _stack_z(z, *thicknesses):
    """Centre heights of items stacked from surface height z with CLEAR between them."""
    out, top = [], z
    for t in thicknesses:
        out.append(top + CLEAR + t/2)
        top += CLEAR + t
    return out, top


# ----------------------------------------------------------------------------- preset A: boutique counter
COUNTER_Z = .915                 # m, 36 in cash-wrap counter [A]
BAG_STACK_XY = (.36, .16)        # centre of the flat-bag stack [A; reach checked]
BAG_PLATE_T = .0025              # m, a lower flat tote as a plate: 4 laminate layers + turn-top board + rope [D]
BAG_MASS = .069                  # kg per tote [D]
BAG_CELLS = 10                   # cells across the 102 mm gusset: 10.2 mm mesh (REPORT §runtime)
HANDLES = True                   # False: DIAGNOSTIC only, the shipping-state tote without its rope handles


class StartStateError(RuntimeError):
    """A preset's start state cannot be used (e.g. the cached state is torn); the message says why."""


FLAP_UP = True                   # shipping state: the bottom flap folded up over the front face (bagnet.shipping_state:
                                 # an authored fold relaxed under the storage pressure, its
                                 # creases aged by their law) on the graded mesh (rows from 0.8 mm at the bottom folds).
                                 # False: the bottom-extended flat state (its perimeter creases remember 0 deg; DIAGNOSTIC)
BOARD_HALF = (.100, .0495, .00025)  # 200 x 99 x 0.5 mm bottom board, 350 g/m2 chipboard [V insert, D]
BOARD_MASS = .007                # kg [D]
CANDLE_HALF = (.045, .045, .055)    # SBS folding carton 90 x 90 x 110 mm [E]
CANDLE_MASS = .42                # kg [E]
MAT_HALF = (.20, .15, .0015)     # 400 x 300 x 3 mm counter mat: where the open bag must stand [A]
MAT_XY = (0., .17)
BOARDS_XY = (-.38, .17)          # stack of three bottom boards, left of the mat [A]
CANDLE_XY = (-.38, -.03)         # boxed candle [A]
CANDLE_XY_PURCHASE = (0., -.07)  # bag-purchase: the customer's candle on the counter in front of the mat, on the centre
                                 # line, where both arms reach both pairs of its side faces. A Franka Hand opens to 80 mm
                                 # and the carton is 90 mm across, so it is held between two hands; at CANDLE_XY the right
                                 # arm reaches neither pair at any seed. Over seeds 0-9999 (+-20 mm, +-30 deg) at
                                 # least 8.9 mm from the mat and 13.9 mm from the customer edge. Position only [A]


def _counter_base():
    """The counter top, the mat and the stack's four lower totes: (top, mat, bodies, origin of the top tote)."""
    from . import bagnet as B
    z = COUNTER_Z
    top = dict(name="counter_top", center=(0., .135, z - .02), half=(.76, .30, .02), material="counter_laminate",
               mu=Pm.MU["paper_table"], fmat="counter")
    mat = dict(name="counter_mat", center=(MAT_XY[0], MAT_XY[1], z + MAT_HALF[2]), half=MAT_HALF,
               material="counter_mat", mu=.6, fmat="mat")
    bodies = [furniture_body(top), furniture_body(mat)]
    # lower four totes of the stack: rigid plates with the flat bag's outline and mass [A]
    sx, sy = BAG_STACK_XY
    zs, ztop = _stack_z(z, *([BAG_PLATE_T]*4))
    if FLAP_UP:
        # the flat totes as they ship: the bottom flap folded up over the front face (bagnet.shipping_state): W x H rectangles
        hl, hw = B.H_BAG/2, B.W_BAG/2
        outline = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
    else:
        # bottom-extended flat state: mouth edge at +y; at -y the extended block bottom ends in a trapezoid
        # (45-degree chamfers of G/2) exactly like the top bag's flat state (bagnet.facet_map). Its bottom-perimeter
        # scores lie flat, so their plastic memory is 0 degrees, which a manufactured tote's are not: an opened bag's
        # bottom then springs back into a rocking bowl and the bag topples
        hl, hw, hb = (B.H_BAG + B.G_BAG/2)/2, B.W_BAG/2, B.W_BAG/2 - B.G_BAG/2
        outline = [(-hb, -hl), (hb, -hl), (hw, -hl + B.G_BAG/2), (hw, hl), (-hw, hl), (-hw, -hl + B.G_BAG/2)]
    for k, zc in enumerate(zs):
        bodies.append(prism_body(f"stack_bag_{k}", outline, BAG_PLATE_T, (sx, sy, zc), BAG_MASS, mu=Pm.MU["paper_paper"], fmat="laminate",
                                 material="bag_plate"))
    # the top tote: full sheet model, mouth toward the robots (+y), lying back face down on the stack
    origin = (sx, sy - (B.H_BAG/2 if FLAP_UP else (B.H_BAG - B.G_BAG/2)/2), ztop)
    return top, mat, bodies, origin


def counter_readiness():
    """Which of the counter preset's two start states exist for the current code, without building anything:
    dict(shipping_state=dict(key, where, sound, reason), handles=dict(key, where) or None). `where` is "shipped",
    "cache" or "missing" (startcache.where). The handle assembly's key needs the shipping state, so it is None until
    that exists."""
    import copy
    from . import bagnet as B, handles as Hd
    st = B.shipping_state_status(BAG_CELLS)
    out = dict(shipping_state={k: st.get(k) for k in ("key", "where", "sound", "reason")}, handles=None)
    if FLAP_UP and HANDLES and st["cached"] and st["sound"]:
        _, _, bodies, origin = _counter_base()
        bag, hinges, membrane, _ = B.shipping_state(origin, cells_per_gusset=BAG_CELLS, seed=0)
        supports = [copy.deepcopy(b) for b in bodies if b.name.startswith(("counter_top", "stack_bag_"))]
        out["handles"] = Hd.assembled(bag, hinges, supports, membrane=membrane, status_only=True)
    return out


def counter(config):
    from . import bagnet as B
    z = COUNTER_Z
    top, mat, bodies, origin = _counter_base()
    spec = SceneSpec("counter", z, dict(MOUNTS), z, [top, mat])
    sx, sy = BAG_STACK_XY
    bag_start = None
    if FLAP_UP:
        import copy
        from . import handles as Hd
        st = B.shipping_state_status(BAG_CELLS)
        if st["cached"] and not st["sound"]:
            raise StartStateError("counter preset unavailable: " + st["reason"])
        bag, hinges, membrane, fold = B.shipping_state(origin, cells_per_gusset=BAG_CELLS, seed=0)   # seeds vary placement only
        if fold.get("splits", 0) or len(bag.x0) != len(B.build(origin, cells_per_gusset=BAG_CELLS).X2):
            raise StartStateError("counter preset unavailable: the tote tore while its shipping state was made "
                                  f"({fold.get('splits')} vertex splits)")
        hand = None
        if HANDLES:
            supports = [copy.deepcopy(b) for b in bodies if b.name.startswith(("counter_top", "stack_bag_"))]
            bag, rods, knots, welds, hinges, hand = Hd.assembled(bag, hinges, supports, membrane=membrane)
            settled = {s.name: s for s in supports}             # the lower totes settled with the bag: the start state
            for b in bodies:                                    # is the whole settled stack
                if b.name in settled and b.kind == "dynamic":
                    b.p, b.R = np.array(settled[b.name].p, float), np.array(settled[b.name].R, float)
            spec.rods = rods
            bodies += knots
        def hinge_hook(ipc, world, h=hinges):
            B.apply_hinge_state(ipc, h)
        hinge_hook.gift_bags_hook = "hinge_state"
        def membrane_hook(ipc, world, m=membrane, sk=bag.start_slack):   # the shipping state's in-plane history (the
            B.apply_membrane_set(ipc, m)                                 # tote is sheet 0): the storage relaxation's
            B.apply_slack(ipc, sk)                                       # set, and the slack of its crumpled triangles
        membrane_hook.gift_bags_hook = "membrane_set"
        spec.ipc_setup += [membrane_hook, hinge_hook]
        if HANDLES:
            def handles_hook(ipc, world, wl=welds, kn=[k.name for k in knots]):
                Hd.install(ipc, wl, kn)
            handles_hook.gift_bags_hook = "handles"
            spec.ipc_setup += [handles_hook]
        bag_start = dict(shipping_state={k: v for k, v in fold.items() if k != "log"},
                         handles={k: v for k, v in hand.items() if k != "log"} if hand else None)
    else:
        bag = B.build(origin, mouth_dir=(0., 1., 0.), seed=0, cells_per_gusset=BAG_CELLS, flap_up=False)
    spec.sheets = [bag]
    # bottom boards (stack of 3) left of the mat, and the boxed candle (left of the mat; bag-purchase: in front of it)
    bz, btop = _stack_z(z, *([2*BOARD_HALF[2]]*3))
    for k, zc in enumerate(bz):
        bodies.append(box_body(f"board_{k}", BOARD_HALF, (BOARDS_XY[0], BOARDS_XY[1], zc), BOARD_MASS, mu=.4, n=(8, 4, 1), fmat="chipboard",
                               material="chipboard"))
    cxy = CANDLE_XY_PURCHASE if config.task == "bag-purchase" else CANDLE_XY
    bodies.append(box_body("candle", CANDLE_HALF, (cxy[0], cxy[1], z + CLEAR + CANDLE_HALF[2]), CANDLE_MASS, mu=.45, fmat="carton",
                           n=(3, 3, 4), material="candle_carton"))
    from .variants import placement
    pl = placement("counter", config.seed)
    place_rigid(pl["stack"], (sx, sy), sheets=[bag], rods=spec.rods,
                bodies=[b for b in bodies if b.name.startswith(("stack_bag_", "knot_"))])
    place_rigid(pl["boards"], BOARDS_XY, bodies=[b for b in bodies if b.name.startswith("board_")])
    place_rigid(pl["candle"], cxy, bodies=[b for b in bodies if b.name == "candle"])
    spec.bodies = bodies
    spec.geometry = dict(mat=dict(center=(MAT_XY[0], MAT_XY[1], z + 2*MAT_HALF[2]), half=MAT_HALF[:2]),
                         bag=dict(W=B.W_BAG, G=B.G_BAG, H=B.H_BAG, turn_top=B.TURN_TOP,
                                  stack_xy=(sx + pl["stack"][0], sy + pl["stack"][1]), stack_yaw=pl["stack"][2],
                                  flap_up=FLAP_UP, handles=bool(FLAP_UP and HANDLES), start_state=bag_start),
                         board=dict(half=BOARD_HALF), candle=dict(half=CANDLE_HALF), placement=pl)
    add_props(spec, spec.bodies, top)            # last: the indices of the items above stay as they were
    spec.ready_pose = HOVER_READY
    return spec


# ----------------------------------------------------------------------------- preset B: home dining table
TABLE_Z = .75                    # m, dining height [A]
BOX_XY = (0., .17)               # gift box centre [A]
FRAME_HALF = (.085, .060, .0075) # photo frame 170 x 120 x 15 mm [E]: the size of a frame for a 4 x 6 in print
                                 # (102 x 152 mm) with 9 mm borders. Catalogue boxes carry real frames (_home_frame)
FRAME_MASS = .35                 # kg [E]


def home(config):
    from . import giftbox as GB
    z = TABLE_Z
    table = dict(name="table_top", center=(0., -.02, z - .02), half=(.75, .45, .02), material="table_wood",
                 mu=Pm.MU["box_table"], fmat="oak")
    spec = SceneSpec("home", z, dict(MOUNTS), z, [table])
    spec.ready_pose = HOVER_READY
    spec.contact_stock = Pm.BOX_COVER
    bodies = [furniture_body(table)]
    from .variants import placement, home_variant
    var = home_variant(config.seed)          # box size, lid clearance, wrap stock (seed 0: the authored present)
    box = GB.box_spec(var["box"], var["clearance"])
    frame_half, frame_mass = _home_frame(box.name)
    wrapped = config.task in ("unwrap-gift", "open-present")
    if wrapped:
        from . import wrapmap as WM
        # the mesh-jitter seed stays 0: seeds vary the present through `var` and its placement
        base, lid, geom, sheet, tapes, geo = WM.relaxed_gift(z, BOX_XY, seed=0, box=box, stock=var["wrap_stock"])
        spec.sheets = [sheet] + tapes
        spec.geometry["wrap"] = geo
    else:
        base, lid, geom = GB.make_box(np.array([BOX_XY[0], BOX_XY[1], z]), box=box)
    pl = placement("home", config.seed)
    place_rigid(pl["box"], BOX_XY, sheets=spec.sheets, bodies=[base, lid])
    if wrapped and any(pl["box"]):
        geo["base_origin"] = base.p - base.R @ base.render["com_offset"]
        geo["yaw"] = pl["box"][2]
    # the photo frame lies inside the base on its floor
    o = base.p - base.R @ base.render["com_offset"]
    frame = box_body("photo_frame", frame_half, o + base.R @ np.array([0., 0., box.wall + CLEAR + frame_half[2]]), frame_mass,
                     mu=.4, n=(4, 3, 1), material="photo_frame", fmat="frame", R=base.R.copy())
    bodies += [base, lid, frame]
    spec.bodies = bodies
    i_base, i_lid = len(bodies) - 3, len(bodies) - 2

    def film(ipc, world):
        f = GB.AirFilm(i_lid, i_base, geom)
        GB.install_film(ipc, f)
        world.films = [f]
    spec.ipc_setup = [film]
    if wrapped:
        from . import bagnet as B

        def wrap_history(ipc, world, h=sheet.start_hinges, m=sheet.start_membrane):
            # the stored present's history (wrapmap.relaxed_gift): the wrap is sheet 0, its triangles come first
            B.apply_membrane_set(ipc, m)
            B.apply_hinge_state(ipc, h)
        wrap_history.gift_bags_hook = "wrap_history"
        spec.ipc_setup.append(wrap_history)

        def tape(ipc, world):
            from . import tape_bonds as TB
            # as laid: bonded where the tape touches the paper in the relaxed start state, every bond unloaded
            world.tape_bonds = TB.bond_tapes(ipc, ipc.sheet_slices[1:len(spec.sheets)], ipc.sheet_slices[0],
                                             Pm.TAPE_ON_WRAP, d0=ipc.caliper["rest_sheet_sheet"], as_laid=True)
        spec.ipc_setup.append(tape)
    spec.geometry.update(box=geom, box_xy=(BOX_XY[0] + pl["box"][0], BOX_XY[1] + pl["box"][1]), box_yaw=pl["box"][2],
                         frame=dict(half=frame_half, mass=frame_mass), base_index=i_base, lid_index=i_lid, placement=pl,
                         variant=dict(var, wrap_stock=var["wrap_stock"] if wrapped else None))
    add_props(spec, spec.bodies, table)          # last: the indices of the items above stay as they were
    return spec


def _home_frame(box_name):
    """(half extents, mass) of the photo frame inside a home-preset box: the authored frame in the authored box, else the
    params.PHOTO_FRAMES entry the catalogue box lists (long side along the box's x)."""
    from . import giftbox as GB
    if box_name == GB.AUTHORED:
        return FRAME_HALF, FRAME_MASS
    f = Pm.PHOTO_FRAMES[Pm.GIFT_BOXES[box_name]["frame"]]
    return tuple(.5*v for v in f["size"]), f["mass"]


# ----------------------------------------------------------------------------- preset C: back-room receiving table
RECEIVING_Z = .86                          # m, stainless work table [A]
RECEIVING_TABLE_Y = .06                    # table centre: back edge 0.115 m in front of the pedestals, as at the counter
RECEIVING_TABLE_HALF = (.75, .375, .02)    # 1.5 x 0.75 m top [A]
CARTON_XY = (0., .15)                      # carton centre 0.40 m ahead of the pedestals, long axis across [A]
# Ready pose at the receiving table: hands 0.30 m above the closed carton's top, 0.33 m in front of each base and 0.25 m
# to its side of the centre line, fingers down, closing along x (a continuous Cartesian path from HOVER_READY, solved
# once; HOVER_READY itself leaves the fingertips 0.7 mm from the carton's end panels at its top height)
RECEIVING_READY = {"left": {"fr3_joint1": .034, "fr3_joint2": -.681, "fr3_joint3": .076, "fr3_joint4": -2.204,
                             "fr3_joint5": .048, "fr3_joint6": 1.524, "fr3_joint7": -2.262},
                   "right": {"fr3_joint1": -.134, "fr3_joint2": -.679, "fr3_joint3": -.011, "fr3_joint4": -2.205,
                             "fr3_joint5": -.007, "fr3_joint6": 1.526, "fr3_joint7": -2.498}}


def receiving(config):
    """A sealed corrugated supply carton (carton.py) on a stainless table: the receiving clerk's start state."""
    from . import carton as CT
    from .variants import placement
    z = RECEIVING_Z
    top = dict(name="table_top", center=(0., RECEIVING_TABLE_Y, z - .02), half=RECEIVING_TABLE_HALF, material="steel_top",
               mu=Pm.MU["carton_table"], fmat="steel")
    spec = SceneSpec("receiving", z, dict(MOUNTS), z, [top])
    st = CT.settled(z, CARTON_XY)
    pl = placement("receiving", config.seed)
    place_rigid(pl["carton"], CARTON_XY, sheets=[st.tape], bodies=st.bodies)
    spec.sheets = [st.tape]
    spec.bodies = [furniture_body(top)] + st.bodies
    spec.ipc_setup = [lambda ipc, world, s=st: CT.install(ipc, s)]
    spec.geometry = dict(carton=st.geometry, carton_xy=(CARTON_XY[0] + pl["carton"][0], CARTON_XY[1] + pl["carton"][1]),
                         carton_yaw=pl["carton"][2], placement=pl)
    add_props(spec, spec.bodies, top)
    spec.ready_pose = RECEIVING_READY
    return spec


BUILDERS = {"dev": dev, "counter": counter, "home": home, "receiving": receiving}


# ----------------------------------------------------------------------------- Unitree G1 configuration (g1.py)
# The humanoid stands on the floor on the robot side of the work surface (+y, where the Franka pedestals stand in the
# arm configurations; the pedestals are not built for it), facing the items (yaw -pi/2: its left hand toward +x).
#   pelvis: (x, y) of the pelvis at import (g1.PELVIS_Z), 0.12 m behind the work surface edge. Home: from 0.17 m a
#           fingers-down hand missed 8 cm above the lid ends by 64 mm; from 0.09 m the settled robot's IK missed them by
#           154 mm. The feet settle onto the floor in the first tenths of a second
#   arms:   posture over the profile's 'whiteboard' pose (arms straight down): shoulders pitched 0.1 rad back and rolled
#           out 0.4 rad, elbows 1.3 rad. Barcode scanning's counter posture (pitch -0.1, roll 0.3; barcode_scanning/g1.py
#           ARMS_DOWN) put the left hand 13.6 mm into the dining table here; this one leaves every arm and hand link
#           58.5 mm (home) / 90.1 mm (counter) from the work surface and 27.1 mm from the rest of the robot, the only one
#           of 60 postures (pitch -0.1..0.3, roll 0.2..0.4, elbow 0.6..1.3) clear by 15 mm of both (MuJoCo distances at
#           the import pose, feet 5 mm lower; re-measured on the settled robot)
G1_ARMS = {"g1/left_shoulder_pitch_joint": .1, "g1/right_shoulder_pitch_joint": .1,
           "g1/left_shoulder_roll_joint": .4, "g1/right_shoulder_roll_joint": -.4,
           "g1/left_elbow_joint": 1.3, "g1/right_elbow_joint": 1.3}
G1_LAYOUT = {
    "counter": dict(pelvis=(.18, .555), yaw=-math.pi/2, arms=G1_ARMS),
    "home": dict(pelvis=(0., .55), yaw=-math.pi/2, arms=G1_ARMS),
}


def build(config):
    from . import g1 as G1
    humanoid = G1.is_g1(config.robot)
    if humanoid and config.preset not in G1_LAYOUT:
        raise ValueError(f"the G1 is laid out for the {sorted(G1_LAYOUT)} presets only (scene.G1_LAYOUT); "
                         f"{config.preset!r} has no standing place, arm posture or reach study for it")
    spec = BUILDERS[config.preset](config)
    if humanoid:
        spec.g1 = dict(G1_LAYOUT[config.preset])
        spec.mounts = {}                 # no pedestals: the humanoid stands on the floor
        spec.ready_pose = None           # the Franka hover pose does not apply; the G1 posture is spec.g1["arms"]
    return spec
