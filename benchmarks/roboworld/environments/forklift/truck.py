"""The forklift truck as one Newton 1.5 articulation (MuJoCo solver), built from the Forklift C asset geometry.

Bodies and joints (truck frame: x forward, y left, z up, origin on the ground under the front axle; every body frame has the
truck frame's orientation at the build pose and its origin at its joint anchor):
  chassis (free) - wheel_fl, wheel_fr (revolute y)                      front drive axle, bolted to the frame
                 - axle (revolute x, +/-AXLE_OSC_LIMIT)                  the oscillating steer axle: its centre pin makes the
                     - knuckle_l, knuckle_r (revolute z)                 three-point stability triangle
                         - wheel_rl, wheel_rr (revolute y)
                 - mast_outer (revolute y: tilt) - mast_int (prismatic z) - mast_inner (prismatic z, = mast_int)
                     - carriage (prismatic z: free lift) - sideshift (prismatic y; forks and load backrest)
                 - steering_wheel (revolute about the column axis) - knob (free spinner bearing)
                 - pedal_inch, pedal_brake, pedal_accel, lever_lift/tilt/shift, dir_lever, park_lever, key (revolute)
                 - seat (prismatic z: suspension)
Equalities: mast_inner = mast_int (inner stage rides 1:1 on the intermediate's chain); knuckle_r = quartic(knuckle_l)
(Ackermann, cot(outer) - cot(inner) = k/L, fit error reported by `ackermann_fit`).
Masses: explicit (params: component estimates within the H2.5A service weight; the chassis takes the remainder with the
CG that reproduces the sheet's axle loads). No shape adds mass (density 0).
Actuation is NOT here: every force on these joints comes from forklift/drivetrain.py, steering.py, hydraulics.py,
controls.py and tyres.py through world.py (joint forces and body wrenches), or from contact with the robot.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from . import params as P
from .geometry import body_meshes, collision_shapes, rim_geometry, body_vertices, PARK_PIN

# ---------------------------------------------------------------------------------------------------- anchors (truck frame)
AXIS_Y, AXIS_X, AXIS_Z = (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)

PEDAL_PIVOTS = {   # measured from the extracted pedal meshes (arms run forward to the pivot under the dash; organ accelerator)
    "pedal_inch": (-0.107, 0.141, 0.862), "pedal_brake": (-0.103, 0.011, 0.862), "pedal_accel": (-0.296, -0.135, 0.692),
}
LEVER_PIVOTS = {"lever_lift": (-0.245, -0.222, 1.132), "lever_tilt": (-0.245, -0.173, 1.132), "lever_shift": (-0.245, -0.127, 1.132)}
DIR_PIVOT = (-0.405, 0.175, 1.293)
PARK_PIVOT = PARK_PIN                 # the handle's cross pin (geometry.PARK_PIN)
KEY_CENTRE = (-0.335, -0.0535, 1.033)
SEAT_ANCHOR = (-0.95, 0.04, 1.030)
# pressing direction of each pedal joint (+y rotation): a hanging pedal (pad below and behind its pivot) swings its pad
# forward with a negative rotation; the organ accelerator (toe ahead of its heel hinge) goes down with a positive one
PEDAL_PRESS_SIGN = {"pedal_inch": -1.0, "pedal_brake": -1.0, "pedal_accel": 1.0}
HORN = dict(r=0.035, thick=0.012, travel=0.004)          # horn button on the hub (drawn part; the switch closes at 2 mm)
# operator-facing conventions of the other controls (sign of the joint angle):
#   cowl levers: knob toward the operator (pull) = negative; direction lever: forward (F) = negative, reverse = positive;
#   parking brake applied = negative; key ON/START = positive


@dataclass
class Truck:
    bodies: dict = field(default_factory=dict)        # name -> body index
    joints: dict = field(default_factory=dict)        # name -> joint index
    shapes: dict = field(default_factory=dict)        # shape index -> (body name, shape label, group)
    anchors: dict = field(default_factory=dict)       # name -> anchor in the truck frame (incl. setup shifts)
    shifts: dict = field(default_factory=dict)        # name -> setup shift of the body (seat rails); meshes are at anchor - shift
    column_axis: tuple = None
    knob_offset: float = None
    ackermann: dict = None
    masses: dict = None
    first_joint: int = None
    last_joint: int = None


def _quat_from_R(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


def ackermann_fit(k=None, L=P.WHEELBASE, inner_max=P.STEER_INNER_MAX):
    """Quartic through the origin, right knuckle angle as a function of the left one (joint convention: +z rotation).
    Left turn of the truck = rear wheels toe clockwise = negative angles, the left (inner) one larger."""
    k = k or 2 * abs(P.KINGPIN[1])
    outer_max = math.atan(1 / (1 / math.tan(inner_max) + k / L))
    ql = np.radians(np.linspace(-math.degrees(inner_max), math.degrees(outer_max), 600))
    qr = []
    for a in ql:
        if a < -1e-9:              # left inner (magnitude -a), right outer
            qr.append(-math.atan(1 / (1 / math.tan(-a) + k / L)))
        elif a > 1e-9:             # left outer, right inner
            c = 1 / math.tan(a) - k / L
            qr.append(math.pi / 2 - math.atan(c) if c > 0 else math.pi / 2 + math.atan(-c))
        else:
            qr.append(0.0)
    qr = np.asarray(qr)
    # value and slope fixed at straight ahead: the exact relation passes through 0 with slope 1 (both wheels turn alike
    # to first order); c2..c4 by least squares over the lock. A free least-squares quartic gives slope 1.045 there, so the
    # two rear wheels' rolling-resistance moments about their kingpins (+/-37 N m at the 0.162 m offset) no longer
    # cancel through the linkage: a standing 1.65 N m on the steering cylinder that its leakage turns into a steady
    # creep off the straight line (E21). The price is a larger error at full lock
    # (`max_error_rad`, at the right-turn stop); near straight ahead the error is under 0.15 deg within +/-10 deg
    A = np.stack([ql ** 2, ql ** 3, ql ** 4], 1)
    c = np.linalg.lstsq(A, qr - ql, rcond=None)[0]
    fit = ql + A @ c
    err = np.abs(fit - qr)
    near = np.abs(ql) <= math.radians(10.0)
    # the right knuckle's joint range is the fitted relation's over the left range (the stops are the cylinder's, on the
    # left knuckle): a range from the exact relation would meet the equality's 3 deg overshoot at the right-turn stop
    return dict(polycoef=[0.0, 1.0, *map(float, c)], max_error_rad=float(err.max()),
                max_error_within_10deg_rad=float(err[near].max()), left_range=(float(ql[0]), float(ql[-1])),
                right_range=(float(fit.min()), float(fit.max())), right_range_exact=(float(qr.min()), float(qr.max())),
                outer_max=float(outer_max))


def _box_inertia(m, size):
    a, b, c = size
    return np.diag([m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12])


def component_masses():
    """Masses, COMs (truck frame) and inertias. The chassis takes the remainder of the service weight with the COM that
    reproduces the sheet's unladen CG (x from axle loads, read; height estimate)."""
    M = body_meshes()

    def centre(name):
        V = body_vertices(M, name)
        return (V.min(0) + V.max(0)) / 2, V.max(0) - V.min(0)

    parts = {}
    for name, m in (("wheel_fl", P.M_WHEEL_FRONT), ("wheel_fr", P.M_WHEEL_FRONT), ("wheel_rl", P.M_WHEEL_REAR),
                    ("wheel_rr", P.M_WHEEL_REAR)):
        c, s = centre(name)
        r, w = s[2] / 2, s[1]
        # solid tyre as a thick ring (inner radius TRUCK_INERTIA["tyre_inner_ratio"] r) + rim
        ri = P.TRUCK_INERTIA["tyre_inner_ratio"] * r
        Iax = 0.5 * m * (r * r + ri ** 2)
        Id = m * (3 * (r * r + ri ** 2) + w * w) / 12
        parts[name] = (m, c, np.diag([Id, Iax, Id]))
    for name, m in (("knuckle_l", P.M_KNUCKLE), ("knuckle_r", P.M_KNUCKLE), ("axle", P.M_AXLE),
                    ("carriage", P.M_CARRIAGE), ("seat", P.M_SEAT)):
        c, s = centre(name)
        parts[name] = (m, c, _box_inertia(m, s))
    for name, m in (("mast_outer", P.M_MAST_OUTER), ("mast_int", P.M_MAST_INT), ("mast_inner", P.M_MAST_INNER)):
        V = body_vertices(M, name)
        c = np.array([np.median(V[:, 0]), 0.0, (V[:, 2].min() + V[:, 2].max()) / 2])
        parts[name] = (m, c, _box_inertia(m, V.max(0) - V.min(0)))
    # sideshift: plate + backrest (P.M_SIDESHIFT) and two forks (P.M_FORK each: blade-weighted COM)
    V = body_vertices(M, "sideshift")
    TI = P.TRUCK_INERTIA
    fork_c = np.array(TI["fork_com"])
    plate_c = np.array(TI["plate_com"])
    msum = P.M_SIDESHIFT + 2 * P.M_FORK
    c = (P.M_SIDESHIFT * plate_c + 2 * P.M_FORK * fork_c) / msum
    parts["sideshift"] = (msum, c, _box_inertia(msum, TI["sideshift_box"]))
    cm = P.CONTROL_MASSES        # park_lever: the handle only (the hidden linkage's inertia is the joint's armature)
    small = {"steering_wheel": cm["steering_wheel"], "knob": cm["knob"], "pedal_inch": cm["pedal_inch"],
             "pedal_brake": cm["pedal_brake"], "pedal_accel": cm["pedal_accel"], "lever_lift": cm["lever"],
             "lever_tilt": cm["lever"], "lever_shift": cm["lever"], "dir_lever": cm["dir_lever"],
             "park_lever": cm["park_lever"], "key": cm["key"]}
    for name, m in small.items():
        c, s = centre(name)
        parts[name] = (m, c, _box_inertia(m, np.maximum(s, 0.02)))
    others = sum(v[0] for v in parts.values())
    mc = P.SERVICE_WEIGHT - others
    target = np.array([P.CG_X_UNLADEN, 0.0, P.CG_Z_UNLADEN]) * P.SERVICE_WEIGHT
    rest = sum(v[0] * np.asarray(v[1]) for v in parts.values())
    cc = (target - rest) / mc
    cc[1] = 0.0
    parts["chassis"] = (mc, cc, _box_inertia(mc, TI["chassis_box"]))
    return parts


def build(builder, *, origin=(0.0, 0.0, 0.0), yaw=0.0, shape_cfg=None, label="truck", seat_rail=None):
    """Add the truck to `builder` at `origin` (ground point under the front axle) with heading `yaw`. Returns Truck."""
    import newton
    import warp as wp
    from newton._src.solvers.mujoco.equality import _add_equality_constraint
    from newton._src.solvers.mujoco.enums import EqType
    M = body_meshes()
    S = collision_shapes(M)
    masses = component_masses()
    T = Truck(masses={k: float(v[0]) for k, v in masses.items()})
    seat_rail = P.SEAT_RAIL if seat_rail is None else float(seat_rail)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    qw = _quat_from_R(Rw)
    o = np.asarray(origin, float)

    def world(p):
        return o + Rw @ np.asarray(p, float)

    # gap: ignored by the engine in this world. Every geom's MuJoCo margin is 0 (Newton zeroes it for multi-CCD), and
    # MuJoCo 3.11 then activates contacts at distance 0 whatever the gap (measured on a resting
    # box at gap 0, 0.004 and 0.05). Passed rather than removed: unset, Newton substitutes builder.rigid_gap (0.1 m)
    cfg = shape_cfg or newton.ModelBuilder.ShapeConfig(density=0.0, mu=P.MU_TRUCK, gap=0.004, margin=0.0)

    def add_body(name, anchor, shift=(0.0, 0.0, 0.0)):
        """shift: a setup offset of the whole body (seat rails) — the body frame and everything on it move by it."""
        m, com, I = masses[name]
        anchor0 = np.asarray(anchor, float)
        anchor = anchor0 + np.asarray(shift, float)
        com_local = np.asarray(com, float) - anchor0
        b = builder.add_link(xform=wp.transform(world(anchor), wp.quat(*qw)), mass=float(m), com=tuple(com_local),
                             inertia=wp.mat33(*np.asarray(I, float).ravel()), lock_inertia=True, label=f"{label}/{name}")
        T.bodies[name] = b
        T.anchors[name] = tuple(map(float, anchor))
        T.shifts[name] = tuple(map(float, shift))
        for s in S.get(name, []):
            _add_shape(builder, b, s, anchor0, cfg, name, T, wp, newton)
        return b

    JA = P.JOINT_ARMATURE

    def joint(kind, parent, child, anchor_parent, axis, **kw):
        """A revolute or prismatic joint. Its limits' stiffness is not set here: World._mujoco_parameters gives every
        hinge and slide joint's limit the solref of JOINT_LIMIT_TIMECONST and JOINT_LIMIT_SOLIMP (Newton's limit_ke /
        limit_kd would be replaced)."""
        pb = T.bodies[parent] if parent else -1
        cb = T.bodies[child]
        pa = np.asarray(anchor_parent, float) - (np.asarray(T.anchors[parent]) if parent else 0.0)
        if parent is None:
            raise ValueError
        f = builder.add_joint_revolute if kind == "rev" else builder.add_joint_prismatic
        rest = kw.pop("rest_rotation", 0.0)       # the body at joint zero turned by this angle about the axis (rad)
        cx = (wp.transform((0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(*axis), -rest)) if rest
              else wp.transform_identity())
        j = f(pb, cb, parent_xform=wp.transform(tuple(pa), wp.quat_identity()), child_xform=cx,
              axis=axis, label=f"{label}/{child}", collision_filter_parent=True, **kw)
        T.joints[child] = j
        return j

    # chassis (free root)
    chassis = add_body("chassis", (0.0, 0.0, 0.0))
    T.first_joint = builder.add_joint_free(chassis, label=f"{label}/chassis")
    T.joints["chassis"] = T.first_joint
    fw = np.asarray(P.FRONT_WHEEL_CENTRE)
    for side, sgn in (("l", 1), ("r", -1)):
        add_body(f"wheel_f{side}", (fw[0], sgn * fw[1], fw[2]))
        joint("rev", "chassis", f"wheel_f{side}", (fw[0], sgn * fw[1], fw[2]), AXIS_Y, armature=P.WHEEL_ARMATURE["front"])
    add_body("axle", P.AXLE_PIVOT)
    joint("rev", "chassis", "axle", P.AXLE_PIVOT, AXIS_X, limit_lower=-P.AXLE_OSC_LIMIT, limit_upper=P.AXLE_OSC_LIMIT,
          armature=P.WHEEL_ARMATURE["axle"])
    ack = ackermann_fit()
    T.ackermann = ack
    kp, rw = np.asarray(P.KINGPIN), np.asarray(P.REAR_WHEEL_CENTRE)
    for side, sgn in (("l", 1), ("r", -1)):
        kpin = (kp[0], sgn * kp[1], kp[2])
        add_body(f"knuckle_{side}", kpin)
        lo, hi = ack["left_range"] if side == "l" else ack["right_range"]
        joint("rev", "axle", f"knuckle_{side}", kpin, AXIS_Z, limit_lower=lo - 0.01, limit_upper=hi + 0.01,
              armature=JA["knuckle"])
        wc = (rw[0], sgn * rw[1], rw[2])
        add_body(f"wheel_r{side}", wc)
        joint("rev", f"knuckle_{side}", f"wheel_r{side}", wc, AXIS_Y, armature=JA["wheel_rear"])
    piv = np.asarray(P.MAST_TILT_PIVOT)
    add_body("mast_outer", piv)
    joint("rev", "chassis", "mast_outer", piv, AXIS_Y, limit_lower=-P.MAST_TILT_BACK, limit_upper=P.MAST_TILT_FWD,
          armature=JA["mast_tilt"])
    add_body("mast_int", piv)
    joint("pri", "mast_outer", "mast_int", piv, AXIS_Z, limit_lower=0.0, limit_upper=P.MAST_STAGE_STROKE,
          armature=JA["mast_stage"])
    add_body("mast_inner", piv)
    joint("pri", "mast_int", "mast_inner", piv, AXIS_Z, limit_lower=0.0, limit_upper=P.MAST_STAGE_STROKE,
          armature=JA["mast_stage"])
    add_body("carriage", piv)
    joint("pri", "mast_inner", "carriage", piv, AXIS_Z, limit_lower=P.CARRIAGE_LOWER, limit_upper=P.CARRIAGE_FREE_LIFT,
          armature=JA["mast_stage"])
    add_body("sideshift", piv)
    joint("pri", "carriage", "sideshift", piv, AXIS_Y, limit_lower=-P.SIDESHIFT_TRAVEL, limit_upper=P.SIDESHIFT_TRAVEL,
          armature=JA["sideshift"])
    c, n, rc, rt = rim_geometry(M)
    T.column_axis = tuple(map(float, n))
    add_body("steering_wheel", c)
    joint("rev", "chassis", "steering_wheel", c, tuple(n), armature=P.STEER_WHEEL_J * JA["steering_wheel_fraction"],
          friction=P.CONTROL_JOINT_FRICTION["steering_wheel"])
    kv = body_vertices(M, "knob")
    kc = (kv.min(0) + kv.max(0)) / 2
    d = kc - c
    T.knob_offset = float(np.linalg.norm(d - (d @ n) * n))
    add_body("knob", kc)
    joint("rev", "steering_wheel", "knob", kc, tuple(n), armature=JA["knob"], friction=P.CONTROL_JOINT_FRICTION["knob"])
    # horn button (drawn part: the asset's hub has a moulded cap): a disk on the hub centre, pressed along the column
    hub = np.concatenate([v[np.unique(f)] for l, v, f in M["steering_wheel"] if l.endswith("hub")])
    htop = c + n * float(((hub - c) @ n).max())
    hc = htop + n * HORN["thick"] / 2
    m_h = P.CONTROL_MASSES["horn"]
    i_h = P.TRUCK_INERTIA["horn_inertia"]
    b_h = builder.add_link(xform=wp.transform(world(hc), wp.quat(*qw)), mass=m_h, com=(0.0, 0.0, 0.0),
                           inertia=wp.mat33(*np.diag([i_h, i_h, i_h]).ravel()), lock_inertia=True, label=f"{label}/horn")
    T.bodies["horn"] = b_h
    T.anchors["horn"] = tuple(map(float, hc))
    qn = _quat_between((0.0, 0.0, 1.0), n)
    sh = builder.add_shape_cylinder(b_h, xform=wp.transform((0.0, 0.0, 0.0), wp.quat(*qn)), radius=HORN["r"],
                                    half_height=HORN["thick"] / 2, cfg=cfg, label="horn:button")
    T.shapes[sh] = ("horn", "button", "truck")
    T.masses["horn"] = m_h
    joint("pri", "steering_wheel", "horn", hc, tuple(n), limit_lower=-HORN["travel"], limit_upper=0.0, armature=JA["horn"])
    pd = {"pedal_inch": P.PEDAL_INCH, "pedal_brake": P.PEDAL_BRAKE, "pedal_accel": P.PEDAL_ACCEL}
    # each pedal's joint zero is its rest and its limit the floor: the hanging pedals' pads meet the toe board at the end of
    # their travel (PEDAL_FLOOR_STOPS; pedal-to-chassis contacts are filtered, so the joint limit is the floor). The
    # inching pedal rests turned back by PEDAL_INCH_RAISE from the asset's pose (its pad at the brake pad's height, as TCM
    # draws them): its body, colliders and drawn mesh turn with it (T.pedal_rest: consumers that place things on a pad
    # from the asset's vertices must turn them by it about the pivot)
    T.pedal_rest = {"pedal_inch": float(P.PEDAL_INCH.get("rest_raise", 0.0))}
    for name, piv_p in PEDAL_PIVOTS.items():
        add_body(name, piv_p)
        sign = PEDAL_PRESS_SIGN[name]
        joint("rev", "chassis", name, piv_p, AXIS_Y, limit_lower=0.0 if sign > 0 else -pd[name]["travel"],
              limit_upper=pd[name]["travel"] if sign > 0 else 0.0, armature=JA["pedal"],
              rest_rotation=-sign * T.pedal_rest.get(name, 0.0))
    for name, piv_l in LEVER_PIVOTS.items():
        add_body(name, piv_l)
        joint("rev", "chassis", name, piv_l, AXIS_Y, limit_lower=-P.LEVER["travel"], limit_upper=P.LEVER["travel"],
              armature=JA["lever"])
    add_body("dir_lever", DIR_PIVOT)
    joint("rev", "chassis", "dir_lever", DIR_PIVOT, tuple(n), limit_lower=-P.DIR_LEVER["travel"],
          limit_upper=P.DIR_LEVER["travel"], armature=JA["dir_lever"])
    add_body("park_lever", PARK_PIVOT)
    joint("rev", "chassis", "park_lever", PARK_PIVOT, AXIS_Y, limit_lower=-P.PARK_LEVER["travel"], limit_upper=0.0,
          armature=JA["park_lever"])
    add_body("key", KEY_CENTRE)
    joint("rev", "chassis", "key", KEY_CENTRE, AXIS_X, limit_lower=0.0, limit_upper=P.KEY["start"], armature=JA["key"])
    add_body("seat", SEAT_ANCHOR, shift=(seat_rail, 0.0, 0.0))
    joint("pri", "chassis", "seat", T.anchors["seat"], AXIS_Z, limit_lower=-P.SEAT_SUSPENSION["travel"] / 2,
          limit_upper=P.SEAT_SUSPENSION["travel"] / 2, armature=JA["seat"])
    T.last_joint = builder.joint_count - 1
    builder.add_articulation(list(range(T.first_joint, T.last_joint + 1)), label=label)
    # equalities
    _add_equality_constraint(builder, EqType.JOINT, joint1=T.joints["mast_inner"], joint2=T.joints["mast_int"],
                             polycoef=[0.0, 1.0, 0.0, 0.0, 0.0], label=f"{label}/eq_inner_stage")
    # the steering linkage: steel links and a steel cylinder rod, stiff (STEER_LINKAGE); MuJoCo's default 0.02 s lets
    # the right knuckle yield 0.036 deg against the left under the same +/-37 N m, a toe mismatch that turns the truck
    # off its line (E21)
    _add_equality_constraint(builder, EqType.JOINT, joint1=T.joints["knuckle_r"], joint2=T.joints["knuckle_l"],
                             polycoef=ack["polycoef"], label=f"{label}/eq_ackermann",
                             custom_attributes={"mujoco:eq_solref": wp.vec2(float(P.STEER_LINKAGE["timeconst"]), 1.0)})
    return T


def _add_shape(builder, body, s, anchor, cfg, name, T, wp, newton):
    kind = s[0]
    tag = s[-1]
    group = "wheel" if name.startswith("wheel") else "truck"
    lab = f"{name}:{tag}"
    if kind == "hull":
        V = np.asarray(s[1], np.float32) - anchor.astype(np.float32)
        mesh = newton.Mesh(V, np.asarray(s[2], np.int32).ravel())
        i = builder.add_shape_convex_hull(body, mesh=mesh, xform=wp.transform_identity(), cfg=cfg, label=lab)
    elif kind == "capsule":
        p0, p1, r = np.asarray(s[1]) - anchor, np.asarray(s[2]) - anchor, float(s[3])
        L = float(np.linalg.norm(p1 - p0))
        z = (p1 - p0) / L
        q = _quat_between((0.0, 0.0, 1.0), z)
        i = builder.add_shape_capsule(body, xform=wp.transform(tuple((p0 + p1) / 2), wp.quat(*q)), radius=r,
                                      half_height=L / 2, cfg=cfg, label=lab)
    elif kind == "box":
        c, h = np.asarray(s[1], float) - anchor, np.asarray(s[2], float)
        i = builder.add_shape_box(body, xform=wp.transform(tuple(map(float, c)), wp.quat_identity()), hx=float(h[0]),
                                  hy=float(h[1]), hz=float(h[2]), cfg=cfg, label=lab)
    elif kind == "cylinder_y":
        c, r, hw = np.asarray(s[1]) - anchor, float(s[2]), float(s[3])
        q = _quat_between((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
        i = builder.add_shape_cylinder(body, xform=wp.transform(tuple(c), wp.quat(*q)), radius=r, half_height=hw,
                                       cfg=cfg, label=lab)
    else:
        raise ValueError(kind)
    T.shapes[i] = (name, tag, group)


def _quat_between(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a); b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if c < -0.999999:
        axis = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return (*axis, 0.0)
    q = np.array([*v, 1.0 + c])
    return tuple(q / np.linalg.norm(q))
