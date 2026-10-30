"""Hole mechanics: hole records, proxy colliders, and the per-tick cutting, walking,
breakthrough, percussion, debris, heat and wear updates shared by every tool.

Holes as geometry. MuJoCo cannot change a body's mesh at runtime, and a box with a hole is not convex. Every drillable
workpiece therefore owns a welded *proxy link* carrying a pool of slots; each slot is a bottom plug (cylinder), a face
pad (box) and a cage of 12 boxes whose inner faces are tangent to the hole radius (a 12-gon of inradius R_h: the
clearance of a round shank is exact towards a face and 3.5 % larger towards a corner; E1a). Hole-capable
items - bits, bolt and anchor shanks, the vacuum nozzle, the brush, the punch point - live on their own welded links
excluded from the drillable solids, and each proxy link is excluded from everything except those item links (Newton
body-pair filters, which the solver emits as MuJoCo excludes). Before each tick the module gives every item near a
workpiece one slot of that workpiece and places in it, in the workpiece frame:
  * hole - the item's shank crosses the mouth of a hole that admits it: the cage along the hole axis and the bottom
           plug at the hole's floor for that item (a cutting bit: its stage's depth; anything else: above the dust);
  * cut  - a bit's tip at an undrilled face, pointing into it: the bottom plug under the tip at the depth cut so far;
  * pad  - any other shank within a few millimetres of a face: a box whose top is that face;
and parks unused slots by clearing their contype/conaffinity (restored when placed). The proxies are children of the
workpiece, so every contact reaction acts on the workpiece itself, implicitly and stably.

Cutting. The bottom plug's solved normal force T gives the feed per revolution the material yields (materials.py),
and the plug recedes by f x (relative turn this tick)/2 pi, never ahead of the tip. The cutting torque M(f) acts
explicitly on the bit and, equal and opposite, on the workpiece, limited to the torque that would bring their relative
spin to zero within the tick (an implicit-friction bound: a stalled bit passes exactly what the motor applies). An
unseated tip's lateral friction is the spinning-chisel law in implicit-damper form, so an unseated drill walks.
"""

import copy
import math
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

from . import params as P
from . import materials as M

N_CAGE = 12
CAGE_THICK = 0.002            # m, radial thickness of a cage box
FACE_MARGIN = 0.004           # m: an item this close to a face gets a pad / cut plug
CO_TOL_M = 0.0003             # m: two cut floors of a stack count as one within this distance along the bit (the lead a
                              # cut floor may take over its tip, _bit_update)
STACK_REACH_M = 0.002         # m: beyond FACE_MARGIN, how far above the next part of a stack a cutting bit already
                              # counts as aimed at it (that part then gives it no pad: _place_item)
CUT_HOLD_M = 2*0.010          # m: a bit holding a cut plug keeps it while its tip is within the plug's own thickness
                              # (2 PLUG_HALF) below the depth cut: the material is there and the plug pushes it back
                              # out (3 mm let a 3.6 m/s impact lose the plug and pass through: _margin)
CAPTURE_TOL = 0.0006          # m: capture beyond the radial clearance = a fastener's chamfer (ISO 4753, ~0.6 P-0.75 mm)
FUNNEL_MODE = True            # chamfer cone contact on metal fasteners: hardened like the walls (2 dt, WALL_SOLIMP). At
                              # the default a 60 N push sank a tip through the cone into both walls and a bolt spun up to
                              # 1053 rad/s in one step; with 4 ms + WALL_SOLIMP a bolt at the combined clearance met 40 kN
                              # (E10, three settings compared)
SPUR_TAU = 0.01              # s: a brad spur returns its tip towards the hole's axis over this time (_spur)
FUNNEL_ABOVE = 0.001         # m: the chamfer cone continues this far above the face (catches a tilted tip)
PLUG_HALF = 0.010             # m, bottom plug half height (a thin disk let a falling bolt tunnel)
PAD_HALF = 0.010              # m, face pad half thickness
PAD_MARGIN = 0.005            # m, a face pad reaches this far beyond its item's footprint (an item moves < 1 mm per tick)
AXIS_FOLLOW_MIN_M = 0.003     # m: below this depth the hole axis is the one frozen at seating


def _rot(q_xyzw):
    return Rotation.from_quat(q_xyzw)


def quat_z_to(a):
    """MuJoCo quaternion (w, x, y, z) rotating +z onto unit vector a."""
    a = np.asarray(a, float)/np.linalg.norm(a)
    c = float(a[2])
    if c > 1 - 1e-12:
        return np.array([1., 0., 0., 0.])
    if c < -1 + 1e-12:
        return np.array([0., 1., 0., 0.])
    axis = np.cross([0., 0., 1.], a)
    axis /= np.linalg.norm(axis)
    h = 0.5*math.acos(max(-1., min(1., c)))
    return np.array([math.cos(h), *(axis*math.sin(h))])


def perp_basis(a):
    a = np.asarray(a, float)
    t1 = np.cross(a, [0., 0., 1.]) if abs(a[2]) < .9 else np.cross(a, [1., 0., 0.])
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(a, t1)


@dataclass
class Hole:
    """One hole in one workpiece, in the workpiece body frame. Stages: one per bit diameter that cut it,
    dict(d_mm, s = tip depth (m), start = depth where the stage began (m), through)."""
    id: int
    workpiece: str
    p0: np.ndarray                     # entry point on the face (m)
    a: np.ndarray                      # unit axis into the material
    face_normal: np.ndarray            # outward normal of the entry face
    thickness: float = math.inf        # material along the axis from the entry face (m)
    stages: list = field(default_factory=list)
    kind: str = "drilled"              # drilled | dimple
    seated_axis: np.ndarray = None
    burr_m: float = 0.
    cap_collapsed: bool = False
    exit_fd: float = 1.
    entry_spall_d_m: float = 0.
    scorch_max_c: float = 20.
    removed_mm3: float = 0.
    dust_loose_mm3: float = 0.
    dust_packed_mm3: float = 0.
    dust_produced_mm3: float = 0.
    dust_evacuated_mm3: float = 0.
    dust_transit_mm3: float = 0.       # dust carried in the turning flutes; settles when the bit stops or withdraws
    dust_compacted: bool = False       # the settled dust was pressed to its compacted bulk density (_insertion_update)
    rebar_damage_m: float = 0.
    max_offset_m: float = 0.           # radial oversize of the swept hole beyond its largest stage's radius (_envelope)
    wander_m: float = 0.               # the largest stage's own wander from where its full diameter began (roundness)
    profile: list = field(default_factory=list)   # [depth_m, lateral offset_m] every 0.5 mm
    tools: list = field(default_factory=list)
    created_tick: int = 0
    last_tick: int = 0
    vacuum_passes: list = field(default_factory=list)   # ticks of completed nozzle passes to the bottom
    brush_strokes: list = field(default_factory=list)
    blocked_at: float = None           # depth of a broken bit stub left in the hole
    events: list = field(default_factory=list)
    energy: dict = field(default_factory=lambda: dict(shaft_j=0., thrust_j=0., cut_j=0., rub_j=0., blow_j=0.))

    @property
    def d_mm(self):
        return max((s["d_mm"] for s in self.stages), default=0.)

    @property
    def depth(self):
        return max((s["s"] for s in self.stages), default=0.)

    @property
    def through(self):
        return any(s["through"] for s in self.stages)

    def through_at(self, d_mm):
        """True when a stage at least d_mm wide runs through (a 3 mm pilot through a sheet is not a 6 mm hole)."""
        return any(s["through"] for s in self.stages if s["d_mm"] >= d_mm - 1e-6)

    def stage(self, d_mm):
        return next((s for s in self.stages if abs(s["d_mm"] - d_mm) < 1e-6), None)

    def full_depth(self, d_mm=None):
        """Depth (m) to which diameter >= d_mm (default: the largest) is cut to full diameter."""
        d_mm = self.d_mm if d_mm is None else d_mm
        if self.kind == "dimple":
            return 0.
        return max((s["s"] - M.point_height(s["d_mm"])*1e-3 for s in self.stages if s["d_mm"] >= d_mm - 1e-6),
                   default=0.)

    def floor_for(self, radius):
        """Depth (m) where an item of `radius` meets the floor or the step to a smaller stage; inf if a stage wide
        enough runs through."""
        ok = [s for s in self.stages if 0.5e-3*s["d_mm"] >= radius - 1e-6]
        if any(s["through"] for s in ok):
            return math.inf
        return max((s["s"] - M.point_height(s["d_mm"])*1e-3 for s in ok), default=0.)

    def pilot_for(self, d_mm, depth):
        """Largest existing diameter < d_mm cut to full diameter beyond `depth` (m), or 0."""
        best = 0.
        for s in self.stages:
            if s["d_mm"] < d_mm - 1e-6 and (s["through"] or s["s"] - M.point_height(s["d_mm"])*1e-3 >= depth - 1e-6):
                best = max(best, s["d_mm"])
        return best

    def dust_column_m(self):
        A = math.pi*(0.5*self.d_mm)**2
        return (self.dust_loose_mm3 + self.dust_packed_mm3)/A*1e-3 if A > 0 else 0.

    def summary(self):
        def r(v, k=5):
            return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), k)
        return dict(id=self.id, workpiece=self.workpiece, kind=self.kind, entry_m=[r(v) for v in self.p0],
                    axis=[r(v, 6) for v in self.a], face_normal=[r(v, 4) for v in self.face_normal],
                    diameter_mm=r(self.d_mm, 3), depth_m=r(self.depth), full_depth_m=r(self.full_depth()),
                    thickness_m=r(self.thickness), through=self.through,
                    stages=[dict(d_mm=r(s["d_mm"], 3), s_m=r(s["s"]), start_m=r(s["start"]), through=s["through"],
                                 max_offset_mm=r(s.get("max_off", 0.)*1e3, 3), ream_mm=r(s.get("ream", 0.)*1e3, 4))
                            for s in self.stages],
                    burr_mm=r(self.burr_m*1e3, 3), cap_collapsed=self.cap_collapsed, exit_fd=r(self.exit_fd, 3),
                    entry_spall_mm=r(self.entry_spall_d_m*1e3, 2), scorch_max_c=r(self.scorch_max_c, 1),
                    removed_mm3=r(self.removed_mm3, 2), dust_column_mm=r(self.dust_column_m()*1e3, 2),
                    dust_loose_mm3=r(self.dust_loose_mm3, 1), dust_packed_mm3=r(self.dust_packed_mm3, 1),
                    dust_produced_mm3=r(self.dust_produced_mm3, 1), dust_evacuated_mm3=r(self.dust_evacuated_mm3, 1),
                    dust_transit_mm3=r(self.dust_transit_mm3, 1),
                    rebar_damage_mm=r(self.rebar_damage_m*1e3, 3), max_offset_mm=r(self.max_offset_m*1e3, 3),
                    roundness=r((self.d_mm + 2e3*self.wander_m)/max(self.d_mm, 1e-9), 4),
                    envelope_oversize_mm=r(self.max_offset_m*1e3, 3),
                    vacuum_passes=list(self.vacuum_passes), brush_strokes=list(self.brush_strokes),
                    blocked_at_m=r(self.blocked_at), tools=list(self.tools), created_tick=self.created_tick,
                    last_tick=self.last_tick, energy_j={k: r(v, 4) for k, v in self.energy.items()},
                    events=list(self.events[-30:]))


@dataclass
class Drillable:
    name: str
    body: int                          # Newton body of the workpiece (a fixed link for static ones)
    proxy_body: int
    material: str                      # steel | aluminium | douglas_fir | plywood | concrete
    center: np.ndarray                 # box centre (body frame)
    half: np.ndarray                   # box half extents (body frame)
    slots: list                        # [{bottom, pad, cage: [...]}] Newton shape ids
    fields: object = None              # ConcreteFields (the wall)
    fields_map: tuple = None           # (origin, R) body frame -> fields frame (x along, y depth, z up)
    static: bool = False
    oiled: bool = False
    holes: list = field(default_factory=list)
    inertia_axis: float = math.inf     # free workpieces: smallest principal inertia (kg m^2), set at bind time

    @property
    def kind(self):
        return "metal" if self.material in M.METALS else ("wood" if self.material in M.WOODS else "concrete")

    def to_fields(self, p_l):
        o, R = self.fields_map
        return R @ (np.asarray(p_l, float) - o)

    def to_fields_dir(self, v_l):
        return self.fields_map[1] @ np.asarray(v_l, float)


@dataclass
class Item:
    """A hole-capable part: the link carrying its shank geom, its axis and tip in that link's frame."""
    name: str
    kind: str                          # bit | bolt | anchor | nozzle | brush | punch
    body: int
    axis: np.ndarray                   # unit, towards the tip (link frame)
    tip: np.ndarray                    # tip point (link frame)
    radius: float                      # shank radius (m)
    length: float                      # shank length behind the tip (m)
    d_mm: float = 0.                   # cutting diameter (bits)
    bit_type: str = ""                 # hss | brad | sds
    spindle: object = None             # tools.Spindle driving this bit (None: not driven)
    stickout: float = 0.08             # free length from chuck to tip (bending)
    wear0: float = 1.
    state: dict = field(default_factory=dict)
    slots: dict = field(default_factory=dict)      # drillable name -> (slot index, mode, hole id or None)


class HoleModule:
    def __init__(self, world):
        self.w = world
        self.drillables, self.items = {}, {}
        self.next_id = 1
        self.events = []
        self.pending = {}               # body -> (6,) world wrench applied before the next step
        self.diag = {}                  # per item: last tick's thrust, torque, feed (owner diagnostics)

    # ------------------------------------------------------------------ registration (scene build time)
    def add_drillable(self, d):
        self.drillables[d.name] = d

    def add_item(self, it):
        it.axis = np.asarray(it.axis, float)/np.linalg.norm(it.axis)
        it.tip = np.asarray(it.tip, float)
        it.state = self.fresh_item_state(it)
        self.items[it.name] = it

    @staticmethod
    def fresh_item_state(it):
        return dict(temp_c=20., max_temp_c=20., wear=float(it.wear0), broken=False, broken_reason=None, burnt=False,
                    contact_depth=0., contact_face=None, chip_depth=0., blows=0, steel_j=0., chipped=False,
                    rub_j=0., at_bottom={}, insertion_m={}, insertion_max_m={}, max_axial_n=0., axial_drag_n=0., last_blow_angle=None,
                    jam_deg=0., removed_seen=0., removed_hole=None, withdraw_mark=None, side_load_n=0.)

    # ------------------------------------------------------------------ binding after finalize
    def bind(self):
        solver = self.w.solver
        m = solver.mj_model
        g2s = np.asarray(solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
        s2g = {int(s): g for g, s in enumerate(g2s) if s >= 0}
        n2m = np.asarray(solver.mjc_body_to_newton.numpy())[0]
        self._n2m = {int(n): j for j, n in enumerate(n2m) if n >= 0}
        self._geoms, self._masks, self._solimp0 = {}, {}, {}
        for d in self.drillables.values():
            for k, slot in enumerate(d.slots):
                g = dict(bottom=s2g[slot["bottom"]], pad=s2g[slot["pad"]], cage=[s2g[s] for s in slot["cage"]],
                         funnel=[s2g[s] for s in slot["funnel"]])
                for gg in [g["bottom"], g["pad"], *g["cage"], *g["funnel"]]:
                    m.geom_sameframe[gg] = 0
                    m.geom_priority[gg] = 2                        # the proxy's contact parameters rule
                    self._solimp0[gg] = m.geom_solimp[gg][:3].copy()
                    m.geom_margin[gg] = 0.
                    self._masks[gg] = (int(m.geom_contype[gg]), int(m.geom_conaffinity[gg]))
                    if self._masks[gg] == (0, 0):
                        raise RuntimeError(f"proxy geom {gg} of {d.name} has no collision mask")
                # an item lying on a drillable's face rests on its pad, whose contact parameters rule (priority 2): the
                # pad carries the round parts' rolling lever arm (condim 6), else an anchor on its side kept rolling on
                # the steel plate for 20 s
                m.geom_condim[g["pad"]] = 6
                m.geom_friction[g["pad"]][1] = self.w._mujoco.mjMINMU
                m.geom_friction[g["pad"]][2] = P.ROLLING_B_M
                self._geoms[(d.name, k)] = g
                self._park(d.name, k)
            if not d.static:
                inertia = np.asarray(self.w.model.body_inertia.numpy()[d.body], float)
                d.inertia_axis = float(np.linalg.eigvalsh(inertia).min())
        self._free = {d.name: list(range(len(d.slots))) for d in self.drillables.values()}
        self._geom_lookup = {}
        for key, g in self._geoms.items():
            self._geom_lookup[g["bottom"]] = (key, "bottom")
            self._geom_lookup[g["pad"]] = (key, "pad")
            for c in [*g["cage"], *g["funnel"]]:
                self._geom_lookup[c] = (key, "cage")
        self._is_proxy = np.zeros(m.ngeom, bool)
        self._is_proxy[list(self._geom_lookup)] = True

    def _park(self, dname, k):
        g = self._geoms[(dname, k)]
        for gg in [g["bottom"], g["pad"], *g["cage"], *g["funnel"]]:
            self._hide(gg)

    def _show(self, gg):
        m = self.w.solver.mj_model
        m.geom_contype[gg], m.geom_conaffinity[gg] = self._masks[gg]

    def _hide(self, gg):
        m = self.w.solver.mj_model
        m.geom_contype[gg] = 0
        m.geom_conaffinity[gg] = 0

    # ------------------------------------------------------------------ frames
    def pose(self, body):
        q = self.w.bq[body]
        return q[:3].astype(float), _rot(q[3:].astype(float))

    def item_world(self, it):
        p, R = self.pose(it.body)
        return p + R.apply(it.tip), R.apply(it.axis)

    def local(self, d, p_world, v_world=None):
        p, R = self.pose(d.body)
        loc = R.inv().apply(np.asarray(p_world, float) - p)
        return loc if v_world is None else (loc, R.inv().apply(np.asarray(v_world, float)))

    def world_point(self, d, p_l):
        p, R = self.pose(d.body)
        return p + R.apply(p_l)

    def world_dir(self, d, v_l):
        return self.pose(d.body)[1].apply(v_l)

    # ------------------------------------------------------------------ box geometry
    def face_under(self, d, tip_l, axis_l, depth=0., margin=FACE_MARGIN):
        """(normal, origin, height) of the face the tip is at, the axis pointing into it, or None. `depth`: how deep
        the tip may be below the face (the depth already cut there, plus the contact's own compliance); `margin`: how
        far above it (FACE_MARGIN plus what the item travels in a tick, _margin)."""
        rel = tip_l - d.center
        best = None
        for ax in range(3):
            for sgn in (-1., 1.):
                n = np.zeros(3)
                n[ax] = sgn
                if float(axis_l @ n) > -0.5:
                    continue
                h = rel[ax]*sgn - d.half[ax]
                if not -(0.001 + depth) <= h <= margin:
                    continue
                if any(abs(rel[i]) > d.half[i] + 1e-4 for i in range(3) if i != ax):
                    continue
                if best is None or h < best[2]:
                    best = (n, d.center + n*d.half[ax], h)
        return best

    def face_near_segment(self, d, a_l, b_l, margin=FACE_MARGIN, inside=0.003, r=0.):
        """(normal, origin, point) of the face nearest the segment a-b (outside or just inside it), or None. `inside`:
        how far through the face the segment may reach and still be on it (an item already on a pad: the pad's own
        thickness, 2 PAD_HALF; otherwise 3 mm, beyond which it is in a hole or a fault). `r`: the item's radius. Heights
        are its SURFACE's: an end's lowest rim point lies r sin(angle between the axis and the face normal) nearer the
        face than the axis. (With the axis heights a 1/2 in shank, r 6.35 mm > FACE_MARGIN, lying on its side got its pad
        only once its side was 2.2 mm inside the face; the pad threw it out and vanished, and a carriage bolt or an
        anchor on the steel plate rocked at +-1-2 rad/s for good.)
        Heights are those of the shank's part OVER the face, within its bounds: a part beyond a face's edge belongs to
        the face round that edge. The segment is first clipped to where its cross-section (r sqrt(1 - u_i^2) about the
        axis along each of the face's two other axes) reaches within those bounds; when an end's lowest rim point lies
        beyond them, the height along the clipped part is the lowest point of each cross-section disc that lies within
        them (_disc_lows), sampled along it. A shank across an edge or end-on at a corner is then on the face (or faces)
        it comes nearest, and the pad's box (the workpiece's own box at that face, _place_pad) carries the edge or the
        corner itself, where the contact is solved. (Heights were taken at the segment's ends over the face's unbounded
        plane, and the face kept only when one axis point lay within 2 mm of its bounds: a carriage bolt swung across the
        top-end edge of a timber board had its tip end 18 mm below the top face's plane 39 mm beyond the board, and its
        head end 52 mm inside the end face's plane above the board, so both faces were dropped, no pad was placed and
        the steel shank passed 8-19 mm into the wood with no contact at all and stayed lodged there. A shank lying
        across a part narrower than itself, or pointing at a corner, got no face either.)"""
        best = None
        a_l, b_l = np.asarray(a_l, float), np.asarray(b_l, float)
        ab = b_l - a_l
        u = ab/max(float(np.linalg.norm(ab)), 1e-12)
        wid = r*np.sqrt(np.maximum(0., 1. - u*u))                 # the shank's half width along each axis
        lo, hi = d.center - d.half, d.center + d.half
        for ax in range(3):
            lat = [i for i in range(3) if i != ax]
            t0, t1 = self._clip(a_l, ab, lat, lo - wid - 1e-6, hi + wid + 1e-6)
            if t0 > t1:
                continue                                           # no part of the shank is over this axis's faces
            a_c = a_l if t0 == 0. else a_l + t0*ab
            b_c = b_l if t1 == 1. else a_l + t1*ab
            for sgn in (-1., 1.):
                n = np.zeros(3)
                n[ax] = sgn
                plane = d.center[ax] + sgn*d.half[ax]
                s = r*math.sqrt(max(0., 1. - float(u[ax])**2))       # axis to the end's lowest rim point, along n
                ha, hb = sgn*(a_c[ax] - plane) - s, sgn*(b_c[ax] - plane) - s
                if min(ha, hb) > margin or max(ha, hb) + 2*s < -inside:
                    continue                                       # every point over the face is beyond either bound
                off = (r*r/s*sgn*float(u[ax]))*u if s > 1e-12 else np.zeros(3)   # ... and across it
                if all(lo[i] - 1e-9 <= c[i] + off[i] <= hi[i] + 1e-9 for c in (a_c, b_c) for i in lat):
                    # both ends' lowest rim points over the face, so every disc's between them: heights are linear
                    if ha*hb < 0:                                  # crosses the face plane
                        if min(ha, hb) < -inside:                  # through it: in a hole (or a fault), never a pad
                            continue
                        t = ha/(ha - hb)
                        p, h = a_c + t*(b_c - a_c), 0.
                    elif ha <= hb:
                        p, h = a_c, ha
                    else:
                        p, h = b_c, hb
                else:
                    ta, tb = self._clip(a_l, ab, lat, lo, hi)      # where the axis itself is over the face
                    T = np.unique(np.r_[t0, t1, np.linspace(t0, t1, 17), [ta, tb] if ta <= tb else []])
                    g = self._disc_lows(a_l + T[:, None]*ab, u, r, ax, sgn, plane, lo, hi, s, off)
                    ok = np.isfinite(g)
                    if not ok.any():
                        continue
                    T, g = T[ok], g[ok]
                    k = int(np.argmin(g))
                    if g[k] < 0. < g.max():                        # crosses the face plane
                        if g[k] < -inside:                         # through it: in a hole (or a fault), never a pad
                            continue
                        j = next(j for j in range(len(g)) if g[j] == 0. or g[j]*g[j + 1] < 0.)
                        t = T[j] if g[j] == 0. else T[j] + (T[j + 1] - T[j])*g[j]/(g[j] - g[j + 1])
                        p, h = a_l + t*ab, 0.
                    else:
                        p, h = a_l + T[k]*ab, float(g[k])
                if not -inside <= h <= margin:
                    continue
                # the face the segment is on or nearest to: with a pad's 20 mm allowance a thin part's far face also
                # qualifies, and preferring the most negative h built the pad on the bottom of a 6 mm bar under a punch
                # point on its top - the pad flipped back once per cycle and pumped a 17-tick limit cycle (3.6-97.8 N
                # in the punch's spring at a 60 N push)
                if best is None or abs(h) < abs(best[3]):
                    best = (n, d.center + n*d.half[ax], p, h)
        return None if best is None else best[:3]

    @staticmethod
    def _clip(a, ab, axes, lo, hi):
        """(t0, t1): the part of the segment a + t ab, t in [0, 1], with lo[i] <= x_i <= hi[i] on the given axes
        (t0 > t1: none)."""
        t0, t1 = 0., 1.
        for i in axes:
            if abs(ab[i]) < 1e-12:
                if not lo[i] <= a[i] <= hi[i]:
                    return 1., 0.
                continue
            ta, tb = (lo[i] - a[i])/ab[i], (hi[i] - a[i])/ab[i]
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        return t0, t1

    @staticmethod
    def _disc_lows(C, u, r, ax, sgn, plane, lo, hi, s, off):
        """For the shank's cross-section discs (radius r, normal u) centred at the rows of C: the height above the face
        (axis ax, side sgn, at `plane`) of each disc's lowest point that lies within the face's bounds lo..hi on the two
        other axes; inf where no part of the disc does. The lowest point of a disc over a rectangle is its lowest rim
        point (axis height - s, at the offset `off` across the axis) when that lies within it; otherwise a vertex of the
        disc cut by the rectangle's sides: where its rim crosses a side, or where an edge of the rectangle's prism (along
        the face normal) passes through the disc."""
        i, j = [k for k in range(3) if k != ax]
        hc = sgn*(C[:, ax] - plane)
        q = C + off
        inn = (q[:, i] >= lo[i] - 1e-9) & (q[:, i] <= hi[i] + 1e-9) & (q[:, j] >= lo[j] - 1e-9) & (q[:, j] <= hi[j] + 1e-9)
        g = np.where(inn, hc - s, np.inf)
        if inn.all():
            return g
        ux, uy, uz = float(u[0]), float(u[1]), float(u[2])         # an orthonormal pair across the axis
        e1 = np.array([uy, -ux, 0.]) if abs(uz) < .9 else np.array([0., uz, -uy])
        e1 /= math.sqrt(float(e1 @ e1))
        e2 = np.array([uy*e1[2] - uz*e1[1], uz*e1[0] - ux*e1[2], ux*e1[1] - uy*e1[0]])
        for k, o in ((i, j), (j, i)):                              # the rim across the sides x_k = lo_k, hi_k
            R0 = r*math.hypot(e1[k], e2[k])
            if R0 < 1e-12:
                continue
            D = np.stack([lo[k] - C[:, k], hi[k] - C[:, k]])      # (side, disc)
            dphi = np.arccos(np.clip(D/R0, -1., 1.))
            phi = math.atan2(e2[k], e1[k]) + np.stack([dphi, -dphi])
            cp, sp = np.cos(phi), np.sin(phi)
            P_o = C[:, o] + r*(cp*e1[o] + sp*e2[o])
            fe = (np.abs(D) <= R0) & (P_o >= lo[o] - 1e-9) & (P_o <= hi[o] + 1e-9)
            h = sgn*(C[:, ax] + r*(cp*e1[ax] + sp*e2[ax]) - plane)
            g = np.minimum(g, np.where(fe, h, np.inf).min(axis=(0, 1)))
        if abs(u[ax]) > 1e-9:                                      # the prism's edges through the disc
            Bi = np.array([lo[i], lo[i], hi[i], hi[i]])[:, None] - C[:, i]
            Bj = np.array([lo[j], hi[j], lo[j], hi[j]])[:, None] - C[:, j]
            dz = -(u[i]*Bi + u[j]*Bj)/u[ax]
            fe = Bi**2 + Bj**2 + dz**2 <= r*r + 1e-12
            g = np.minimum(g, np.where(fe, sgn*(C[:, ax] + dz - plane), np.inf).min(axis=0))
        return g

    def thickness_along(self, d, p0, a):
        rel = p0 - d.center
        t_exit = math.inf
        for i in range(3):
            if abs(a[i]) > 1e-9:
                for bound in (-d.half[i], d.half[i]):
                    t = (bound - rel[i])/a[i]
                    if t > 1e-6:
                        q = rel + t*a
                        if all(abs(q[j]) <= d.half[j] + 1e-6 for j in range(3) if j != i):
                            t_exit = min(t_exit, t)
        if d.fields is not None:
            return math.inf                                       # the wall: holes are blind (200 mm)
        return t_exit

    def engaged_hole(self, d, it, tip_l, axis_l):
        """(hole, offset at the mouth) for the hole whose mouth the item's shank crosses and which admits it."""
        cutting = it.kind == "bit" and it.spindle is not None and not it.state["broken"]
        r_item = 0.5e-3*it.d_mm if it.kind == "bit" else it.radius
        base_l = tip_l - axis_l*it.length
        best = None
        for h in d.holes:
            if float(axis_l @ h.a) < math.cos(math.radians(40.)):
                continue
            z_tip = float((tip_l - h.p0) @ h.a)
            if z_tip < -FACE_MARGIN:
                continue
            z_base = float((base_l - h.p0) @ h.a)
            if z_base > 0.002 and not h.through:                   # the whole shank is past the mouth of a blind hole?
                if z_base > h.depth + 0.002:
                    continue
            if z_tip > 0. and z_base < 0.:
                t = z_tip/(z_tip - z_base)
                q = tip_l + t*(base_l - tip_l)
            else:
                q = tip_l
            rel = q - h.p0
            rho = float(np.linalg.norm(rel - float(rel @ h.a)*h.a))
            R_h = 0.5e-3*h.d_mm + (0. if cutting else h.max_offset_m)       # the swept hole (see _place_hole)
            cur = it.slots.get(d.name)
            held = cur is not None and cur[1] == "hole" and cur[2] == h.id
            # An item that stood in this hole's mouth a tick ago and whose tip is now past the rim's reach can only be in
            # this hole: it is handed to the hole like one already held. (Judged as a fresh capture, an item 0.3 mm off
            # the axis fails; the mouth's rim no longer reaches it either, 3 mm in, so it is left with no proxy at all
            # and passes through the wall with no drag: an anchor struck at 4 m/s, 4 mm a tick, went 93 mm in one blow
            # of 8.6 J against a clip that drags 500 N.)
            held = held or (cur is not None and cur[1] == "mouth" and cur[2] == h.id and z_tip > 0.003)
            own = h.stage(it.d_mm) if cutting else None
            if own is not None and not own["through"] and not held and z_tip > own["s"] + 0.002:
                # a cutting bit is never handed to a hole whose floor for it stands above its tip: it is in another
                # cavity (two records side by side). The plug placed there stood 7.8 mm inside a 6 mm drill and threw
                # 32 kN at it (panel-holes)
                continue
            # the bore the smaller drills left, for a bit that is enlarging this hole: from its first touch until its
            # own stage is through
            r_pilot = max((0.5e-3*s_["d_mm"] for s_ in h.stages if s_["d_mm"] < it.d_mm - 1e-6), default=None) \
                if cutting and h.kind != "dimple" and (own is None or not own["through"]) else None
            if h.kind == "dimple":
                if not cutting or R_h < 0.5e-3*M.chisel_length(it.d_mm) or rho > R_h:
                    continue
            elif r_pilot is not None:
                # an enlarging bit's point must fall in the mouth of the bore it follows. (Compared with the hole's
                # LARGEST stage, which is the bit's own from the tick after _stage_for has made it, a 6 mm drill
                # 0.49 mm off a 3 mm pilot's axis, its point just at the face of the plywood backer under a sheet,
                # failed the tighter rule below, cut the backer as a bare face and left a second hole record 0.4 mm
                # beside the pilot's; withdrawn, it was handed to the pilot's record, whose floor stood 7.8 mm above
                # its tip: 26 kN of side load, bit broken.)
                if rho > r_pilot and not (held and z_tip > CAPTURE_TOL and rho < R_h + r_item):
                    continue
            else:
                lim = max(R_h - r_item, 0.) + CAPTURE_TOL
                if rho > lim:
                    # capture is decided at the mouth; an item already past its chamfer in this hole stays in it while
                    # its shank overlaps the hole (a soft cage let a shank pushed sideways read 1.6 mm off, the hole
                    # released it and a face pad was placed through the shank 21 mm deep: E10)
                    if not (held and z_tip > CAPTURE_TOL and rho < R_h + r_item):
                        continue
                elif not held and z_tip > 0.:
                    # a fresh capture must fit along the shank already inside, not only where its axis crosses the
                    # mouth: a bolt leaning 20 deg in the mouth (held by the rim's walls) was handed to the vertical
                    # wall it could not fit and thrown at 2 m/s (E10 slide)
                    rel_t = tip_l - h.p0
                    rho_t = float(np.linalg.norm(rel_t - float(rel_t @ h.a)*h.a))
                    if rho_t > lim:
                        continue
            if best is None or rho < best[1]:
                best = (h, rho)
        return best

    # ------------------------------------------------------------------ slots
    def _claim(self, it, d):
        cur = it.slots.get(d.name)
        if cur is not None:
            return cur[0]
        free = self._free[d.name]
        if not free:
            # every proxy slot of this workpiece is in use: the item would pass through material that is there. That
            # is a fault of the world, never a silent pass-through; the pool is sized per
            # workpiece at scene build (scene.add_drillable n_slots)
            self.events.append(dict(type="slot_pool_exhausted", workpiece=d.name, item=it.name, tick=self.w.tick))
            raise RuntimeError(f"hole proxy slots exhausted on {d.name} ({len(d.slots)} in use) when {it.name} "
                               f"reached it at tick {self.w.tick}")
        k = free.pop(0)
        it.slots[d.name] = (k, None, None)
        return k

    def _release(self, it, dname):
        cur = it.slots.pop(dname, None)
        if cur is not None:
            self._park(dname, cur[0])
            self._free[dname].append(cur[0])

    def _set_geom(self, gg, pos, size, typ, quat):
        m = self.w.solver.mj_model
        m.geom_pos[gg] = pos
        m.geom_quat[gg] = quat
        if typ == "cyl":
            m.geom_size[gg][0], m.geom_size[gg][1] = size
            m.geom_rbound[gg] = math.hypot(size[0], size[1])
            m.geom_aabb[gg] = [0, 0, 0, size[0], size[0], size[1]]
        else:
            m.geom_size[gg][:3] = size
            m.geom_rbound[gg] = float(np.linalg.norm(size))
            m.geom_aabb[gg] = [0, 0, 0, *size]
        self._show(gg)

    def _mu(self, d):
        return P.MU["steel_wood"] if d.kind == "wood" else P.MU["steel_concrete"] if d.kind == "concrete" else P.MU["steel_steel"]

    def _place_pad(self, it, d, k, n, origin, point, seg=None):
        """The face under an item that is on it but over no hole: a box whose top is the face plane, PAD_HALF*2 deep,
        covering the item's footprint (its shank's projection on the face plus PAD_MARGIN) and never a hole of this
        workpiece. (A pad over the whole face reached into the holes, where another item's shank sat: proxies are not
        filtered item by item, so one item's pad would have pushed another item out of its hole: E10 two-item case.)"""
        g = self._geoms[(d.name, k)]
        m = self.w.solver.mj_model
        self._hide(g["bottom"])
        for c in [*g["cage"], *g["funnel"]]:
            self._hide(c)
        ax = int(np.argmax(np.abs(n)))
        ij = [i for i in range(3) if i != ax]
        lo, hi = d.center - d.half, d.center + d.half
        if seg is not None:
            r_item = (0.5e-3*it.d_mm if it.kind == "bit" else it.radius) + PAD_MARGIN
            pts = np.asarray(seg, float)
            lo = np.maximum(lo, pts.min(axis=0) - r_item)
            hi = np.minimum(hi, pts.max(axis=0) + r_item)
            anchor = np.asarray(point, float)
            for ex_lo, ex_hi in self._hole_footprints(d, n, origin):
                if any(ex_hi[i] <= lo[i] or ex_lo[i] >= hi[i] for i in ij):
                    continue                                       # this hole is outside the pad
                cuts = []
                for i in ij:
                    if ex_lo[i] > anchor[i]:
                        cuts.append((i, "hi", ex_lo[i]))
                    if ex_hi[i] < anchor[i]:
                        cuts.append((i, "lo", ex_hi[i]))
                if not cuts:
                    continue                                       # the item stands over it: the mouth logic's case
                def area(c):
                    l2, h2 = lo.copy(), hi.copy()
                    (l2 if c[1] == "lo" else h2)[c[0]] = c[2]
                    return float(np.prod([max(h2[i] - l2[i], 0.) for i in ij]))
                i, side, v = max(cuts, key=area)
                if side == "hi":
                    hi[i] = v
                else:
                    lo[i] = v
        half = 0.5*(hi - lo)
        centre = 0.5*(hi + lo)
        half[ax] = PAD_HALF
        centre[ax] = origin[ax] - n[ax]*PAD_HALF                 # the pad's top is the face plane
        self._set_geom(g["pad"], centre, tuple(half), "box", np.array([1., 0., 0., 0.]))
        m.geom_friction[g["pad"]][0] = self._mu(d)
        self._stiffen(g["pad"], d, it)
        it.slots[d.name] = (k, "pad", None)

    def _hole_footprints(self, d, n, origin):
        """Axis-aligned boxes (workpiece frame) that a face pad on face (n, origin) must not cover: every drilled hole's
        volume within the pad's depth below that face, as the bounding box of its axis there widened by its radius
        (with its oversize and, for concrete, its entry spall)."""
        out = []
        ax = int(np.argmax(np.abs(n)))
        for h in d.holes:
            if h.kind == "dimple":
                continue
            r = 0.5e-3*h.d_mm + h.max_offset_m
            if d.kind == "concrete":
                r = max(r, 0.5*h.entry_spall_d_m)
            L = h.thickness if (h.through and math.isfinite(h.thickness)) else h.depth
            zs = np.linspace(0., max(L, 0.), 9)
            pts = h.p0[None, :] + zs[:, None]*h.a[None, :]
            depth_below = (origin[ax] - pts[:, ax])*n[ax]          # how far below this face each axis point lies
            sel = pts[(depth_below >= -r) & (depth_below <= 2*PAD_HALF + r)]
            if not len(sel):
                continue
            out.append((sel.min(axis=0) - r, sel.max(axis=0) + r))
        return out

    def _place_cut(self, it, d, k, n, origin, tip_l):
        g = self._geoms[(d.name, k)]
        m = self.w.solver.mj_model
        self._hide(g["pad"])
        for c in [*g["cage"], *g["funnel"]]:
            self._hide(c)
        a = -n
        foot = tip_l - n*float((tip_l - origin) @ n)
        s = it.state["contact_depth"]
        R = max(0.5e-3*it.d_mm + 0.002, 0.004)
        self._set_geom(g["bottom"], foot + a*(s + PLUG_HALF), (R, PLUG_HALF), "cyl", quat_z_to(a))
        m.geom_friction[g["bottom"]][0] = 0.                     # a spinning tip's lateral friction: the walking law
        self._stiffen(g["bottom"], d, it)
        it.slots[d.name] = (k, "cut", None)

    def _stiffen(self, gg, d, it, wall=False):
        """Proxy contact. Hole walls in metal carrying a fastener get the hardened contact (2 dt, WALL_SOLIMP): at the
        repo's rigid contact a 19 g shank sank 0.1 mm into each wall at 10 N and an M8 bolt passed two 9 mm holes 1.2 mm
        out of line (E10). Faces, rims, the chamfer cone and plugs keep the repo's rigid contact (CONTACT_TIMECONST,
        default impedance): a shank's flat end on a hardened single cylinder-box contact chattered while sliding and
        was thrown up at 2.7-3.7 m/s (E10, with 2 dt and with WALL_SOLIMP). Bits keep the validated cutting contact."""
        m = self.w.solver.mj_model
        if wall and d.kind == "metal" and it.kind != "bit":
            m.geom_solref[gg] = (2*self.w.dt, 1.) if wall is True else (P.CONTACT_TIMECONST, 1.)
            m.geom_solimp[gg][:3] = P.WALL_SOLIMP
        elif it.kind == "punch" and d.kind == "metal" and not wall:
            # the centre punch's point on metal carries the whole hand push (60-130 N) on a tool of 70 g: at the repo's
            # rigid contact, whose stiffness goes with the mass behind it, the point stood 1.13 mm inside the bar at
            # 96 N and the strike found no face under it (no dimple; at 131 g it stood 0.6 mm in). A rounded tip on a
            # face is one well-defined contact
            m.geom_solref[gg] = (2*self.w.dt, 1.)
            m.geom_solimp[gg][:3] = P.WALL_SOLIMP
        else:
            m.geom_solref[gg] = (P.CONTACT_TIMECONST, 1.)
            m.geom_solimp[gg][:3] = self._solimp0[gg]

    def _place_hole(self, it, d, k, h, floor, rho=0., tip_l=None):
        """Hole proxies for an item captured by hole h: the floor plug, the wall (12 boxes tangent to the hole's radius,
        from the chamfer depth down) and the mouth's 45-degree chamfer cone (12 inclined facets from 1 mm above the face
        to CAPTURE_TOL deep). An item captured off centre (its full shank overlapping the wall, within CAPTURE_TOL)
        lands with its tip edge on the cone; the solver resolves the inclined contact - axial support and the lateral
        push towards the axis together, implicitly - which is the fastener's chamfer meeting the sharp rim. (An explicit
        wedge force with a stop plug either drove the shank across its clearance or was too weak to centre it, and a
        full-size cage around an overlapping shank threw it at 1.8 m/s: E10.)"""
        g = self._geoms[(d.name, k)]
        m = self.w.solver.mj_model
        self._hide(g["pad"])
        a = h.a
        R_h = 0.5e-3*h.d_mm
        # the hole is the bit's swept envelope: a bit that wandered while cutting left it oversize by its largest
        # excursion. Walls use that radius for every item except a bit still cutting this hole (a lunging bit at the
        # breakthrough of a lobed 6 mm hole, 1.56 mm wander, met a nominal-radius wall with 1.4 kN and snapped: E6)
        stage_here = h.stage(it.d_mm) if it.kind == "bit" else None
        if not (stage_here is not None and not stage_here["through"]):
            R_h += h.max_offset_m
        else:
            # concrete worn away by this bit's own side load (_ream): really gone. In metal and wood the bit has swept its
            # own stage's wander (stage max_off, recorded only there): a 6 mm bit that had wandered 0.565 mm from a
            # pilot-seated backer hole's axis, held at its nominal radius there and by the panel hole above it on the
            # other side, was pinched at zero clearance and broke at 1.7 kN (panel-holes E12)
            R_h += stage_here.get("ream", 0.) + stage_here.get("max_off", 0.)
        # a drill runs in its own hole with the hole's oversize as clearance (a zero-clearance cage turned 40 um of
        # float error into 1.6 kN of side load)
        R_c = R_h + (P.BIT_HOLE_CLEARANCE_M if it.kind == "bit" else 0.)
        r_item = 0.5e-3*it.d_mm if it.kind == "bit" else it.radius
        key = f"{d.name}/{h.id}"
        held = it.state.setdefault("contain", {})
        cur = it.slots.get(d.name)
        fresh = cur is None or cur[1] != "hole" or cur[2] != h.id
        z_tip = float((tip_l - h.p0) @ a) if tip_l is not None else math.inf
        excess = rho + r_item - R_c
        if excess > 1e-5 and z_tip > CAPTURE_TOL and (fresh or key in held):
            # found past the chamfer and overlapping at capture (a part placed there): contain it, closing only as it
            # centres. Never opened for an item pressed into the wall later (a ratcheting cage let a shank be pushed
            # out through the wall: E10)
            held[key] = min(held.get(key, math.inf), rho + r_item + 5e-5)
            R_c = max(R_c, held[key])
        else:
            held.pop(key, None)
        if math.isinf(floor):
            self._hide(g["bottom"])
        else:
            self._set_geom(g["bottom"], h.p0 + a*(floor + PLUG_HALF), (R_h + CAGE_THICK, PLUG_HALF), "cyl", quat_z_to(a))
            m.geom_friction[g["bottom"]][0] = 0. if it.kind == "bit" else self._mu(d)
            self._stiffen(g["bottom"], d, it)
        t1, t2 = perp_basis(a)
        mu = P.MU["bit_wall"] if it.kind == "bit" else self._mu(d)
        # the mouth: a 45-degree chamfer cone from FUNNEL_ABOVE above the face to CAPTURE_TOL deep (metal and wood);
        # a concrete hole's mouth is its entry spall crater (entry_spall_d_m, from the hole's creation), a cone at the
        # concrete breakout slope SPALL_CRATER_SLOPE_DEG from the face down to the hole's wall: there is no rim at the
        # hole's radius at the face (a chamfer drawn there carried 3.6 kN on a tilted hammer bit)
        z0, r0, z1 = -FUNNEL_ABOVE, R_c + CAPTURE_TOL + FUNNEL_ABOVE, CAPTURE_TOL
        if d.kind == "concrete" and 0.5*h.entry_spall_d_m > R_c + CAPTURE_TOL:
            r0 = 0.5*h.entry_spall_d_m
            z0 = 0.
            z1 = max(CAPTURE_TOL, (r0 - 0.5e-3*h.d_mm)*math.tan(math.radians(P.SPALL_CRATER_SLOPE_DEG)))
        # the wall, from the bottom of the mouth down
        top = z1
        bottom = (h.thickness + 0.002) if h.through else (h.depth + 0.002)
        if not math.isfinite(bottom):
            bottom = h.depth + 0.002
        bottom = max(bottom, top + 0.002)
        L = bottom - top
        hw = 1.1*(R_c + CAGE_THICK)*math.tan(math.pi/N_CAGE)
        for j, c in enumerate(g["cage"]):
            ang = 2*math.pi*j/N_CAGE
            radial = math.cos(ang)*t1 + math.sin(ang)*t2
            pos = h.p0 + a*(top + 0.5*L) + (R_c + 0.5*CAGE_THICK)*radial
            q = Rotation.from_matrix(np.column_stack([radial, np.cross(a, radial), a])).as_quat()
            self._set_geom(c, pos, (0.5*CAGE_THICK, hw, 0.5*L), "box", np.array([q[3], q[0], q[1], q[2]]))
            self._stiffen(c, d, it, wall=True)
            m.geom_friction[c][0] = mu
        # the mouth cone: radius r0 at z0 narrowing to R_c at z1 (the chamfer: 45 degrees, r0 - R_c = z1 - z0)
        kc = (r0 - R_c)/(z1 - z0)                              # radial narrowing per unit depth
        Lg = (z1 - z0)*math.sqrt(1. + kc*kc)
        zm, rm = 0.5*(z0 + z1), 0.5*(r0 + R_c)
        hwf = 1.1*(r0 + CAGE_THICK)*math.tan(math.pi/N_CAGE)
        for j, c in enumerate(g["funnel"]):
            ang = 2*math.pi*j/N_CAGE
            radial = math.cos(ang)*t1 + math.sin(ang)*t2
            tang = np.cross(a, radial)
            u = (radial + kc*a)/math.sqrt(1. + kc*kc)          # into the cone's material (out and down)
            gen = np.cross(u, tang)                            # along the generator
            surf = h.p0 + a*zm + rm*radial
            pos = surf + 0.5*CAGE_THICK*u
            q = Rotation.from_matrix(np.column_stack([u, tang, gen])).as_quat()
            self._set_geom(c, pos, (0.5*CAGE_THICK, hwf, 0.5*Lg), "box", np.array([q[3], q[0], q[1], q[2]]))
            self._stiffen(c, d, it, wall=FUNNEL_MODE)
            m.geom_friction[c][0] = mu
        it.slots[d.name] = (k, "hole", h.id)

    def mouth_hole(self, d, it, tip_l, axis_l):
        """(hole, rho) for the hole whose mouth an uncaptured item's tip overlaps (axis within 40 deg of the hole's, tip
        at the face): the tip rests on the rim around the hole, not on a face drawn over it."""
        r_item = 0.5e-3*it.d_mm if it.kind == "bit" else it.radius
        best = None
        for h in d.holes:
            if h.kind == "dimple" or float(axis_l @ h.a) < math.cos(math.radians(40.)):
                continue
            rel = tip_l - h.p0
            z_tip = float(rel @ h.a)
            if not -FACE_MARGIN <= z_tip <= 0.003:
                continue
            rho = float(np.linalg.norm(rel - z_tip*h.a))
            if rho < 0.5e-3*h.d_mm + h.max_offset_m + r_item and (best is None or rho < best[1]):
                best = (h, rho)
        return best

    def _place_mouth(self, it, d, k, h):
        """The rim around a hole's mouth: the cage boxes, their inner faces on the hole's 12-gon (inradius R_h), made
        thick enough radially to carry the tip's whole footprint, their tops in the face plane."""
        g = self._geoms[(d.name, k)]
        m = self.w.solver.mj_model
        self._hide(g["pad"])
        self._hide(g["bottom"])
        for c in g["funnel"]:
            self._hide(c)
        a = h.a
        R_h = 0.5e-3*h.d_mm + h.max_offset_m
        r_item = 0.5e-3*it.d_mm if it.kind == "bit" else it.radius
        ax = int(np.argmax(np.abs(h.face_normal)))
        room = min(d.half[i] - abs(float(h.p0[i] - d.center[i])) for i in range(3) if i != ax) - R_h
        Tr = max(0.001, min(2*r_item + 0.002, room))
        depth = min(h.thickness, 0.02) if math.isfinite(h.thickness) else 0.02
        t1, t2 = perp_basis(a)
        hw = 1.1*(R_h + Tr)*math.tan(math.pi/N_CAGE)
        for j, c in enumerate(g["cage"]):
            ang = 2*math.pi*j/N_CAGE
            radial = math.cos(ang)*t1 + math.sin(ang)*t2
            pos = h.p0 + a*(0.5*depth) + (R_h + 0.5*Tr)*radial
            q = Rotation.from_matrix(np.column_stack([radial, np.cross(a, radial), a])).as_quat()
            self._set_geom(c, pos, (0.5*Tr, hw, 0.5*depth), "box", np.array([q[3], q[0], q[1], q[2]]))
            self._stiffen(c, d, it)
            m.geom_friction[c][0] = self._mu(d)
        it.slots[d.name] = (k, "mouth", h.id)

    # ------------------------------------------------------------------ per tick: before the step
    def _cut_owner(self, it, tip_w, ax_w):
        """The workpiece whose face a cutting bit's tip is furthest into (stacked parts: a sheet on a backer, two plates,
        a ledger on concrete). Only it may own the bit's cut; a face hidden under it is reached by drilling through.
        (Every face near the tip claiming a cut reset the bit's single cut depth to its own, and a backed sheet was
        never cut: E6.)"""
        if not (it.kind == "bit" and it.spindle is not None and not it.state["broken"]):
            return None
        for dn, (k, mode, hid) in it.slots.items():
            if mode == "hole":
                d = self.drillables[dn]
                h = next((x for x in d.holes if x.id == hid), None)
                stage = h.stage(it.d_mm) if h is not None else None
                if stage is not None and not stage["through"] and \
                        float((self.local(d, tip_w) - h.p0) @ h.a) < h.thickness - 1e-4:
                    return dn            # still cutting its stage inside this part: nothing beyond it may cut
        best = None
        for d in self.drillables.values():
            tip_l, ax_l = self.local(d, tip_w, ax_w)
            cf = it.state["contact_face"]
            f = self.face_under(d, tip_l, ax_l, depth=it.state["contact_depth"] + CUT_HOLD_M if cf and cf[0] == d.name
                                else 0., margin=self._margin(it))
            if f is not None and (best is None or f[2] < best[1]):
                best = (d.name, f[2])
        return None if best is None else best[0]

    def pre_step(self):
        """Place proxies for every item near a workpiece; queue last tick's explicit wrenches."""
        for it in self.items.values():
            tip_w, ax_w = self.item_world(it)
            it.state["approach_m"] = float(np.linalg.norm(self._point_velocity(it.body, tip_w)))*self.w.dt
            owner = self._cut_owner(it, tip_w, ax_w)
            for d in self.drillables.values():
                tip_l, ax_l = self.local(d, tip_w, ax_w)
                base_l = tip_l - ax_l*it.length
                lo = np.minimum(tip_l, base_l) - d.center
                hi = np.maximum(tip_l, base_l) - d.center
                if (lo > d.half + 0.03).any() or (hi < -d.half - 0.03).any():
                    if d.name in it.slots:
                        self._leave(it, d)
                    continue
                self._place_item(it, d, tip_l, ax_l, base_l, may_cut=owner is None or owner == d.name)
        for body, wr in self.pending.items():
            self.w.body_wrench[body] += wr
        self.pending = {}

    @staticmethod
    def _margin(it):
        """How far from a face an item gets its pad or cut plug: FACE_MARGIN plus twice what its tip travels in a tick,
        so that a fast item meets the proxy at the face instead of being found already inside the material (a bit
        swung at a slab at 3.6 m/s penetrated its cut plug 4.4 mm, beyond the 3 mm the module then allowed, lost the
        plug and passed 160 mm into the concrete)."""
        return FACE_MARGIN + 2.*it.state.get("approach_m", 0.)

    def _leave(self, it, d):
        for key in [x for x in it.state.get("contain", {}) if x.startswith(d.name + "/")]:
            it.state["contain"].pop(key)
        cur = it.slots.get(d.name)
        if cur is not None and cur[1] == "hole" and it.kind == "bit":
            h = next((x for x in d.holes if x.id == cur[2]), None)
            if h is not None:
                self._settle_dust(h, P.WITHDRAW_CARRIES)
        self._release(it, d.name)
        st = it.state
        if st["contact_face"] and st["contact_face"][0] == d.name:
            st["contact_face"] = None
            st["contact_depth"] = 0.
        st["at_bottom"].pop(d.name, None)

    def _place_item(self, it, d, tip_l, ax_l, base_l, may_cut=True):
        st = it.state
        cutting = it.kind == "bit" and it.spindle is not None and not st["broken"]
        hit = self.engaged_hole(d, it, tip_l, ax_l)
        if hit is not None:
            h = hit[0]
            if cutting and h.kind == "dimple":
                h.kind = "drilled"
                h.seated_axis = h.a.copy()
                h.events.append(dict(tick=self.w.tick, event="seated_in_dimple", d_mm=it.d_mm))
            k = self._claim(it, d)
            if k is None:
                return
            r_item = 0.5e-3*it.d_mm if it.kind == "bit" else it.radius
            if cutting:
                stage = self._stage_for(it, h, tip_l)
                own = h.stage(it.d_mm)
                if stage is not None and not stage["through"]:
                    floor = stage["s"]
                elif own is not None and not own["through"]:
                    # a bit withdrawn more than 2 mm from its own floor (_stage_for: None) still meets that floor with its
                    # point when it comes back: the cone it cut fits it. floor_for() is the full-diameter depth, a point
                    # height (~4 mm on 13.3 mm) shallower: a plug there held the returning tip short of the 2 mm window
                    # for good, and on a just-seated hole (0.9 mm) it stood above the face
                    floor = own["s"]
                else:
                    floor = h.floor_for(r_item)
            else:
                floor = h.floor_for(r_item)
                if math.isfinite(floor):
                    floor = max(floor - h.dust_column_m(), 0.)
            if h.blocked_at is not None:
                floor = min(floor, h.blocked_at)
            self._place_hole(it, d, k, h, floor, rho=hit[1], tip_l=tip_l)
            if st["contact_face"] and st["contact_face"][0] == d.name:
                st["contact_face"], st["contact_depth"] = None, 0.
            return
        if cutting and may_cut:
            cf = st["contact_face"]
            f = self.face_under(d, tip_l, ax_l, depth=st["contact_depth"] + CUT_HOLD_M if cf and cf[0] == d.name else 0.,
                                margin=self._margin(it))
            if f is not None:
                k = self._claim(it, d)
                if k is None:
                    return
                n, origin, h_tip = f
                cf = st["contact_face"]
                if cf is None or cf[0] != d.name or np.abs(cf[1] - n).max() > 1e-6:
                    st["contact_face"] = (d.name, n.copy())
                    # KINEMATIC WRITE (declared): a cut that starts with the tip already below this face (at most the
                    # 1 mm face_under allows) starts at the tip. That happens when the bit comes through the part
                    # above in a stack: two free parts pressed together stand 0.3-0.8 mm inside each other under a
                    # clamp (the contact's softness, not material), and that overlap was cut with the part above. A plug
                    # at this face would stand that far above the tip
                    # Only then: a tip that meets a face from the side, or a part pushed against a standing bit, starts
                    # at the face and is carried by it. (Without this condition a bar carried under the drill press's
                    # stopped bit, its top face 0.87 mm above the tip, got a 0.87 mm "drilled" hole with nothing
                    # removed.)
                    st["contact_depth"] = max(0., -float(h_tip)) if self._arriving_through(it, d, -float(h_tip)) else 0.
                self._place_cut(it, d, k, n, origin, tip_l)
                return
        if cutting and not may_cut and \
                self.face_under(d, tip_l, ax_l, margin=self._margin(it) + STACK_REACH_M) is not None:
            # the next part of a stack, under the part this bit is still cutting: no pad on its face. The cut owner's
            # own floor carries the tip until that part is through; then this part owns the cut. A pad here stood
            # 0.3 mm INSIDE the clamped part above (the two sink into each other under the clamp load), took 56 N of
            # an 80 N push and left the hole's floor 24 N, under a 3 mm pilot's chisel threshold: the bit stopped
            # 0.3 mm short of coming through, for good (panel-holes)
            if d.name in it.slots:
                self._leave(it, d)
            return
        if not cutting:
            mh = self.mouth_hole(d, it, tip_l, ax_l)
            if mh is not None:
                k = self._claim(it, d)
                if k is None:
                    return
                self._place_mouth(it, d, k, mh[0])
                if st["contact_face"] and st["contact_face"][0] == d.name:
                    st["contact_face"], st["contact_depth"] = None, 0.
                return
        cur = it.slots.get(d.name)
        f = self.face_near_segment(d, tip_l, base_l, margin=self._margin(it),
                                   inside=2*PAD_HALF if cur is not None and cur[1] == "pad" else 0.003,
                                   r=0.5e-3*it.d_mm if it.kind == "bit" else it.radius)
        if f is not None:
            k = self._claim(it, d)
            if k is None:
                return
            self._place_pad(it, d, k, *f, seg=(tip_l, base_l))
            if st["contact_face"] and st["contact_face"][0] == d.name:
                st["contact_face"], st["contact_depth"] = None, 0.
            return
        if d.name in it.slots:
            self._leave(it, d)

    def _arriving_through(self, it, d, depth_below_face):
        """True when the bit reaches d's face while it is cutting forward through a part that lies ABOVE d: that part
        holds an unfinished stage of this bit and the tip stands at that stage's floor (within 1 mm). Not true for a
        bit being withdrawn, for a part above the one the bit is in, or for a bit that has finished the part above.
        (Asking only for a hole slot in any other part, a spinning 6 mm drill withdrawn from the backer gave the
        sheet ABOVE a second, empty hole record 0.6 mm beside the finished hole.)"""
        tip_w, a_w = self.item_world(it)
        for dn, v in it.slots.items():
            if dn == d.name or v[1] != "hole":
                continue
            g = self._floor_gap(it, dn, v, tip_w, a_w)
            if g is None or abs(g) > 0.001:
                continue
            o = self.drillables[dn]
            h = next(x for x in o.holes if x.id == v[2])
            if float((self.local(o, tip_w) - h.p0) @ h.a) > depth_below_face + 1e-4:      # o's face is above d's
                return True
        return False

    def _stage_for(self, it, h, tip_l):
        """The stage this bit cuts in hole h (created when an enlarging bit first meets the hole), or None when the
        bit is withdrawn from its floor.
        A new stage starts where the enlarging point first touches the existing bore (see below) and then deepens only
        by the cutting law."""
        stage = h.stage(it.d_mm)
        z = float((tip_l - h.p0) @ h.a)
        if stage is None and it.d_mm > h.d_mm - 1e-6:
            # the depth at which the new point first touches what the smaller stages left: the rim of a pilot bore
            # (its radius over tan of the half point angle, the pilot's own point height) or the bottom of a dimple
            # shallower than that. Geometry, not the tip's position at this moment: a stage that started "at the
            # tip" let a 6 mm drill that was not even turning sink 1.5 mm through a 3 mm pilot in steel sheet
            s0 = max((min(s_["s"], M.point_height(s_["d_mm"])*1e-3) for s_ in h.stages
                      if s_["d_mm"] < it.d_mm - 1e-6), default=0.)
            stage = dict(d_mm=float(it.d_mm), s=s0, start=s0, through=False)
            h.stages.append(stage)
            self._envelope(h)                                    # a larger stage absorbs a smaller one's wander
            h.tools.append(it.name)
            h.events.append(dict(tick=self.w.tick, event="stage", d_mm=it.d_mm, depth_mm=round(stage["s"]*1e3, 3)))
        if stage is None:
            return None
        if stage["through"]:
            return stage
        return stage if z >= stage["s"] - 0.002 else None

    # ------------------------------------------------------------------ per tick: after the step
    def post_step(self):
        dt = self.w.dt
        forces = self._slot_forces()
        for it in self.items.values():
            if it.kind == "bit" and it.spindle is not None:
                self._bit_update(it, forces, dt)
            elif it.kind in ("nozzle", "brush"):
                self._cleaning_update(it)
            elif it.kind in ("bolt", "anchor"):
                self._insertion_update(it, forces)

    def _slot_forces(self):
        """{(drillable, slot): dict(bottom=N, bottom_vec, cage=(3,) world force on the item, pad=N)}."""
        import mujoco
        m, d = self.w.solver.mj_model, self.w.solver.mj_data
        out = {}
        n = d.ncon
        if not n:
            return out
        geom = np.asarray(d.contact.geom[:n])
        hit0 = self._is_proxy[geom[:, 0]]
        hit1 = self._is_proxy[geom[:, 1]]
        idx = np.flatnonzero(hit0 | hit1)
        if not len(idx):
            return out
        frames = np.asarray(d.contact.frame[:n]).reshape(n, 3, 3)
        f6 = np.zeros(6)
        for i in idx:
            g0, g1 = int(geom[i, 0]), int(geom[i, 1])
            key, part = self._geom_lookup[g0 if hit0[i] else g1]
            mujoco.mj_contactForce(m, d, i, f6)
            # force of geom0 on geom1 along the frame (normal points from geom0 to geom1)
            fw = frames[i].T @ f6[:3]
            f_on_item = fw if hit0[i] else -fw
            e = out.setdefault(key, dict(bottom=0., bottom_vec=np.zeros(3), cage=np.zeros(3), pad=0., cage_abs=0.,
                                         cage_mom=np.zeros(3)))
            if part == "bottom":
                e["bottom"] += abs(float(f6[0]))
                e["bottom_vec"] += f_on_item
            elif part == "cage":
                e["cage"] += f_on_item
                e["cage_abs"] += abs(float(f6[0]))                  # wall normal forces, summed as magnitudes
                e["cage_mom"] += np.cross(np.asarray(d.contact.pos[i], float), f_on_item)   # about the world origin
            else:
                e["pad"] += abs(float(f6[0]))
        return out

    # ------------------------------------------------------------------ cutting
    def _engaged(self, it):
        """(drillable name, (slot, mode, hole id)) of the cut the bit's tip stands on. With one part that is the part;
        in a stack it is the lowest part the point has reached (_engaged_lowest), UNLESS a part above still holds an
        unfinished stage of this bit whose floor lies above that part's floor: then the tip stands on the upper
        part's floor and the upper part is the one being cut (a 6 mm drill opening a 3 mm pilot in a sheet stands on
        the sheet's rim, although the pilot's point left a dimple in the backer below)."""
        eng = self._engaged_lowest(it)
        if eng is None:
            return None
        tip_w, a_w = self.item_world(it)
        g0 = self._floor_gap(it, eng[0], eng[1], tip_w, a_w)
        best = (math.inf if g0 is None else g0, eng)
        for dn, v in it.slots.items():
            if dn == eng[0] or v[1] != "hole":
                continue
            g = self._floor_gap(it, dn, v, tip_w, a_w)
            if g is not None and g < best[0] - CO_TOL_M:
                best = (g, (dn, v))
        return best[1]

    def _engaged_lowest(self, it):
        """(drillable name, (slot, mode, hole id)) the bit's tip is engaged with, preferring a cut slot."""
        engaged = [(dn, v) for dn, v in it.slots.items() if v[1] in ("cut", "hole")]
        if not engaged:
            return None
        cuts = [(dn, v) for dn, v in engaged if v[1] == "cut"]
        if cuts:
            # with stacked workpieces (sheet on a backer, two plates, a ledger on concrete) every face near the tip gets
            # a cut slot; the one being cut is the face the tip is furthest into (picking the first in the dict cut
            # the backer's unreached face and left a backed sheet untouched: E6)
            tip_w, a_w = self.item_world(it)

            def depth(dn):
                d = self.drillables[dn]
                tip_l, a_l = self.local(d, tip_w, a_w)
                f = self.face_under(d, tip_l, a_l, depth=0.05)
                return -1e9 if f is None else -f[2]
            return max(cuts, key=lambda c: depth(c[0]))
        # deepest-tip hole: the last drillable along the bit
        tip_w, a_w = self.item_world(it)
        best = None
        for dn, v in engaged:
            d = self.drillables[dn]
            h = next(x for x in d.holes if x.id == v[2])
            z = float((self.local(d, tip_w) - h.p0) @ h.a)
            if not h.through or z < h.thickness:
                if best is None or z < best[0]:
                    best = (z, dn, v)
        if best is None:
            return engaged[-1]
        return best[1], best[2]

    def _floor_gap(self, it, dn, v, tip_w, a_w):
        """How far ahead of the tip, along the bit, the floor of the unfinished cut in slot v of workpiece dn lies (m;
        negative: the tip is pressed into it); None when that slot holds no unfinished cut of this bit."""
        k, mode, hid = v
        d = self.drillables[dn]
        tip_l, a_l = self.local(d, tip_w, a_w)
        if mode == "cut":
            cf = it.state["contact_face"]
            f = self.face_under(d, tip_l, a_l, depth=0.05)
            if f is None or not cf or cf[0] != dn:
                return None
            return it.state["contact_depth"] + float(f[2])
        if mode == "hole":
            h = next((x for x in d.holes if x.id == hid), None)
            stage = h.stage(it.d_mm) if h is not None else None
            if stage is None or stage["through"]:
                return None
            return stage["s"] - float((tip_l - h.p0) @ h.a)
        return None

    def _rings(self, it, eng):
        """The unfinished stages this bit has left in the parts ABOVE the one its point is cutting (eng): the ring
        each part still holds around the point, which the lips are cutting while the point is already in the part
        below (a sheet on its backer, a bracket on a bar, a ledger on the wall). A ring counts once its own floor has
        come down to the floor the tip stands on (within CO_TOL_M). Until then the tip stands on the upper part's own
        floor and that part is the one being cut (_engaged). The rings' thrust and torque are charged by their own
        material's law together with the engaged part's (_bit_update), and their floors advance with its floor
        (_tie_rings); nothing is cut for free. (Following the tip uncharged, a 6 mm drill went through a piloted
        1.5 mm steel sheet in 0.25 s, removing no steel, because the pilot had left a dimple in the backer and the
        point counted as "in the part below" from the start.)"""
        tip_w, a_w = self.item_world(it)
        g0 = self._floor_gap(it, eng[0], eng[1], tip_w, a_w)
        if g0 is None:
            return []
        out = []
        for dn, v in it.slots.items():
            if dn == eng[0] or v[1] != "hole":
                continue
            g = self._floor_gap(it, dn, v, tip_w, a_w)
            if g is None or g > g0 + CO_TOL_M:
                continue
            d = self.drillables[dn]
            h = next(x for x in d.holes if x.id == v[2])
            stage = h.stage(it.d_mm)
            z = float((self.local(d, tip_w) - h.p0) @ h.a)
            if math.isinf(h.thickness) or z < h.thickness - CO_TOL_M:
                continue                                        # the point is still inside that part: not a ring
            out.append(dict(d=d, h=h, stage=stage, k=v[0], z=z))
        return out

    def _ring_geometry(self, it, r):
        """(lam_T, lam_M, chi, inner radius mm) of a ring: the part of the lips still in that part."""
        s_u = r["stage"]["s"]
        dp = r["h"].pilot_for(it.d_mm, s_u)
        lt, lm, ch = M.engagement(it.d_mm, dp, s_u*1e3, (r["h"].thickness - s_u)*1e3)
        r_in = max(0.5*dp, min(0.5*it.d_mm, max(s_u - r["h"].thickness, 0.)*1e3*math.tan(0.5*P.POINT_ANGLE)))
        return lt, lm, (0. if r["h"].cap_collapsed else ch), r_in

    def _tie_rings(self, it, dname, rings, thrusts=None):
        """After the engaged part's floor has advanced: every ring's floor comes down with it (never back), the steel
        or wood that the lips took out of the ring is booked to the ring's own hole, and the ring reads through once
        the point has cleared its part. Under a concrete part cut by blows (a timber ledger on the wall) the ring is
        tied without a charge: its resistance is neglected against the concrete's (declared)."""
        if not rings or dname not in it.slots:
            return
        tip_w, a_w = self.item_world(it)
        g0 = self._floor_gap(it, dname, it.slots[dname], tip_w, a_w)
        if g0 is None:
            return
        for i, r in enumerate(rings):
            st_u = r["stage"]
            if st_u["through"]:
                continue
            s_new = r["z"] + g0
            if s_new > st_u["s"]:
                r_in = self._ring_geometry(it, r)[3]
                r["h"].removed_mm3 += math.pi*max((0.5*it.d_mm)**2 - r_in**2, 0.)*(s_new - st_u["s"])*1e3
                r["h"].last_tick = self.w.tick
                st_u["s"] = s_new
            self._exit_checks(r["d"], r["h"], st_u, 0. if thrusts is None else thrusts[i])

    def _bit_update(self, it, forces, dt):
        st = it.state
        sp = it.spindle
        eng = None if st["broken"] else self._engaged(it)
        rings = [] if eng is None else self._rings(it, eng)
        if eng is None:
            self._heat(it, 0., dt, False)
            st["bend_f"] = 0.                                # out of every hole: nothing bends the bit
            self.diag[it.name] = dict(thrust_n=0., torque_nm=0., feed_mm_rev=0., mode=None)
            return
        dname, (k, mode, hid) = eng
        d = self.drillables[dname]
        f = forces.get((dname, k), {})
        # the thrust is what the tip's floor carries: the engaged part's plug and, where a ring's plug stands at the same
        # place, that plug's share of the same contact
        T = float(f.get("bottom", 0.)) + sum(float(forces.get((r["d"].name, r["k"]), {}).get("bottom", 0.)) for r in rings)
        tip_w, a_w = self.item_world(it)
        tip_l, a_l = self.local(d, tip_w, a_w)
        bqd = self.w.bqd
        w_bit = bqd[it.body][3:].astype(float)
        w_wp = np.zeros(3) if d.static else bqd[d.body][3:].astype(float)
        w_rel = float((w_bit - w_wp) @ a_w)                     # + : forward (right-hand cutting) rotation
        dtheta = w_rel*dt
        turn = max(dtheta, 0.)/(2*math.pi)
        percussive = getattr(sp, "percussive", False)
        if mode == "hole":
            h_ = next((x for x in d.holes if x.id == hid), None)
            if h_ is not None and abs(w_rel) < 1. and h_.dust_transit_mm3 > 0:
                self._settle_dust(h_, 0.)              # the flutes stopped: what they carried drops to the bottom
            elif h_ is not None and abs(w_rel) >= 1. and d.kind == "concrete" and \
                    float((tip_l - h_.p0) @ h_.a) > h_.depth - 0.002 and h_.dust_loose_mm3 + h_.dust_packed_mm3 > 0:
                # a turning bit back on the floor takes the settled dust up into its flutes again
                h_.dust_transit_mm3 += h_.dust_loose_mm3 + h_.dust_packed_mm3
                h_.dust_loose_mm3 = h_.dust_packed_mm3 = 0.
        h, stage = None, None
        if mode == "hole":
            h = next(x for x in d.holes if x.id == hid)
            stage = self._stage_for(it, h, tip_l)
            if stage is None or stage["through"]:
                self._heat(it, 0., dt, d.oiled)
                self._track_withdrawal(it, h, tip_l)
                self._lateral_check(it, d, h, f, a_w)
                self.diag[it.name] = dict(thrust_n=T, torque_nm=0., feed_mm_rev=0., mode="in_hole", workpiece=d.name)
                return
            s = stage["s"]
            depth_mm = s*1e3
            dp_mm = h.pilot_for(it.d_mm, s)
            exit_mm = None if math.isinf(h.thickness) else (h.thickness - s)*1e3
        else:
            s = st["contact_depth"]
            depth_mm, dp_mm, exit_mm = max(s*1e3, 0.02), 0., None
        W = st["wear"]
        lam_t, lam_m, chi = M.engagement(it.d_mm, dp_mm, depth_mm, exit_mm)
        if h is not None and h.cap_collapsed:
            chi = 0.
        # ---- the law: feed per revolution the solved thrust supports, and torque as a function of the chip cut
        feed, scrape = 0., False
        torque_of = lambda f: 0.
        if percussive and d.kind == "concrete":
            scrape = sp.mode == "rotary"
        elif percussive and d.kind == "wood":
            feed = M.wood_feed(P.KAPPA_SDS_WOOD, it.d_mm, T, max(lam_t, 0.05))
            torque_of = lambda f: M.wood_torque(P.KAPPA_SDS_WOOD, it.d_mm, f, max(lam_m, 0.05))
        elif d.kind == "metal":
            Wm = W*(P.SDS_IN_METAL_FACTOR if percussive else 1.)
            feed = M.metal_feed(d.material, it.d_mm, T, Wm, max(lam_t, 1e-3), chi)
            torque_of = lambda f: M.metal_torque(d.material, it.d_mm, f, Wm, max(lam_m, 1e-3))
        elif d.kind == "wood":
            kap = M.WOODS[d.material]
            feed = M.wood_feed(kap, it.d_mm, T, max(lam_t, 1e-3))
            torque_of = lambda f: M.wood_torque(kap, it.d_mm, f, max(lam_m, 1e-3))
        else:                                                    # a twist drill in concrete: scraping, no blows
            scrape = True
        # ---- a stack: the rings the lips are still cutting in the parts above are charged by their own material's
        # law at the same feed. The feed is the one at which the engaged part's thrust and the rings' thrust together
        # equal the solved thrust; the torque is their sum (steel on plywood: the steel ring sets the feed).
        ring_terms = []
        if rings and not scrape and not (percussive and d.kind == "concrete"):
            def _law(dd, lt, lm, ch):
                if dd.kind == "metal":
                    Wm_ = W*(P.SDS_IN_METAL_FACTOR if percussive else 1.)
                    return (lambda f_: M.metal_thrust(dd.material, it.d_mm, f_, Wm_, lt, ch),
                            lambda f_: M.metal_torque(dd.material, it.d_mm, f_, Wm_, lm))
                if dd.kind == "wood":
                    kap_ = P.KAPPA_SDS_WOOD if percussive else M.WOODS[dd.material]
                    return (lambda f_: M.wood_thrust(kap_, it.d_mm, f_, lt), lambda f_: M.wood_torque(kap_, it.d_mm, f_, lm))
                return None
            own = _law(d, lam_t, lam_m, chi)
            ring_terms = [_law(r["d"], *self._ring_geometry(it, r)[:3]) for r in rings]
            if own is not None and all(t is not None for t in ring_terms):
                thrust_of = lambda f_: own[0](f_) + sum(t[0](f_) for t in ring_terms)
                feed = M.feed_for_thrust(thrust_of, T, P.FEED_GRAB_MAX_D*it.d_mm)
                torque_of = lambda f_: own[1](f_) + sum(t[1](f_) for t in ring_terms)
            else:
                ring_terms = []
        # ---- advance the floor, never ahead of the tip; the chip actually cut sets the torque
        ds = 0.
        if turn > 0 and feed > 0:
            # no chip thicker than the lips can take (FEED_MAX_D x d per revolution): beyond it a drill jams, it does not
            # cut faster; this bounds impacts, where the tip briefly sinks into its soft floor contact
            cap_d = P.FEED_GRAB_MAX_D if (h is not None and h.cap_collapsed) else P.FEED_MAX_D
            ds = min(feed, cap_d*it.d_mm)*1e-3*turn
            if mode == "cut":
                n_c = st["contact_face"][1]
                origin = d.center + n_c*float(np.abs(n_c) @ d.half)
                z_tip = float((tip_l - origin) @ (-n_c))           # tip depth below the face
            else:
                z_tip = float((tip_l - h.p0) @ h.a)
            ds = max(0., min(s + ds, z_tip + 0.0003) - s)
        f_eff = ds*1e3/turn if turn > 1e-12 else feed
        if d.kind == "concrete":
            M_cut = P.CONCRETE_ROT_MU*T*(it.d_mm*1e-3)/3.
        else:
            M_cut = torque_of(f_eff)
            if f_eff <= 1e-6 and T > 0.5:
                M_cut = max(M_cut, M.rubbing_torque(T, it.d_mm))
        if not percussive and h is not None:                     # chip packing since the last withdrawal
            z0 = P.CHIP_ONSET_D*it.d_mm*1e-3
            beta = math.log(2.)/((P.CHIP_DOUBLE_DEPTH_D - P.CHIP_ONSET_D)*it.d_mm*1e-3)
            M_cut *= math.exp(beta*max(st["chip_depth"] - z0, 0.))
        if h is not None and d.kind == "concrete" and self.rebar_overlap(d, h, stage):
            # an edge caught on a bar: the jam torque grows with the turn made against the bar
            M_cut += P.JAM_RATE_NM_PER_DEG*st["jam_deg"]
            st["jam_deg"] += math.degrees(max(dtheta, 0.))
        else:
            st["jam_deg"] = 0.
        # ---- torque: the cut (and a jam) resists the relative spin like friction, up to the cutting torque M_cut. The
        # torque applied is the one that leaves bit and workpiece without relative spin at the end of the tick, clipped
        # to M_cut: w' = w + dt (m/I_bit - M inv) = 0, with m the torque the motor applies to a stopped bit in this
        # tick (its command through its slope, up to its cap: the stall torque at a full trigger). Below M_cut the bit
        # is slowed or held: an overloaded bit STALLS, the motor sits at its stall torque (and the rotary hammer's
        # overload clutch interrupts the drive). Two other bounds fail:
        #   * |w| (I + c dt)/dt, no motor term, no torque at standstill: the motor's drive term turned an overloaded
        #     bit on at about a fifth of its no-load speed whatever the load (c dt w_t/(I + 2 c dt): 207 rpm for the
        #     hammer), so no drill could stall, and a bit caught on a bar ground 3.6 mm into it in 40 s;
        #   * for jams, plus the motor's torque of the PREVIOUS tick, the direction from the bit's own velocity: a
        #     standing bit creeping backwards at 0.5 rad/s got hold and motor added up on alternate ticks: 0 and
        #     390-470 rpm in turn, 0.9 mm into the bar in 4.5 s, the motor torque swinging between 8 and 13.9 N m.
        I_bit = sp.inertia_about_axis()
        inv = 1./I_bit + (0. if d.static else 1./max(d.inertia_axis, 1e-6))
        m_motor = math.copysign(min(abs(sp.c*sp.cmd), sp.cap), sp.cmd) if sp.cmd else 0.
        need = -(w_rel/dt + m_motor/I_bit)/inv                   # signed torque on the bit about its axis
        M_app = min(M_cut, abs(need))
        sgn = -math.copysign(1., need) if need else 0.
        ring_thrusts = None
        if M_app > 0. and sgn != 0.:
            self._torque(it.body, -sgn*M_app*a_w)
            shares = [1.]
            if ring_terms:                                       # each part of a stack takes the torque of its own cut
                tq = [own[1](f_eff)] + [t[1](f_eff) for t in ring_terms]
                tot = sum(tq)
                shares = [x/tot for x in tq] if tot > 1e-12 else [1.] + [0.]*len(ring_terms)
            if not d.static:
                self._torque(d.body, sgn*M_app*shares[0]*a_w)
            for r, sh in zip(rings, shares[1:]):
                if not r["d"].static:
                    self._torque(r["d"].body, sgn*M_app*sh*a_w)
        if ring_terms:
            ring_thrusts = [t[0](f_eff) for t in ring_terms]
        power = M_app*abs(w_rel)
        if scrape and dtheta > 0:                                # scraping concrete: rotation work removes material
            A = math.pi*(0.5*it.d_mm)**2
            z_tip = float((tip_l - h.p0) @ h.a) if h is not None else s
            ds = max(0., min(M_app*dtheta/P.CONCRETE_ROT_EV_J_MM3/A*1e-3, z_tip + 0.0003 - s))
            feed = f_eff = ds*1e3/max(turn, 1e-12)
        if ds > 0.:
            if mode == "cut":
                st["contact_depth"] = s + ds
            else:
                stage["s"] = s + ds
                st["chip_depth"] += ds
                if stage is h.stages[0] and stage["s"] > AXIS_FOLLOW_MIN_M and d.kind != "concrete":
                    self._follow_axis(d, h, tip_l)
            dv = math.pi*0.25*(it.d_mm**2 - dp_mm**2)*ds*1e3
            if d.kind != "concrete":
                # what the cone has reached in this part: from the pilot's radius (or the radius that has already left
                # through the far face) out to the radius the cone has at the entry face. Booking the full annulus at
                # every depth counted the point's cone twice (a 6 mm hole through a 1.5 mm sheet read 55 mm3 where
                # its cylinder holds 42), and in a stack it charged this part with what the rings above hold
                tan_h = math.tan(0.5*P.POINT_ANGLE)
                r_o = min(0.5*it.d_mm, max(depth_mm, 0.)*tan_h)
                r_i = 0.5*dp_mm if exit_mm is None else max(0.5*dp_mm, min(0.5*it.d_mm, max(-exit_mm, 0.)*tan_h))
                dv = math.pi*max(r_o**2 - r_i**2, 0.)*ds*1e3
            if h is not None:
                h.removed_mm3 += dv
                h.last_tick = self.w.tick
                if d.kind == "concrete":
                    self._dust(h, dv, rotating=True)
        if h is not None:
            h.energy["shaft_j"] += power*dt
            h.energy["thrust_j"] += T*ds
            h.energy["cut_j" if ds > 0 else "rub_j"] += power*dt + T*ds
        frac = P.HEAT_TO_TIP[0] if d.kind == "metal" else P.HEAT_TO_TIP[1]
        # the cutting corners run hotter than the lumped tip by the tool-chip interface rise of the chip being cut
        # (Cook's formula: grows with the corner speed as V^(1/3)). Cook's rise is a steady-state value for an
        # established chip, so it reads the feed per revolution averaged over one revolution (each lip takes its chip
        # once per half turn): a one-tick sink of the tip into its floor contact at first touch is not a chip (it gave
        # 2.54 mm/rev for 5 ms and a 714 K rise at a 12.7 mm drill's entry). Rubbing cuts no chip and adds none. The
        # chip's specific energy is the sharp tool's (W = 1): a worn edge's extra rubbing heat already feeds the node
        # through the cutting power, and letting W into the rise closed an instant loop with the thermal wear (wear ->
        # hotter edge -> faster wear) that took W to 9e5 and the edge to 5e8 C within 40 ms of a 3000 rpm entry.
        # The rise applies once the lips have cut one full revolution without a break (an established chip): at
        # 3000 rpm a revolution is 20 ticks, and the first-touch sink then still dominated the average and burnt a
        # 9 mm bit at the fourth tick of its entry (E7).
        edge = 0.
        if d.kind == "metal" and it.bit_type in ("hss", "brad") and not percussive and abs(w_rel) > 1.:
            a_rev = dt/(2.*math.pi/abs(w_rel) + dt)
            st["edge_feed"] = st.get("edge_feed", 0.) + a_rev*((f_eff if ds > 0. else 0.) - st.get("edge_feed", 0.))
            st["chip_rev"] = st.get("chip_rev", 0.) + abs(w_rel)*dt/(2.*math.pi) if ds > 0. else 0.
            if st["chip_rev"] >= 1.:
                edge = M.edge_temperature_rise(d.material, it.d_mm, st["edge_feed"], abs(w_rel)*0.5e-3*it.d_mm, 1.)
        else:
            st["edge_feed"] = 0.
            st["chip_rev"] = 0.
        self._heat(it, frac*(power + T*ds/dt), dt, d.oiled, edge_rise=edge, tool_share=frac)
        if d.kind == "wood" and h is not None:
            if f_eff < 0.02 and T > 1. and abs(w_rel) > 1.:
                st["rub_j"] += power*dt
                h.scorch_max_c = max(h.scorch_max_c, 20. + P.WOOD_SCORCH_K_PER_J*st["rub_j"])
            else:
                st["rub_j"] = max(0., st["rub_j"] - P.WOOD_RUB_RECOVER_J_S*dt)
        # ---- lateral: walking while unseated; seating; exit; profile
        if mode == "cut":
            self._walk(it, d, T, w_rel, tip_w, a_w, dt)
            s_seat = 0.5*M.chisel_length(it.d_mm)*1e-3/math.tan(0.5*P.POINT_ANGLE) + 1e-4
            if it.bit_type == "brad" and d.kind == "wood":
                s_seat = P.BRAD_SPUR_PIERCE_M            # the sharp centre spur seats the bit as it pierces
            if st["contact_depth"] >= s_seat:
                self._create_hole(it, d, tip_l, a_l)
        else:
            if it.bit_type == "brad" and d.kind == "wood" and stage["s"] < M.point_height(it.d_mm)*1e-3:
                self._spur(it, d, h, stage["s"], tip_w, dt)
            self._exit_checks(d, h, stage, T)
            off = self._offset(h, tip_l)
            if not h.profile or stage["s"] - h.profile[-1][0] >= 5e-4:
                h.profile.append([float(stage["s"]), float(off)])
            if stage["s"] > M.point_height(stage["d_mm"])*1e-3 and d.kind != "concrete":
                # (once the full diameter has entered the face: before that the point is still centring itself in a
                # pilot, 0.3 mm of "wander" before any wall is cut)
                stage["max_off"] = max(stage.get("max_off", 0.), off)
                # the stage's own wander, from where its full diameter began cutting: a 6 mm drill centred in a pilot
                # that wandered 0.7 mm cuts a round hole 0.7 mm off the mark, not a hole 1.4 mm oversize
                ov = self._offset_vec(h, tip_l)
                c0 = stage.setdefault("c0", [float(x) for x in ov])
                stage["wander"] = max(stage.get("wander", 0.), float(np.linalg.norm(ov - np.asarray(c0))))
                self._envelope(h)
            self._track_withdrawal(it, h, tip_l)
        self._lateral_check(it, d, h, f, a_w)
        if M_app > M.bit_torsion_limit(it.d_mm) and it.bit_type in ("hss", "brad"):
            self._break(it, d, h, f"torsion {M_app:.2f} N m > {M.bit_torsion_limit(it.d_mm):.2f} N m")
        self._wear(it, h)
        self._tie_rings(it, dname, rings, ring_thrusts)
        self.diag[it.name] = dict(thrust_n=T, torque_nm=M_app, torque_law_nm=M_cut, feed_mm_rev=f_eff, feed_law_mm_rev=feed,
                                  rings=[r["d"].name for r in rings],
                                  mode=mode,
                                  w_rel_rad_s=w_rel, depth_m=(stage["s"] if stage else st["contact_depth"]),
                                  workpiece=d.name, lam_t=lam_t, chi=chi, pilot_mm=dp_mm,
                                  lateral_n=st["side_load_n"], wear=W, chip_depth_m=st["chip_depth"],
                                  tip_below_floor_m=self._tip_below_floor(d, h, stage, tip_l, st))

    @staticmethod
    def _tip_below_floor(d, h, stage, tip_l, st):
        """How far the tip sits beyond the floor proxy's face: the contact's compliance under the thrust."""
        if h is not None and stage is not None:
            return float((tip_l - h.p0) @ h.a) - stage["s"]
        cf = st["contact_face"]
        if cf is None:
            return 0.
        n = cf[1]
        origin = d.center + n*float(np.abs(n) @ d.half)
        return float((tip_l - origin) @ (-n)) - st["contact_depth"]

    def _follow_axis(self, d, h, tip_l):
        """The primary stage's hole axis follows its bit's tip: the line from the mouth to the tip.
        KINEMATIC WRITE (declared): the hole is kept straight, from its mouth to the tip, and its
        axis and thickness are rewritten each tick the first stage deepens beyond AXIS_FOLLOW_MIN_M (metal and wood,
        within 20 deg of the seated axis; never concrete, whose percussive hole keeps its seated axis). A curved hole is
        therefore recorded as the chord from mouth to tip."""
        v = tip_l - h.p0
        n = float(np.linalg.norm(v))
        if n < AXIS_FOLLOW_MIN_M:
            return
        a = v/n
        if float(a @ h.seated_axis) > math.cos(math.radians(20.)):
            h.a = a
            h.thickness = self.thickness_along(d, h.p0, a)

    @staticmethod
    def _envelope(h):
        """The hole's swept envelope, per stage: each bit that cut a stage at its full diameter swept a circle of its
        own radius plus its own largest excursion from the hole axis (stage max_off); the hole's oversize (max_offset_m)
        is the widest of those beyond the largest stage's radius. A 3 mm pilot that wandered 0.4 mm sweeps 1.9 mm,
        inside the 6 mm drill's 3.0 mm, so it does not count against the finished hole."""
        r_max = max((0.5e-3*st["d_mm"] for st in h.stages), default=0.)
        env = max((0.5e-3*st["d_mm"] + st.get("max_off", 0.) + st.get("ream", 0.) for st in h.stages), default=0.)
        h.max_offset_m = max(0., env - r_max)
        big = max(h.stages, key=lambda st: st["d_mm"])
        h.wander_m = float(big.get("wander", 0.))

    @staticmethod
    def _offset(h, tip_l):
        rel = tip_l - h.p0
        return float(np.linalg.norm(rel - float(rel @ h.a)*h.a))

    @staticmethod
    def _offset_vec(h, tip_l):
        rel = tip_l - h.p0
        return rel - float(rel @ h.a)*h.a

    def _walk(self, it, d, T, w_rel, tip_w, a_w, dt):
        """Spinning-chisel lateral friction (implicit-damper form, capped at mu T) and the lip-asymmetry force."""
        n_w = self.world_dir(d, it.state["contact_face"][1])
        v_rel = self._point_velocity(it.body, tip_w) - (np.zeros(3) if d.static else self._point_velocity(d.body, tip_w))
        v_lat = v_rel - n_w*float(v_rel @ n_w)
        r_c = 0.25*M.chisel_length(it.d_mm)*1e-3
        c = P.TIP_MU*T/max(abs(w_rel)*r_c, 1e-5)
        F = -c*v_lat/(1. + c*dt/it.spindle.walking_mass())
        nf = float(np.linalg.norm(F))
        if nf > P.TIP_MU*T:
            F *= P.TIP_MU*T/nf
        if abs(w_rel) > 1.:
            R = self.pose(it.body)[1]
            e = R.apply(perp_basis(it.axis)[0])                 # a direction fixed in the bit: turns with it
            e -= n_w*float(e @ n_w)
            ne = float(np.linalg.norm(e))
            if ne > 1e-9:
                F = F + P.ASYM_GAIN*T*P.LIP_ERROR_M/(it.d_mm*1e-3)*e/ne
        self._force(it.body, F, tip_w)
        if not d.static:
            self._force(d.body, -F, tip_w)

    def _spur(self, it, d, h, s, tip_w, dt):
        """A brad point's centre spur, embedded ahead of the lips, holds the tip on the hole's axis until the bit's full
        diameter is in the wood. It is a short dowel in timber: held while the lateral load stays within its embedment
        capacity f_h d_s L_e (EN 1995-1-1 eq. 8.32), ploughing through the wood at that force beyond it. Held means the
        force that returns the tip towards the axis within SPUR_TAU, sized by the tip's own effective lateral mass
        (from the mass matrix at the tip: the drill and whatever holds it), at half the one-tick value so that it stays
        stable. Without it a 4 N side load at the housing slid the unseated 9/16 in brad point 2.4 mm."""
        d_s = P.BRAD_SPUR_D_FRAC*it.d_mm                                 # mm
        L_e = min(max(s, 0.), P.BRAD_SPUR_LEN_M)
        if L_e <= 0. or d_s <= 0.:
            return
        f_h = P.WOOD_EMBED_K*(1. - 0.01*d_s)*P.DENSITY.get(d.material, P.WOOD_DENSITY)    # N/mm2
        F_cap = f_h*d_s*L_e*1e3                                          # N/mm2 x mm x mm
        p0, a_w = self.world_point(d, h.p0), self.world_dir(d, h.a)
        r = tip_w - p0
        off = r - a_w*float(r @ a_w)
        v = self._point_velocity(it.body, tip_w) - (np.zeros(3) if d.static else self._point_velocity(d.body, tip_w))
        v_lat = v - a_w*float(v @ a_w)
        Pl = np.eye(3) - np.outer(a_w, a_w)
        A = Pl @ self._point_mobility(it.body, tip_w) @ Pl
        if not d.static:
            A = A + Pl @ self._point_mobility(d.body, tip_w) @ Pl
        lam = float(np.linalg.eigvalsh(A)[-1])                           # 1/kg: the most mobile lateral direction
        if lam <= 0.:
            return
        F = 0.5/lam*(-off/SPUR_TAU - v_lat)/dt
        nf = float(np.linalg.norm(F))
        if nf > F_cap:
            F *= F_cap/nf
        it.state["spur_n"] = round(float(np.linalg.norm(F)), 3)
        self._force(it.body, F, tip_w)
        if not d.static:
            self._force(d.body, -F, tip_w)

    def _point_mobility(self, body, p_w):
        """J M^-1 J^T at a world point on a Newton body (3 x 3, 1/kg): the point's inverse effective mass."""
        w = self.w
        m, dd, mujoco = w.solver.mj_model, w.solver.mj_data, w._mujoco
        if not hasattr(self, "_n2m"):
            self._n2m = {n: j for j, n in w._mj_to_newton.items()}
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, dd, jacp, jacr, np.asarray(p_w, float), self._n2m[body])
        X = np.zeros((3, m.nv))
        mujoco.mj_solveM(m, dd, X, jacp)
        return jacp @ X.T

    def _create_hole(self, it, d, tip_l, a_l):
        n = it.state["contact_face"][1]
        origin = d.center + n*float(np.abs(n) @ d.half)
        a = a_l/np.linalg.norm(a_l)
        foot = tip_l - a*float((tip_l - origin) @ n)/float(a @ n)          # where the bit axis meets the face
        depth = it.state["contact_depth"]
        # One hole whose start wandered is one record. A tip that leaves a start before its full diameter is in the
        # part, and seats again within its own radius of it, goes on with that start: the two cones overlap and
        # the deeper one takes the other in. (With a second record, at panel-holes' first mark the pilot met
        # the backer 0.38 mm off as it broke through the sheet, cut 0.3 mm, and seated again 0.62 mm away; the
        # 6 mm drill then enlarged both, was pushed about between them and left the sheet's hole lobed, roundness
        # 1.12 against the 1.10 allowed: E12.)
        for h0 in d.holes:
            if h0.kind != "drilled" or len(h0.stages) != 1 or h0.stages[0]["through"]:
                continue
            st0 = h0.stages[0]
            if abs(st0["d_mm"] - it.d_mm) > 1e-6 or st0["s"] >= M.point_height(it.d_mm)*1e-3:
                continue
            r0 = foot - h0.p0
            off = float(np.linalg.norm(r0 - n*float(r0 @ n)))
            if off < 0.5e-3*it.d_mm:
                h0.events.append(dict(tick=self.w.tick, event="reseated", moved_mm=round(off*1e3, 3),
                                      depth_mm=round(depth*1e3, 3)))
                h0.p0, h0.a, h0.seated_axis = foot.copy(), a.copy(), a.copy()
                st0["s"] = max(st0["s"], depth)
                h0.last_tick = self.w.tick
                if it.name not in h0.tools:
                    h0.tools.append(it.name)
                it.state["contact_depth"] = 0.
                it.state["contact_face"] = None
                it.state["chip_depth"] = depth
                self.events.append(dict(type="hole_reseated", workpiece=d.name, hole=h0.id, tick=self.w.tick,
                                        tool=it.name, moved_mm=round(off*1e3, 3)))
                return h0
        h = Hole(self.next_id, d.name, foot.copy(), a.copy(), n.copy(), thickness=self.thickness_along(d, foot, a),
                 stages=[dict(d_mm=float(it.d_mm), s=depth, start=0., through=False)], seated_axis=a.copy(),
                 created_tick=self.w.tick, last_tick=self.w.tick, tools=[it.name])
        self.next_id += 1
        d.holes.append(h)
        it.state["contact_depth"] = 0.
        it.state["contact_face"] = None
        it.state["chip_depth"] = depth
        h.events.append(dict(tick=self.w.tick, event="seated", depth_mm=round(depth*1e3, 3)))
        if d.kind == "concrete":
            lat = float(self.diag.get(it.name, {}).get("lateral_n", 0.) or 0.)
            h.entry_spall_d_m = (P.SPALL_DIAMETER_FACTOR if lat < P.SPALL_LATERAL_N else
                                 P.SPALL_DIAMETER_FACTOR_LATERAL)*it.d_mm*1e-3
        self.events.append(dict(type="hole_created", workpiece=d.name, hole=h.id, tick=self.w.tick, tool=it.name,
                                d_mm=it.d_mm))
        return h

    def _exit_checks(self, d, h, stage, T):
        if math.isinf(h.thickness) or stage["through"]:
            return
        s = stage["s"]
        h_r = h.thickness - s
        backed = self._exit_backed(d, h)
        # a stage enlarging a hole whose smaller stage already runs through has no central cap left to collapse
        over_pilot = any(st["through"] and st["d_mm"] < stage["d_mm"] - 1e-6 for st in h.stages)
        if d.kind == "metal" and not h.cap_collapsed and h_r > 0 and not over_pilot:
            y = M.METALS[d.material]["yield_mpa"]
            if not backed and h_r <= M.cap_collapse_thickness(T, y):
                h.cap_collapsed = True
                h.burr_m = max(h.burr_m, M.exit_burr_height(h_r))    # the cap leaves as the drill cap; its rim stays
                h.events.append(dict(tick=self.w.tick, event="cap_collapse", remaining_mm=round(h_r*1e3, 3),
                                     thrust_n=round(T, 1), burr_mm=round(M.exit_burr_height(h_r)*1e3, 3)))
        if d.kind == "wood" and h_r > 0 and not backed:
            Fc = M.tearout_threshold(h_r)
            if T > Fc:
                h.exit_fd = min(P.FD_MAX, max(h.exit_fd, 1. + P.FD_SLOPE*(T - Fc)))
                if not any(e.get("event") == "tearout_onset" for e in h.events):
                    h.events.append(dict(tick=self.w.tick, event="tearout_onset", remaining_mm=round(h_r*1e3, 3),
                                         thrust_n=round(T, 1), threshold_n=round(Fc, 1)))
        if s >= h.thickness + M.point_height(stage["d_mm"])*1e-3:
            stage["through"] = True
            if d.kind == "metal" and stage is h.stages[0]:
                h.burr_m = max(h.burr_m, P.BURR_MIN_M)
            h.events.append(dict(tick=self.w.tick, event="through", d_mm=stage["d_mm"], backed=backed))
            self.events.append(dict(type="through", workpiece=d.name, hole=h.id, tick=self.w.tick))

    def _exit_backed(self, d, h):
        """True when another solid touches the exit face within 1.5 d of the exit point (a backer, the table, the next
        part of a stack). Re-evaluated every 10 ticks."""
        cache = self.__dict__.setdefault("_backed_cache", {})
        key = (d.name, h.id)
        if key in cache and self.w.tick - cache[key][0] < 10:
            return cache[key][1]
        m, dd = self.w.solver.mj_model, self.w.solver.mj_data
        exit_w = self.world_point(d, h.p0 + h.a*h.thickness)
        n_exit_w = self.world_dir(d, h.a)
        mjb = self._n2m.get(d.body, 0)
        backed = False
        r = 1.5*h.d_mm*1e-3 + 0.004
        for i in range(dd.ncon):
            g = dd.contact.geom[i]
            if self._is_proxy[g[0]] or self._is_proxy[g[1]]:
                continue
            if mjb not in (int(m.geom_bodyid[g[0]]), int(m.geom_bodyid[g[1]])):
                continue
            pos = np.asarray(dd.contact.pos[i])
            if np.linalg.norm(pos - exit_w) < r and abs(float((pos - exit_w) @ n_exit_w)) < 0.003:
                backed = True
                break
        if not backed:
            # another workpiece's solid just beyond the exit face (welded and static parts make no contacts with each
            # other: a static sheet on a static plywood backer read as unbacked, E6)
            t1, t2 = perp_basis(n_exit_w)
            rr = 0.7*0.5e-3*h.d_mm
            probes = [exit_w + n_exit_w*0.0005 + o for o in (0.*t1, rr*t1, -rr*t1, rr*t2, -rr*t2)]
            for d2 in self.drillables.values():
                if d2 is d:
                    continue
                for p_w in probes:
                    rel = self.local(d2, p_w) - d2.center
                    if np.all(np.abs(rel) <= d2.half + 3e-4):
                        backed = True
                        break
                if backed:
                    break
        cache[key] = (self.w.tick, backed)
        return backed

    def _track_withdrawal(self, it, h, tip_l):
        z = float((tip_l - h.p0) @ h.a)
        st = it.state
        if z < 0.5*max(h.depth, 1e-6):
            st["chip_depth"] = 0.                       # flutes cleared by a withdrawal
            if st["withdraw_mark"] != h.id:
                self._settle_dust(h, P.WITHDRAW_CARRIES)
                if h.dust_loose_mm3 > 0:
                    carried = P.WITHDRAW_CARRIES*h.dust_loose_mm3
                    h.dust_loose_mm3 -= carried
                    h.dust_evacuated_mm3 += carried
                st["withdraw_mark"] = h.id
        elif z > 0.8*h.depth:
            st["withdraw_mark"] = None

    def _bending_now(self, it, d, M_raw, L, tip_w, a_w):
        """The bending moment the bit carries, given the moment M_raw that this tick's wall forces put on it.
        A sustained load is carried in full. An impact is not: the rigid-contact solve stops a tool that rattles
        sideways in its hole within one tick, a force of hundreds of newtons for a millisecond, while a real bit
        bends. The bit is a cantilever spring k = 3 E I / L^3 between the wall and the mass that moves with the tool
        (m_eff, the inverse of the lateral mobility at the tip, of the tool and of a workpiece that can move): an
        impulse J loads it to J sqrt(k / m_eff), which is what a first-order lag of time constant
        tau = sqrt(m_eff / k) returns for a one-tick pulse, and a load that lasts passes it unchanged.
        (Without this a rotary hammer drilling through the ledger into the wall, its bending moment 4 N m on
        average, broke its 13.3 mm bit on a single tick at 640 N of side force, 138.8 N m against the body's
        136.7: E12 ledger-anchor, seed 1, the third hole.)"""
        st = it.state
        if M_raw <= 0. and st.get("bend_f", 0.) < 1e-6:
            st["bend_f"] = 0.
            return 0.
        if self.w.tick - st.get("bend_tau_tick", -10**9) >= 100:
            I = P.FLUTE_I_FACTOR*math.pi*(it.d_mm*1e-3)**4/64.
            k_b = 3.*P.STEEL_E_PA*I/max(L, 1e-3)**3
            t1, t2 = perp_basis(a_w)
            W = self._point_mobility(it.body, tip_w)
            if d is not None and not d.static:
                W = W + self._point_mobility(d.body, tip_w)
            mob = 0.5*(float(t1 @ W @ t1) + float(t2 @ W @ t2))
            st["bend_tau"] = min(math.sqrt(1./(max(mob, 1e-9)*k_b)), P.BEND_TAU_MAX_S)
            st["bend_tau_tick"] = self.w.tick
        dt = float(self.w.dt)
        st["bend_f"] = st.get("bend_f", 0.) + dt/(st["bend_tau"] + dt)*(M_raw - st.get("bend_f", 0.))
        return st["bend_f"]

    def _lateral_check(self, it, d, h, f, a_w):
        lat = np.asarray(f.get("cage", np.zeros(3)), float)
        F_lat = float(np.linalg.norm(lat - a_w*float(lat @ a_w)))
        it.state["side_load_n"] = F_lat
        tip_w = self.item_world(it)[0]
        if it.bit_type in ("hss", "brad"):
            M_b = self._bending_now(it, d, F_lat*it.stickout, it.stickout, tip_w, a_w)
            it.state["bending_nm"], it.state["bending_raw_nm"] = M_b, F_lat*it.stickout
            if M_b > M.bit_bending_limit(it.d_mm):
                self._break(it, d, h, f"bending {M_b:.2f} N m > {M.bit_bending_limit(it.d_mm):.2f} N m")
        elif it.bit_type == "sds" and h is not None:
            # the bending moment the hole's wall contacts put on the bit at the chuck end of its free length (a wedge
            # couple has a small net side force but a large moment); beyond the body's bending strength it breaks
            base = tip_w - a_w*it.length
            Mo = np.asarray(f.get("cage_mom", np.zeros(3)), float) - np.cross(base, lat)
            M_raw = float(np.linalg.norm(Mo - a_w*float(Mo @ a_w)))
            M_b = self._bending_now(it, d, M_raw, it.length, tip_w, a_w)
            it.state["bending_nm"], it.state["bending_raw_nm"] = M_b, M_raw
            lim = self.sds_bending_limit(it.d_mm)
            if M_b > lim:
                self._break(it, d, h, f"bending {M_b:.2f} N m > {lim:.2f} N m (SDS body)")
        if h is not None and d.kind == "concrete" and not it.state["broken"]:
            self._ream(it, d, h, f)

    @staticmethod
    def sds_bending_limit(d_mm):
        """Bending moment (N m) at which an SDS bit's fluted body yields: sigma x I_eff/(d/2), the same flute section
        factor as the HSS drills (FLUTE_I_FACTOR), with the body steel's strength SDS_BODY_BEND_STRENGTH."""
        I = P.FLUTE_I_FACTOR*math.pi*(d_mm*1e-3)**4/64.
        return P.SDS_BODY_BEND_STRENGTH*I/(0.5*d_mm*1e-3)

    def _ream(self, it, d, h, f):
        """A spinning bit pressed against its concrete hole's wall wears the wall away (a tilting or side-loaded bit
        reams its hole). The friction work of the wall contacts, mu_wall x (sum of their normal forces) x the bit's
        surface speed, removes concrete at the rotary-scraping specific energy CONCRETE_ROT_EV_J_MM3 (the energy the
        model already charges a scraping face); the volume comes off the loaded side as a crescent over REAM_ZONE_D bit
        diameters of the hole's length (crescent area ~ 2 R x the radial wear), and the hole's radius grows by that
        wear (stage 'ream', which the wall proxies follow next tick and _envelope records as the hole's oversize). The
        worn concrete becomes dust in the flutes. No force is applied here: the solver's wall contacts carry the load
        and their friction already dissipates this work."""
        F = float(f.get("cage_abs", 0.))
        if F <= 0. or it.spindle is None:
            return
        stage = h.stage(it.d_mm)
        if stage is None:
            return
        a_w = self.item_world(it)[1]
        w_wp = np.zeros(3) if d.static else self.w.bqd[d.body][3:].astype(float)
        w_rel = abs(float((self.w.bqd[it.body][3:].astype(float) - w_wp) @ a_w))
        if w_rel < 1.:
            return
        R = 0.5e-3*it.d_mm
        dV = P.MU["bit_wall"]*F*w_rel*R*self.w.dt/(P.CONCRETE_ROT_EV_J_MM3*1e9)          # m^3
        dr = dV/(2.*R*P.REAM_ZONE_D*it.d_mm*1e-3)
        stage["ream"] = stage.get("ream", 0.) + dr
        h.removed_mm3 += dV*1e9
        self._dust(h, dV*1e9, rotating=True)
        self._envelope(h)

    # ------------------------------------------------------------------ percussion (called by the rotary hammer)
    def blow(self, it, energy_j):
        """One delivered blow at the bit's tip; returns the removed volume (mm^3), 0 for a blank blow. A reduced model:
        the blow's energy is booked as removed concrete (the hole's depth is written from it), and the only force of a
        blow is the recoil on the housing (tools.RotaryHammer); no impulse is applied to the bit itself."""
        st = it.state
        if st["broken"]:
            return 0.
        eng = self._engaged(it)
        if eng is None:
            return 0.
        dname, (k, mode, hid) = eng
        d = self.drillables[dname]
        tip_w, a_w = self.item_world(it)
        tip_l, a_l = self.local(d, tip_w, a_w)
        A = math.pi*(0.5*it.d_mm)**2
        ang = it.spindle.angle()
        last = st["last_blow_angle"]
        st["last_blow_angle"] = ang
        index = 1. if last is None else min(1., max(P.HAMMER_NO_INDEX_EFF,
                                                    abs(ang - last)/math.radians(P.HAMMER_INDEX_DEG)))
        st["blows"] += 1
        if d.kind != "concrete":
            return 0.
        if mode == "cut":
            n = st["contact_face"][1]
            origin = d.center + n*float(np.abs(n) @ d.half)
            if float((tip_l - origin) @ (-n)) < st["contact_depth"] - 0.0015:
                return 0.
            fr = self._fractions(d, tip_l, a_l, it)
            dv = M.blow_removal_mm3(energy_j*index/st["wear"], fr, it.d_mm)
            st["contact_depth"] += dv/A*1e-3
            st["pending_dv"] = st.get("pending_dv", 0.) + dv
            s_seat = 0.5*M.chisel_length(it.d_mm)*1e-3/math.tan(0.5*P.POINT_ANGLE) + 1e-4
            if st["contact_depth"] >= s_seat:
                h = self._create_hole(it, d, tip_l, a_l)
                h.removed_mm3 += st["pending_dv"]
                h.energy["blow_j"] += energy_j
                self._dust(h, st["pending_dv"], rotating=index > P.HAMMER_NO_INDEX_EFF)
                st["pending_dv"] = 0.
            return dv
        h = next(x for x in d.holes if x.id == hid)
        stage = self._stage_for(it, h, tip_l)
        if stage is None:
            return 0.
        z_tip = float((tip_l - h.p0) @ h.a)
        if z_tip < stage["s"] - 0.0015:                # not on the floor: a blank blow
            return 0.
        centre = h.p0 + h.a*stage["s"]
        fr = self._fractions(d, centre, h.a, it)
        cushion = math.exp(-h.dust_column_m()/P.DUST_CUSHION_M)   # dust under the bit absorbs the blow
        dv = M.blow_removal_mm3(energy_j*index*cushion/st["wear"], fr, it.d_mm)
        ds = dv/A*1e-3
        stage["s"] += ds
        st["chip_depth"] += ds
        h.removed_mm3 += dv
        h.energy["blow_j"] += energy_j
        h.last_tick = self.w.tick
        self._dust(h, dv, rotating=index > P.HAMMER_NO_INDEX_EFF)
        if fr["steel"] > 0:
            st["steel_j"] += energy_j*fr["steel"]
            self.rebar_overlap(d, h, stage)                     # records the deepest cut of the whole face into the bar
            if st["steel_j"] > P.CARBIDE_CHIP_J and not st["chipped"]:
                st["chipped"] = True
                st["wear"] += 0.3
                h.events.append(dict(tick=self.w.tick, event="carbide_chipped"))
            if not any(e.get("event") == "rebar_contact" for e in h.events):
                h.events.append(dict(tick=self.w.tick, event="rebar_contact", depth_mm=round(stage["s"]*1e3, 2)))
        return dv

    def _dust(self, h, dv_mm3, rotating):
        """Concrete dust: produced = bulking x removed. Turning flutes carry all but DUST_RESIDUAL_FRACTION out to the
        shroud; that fraction rides in the flutes (in transit) and settles at the bottom when the bit stops turning
        or is withdrawn (_settle_dust). Without rotation everything stays at the bottom at once."""
        produced = P.DUST_BULKING*dv_mm3
        h.dust_produced_mm3 += produced
        if rotating:
            h.dust_transit_mm3 += P.DUST_RESIDUAL_FRACTION*produced
            h.dust_evacuated_mm3 += (1. - P.DUST_RESIDUAL_FRACTION)*produced
        else:
            h.dust_packed_mm3 += 0.5*produced
            h.dust_loose_mm3 += 0.5*produced

    @staticmethod
    def _settle_dust(h, carried_out):
        """Dust in transit settles: `carried_out` of it leaves with a withdrawing bit, the rest drops to the bottom."""
        if h.dust_transit_mm3 <= 0.:
            return
        out = carried_out*h.dust_transit_mm3
        rest = h.dust_transit_mm3 - out
        h.dust_evacuated_mm3 += out
        h.dust_packed_mm3 += 0.5*rest
        h.dust_loose_mm3 += 0.5*rest
        h.dust_transit_mm3 = 0.

    def _fractions(self, d, centre_l, axis_l, it):
        """Mortar / aggregate / steel under the bit's face with its tip at centre_l. Steel is the share of the whole
        conical face (equal-area samples, _face_cut) inside a bar; mortar and aggregate share the rest as the
        aggregate sampler finds them."""
        if d.fields is None:
            return dict(mortar=1., aggregate=0., steel=0.)
        fr = d.fields.face_fractions(d.to_fields(centre_l), d.to_fields_dir(axis_l), 0.5e-3*it.d_mm)
        if not d.fields.bars:
            return fr
        steel = self._face_cut(d, centre_l, axis_l, it.d_mm)[1]
        rest = fr["mortar"] + fr["aggregate"]
        share = fr["mortar"]/rest if rest > 0. else 1.
        return dict(mortar=(1. - steel)*share, aggregate=(1. - steel)*(1. - share), steel=steel)

    # ------------------------------------------------------------------ reinforcement: the bit's whole face
    FACE_RINGS = 4                      # equal-area rings of face samples (12 azimuths each) + the tip + the rim
    FACE_AZ = 12

    def _face_points(self, d, tip_l, axis_l, d_mm):
        """Points (wall-fields frame) on the face of a bit of d_mm whose tip is at tip_l: the cone from the tip to the
        full-diameter corners (M.point_height, as the hole record's full depth), equal-area rings plus the tip and the
        rim. Returns (points, mask of the equal-area samples)."""
        a = np.asarray(axis_l, float)/np.linalg.norm(axis_l)
        t1, t2 = perp_basis(a)
        R = 0.5e-3*d_mm
        ph = M.point_height(d_mm)*1e-3
        k = self.FACE_RINGS
        rho = np.concatenate([[0.], R*np.sqrt((np.arange(k) + 0.5)/k), [R]])
        phi = 2*math.pi*np.arange(self.FACE_AZ)/self.FACE_AZ
        pts, eq = [np.asarray(tip_l, float)], [False]
        for i, r in enumerate(rho[1:], start=1):
            ring = (np.cos(phi)[:, None]*t1 + np.sin(phi)[:, None]*t2)*r - a*(ph*r/R)
            pts += list(np.asarray(tip_l, float) + ring)
            eq += [i <= k]*self.FACE_AZ
        P_l = np.asarray(pts)
        o, Rm = d.fields_map
        return (P_l - o) @ Rm.T, np.asarray(eq)

    def _face_cut(self, d, tip_l, axis_l, d_mm):
        """(depth (m) the bit's face, tip at tip_l, has cut into a bar: the largest steel chord behind any face point
        along the axis; share of the face in steel)."""
        if d.fields is None or not d.fields.bars:
            return 0., 0.
        pts, eq = self._face_points(d, tip_l, axis_l, d_mm)
        cut, inside = d.fields.face_cut(pts, d.to_fields_dir(axis_l))
        return float(cut.max()), float(inside[eq].mean())

    def rebar_overlap(self, d, h, stage):
        """True when the bit's face at this stage's floor has cut into a bar - anywhere on its face, not only on its
        axis (an edge that meets the bar catches it: the jam). Records the deepest cut as the hole's rebar damage.
        (Judged at the axis point, a bit overlapping a bar by 12 of its 13.3 mm chipped and drilled on through it to
        112 mm with 1.8 mm of recorded damage.)"""
        if d.fields is None or stage is None or not d.fields.bars:
            return False
        cache = self.__dict__.setdefault("_cut_cache", {})
        key = (d.name, h.id, stage["d_mm"])
        if key in cache and cache[key][0] == stage["s"]:
            return cache[key][1] > 0.
        cut = self._face_cut(d, h.p0 + h.a*stage["s"], h.a, stage["d_mm"])[0]
        cache[key] = (stage["s"], cut)
        if cut > h.rebar_damage_m:
            h.rebar_damage_m = cut
        if cut > 0. and not any(e.get("event") == "rebar_contact" for e in h.events):
            h.events.append(dict(tick=self.w.tick, event="rebar_contact", depth_mm=round(stage["s"]*1e3, 2)))
        return cut > 0.

    # ------------------------------------------------------------------ passive items
    def vacuum_running(self):
        """Is the vacuum the nozzle hangs on running? The station's vacuum unit (scene.tools['vacuum'], switched by
        hand) if there is one; otherwise a bench's declared assumption world.vacuum_assumed_on (default: off)."""
        vac = getattr(self.w.scene, "tools", {}).get("vacuum")
        if vac is not None and hasattr(vac, "running"):
            return bool(vac.running(self.w))
        return bool(getattr(self.w, "vacuum_assumed_on", False))

    def _cleaning_update(self, it):
        tip_w, ax_w = self.item_world(it)
        for dn, (k, mode, hid) in list(it.slots.items()):
            if mode != "hole":
                continue
            d = self.drillables[dn]
            h = next(x for x in d.holes if x.id == hid)
            z = float((self.local(d, tip_w) - h.p0) @ h.a)
            floor = h.floor_for(it.radius)
            if math.isinf(floor):
                continue
            ab = it.state["at_bottom"]
            key = f"{dn}/{h.id}"
            if it.kind == "nozzle":
                if z >= floor - h.dust_column_m() - P.NOZZLE_BOTTOM_TOL_M:
                    # suction lifts loose dust only while the vacuum runs: a pass counts if the vacuum was running
                    # while the nozzle was at the bottom (a nozzle on a switched-off vacuum is a stick in a hole)
                    if self.vacuum_running():
                        ab[key] = True
                    elif not ab.get(key) and ab.get(key + "/off") is None:
                        ab[key + "/off"] = self.w.tick
                        h.events.append(dict(tick=self.w.tick, event="nozzle_at_bottom_vacuum_off"))
                elif z < 0.5*floor and (ab.get(key) or ab.get(key + "/off") is not None):
                    ab.pop(key + "/off", None)
                    if not ab.get(key):
                        continue
                    ab[key] = False
                    removed = P.VACUUM_PASS_REMOVES*h.dust_loose_mm3
                    h.dust_loose_mm3 -= removed
                    h.dust_evacuated_mm3 += removed
                    h.vacuum_passes.append(self.w.tick)
                    h.events.append(dict(tick=self.w.tick, event="vacuum_pass", removed_mm3=round(removed, 1)))
            else:
                if z >= P.BRUSH_DEPTH_FRACTION*floor - h.dust_column_m():
                    ab[key] = True
                elif ab.get(key) and z < 0.5*floor:
                    ab[key] = False
                    moved = P.BRUSH_STROKE_LOOSENS*h.dust_packed_mm3
                    h.dust_packed_mm3 -= moved
                    h.dust_loose_mm3 += moved
                    h.brush_strokes.append(self.w.tick)
                    h.events.append(dict(tick=self.w.tick, event="brush_stroke", loosened_mm3=round(moved, 1)))

    def _insertion_update(self, it, forces):
        st = it.state
        tip_w, ax_w = self.item_world(it)
        clip_best = None
        for dn, (k, mode, hid) in it.slots.items():
            if mode != "hole":
                continue
            d = self.drillables[dn]
            h = next(x for x in d.holes if x.id == hid)
            z = float((self.local(d, tip_w) - h.p0) @ h.a)
            st["insertion_m"][f"{dn}/{h.id}"] = z
            # the driven depth: the deepest the tip has STAYED in this hole (params.DRIVEN_DWELL), not the deepest
            # point of a blow's elastic overshoot
            key_ = f"{dn}/{h.id}"
            deep = st.setdefault("insertion_max_m", {})
            dwell = st.setdefault("insertion_dwell", {})
            z0_, n_ = dwell.get(key_, (z, 0))
            if abs(z - z0_) <= P.DRIVEN_DWELL["band_m"]:
                n_ += 1
            else:
                z0_, n_ = z, 0
            dwell[key_] = (z0_, n_)
            if key_ not in deep:
                deep[key_] = z
            elif n_*self.w.dt >= P.DRIVEN_DWELL["seconds"]:
                deep[key_] = max(deep[key_], z0_)
            f = forces.get((dn, k), {})
            a_w = self.world_dir(d, h.a)
            axial = abs(float(np.asarray(f.get("cage", np.zeros(3))) @ a_w))
            clip = 0.
            if it.kind == "anchor" and d.kind == "concrete":
                c = self._clip_in_hole(it, d, h, z)
                if clip_best is None or c["engaged"] > clip_best["engaged"]:
                    clip_best = c
                if c["engaged"] > 0.:
                    clip = self._clip_drag(it, d, a_w, tip_w)
            st["axial_drag_n"] = axial + clip
            st["max_axial_n"] = max(st["max_axial_n"], axial + clip)
            if d.kind == "concrete" and not h.dust_compacted and h.dust_loose_mm3 + h.dust_packed_mm3 > 0.:
                # a part driven onto the dust at the bottom (the floor plug sits on the dust column) presses it from its
                # loose to its compacted bulk density: the column shortens by DUST_COMPACT_RATIO, once
                A = math.pi*(0.5e-3*h.d_mm)**2
                if float(f.get("bottom", 0.)) > P.DUST_COMPACT_PA*A:
                    h.dust_loose_mm3 *= P.DUST_COMPACT_RATIO
                    h.dust_packed_mm3 *= P.DUST_COMPACT_RATIO
                    h.dust_compacted = True
                    h.events.append(dict(tick=self.w.tick, event="dust_compacted", item=it.name,
                                         column_mm=round(h.dust_column_m()*1e3, 2)))
        if it.kind == "anchor":
            st["clip"] = clip_best

    def _clip_in_hole(self, it, d, h, z_tip):
        """Where the wedge anchor's expansion clip sits in hole h: the share of its length inside the hole (it spans
        2 mm to 2 mm + CLIP_LEN_M from the tip as delivered and stays put on the wall while the cone is drawn through it
        by s), the radius of the hole it meets (the swept hole every item meets: the stage's radius plus the hole's
        oversize), the tip's depth, and whether the hole is in fixed concrete (parts.WedgeAnchor holds the stud by joint
        friction on its free joint, i.e. against the world)."""
        s = it.state.get("setting", {}).get("s_m", 0.)
        z_hi = z_tip - 0.002 + s
        z_lo = z_hi - P.CLIP_LEN_M
        inside = max(0., min(z_hi, h.depth) - max(z_lo, 0.))
        return dict(key=f"{d.name}/{h.id}", engaged=round(min(1., inside/P.CLIP_LEN_M), 6),
                    R_m=0.5e-3*h.d_mm + h.max_offset_m, concrete=bool(d.static), z_tip=z_tip, clip_depth_m=round(z_hi, 6))

    def _clip_drag(self, it, d, a_w, tip_w):
        """The wedge anchor clip's hold on the stud along the hole (N): the ring's drag while it is driven in, the cone's
        resistance once set. It is applied by parts.WedgeAnchor as implicit Coulomb friction on the stud's free joint;
        here it is only reported."""
        return float(it.state.get("setting", {}).get("hold_n", 0.))

    # ------------------------------------------------------------------ heat, wear, breakage
    def _heat(self, it, q_w, dt, oiled, edge_rise=0., tool_share=0.):
        """Lumped tip node (heated length 2.5 d): C dT/dt = q - G (T - 20 C), updated exactly for a constant q, kept in
        state "bulk_temp_c". "temp_c" is the hottest point of the tip: the node, or the cutting corners' tool-chip
        interface while a chip is cut, whichever is hotter. The interface sits edge_rise (Cook, K) above ambient; a
        tool already at the node temperature raises it by tool_share of that excess only (two bodies in contact at a
        heat source: the interface rises with the tool's base temperature by the share of the heat entering the
        tool, Blok's partition, the share HEAT_TO_TIP the node is fed with). Burning and thermal wear read temp_c."""
        st = it.state
        r = 0.5e-3*(it.d_mm if it.d_mm > 0 else 2e3*it.radius)
        L = 5.*r
        C = P.HSS_DENSITY*0.6*math.pi*r*r*L*P.HSS_CP
        G = P.TIP_H_W_M2K*(P.OIL_H_FACTOR if oiled else 1.)*2*math.pi*r*L + P.HSS_K*math.pi*r*r/(3*L)
        T_inf = 20. + q_w/G
        bulk = st.get("bulk_temp_c", st["temp_c"])
        st["bulk_temp_c"] = T_inf + (bulk - T_inf)*math.exp(-G*dt/C)
        st["edge_rise_k"] = edge_rise
        interface = 20. + edge_rise + tool_share*(st["bulk_temp_c"] - 20.) if edge_rise > 0. else -math.inf
        st["temp_c"] = max(st["bulk_temp_c"], interface)
        st["max_temp_c"] = max(st["max_temp_c"], st["temp_c"])
        if st["temp_c"] > P.HSS_BURN_C and not st["burnt"] and it.bit_type in ("hss", "brad"):
            st["burnt"] = True
            st["wear"] += P.BURN_WEAR_STEP
            self.events.append(dict(type="bit_burnt", item=it.name, tick=self.w.tick, temp_c=round(st["temp_c"], 1)))

    def _wear(self, it, h):
        if h is None:
            return
        st = it.state
        dv = h.removed_mm3 - st["removed_seen"] if st["removed_hole"] == h.id else 0.
        st["removed_seen"], st["removed_hole"] = h.removed_mm3, h.id
        if dv <= 0:
            return
        if it.bit_type in ("hss", "brad"):
            T = st["temp_c"]
            factor = math.exp((T - P.HSS_SOFTEN_C)/50.) if T > P.HSS_SOFTEN_C else 1.
            st["wear"] += P.WEAR_PER_MM3*dv*factor
        elif it.bit_type == "sds":
            st["wear"] += P.SDS_WEAR_PER_MM3*dv

    def _break(self, it, d, h, why):
        if it.state["broken"]:
            return
        it.state["broken"] = True
        it.state["broken_reason"] = why
        if h is not None and d is not None:
            z = float((self.local(d, self.item_world(it)[0]) - h.p0) @ h.a)
            stub = min(max(z, 0.), 2.5*it.d_mm*1e-3)                 # the stub (flute length in the hole) stays
            h.blocked_at = max(0., z - stub)
            it.state["stub_m"] = stub                                  # the piece the bit loses (world.on_bit_broken)
            h.events.append(dict(tick=self.w.tick, event="bit_broken", item=it.name, reason=why))
        elif why.startswith("bending"):
            # out of any hole a side load at the tip bends the bit as a cantilever, whose largest moment is at its root:
            # it snaps at the chuck and loses its whole free length
            it.state["stub_m"] = float(it.stickout)
        self.events.append(dict(type="bit_broken", item=it.name, tick=self.w.tick, reason=why))
        if hasattr(self.w, "on_bit_broken"):
            self.w.on_bit_broken(it)

    # ------------------------------------------------------------------ wrench helpers
    def _point_velocity(self, body, p_w):
        qd = self.w.bqd[body].astype(float)
        q = self.w.bq[body].astype(float)
        com = q[:3] + _rot(q[3:]).apply(self.w.body_com[body])
        return qd[:3] + np.cross(qd[3:], np.asarray(p_w) - com)

    def _force(self, body, F, p_w):
        q = self.w.bq[body].astype(float)
        com = q[:3] + _rot(q[3:]).apply(self.w.body_com[body])
        wr = self.pending.setdefault(body, np.zeros(6))
        wr[:3] += F
        wr[3:] += np.cross(np.asarray(p_w) - com, F)

    def _torque(self, body, tau):
        wr = self.pending.setdefault(body, np.zeros(6))
        wr[3:] += tau

    # ------------------------------------------------------------------ dimples (centre punch)
    def make_dimple(self, d, p_l, n_l, depth):
        """A 90-degree conical dimple of `depth` (m) at the face point nearest p_l on the face with outward normal
        n_l. A second strike within the dimple deepens it (volumes add)."""
        origin = d.center + n_l*float(np.abs(n_l) @ d.half)
        foot = p_l - n_l*float((p_l - origin) @ n_l)
        for h in d.holes:
            if h.kind == "dimple" and np.linalg.norm(h.p0 - foot) < 0.5e-3*h.d_mm:
                st = h.stages[0]
                new = (st["s"]**3 + depth**3)**(1/3)
                st["s"] = new
                st["d_mm"] = 2e3*new*math.tan(P.PUNCH_TIP_HALF_ANGLE)
                h.events.append(dict(tick=self.w.tick, event="restrike", depth_mm=round(new*1e3, 4)))
                return h
        dmm = 2e3*depth*math.tan(P.PUNCH_TIP_HALF_ANGLE)
        h = Hole(self.next_id, d.name, foot, -n_l.copy(), n_l.copy(), thickness=self.thickness_along(d, foot, -n_l),
                 stages=[dict(d_mm=dmm, s=depth, start=0., through=False)], kind="dimple", created_tick=self.w.tick,
                 last_tick=self.w.tick, tools=["centre_punch"])
        self.next_id += 1
        d.holes.append(h)
        self.events.append(dict(type="dimple", workpiece=d.name, hole=h.id, tick=self.w.tick,
                                depth_mm=round(depth*1e3, 4), d_mm=round(dmm, 4)))
        return h

    # ------------------------------------------------------------------ checkpoints and summaries
    def snapshot(self):
        return copy.deepcopy(dict(holes={n: d.holes for n, d in self.drillables.items()}, next_id=self.next_id,
                                  items={n: it.state for n, it in self.items.items()}, events=self.events,
                                  diag=self.diag))

    def restore(self, snap):
        """Restore hole records and item states, park every slot and re-place proxies for the restored poses (the
        caller restores body poses first)."""
        snap = copy.deepcopy(snap)
        for n, holes in snap["holes"].items():
            self.drillables[n].holes = holes
        self.next_id = snap["next_id"]
        self.events = snap["events"]
        self.diag = snap["diag"]
        for key in self._geoms:
            self._park(*key)
        self._free = {d.name: list(range(len(d.slots))) for d in self.drillables.values()}
        for n, state in snap["items"].items():
            self.items[n].state = state
            self.items[n].slots = {}
        self.pending = {}
        self.__dict__.pop("_backed_cache", None)
        self.__dict__.pop("_cut_cache", None)
        saved = self.w.body_wrench.copy()
        self.pre_step()
        self.w.body_wrench[:] = saved

    def summary(self):
        return {n: [h.summary() for h in d.holes] for n, d in self.drillables.items()}

    def item_summary(self):
        out = {}
        for n, it in self.items.items():
            st = it.state
            diag = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.diag.get(n, {}).items()}
            out[n] = dict(kind=it.kind, d_mm=it.d_mm, temp_c=round(st["temp_c"], 2),
                          max_temp_c=round(st["max_temp_c"], 2), wear=round(st["wear"], 5), broken=st["broken"],
                          broken_reason=st["broken_reason"], burnt=st["burnt"], chipped=st["chipped"],
                          blows=st["blows"], slots={dn: [v[1], v[2]] for dn, v in it.slots.items()},
                          insertion_m={k: round(v, 5) for k, v in st["insertion_m"].items()},
                          max_axial_n=round(st["max_axial_n"], 3), diag=diag)
        return out
