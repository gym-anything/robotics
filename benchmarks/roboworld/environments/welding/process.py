"""One weld in progress: grid + thermal solver + arc source + heat sources + pool + porosity, in the weld frame.

Used identically by the component probes (a kinematic phantom torch path, ASSISTED) and by the robot world (the
torch on the robot's flange). The caller supplies, each robot tick, the contact tip position and the wire axis in the
weld frame, the travel velocity and whether the welding circuit is closed; the process does the rest.

Weld frame (all presets A): x along the weld, z the base plate's outward normal, origin at the root of the joint's
start. Coupon builders below place the parts in it and grade the grid around the joint.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import params as PR
from . import thermal as TH
from .arc import CVSource, Settings, attach, air_fraction, porosity_fraction
from .pool import Joint, Pool

BODY_BASE, BODY_WEB = 1, 2
TIP_RECESS = 3.2e-3        # m: contact tip recessed 1/8 in inside the nozzle for spray (Tregaskiss CA3, Bernard)


@dataclass
class Coupon:
    kind: str                  # "tee" | "plate"
    length: float              # weld length (bracket length) m
    base: tuple                # (length x, width y, thickness z) m
    base_x0: float             # base plate start x (weld frame)
    base_y: tuple              # (y_min, y_max)
    web: tuple = None          # (length x, thickness y, height z) for tee
    fine: float = 2.5e-4       # fine cell (y, z) m
    fine_x: float = 5e-4       # cell along the weld m


def tee_coupon(length=.150, base=(.200, .150, .010), web=(.150, .008, .075)):
    return Coupon("tee", length, base, -(base[0] - length)/2, (-base[1]/2 - web[1]/2, base[1]/2 - web[1]/2), web)


def plate_coupon(length=.150, base=(.200, .100, .010)):
    return Coupon("plate", length, base, -(base[0] - length)/2, (-base[1]/2, base[1]/2))


def _band(a, b, cuts, cell):
    """Cell widths over [a, b] near `cell` with every position in `cuts` on a cell edge. One band of round(length/cell)
    equal cells when that already puts the cuts on edges (the default 0.25 / 0.5 mm grid, unchanged), else one band
    per interval between cuts."""
    n = max(1, int(round((b - a)/cell)))
    w = (b - a)/n
    if all(abs((c - a)/w - round((c - a)/w)) < 1e-6 for c in cuts):
        return TH.graded([(b - a, cell)])
    pts = [a] + sorted(c for c in cuts if a < c < b) + [b]
    return TH.graded([(q - p, cell) for p, q in zip(pts[:-1], pts[1:])])


def build_grid(c):
    """Graded grid around the joint: fine cells in the weld band, geometric growth outward. The band has a cell edge on
    each of the coupon's faces in it (the bracket's back and front faces, the plate top, the bracket's ends) at any cell
    size: a single band of round(length/cell) cells puts the faces inside cells unless the cell size divides the band's
    offsets, and at 0.35 mm the joint's faces are smeared over a straddling layer (E12)."""
    f, fx = c.fine, c.fine_x
    bx0, bx1 = c.base_x0, c.base_x0 + c.base[0]
    # x: fine from -3 mm to length+3 mm, coarser to the plate ends
    xa, xb = -3e-3, c.length + 3e-3
    dx = np.concatenate([TH.ramp(xa - bx0, fx, 2.5e-3)[::-1], _band(xa, xb, (0., c.length), fx),
                         TH.ramp(bx1 - xb, fx, 2.5e-3)])
    y0, y1 = c.base_y
    if c.kind == "tee":
        ya, yb = -c.web[1] - 2e-3, 16e-3
        za, zb = -c.base[2], 16e-3
        ztop = c.web[2]
        ycuts = (-c.web[1], 0.)
    else:
        ya, yb = -14e-3, 14e-3
        za, zb = -c.base[2], 12e-3
        ztop = zb + 4e-3
        ycuts = ()
    dy = np.concatenate([TH.ramp(ya - y0, f, 3e-3)[::-1], _band(ya, yb, ycuts, f), TH.ramp(y1 - yb, f, 3e-3)])
    zband = _band(za, zb, (0.,), f)
    dz = np.concatenate([zband, TH.ramp(ztop - zb, f, 3e-3)]) if ztop > zb else zband
    return TH.Grid(dx, dy, dz, origin=(bx0, y0, za)), (ya, yb, za, zb)

class WeldProcess:
    # the world may let the wire strike on grounded bodies outside the grid (world._process_tick): the source books such
    # a stray arc's metal and heat (arc.ArcState.stray_*) and this process leaves the coupon alone meanwhile
    books_stray_arcs = True

    def __init__(self, coupon, settings=None, device="cuda:0", draft_m_s=0., gravity=(0., 0., -9.81),
                 pool_every=5, root_gap=0.):
        self.c = coupon
        self.g, box = build_grid(coupon)
        self.s = s = TH.ThermalSolver(self.g, device=device)
        c = coupon
        bx0 = c.base_x0
        s.box((bx0, c.base_y[0], -c.base[2]), (bx0 + c.base[0], c.base_y[1], 0.), TH.STEEL, BODY_BASE)
        if c.kind == "tee":
            s.box((0., -c.web[1] - root_gap*0, root_gap), (c.length, -1e-9, root_gap + c.web[2]), TH.STEEL, BODY_WEB)
            self.joint = Joint.tee_near_side(14e-3, 14e-3, box)
        else:
            self.joint = Joint.plate(-12e-3, 12e-3, box)
        # base plate underside rests on the fixture table: sink faces (z- face of the lowest base cells)
        k0 = 0
        g = self.g
        I, J = np.meshgrid(np.arange(g.nx), np.arange(g.ny), indexing="ij")
        under = g.index(I.ravel(), J.ravel(), np.full(I.size, k0))
        under = under[s.mat[under] != TH.AIR]
        s.set_sink_face(under, 4)
        self.original_metal = s.mat != TH.AIR
        self._vols = self.g.volumes()
        self._v0 = float(np.sum(self._vols[self.original_metal]))      # metal at the start (the mass balance's base)
        s.upload()
        self.pool = Pool(s, self.joint, self.original_metal)
        self.src = CVSource(settings or Settings())
        self.draft = draft_m_s
        self.gravity = np.asarray(gravity, float)
        self.dt_th = s.stable_dt()
        self.pool_every = pool_every
        self.ticks = 0
        self.time = 0.
        self.root = None
        self.root_body = 0
        self.x_air = 0.
        self.travel_angle = 0.
        self.work_angle = None
        self.e_input_arc = 0.            # eta U I integrated (J): what the arc delivers to the work
        self.circuit_closed = True
        self.events = []
        h20 = s.h_of(TH.WELD, 20.)
        rho = PR.v(PR.STEEL_RHO)
        self.h_drop = (s.h_of(TH.WELD, PR.v(PR.DROPLET_T)) - h20)/rho          # J/kg carried by droplets
        self.h_cell = (self.pool.h_new_weld - h20)/rho                           # J/kg carried by new weld cells
        self.last_pool = {}
        self.ever_hot = False
        self.axis_last = None
        self.parc_I = 0.                 # arc current and root as felt by the pool surface (lagged by POOL_TAU)
        self.parc_root = None
        self.detached = {}               # body id -> saved cells of a part out of its fit-up pose (detach_part)

    # ---------------------------------------------------------------- one robot tick
    def tick(self, dt, tip, axis, velocity=(0., 0., 0.), circuit_closed=True, attached=None):
        """tip: contact-tip end (weld frame, m); axis: unit wire direction toward the work; velocity: torch velocity
        (weld frame, m/s); attached: a precomputed arc.attach(...) result for this tip and axis (optional)."""
        s, src = self.s, self.src
        st = src.st
        axis = np.asarray(axis, float)/np.linalg.norm(axis)
        tip = np.asarray(tip, float)
        self.axis_last = axis
        self.circuit_closed = circuit_closed
        point, dist, body = attach(self.g, s.mat, s.body, tip, axis) if attached is None else attached
        touching = math.isfinite(dist) and dist <= st.l_s + 1e-4
        # torch angles: travel angle (signed: + push) from the wire axis' component along the weld; work angle in
        # the plane normal to the weld, from the base plate (y-z plane, measured from +y toward +z)
        # travel angle: the torch's tilt along the weld line (x), positive (push) when the wire leans toward the
        # direction of travel along the weld (+x while not moving along it). Taken from the wire axis against the
        # weld line, not against the tip's velocity: a torch retracting along its own axis is not at a 90 deg
        # travel angle (that would read as gross porosity)
        v = np.asarray(velocity, float)
        sgn = -1. if v[0] < -1e-4 else 1.
        self.travel_angle = math.degrees(math.asin(np.clip(sgn*axis[0], -1, 1)))
        ayz = -axis[1:]
        if np.linalg.norm(ayz) > 1e-6:
            self.work_angle = math.degrees(math.atan2(ayz[1], ayz[0]))
        ntwd = dist - TIP_RECESS if math.isfinite(dist) else 1.
        self.x_air = air_fraction(ntwd, self.travel_angle, self.draft, st.gas_on)
        st.x_air = self.x_air
        # the air under the arc decides whether spray is possible (argon fraction), so it is known before the source
        # steps
        src.update(dt, dist, circuit_closed, touching, self.x_air)
        # an arc on a grounded body outside the grid (a stray arc: the table, a clamp) neither heats the coupon nor
        # feeds or presses its pool; the source books its metal and heat (arc.ArcState.stray_m3 / stray_j)
        on_coupon = bool(st.arc and st.stray_node is None and point is not None)
        self.root = point if on_coupon else None
        self.root_body = body if on_coupon else 0
        if on_coupon:
            Q, mdot = src.heat()
            q_drop_cyl = mdot*(self.h_drop - self.h_cell)            # the new cells carry mdot*h_cell (pool)
            q_surf = max(0., Q - mdot*self.h_drop)
            self.e_input_arc += Q*dt
            s.weights("gauss", point, axis, PR.v(PR.ARC_SIGMA))
            # droplet heat: spray droplets drive into the pool below the arc root (a cylinder, Kumar & Bhaduri form);
            # in short circuit each droplet merges into the pool under the wire when the wire touches it, so its
            # enthalpy enters a surface layer one wire diameter deep under the wire (spread over the whole arc
            # Gaussian, a 175 A tack never melts the joint). Globular drops,
            # detached by gravity at low speed, land on the pool surface under the wire as short-circuit drops do
            # [estimate]. The mode follows from the operating point (arc.CVSource.transfer_mode), not a command
            depth = PR.v(PR.DROPLET_CYL_DEPTH) if st.mode == "spray" else PR.v(PR.WIRE_D)
            s.weights("cyl", point - axis*2.5e-4, axis, 1.0e-3, depth)
        else:
            q_surf = q_drop_cyl = 0.
        if on_coupon:
            self.ever_hot = True
        if not self.ever_hot:
            # the coupon has never been heated: an all-ambient field with no source does not change (exact skip)
            self.ticks += 1
            self.time += dt
            return
        n = max(1, int(math.ceil(dt/self.dt_th - 1e-9)))
        h = dt/n
        for _ in range(n):
            if q_surf > 0:
                s.apply("gauss", q_surf)
            if q_drop_cyl > 0:
                s.apply("cyl", q_drop_cyl)
            s.step(h)
        lag = 1. - math.exp(-dt/PR.v(PR.POOL_TAU))
        if on_coupon:
            self.parc_root = point.copy() if self.parc_root is None else self.parc_root + (point - self.parc_root)*lag
        self.parc_I += ((st.current if on_coupon else 0.) - self.parc_I)*lag
        self.ticks += 1
        self.time += dt
        if self.ticks % self.pool_every == 0:
            self.update_pool()

    def update_pool(self):
        st = self.src.st
        self.pool.drop_axis = self.axis_last
        self.last_pool = self.pool.update(st.deposited_m3, self.parc_root, self.parc_I, self.gravity,
                                          root_y=None if self.root is None else float(self.root[1]),
                                          drop=self.root)
        if self.last_pool.get("new_j") or self.last_pool.get("removed_j"):
            # the cells the pool filled hold their enthalpy but still an air cell's 20 C: the next conduction step would
            # read them at 20 C, draw their neighbours' heat into them and overshoot on inversion (a parked pool's
            # hottest cell would rise with the arc off). Temperatures are made consistent with the enthalpies now
            self.s._invert()
        for i, sl in enumerate(self.pool.slices):
            if sl.liquid and sl.freeze_t is not None:
                # melted again (a fillet over a tack): its pores melt out and its gas history starts again; a slice
                # that kept its first freeze time would freeze no porosity over the tacks in a no-gas weld
                sl.freeze_t, sl.x_air_max, sl.porosity = None, 0., 0.
            # nitrogen and oxygen enter the pool through the arc (dissociated in the plasma, carried by the droplets);
            # a pool open to air after the arc is out takes up almost none (molecular N2 at Sieverts' equilibrium,
            # ~0.04 wt% in air, through a mass-transfer coefficient ~1e-4 m/s: < 0.001 wt% into a 3 mm pool in
            # 0.5 s, derived). So the air that counts is the air under a burning arc (otherwise lifting the torch off a
            # still-liquid crater would read as 15 % porosity)
            if sl.liquid and st.arc and self.root is not None:
                sl.x_air_max = max(sl.x_air_max, self.x_air)
            elif sl.frozen and sl.freeze_t is None:
                sl.freeze_t = self.time
                sl.porosity = porosity_fraction(sl.x_air_max)

    # ---------------------------------------------------------------- a part leaving its fit-up pose
    def _cell_volumes32(self, idx):
        """Cell volumes as the GPU's stored-energy reduction computes them (float32 product), float64."""
        g = self.g
        k = idx % g.nz
        j = (idx//g.nz) % g.ny
        i = idx//(g.ny*g.nz)
        return (g.dx[i].astype(np.float32)*g.dy[j].astype(np.float32)*g.dz[k].astype(np.float32)).astype(np.float64)

    def detach_part(self, body):
        """Take a part's cells out of the grid: the part has left the fit-up pose the grid describes (the grid is fixed
        in the base plate's frame), so its cells would otherwise stay standing where it was -- the arc would weld to
        them, the wire stop on them and the cameras draw them. They become air; their state is
        kept to put back if the part returns (its heat does not change meanwhile: the loose bracket is cold in every
        start state); their energy and volume are booked in ledger slots 6 and 7. Returns the number of cells."""
        s = self.s
        if body in self.detached:
            return 0
        idx = np.nonzero((s.body == body) & (s.mat != TH.AIR))[0].astype(np.int32)
        if len(idx) == 0:
            return 0
        saved = {n: s.gather(n, idx) for n in ("H", "T", "fl", "peakT", "peakf")}
        saved.update(idx=idx, mat=s.mat[idx].copy(), body=s.body[idx].copy())
        vol = self._cell_volumes32(idx)
        e = float(np.sum(saved["H"]*vol))
        v = float(np.sum(self.g.volumes()[idx]))
        n = len(idx)
        s.scatter("mat", idx, np.zeros(n, np.uint8))
        s.scatter("body", idx, np.zeros(n, np.uint8))
        s.scatter("H", idx, np.zeros(n, np.float32))
        s.scatter("T", idx, np.full(n, s.t_amb, np.float32))
        for f in ("fl", "peakf"):
            s.scatter(f, idx, np.zeros(n, np.float32))
        s.scatter("peakT", idx, np.full(n, s.t_amb, np.float32))
        # the pool reads the original-metal mask to find the joint's original surface: the detached part is no
        # longer there (without this the pool refills the part's cells and makes 472 mm3 of metal from 129 mm3 of
        # filler)
        self.original_metal[idx] = False
        s.set_ledger_extra(6, e)
        s.set_ledger_extra(7, v)
        self.detached[body] = dict(saved, energy_j=e, volume_m3=v)
        self.events.append(dict(t=self.time, event="part_detached", body=int(body), cells=n))
        return n

    def attach_part(self, body):
        """Put a detached part's cells back (it is at its fit-up pose again). A cell that has become metal meanwhile
        (weld metal laid where the part stood) keeps its metal; the part's cell there stays out of the grid."""
        d = self.detached.pop(body, None)
        if d is None:
            return 0
        s = self.s
        free = s.mat[d["idx"]] == TH.AIR
        idx = d["idx"][free]
        if len(idx):
            for n in ("T", "fl", "peakT", "peakf"):
                s.scatter(n, idx, d[n][free].astype(np.float32))
            h = d["H"][free]
            h32 = h.astype(np.float32)
            s.scatter("H", idx, h32)
            s.scatter("Hlo", idx, (h - h32.astype(np.float64)).astype(np.float32))
            s.scatter("mat", idx, d["mat"][free])
            s.scatter("body", idx, d["body"][free])
            self.original_metal[idx] = True
        vol = self._cell_volumes32(idx)
        e = float(np.sum(d["H"][free]*vol))
        v = float(np.sum(self.g.volumes()[idx]))
        s.set_ledger_extra(6, -e)
        s.set_ledger_extra(7, -v)
        self.events.append(dict(t=self.time, event="part_attached", body=int(body), cells=int(len(idx)),
                                cells_lost=int((~free).sum())))
        return len(idx)

    # ---------------------------------------------------------------- checkpoints
    _KEEP = ("ticks", "time", "root", "root_body", "x_air", "travel_angle", "work_angle", "e_input_arc",
             "circuit_closed", "parc_I", "parc_root", "ever_hot", "detached")

    def snapshot(self):
        import copy
        return dict(solver=self.s.snapshot(), pool=self.pool.snapshot(), st=copy.deepcopy(self.src.st),
                    settings=copy.deepcopy(self.src.s), events=list(self.events),
                    original_metal=self.original_metal.copy(),
                    **{k: copy.deepcopy(getattr(self, k)) for k in self._KEEP})

    def restore(self, snap):
        import copy
        self.s.restore(snap["solver"])
        self.pool.restore(snap["pool"])
        self.src.st = copy.deepcopy(snap["st"])
        self.src.s = copy.deepcopy(snap["settings"])
        self.events = list(snap["events"])
        for k in self._KEEP:
            setattr(self, k, copy.deepcopy(snap[k]) if k in snap else {} if k == "detached" else getattr(self, k))
        if "original_metal" in snap:
            self.original_metal[:] = snap["original_metal"]       # in place: the pool holds the same array

    # ---------------------------------------------------------------- state
    def arc_command(self, on=None, sensing=None, wfs_m_min=None, voltage_v=None, gas_on=None, inch=None):
        st, s = self.src.st, self.src.s
        if inch is not None:
            st.inch = {"forward": 1, "backward": -1, "stop": 0}[inch]
        if wfs_m_min is not None:
            s.wfs_m_min = float(wfs_m_min)
        if voltage_v is not None:
            s.voltage_v = float(voltage_v)
        if gas_on is not None:
            st.gas_on = bool(gas_on)
        if sensing is not None:
            st.sensing = bool(sensing)
        if on is not None:
            if on and not st.on_cmd and st.fault != "burnback":
                st.fault = ""                       # a burnback (wire welded to the contact tip) stays until reset
            st.on_cmd = bool(on)

    def ledger(self):
        led = self.s.ledger()
        led["arc_delivered_j"] = self.e_input_arc
        led["wire_fed_m3"] = self.src.st.fed_m3
        led["deposited_m3"] = self.src.st.deposited_m3
        led["spatter_m3"] = self.src.st.spatter_m3
        # stray arcs (on grounded bodies outside the grid): the wire metal melted onto them and the heat they received
        led["stray_arc_m3"] = self.src.st.stray_m3
        led["stray_arc_j"] = self.src.st.stray_j
        # the wire balance: every wire fed is deposited, spattered, melted onto a stray body or still on the stick-out
        # (zero to rounding)
        st = self.src.st
        led["wire_balance_m3"] = (st.fed_m3 - st.deposited_m3 - st.spatter_m3 - st.stray_m3
                                  - (st.l_s - st.l_s0)*self.src.aw)
        # mass balance: metal now minus metal at the start must equal the filler deposited, less the solidification
        # shrinkage the pool has drawn (the crater) -- a cell-level check of the pool's volume bookkeeping
        v_now = float(np.sum(self._vols[self.s.mat != TH.AIR]))
        # cells of a part that left its fit-up pose are out of the grid (detach_part): their volume still counts
        led["metal_volume_change_m3"] = v_now + led["detached_m3"] - self._v0
        led["pool_frozen_m3"] = self.pool.frozen_volume
        led["pool_liquid_expansion_m3"] = self.pool.liquid_expansion
        led["pool_arrested_m3"] = getattr(self.pool, "arrested_m3", 0.)
        led["pool_unplaced_m3"] = self.pool.excess          # filler an overfull pool has not yet spread to hold
        return led
