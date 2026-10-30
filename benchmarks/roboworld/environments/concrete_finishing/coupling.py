"""How the wet materials and the rigid world act on each other (M2, M7).

The paste is not a MuJoCo collider. Every contactor (a tool's sole, a fingertip pad) has a SUPPORT PLATE: a frictionless
static box of MuJoCo's world body, moved every tick by writing its geom_pos / geom_quat, parallel
to the region's base and centred under the sole, whose top face is where the paste under the sole bears. MuJoCo's
implicit contact between the sole's collision part and its plate carries the tool; nothing else touches the plate
(contype/conaffinity bits of its own). Where the plate stands is decided by the paste every tick:

  * the sole's height field over its footprint (paste.footprint) is taken relative to its lowest point, s_rel; a cell
    reaches the sole at plate level L if h >= L + s_rel;
  * the plate carries what the cells reaching the sole can bear, C(L) = sum p_cap A (paste.Region.capacity: Prandtl /
    Green through the bulk, or the squeeze of a weak surface layer);
  * against the plate's contact force F of the last tick (filtered over 5 ms): the level is the highest at which
    C(L) >= F (the bearing equilibrium of M7); when even every cell under the sole cannot carry F, the paste
    flows and the plate sinks at (F - C)/C_Y (M2's yielding law); the plate moves toward its level by at most
    0.2 mm a tick down and 0.02 mm a tick up, and never rises into a sole it does not touch.

After each step, with the sole where the solver put it:
  * paste standing above the sole is cut into the tool's roll (ploughed); the roll fills cells below the sole where
    the sole bears (within 0.3 mm of its lowest point); what exceeds the roll's prism (height W sin(alpha) + lift)
    runs to the blade's ends and is left there; when the tool reverses or lifts, the roll is set down ahead of its
    leading edge, except on tools that carry (knives, margin trowel), whose load stays on the blade as long as the
    paste's yield stress holds it there (params.DROP_STICK);
  * the surface under a sliding sole is worked: aggregate embedded and paste brought up, texture smoothed toward
    the tool's own, densified, sealed (steel pitched), bleed water worked in if a film stood there, joints
    compacted, sealant pressed against the joint walls;
  * the drag of the interface, the ploughing and the roll, and the tack of a sole lifting off, act on the tool as
    explicit forces, regularised so that their slope never exceeds the explicit limit for the tool's mass.

The tool and the plate meet in MuJoCo's own contact solver, so the support is as stiff as any rigid contact and a
sliding tool never loses it to an explicit spring's lag (M2 scheme B) or to a jumping piston (M7).
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation
from . import params as P
from .paste import footprint, green, Sole
from .transfer import Packet, sample, remove, deposit, all_regions, catch_deposit, ContactWindow

PLATE_HALF_T = 0.010
C_Y = 2000.0/0.032           # N s/m3: yielding resistance of the paste under an overloaded plate, per unit of bearing
                             # area (M2's 2000 N s/m for the 406 x 79 mm float, 0.032 m2): a needle of 32 mm2 sinks at
                             # its bearing load, not at a float's
DOWN_RATE = 2.0e-4           # m per tick: the plate sinks at most this much in one 0.5 ms tick
UP_RATE = 2.0e-5             # m per tick: and rises at most this much
BASE_PEN_MAX = 1.0e-3        # m: a sole this far into the rigid base is a soft contact's sinking; deeper, a lattice
                             # artifact of a part of the sole beside a raised base
CLIMB_FLOOR = 1.0e-3         # m/s: a sliding sole's plate rises at least this fast
CLIMB_FROM = 5.0e-3          # m/s: the climb limit applies to a sole advancing at least this fast (a tremble's advance is
                             # under 1 mm/s once low-passed: ADVANCE_TAU)
F_TAU = 0.005                # s: filter of the plate force
BEAR_BAND = 3.0e-4           # m: a cell is under the bearing part of the sole within this of its lowest point
SPILL_BAND = 0.020           # m: width over which a roll's overflow stands beyond the blade's ends
SMEAR_TAU = 5.0e3            # Pa: mortar softer than this squeezes onto the block faces when tooled (BIA TN 7B)
YIELD_LEAD = 2.0e-5          # m: how far a yielding plate may sink ahead of its sole's lowest point
WORK_SPEED = 0.001           # m/s: a sole slower than this does not work the surface (embed, bring up paste, densify)
HEAVE_SPEED = 0.005          # m/s: a sole sliding slower than this is pressing (what it displaces heaves around it)
LIFT_OFF_S = 0.05            # s without contact after which a roll is set down (non-carrying tools)
ADVANCE_TAU = 0.10           # s: a sole's advance is its in-plane velocity low-passed twice over this (a 1 mm, 8 Hz
                             # hand tremble, 50 mm/s at its peaks, passes as 1.9 mm/s; one stage passed 9.7 mm/s, over
                             # HEAVE_SPEED): paste cut by a sole that trembles in place without advancing is
                             # pressed out around it, not pushed ahead as a roll, and a tremble's reversal is no stroke's
                             # reversal (a trembling edger pressed into the pad fed its roll and stacked 70 mm piles)
SPILL_WIDEN = 0              # local 20 mm exit band only; excess stays in the finite roll, never teleports 320 mm
                             # bands 2, 4 ... 2^SPILL_WIDEN times SPILL_BAND, and what is left stays in the roll
V_REG = 0.005                # m/s: drag regularisation (the slope is also bounded by the explicit limit)
SIDE_ON = 0.2                # a tool's other sides take over from its working sole when the sole's normal is within this
                             # (cosine) of the region's plane: where the sole itself stops working a region (_gap)
FACE_HYST = 0.05             # cosine by which another side must face a region more squarely before the tool turns onto it
RIDE_FLAT = math.cos(math.radians(10.0))   # a sliding sole within 10 degrees of flat is lifted by paste it cannot plough
                                           # (_rides; estimate: pitched further, a blade shears the paste onto its face)


class Contactor:
    """One sole and its plate. `body` is the Newton body index; (p_ls, R_ls) the sole frame in the body frame."""

    def __init__(self, name, body, p_ls, R_ls, sole, mass, *, tool=None, geom_sole=(), plate_geom=None, bear=None):
        self.name, self.body, self.sole, self.tool = name, int(body), sole, tool
        self.bear = bear if bear is not None else sole     # the flat part the plate carries (the edger's blade)
        self.p_ls, self.R_ls = np.asarray(p_ls, float), np.asarray(R_ls, float)
        self.mass = float(mass)
        self.geom_sole = list(geom_sole)
        self.plate_geom = plate_geom
        self.level = {}            # region name -> plate level (normal coordinate of the sole's lowest point)
        self.F = 0.0               # filtered plate force
        self.F_raw = 0.0
        self.region = None         # the region the plate serves now
        self.roll = 0.0            # m3 of paste pushed ahead of / carried on the blade
        self.roll_mat = None
        self.packet = None
        self.source_region = None
        self.lead = None           # unit in-plane direction (region frame) the roll is pushed toward
        self.v_adv = np.zeros(2)   # the sole's advance: in-plane velocity (region frame) low-passed twice over ADVANCE_TAU
        self.v_lp = np.zeros(2)    # its first stage
        self.ref_level = None      # the paste's surface (region normal) under the sole where it came onto it this time:
                                   # the roll rides on it (median over the footprint: a ridge beside it does not count)
        self.no_contact_s = 1.0
        self.on_base = False       # the sole bears on the region's rigid base (bare board, slab, block faces, form tops)
        self.base_level = 0.0      # and the base's level where it does
        self.ploughed = 0.0        # m3 cut by the edge in the last tick
        self.rides = False         # the paste in its way was too strong to plough in the last tick (Coupler._rides)
        self.contact = dict(area=0.0, capacity=0.0, cells=0, pressure=0.0)
        self.tack_area = 0.0
        self.tack_sep = None
        self.work = 0.0
        self.peak_force = 0.0
        self.wrench = np.zeros(6)
        self.point = None
        self.side_of = None        # a tool's side contactor: its working sole's contactor (they share the plate)
        self.faces = ()            # and the tool's other sides (tools.side_faces), one of which it serves at a time
        self.face = None
        self.face_name = None

    def use_face(self, k):
        """A side contactor turns to face k of the tool (its frame, profile sole and name)."""
        f = self.faces[k]
        self.face, self.face_name = k, f["name"]
        self.p_ls, self.R_ls = np.asarray(f["p_ls"], float), np.asarray(f["R_ls"], float)
        self.sole = self.bear = f["sole"]

    # ---------------------------------------------------------------- pose
    def sole_pose(self, q):
        """World pose of the sole frame from the body's pose row (x, y, z, qx, qy, qz, qw)."""
        from scipy.spatial.transform import Rotation
        Rb = Rotation.from_quat(q[3:7]).as_matrix()
        return q[:3] + Rb @ self.p_ls, Rb @ self.R_ls

    def snapshot(self):
        return dict(level=dict(self.level), F=self.F, roll=self.roll, roll_mat=self.roll_mat, lead=None if self.lead is None else self.lead.copy(),
                    no_contact_s=self.no_contact_s, region=None if self.region is None else self.region.name,
                    work=self.work, peak=self.peak_force, v_adv=self.v_adv.copy(), v_lp=self.v_lp.copy(),
                    ref_level=self.ref_level, packet=None if self.packet is None else self.packet.snapshot(),
                    source_region=self.source_region, rides=self.rides, face=self.face)

    def restore(self, s, regions):
        self.level = dict(s["level"]); self.F = s["F"]; self.roll = s["roll"]; self.roll_mat = s["roll_mat"]
        self.lead = None if s["lead"] is None else np.array(s["lead"])
        self.no_contact_s = s["no_contact_s"]; self.region = regions.get(s["region"]) if s["region"] else None
        self.work = s["work"]; self.peak_force = s["peak"]
        self.v_adv = np.array(s.get("v_adv", (0.0, 0.0)), float)
        self.v_lp = np.array(s.get("v_lp", (0.0, 0.0)), float)
        self.ref_level = s.get("ref_level")
        self.packet = Packet.restore(s["packet"]) if s.get("packet") else None
        self.source_region = s.get("source_region")
        self.rides = bool(s.get("rides", False))
        if self.faces and s.get("face") is not None:
            self.use_face(int(s["face"]))


class Coupler:
    """All contactors of a world against all its regions."""

    def __init__(self, regions, contactors, dt):
        self.regions = regions
        self.contactors = contactors
        self.dt = dt
        self.log = dict(cut_m3=0.0, filled_m3=0.0, spilled_m3=0.0, set_down_m3=0.0, carried_m3=0.0)
        self.body_com = None                 # (n, 3) centre of mass in each body's frame (the world sets it)
        self.falling = []
        self.rag_packet = None

    # ---------------------------------------------------------------- which region a sole works
    def _near(self, c, p, R):
        """The region whose surface the sole is on or just above (within 15 mm of its paste or base), if any."""
        best = None
        for reg in all_regions(self.regions):
            gap = self._gap(c.sole, reg, p, R)
            if gap is not None and (best is None or gap < best[1]):
                best = (reg, gap)
        return None if best is None else self._root(best[0])

    def _gap(self, sole, reg, p, R):
        """|gap| between a sole at (p, R) and the region's surface under its centre, when the sole faces the region, its
        centre is no more than 60 mm into its paste or base and its lowest corner no more than 20 mm above it; else None.
        (Judged at the centre alone, a tool tilted far off flat had a corner deep in the paste before it was coupled: a
        float rolling off its side onto its sole was 36 mm into the pad when its sole took over.)"""
        pr = reg.to_region(p)
        lu, lv = reg.extent()
        m = max(sole.hx, sole.hy) + 0.01
        if pr[0] < -m or pr[0] > lu + m or pr[1] < -m or pr[1] > lv + m:
            return None
        rr = reg.R.T @ R
        if rr[2, 2] < 0.2:
            return None
        i = int(np.clip(pr[0]/reg.cell, 0, reg.nu - 1)); j = int(np.clip(pr[1]/reg.cell, 0, reg.nv - 1))
        if not reg.active[i, j]:
            return None
        gap = pr[2] - max(float(reg.b[i, j] + reg.tb[i, j] + reg.ts[i, j] + reg.bl[i, j]), float(reg.support_height((i, j))))
        drop = abs(float(rr[2, 0]))*sole.hx + abs(float(rr[2, 1]))*sole.hy     # the lowest corner, below the centre
        return abs(gap) if -0.060 < gap and gap - drop < 0.020 else None

    def _near_side(self, c, qb):
        """A tool's side contactor: the region one of the tool's other sides is on or just above, while its working sole
        is turned away from that region (SIDE_ON), and which side: the one facing the region most squarely (kept until
        another faces it FACE_HYST more squarely). (region, face index) or None."""
        Rb = Rotation.from_quat(qb[3:7]).as_matrix()
        best = None
        for reg in all_regions(self.regions):
            n_t = Rb.T @ reg.N                                       # the region's normal in the tool's frame
            if n_t[2] >= SIDE_ON:
                continue                                             # the working sole faces it: its own contactor's
            zts = [float(n_t @ f["R_ls"][:, 2]) for f in c.faces]
            k = int(np.argmax(zts))
            if c.face is not None and zts[c.face] > zts[k] - FACE_HYST:
                k = c.face
            f = c.faces[k]
            gap = self._gap(f["sole"], reg, qb[:3] + Rb @ f["p_ls"], Rb @ f["R_ls"])
            if gap is not None and (best is None or gap < best[1]):
                best = (reg, gap, k)
        return None if best is None else (self._root(best[0]), best[2])

    def _root(self, reg):
        if hasattr(reg, "catch_root"):
            if reg.catch_floor:
                # Stable identity as the contact crosses numerical tile boundaries.
                return next(r for r in reg.catch_root.catches.values() if r.catch_floor)
            return reg.catch_root
        return reg

    # ---------------------------------------------------------------- before the step: plates
    def place_plates(self, world, q):
        """Set every plate from the paste under its sole and the force it carried last tick."""
        out = {}
        state = getattr(world, "state", None)
        qd = state.body_qd.numpy().astype(float) if state is not None else None
        for c in self.contactors:
            if c.tool is not None and c.tool.kind == "knife":
                # Quasi-static blade compliance; body/grip remain rigid. Limit the
                # linear beam approximation to 3 mm deflection (small rotations).
                c.sole.flex = min(0.003, max(c.F, 0.)/(P.FLEX_K*(2*c.sole.hx/.152)))
            if c.side_of is None:
                p, R = c.sole_pose(q[c.body])
                reg = self._near(c, p, R)
            else:
                # the tool's other sides, while its working sole is not on a region (the two share one plate)
                found = None if c.side_of.region is not None else self._near_side(c, q[c.body])
                reg = None if found is None else found[0]
                if found is not None and found[1] != c.face:
                    if c.region is not None:
                        self._release(c)                             # turned onto another side: as if it left
                        c.region = None
                    c.use_face(found[1])
                p, R = c.sole_pose(q[c.body])
            if reg is None:
                if c.region is not None:
                    self._release(c)
                c.region = None
                out[c.name] = None
                continue
            if c.region is not reg:
                if c.region is not None:
                    self._release(c)
                c.region = reg
                c.v_adv, c.v_lp = np.zeros(2), np.zeros(2)
                c.level.pop(reg.name, None)
                c.ref_level = None
            reg = ContactWindow(reg, c, p, R)
            reg.name = c.region.name
            fp = footprint(reg, c.bear, p, R)
            if fp is None:
                out[c.name] = None
                continue
            sl, inside, s, x, y = fp
            cover = reg.last_cover
            h = reg.h_at(sl).astype(float)
            t = reg.t_at(sl).astype(float)
            ins = inside & np.isfinite(s)
            if not ins.any():
                out[c.name] = None
                continue
            s_low = float(np.nanmin(np.where(ins, s, np.nan)))
            s_rel = np.where(ins, s - s_low, np.inf)
            solid = reg.support_height(sl)
            fibrous = solid > reg.b[sl] + 1e-8
            reach = np.where(ins & ((t > 1e-5) | fibrous), np.maximum(h, solid) - s_rel, -np.inf)       # plate level at which each paste cell meets the sole
            top_level = float(reach.max()) if np.isfinite(reach.max()) else -np.inf
            if not np.isfinite(top_level):
                out[c.name] = None                                       # no paste under this sole: the base is MuJoCo's
                c.level.pop(reg.name, None)
                c.ref_level = None
                continue
            L = c.level.get(reg.name)
            self._cell = reg.cell
            width = self._width(c, ins, reach, top_level)
            n_bear = max(int(((reach > top_level - BEAR_BAND) & ins).sum()), 1)
            length = float(min(max(n_bear*reg.cell**2/width, width), 2*max(c.bear.hx, c.bear.hy)))
            # embedment: the paste around the footprint stands this far above the sole's lowest point
            around = (~inside) & (reg.t_at(sl) > 1e-5)
            depth = max(0.0, float(np.median(h[around])) - (L if L is not None else s_low)) if around.any() else 0.0
            cap_cell = reg.capacity(sl, width, length, depth)*reg.cell**2*np.where(inside, cover, 0.0)   # N per cell when it bears
            F = c.F
            order = np.argsort(reach, axis=None)[::-1]
            rv = reach.ravel()[order]
            cv = np.where(np.isfinite(rv), cap_cell.ravel()[order], 0.0)
            cum = np.cumsum(cv)
            total = float(cum[-1]) if len(cum) else 0.0
            # the sole's lowest point is sampled at cell centres to first order in its tilt: within a cell's run of the
            # sole's slope of the true edge, above or below it. Pinned 0.05 mm above a sample that fell 0.07 mm below a
            # pitched trowel's real edge, the plate never touched it, read no force, followed it down tick by tick, and
            # the trowel cut through the whole pad (A2: 3.4 L in 0.2 s, set down 2 mm from a form)
            zt = reg.R.T @ R[:, 2]
            slack = 5e-5 + reg.cell*math.sqrt(max(0.0, 1.0 - float(zt[2])**2))
            # how far ahead of its sampled lowest point a yielding plate may sink this tick: YIELD_LEAD, or, for a sole
            # sinking faster than that, as far as the sole's real lowest point (up to `slack` below the sample) travels in
            # the tick. Within a fixed YIELD_LEAD (4 cm/s) and DOWN_RATE (0.4 m/s), a faster sole ran into its plate every
            # tick and the solver's contact damping against a plate it sees at rest, not the paste, set the force: a
            # fingertip pushed into 5-minute concrete at 20 cm/s met 44 N where the yielding law gives 7, and a hand moved
            # into the pad at 0.5 m/s tripped the FR3's 100 N reflex. Kept out of a sinking sole's way
            # where the law lets the paste give, the plate meets it only while the force is below the law's: on average
            # the sole meets C + C_Y A v
            advance = self._sink_speed(c, q, qd, p, reg)*self.dt
            lead = max(YIELD_LEAD, advance + slack) if advance > YIELD_LEAD else YIELD_LEAD
            if F <= 1e-6 or L is None:
                target = top_level
            elif F <= total:
                k = int(np.searchsorted(cum, F))
                target = float(rv[min(k, len(rv) - 1)])
            else:
                base_level = float(rv[np.isfinite(rv)].min()) if np.isfinite(rv).any() else top_level
                area_b = max(float(ins.sum())*reg.cell**2, reg.cell**2)
                target = min(L if L is not None else top_level, base_level) - (F - total)/(C_Y*area_b)*self.dt
                # yielding paste gives way under the sole, no faster: the plate never drops away below the sole (it
                # did, and a light needle lost and regained its support every few ticks, carrying the right load only on
                # average)
                target = max(target, s_low - lead)
            down = max(DOWN_RATE, lead)
            if L is None:
                L = min(target, s_low + max(1e-4, slack))               # first touch: never above where the sole is
                on = ins & (t > 1e-5)
                c.ref_level = float(np.median(h[on])) if on.any() else None
            else:
                if c.F_raw <= 1e-6:
                    L = min(target, max(L, s_low - 2e-4) + UP_RATE) if target > L else max(target, L - down)
                    L = min(L, s_low + slack)
                else:
                    L = L + float(np.clip(target - L, -down, self._climb(c, zt)))
            # never below the rigid base under the bearing part of the sole
            base_top = float(np.nanmax(np.where(ins, reg.support_height(sl) - s_rel, np.nan)))
            L = max(L, base_top)
            c.level[reg.name] = L
            c.contact = dict(area=float(ins.sum())*reg.cell**2, capacity=total, cells=int(ins.sum()), width=width, length=length,
                             depth=depth)
            # the plate: a box parallel to the region's base, top face at L, centred under the sole's lowest point
            pr = reg.to_region(p)
            collision_offset = 0.
            if c.tool is not None and c.tool.kind == "knife":
                # The solver collider is rigid; lower its support by the elastic
                # sole's displacement. Otherwise the bent sole would float above
                # a plate still supporting the undeformed edge (double support).
                # Use the continuous rigid blade edge, not its nearest cell
                # centre: at 20 degrees the old sampled edge left up to 0.36 mm
                # of extra coating solely from grid phase on the 1 mm board.
                rr = reg.R.T @ R
                rigid_low = pr[2] - abs(rr[2, 0])*c.bear.hx - abs(rr[2, 1])*c.bear.hy
                collision_offset = float(rigid_low)-s_low
            centre = reg.o + reg.R @ np.array([pr[0], pr[1], L + collision_offset - PLATE_HALF_T])
            xr = reg.R.T @ R[:, 0]                                        # the sole's x axis in the region frame
            yaw = math.atan2(xr[1], xr[0]) if abs(xr[0]) + abs(xr[1]) > 1e-6 else 0.0
            cz, sz = math.cos(yaw), math.sin(yaw)
            Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            out[c.name] = (centre, reg.R @ Rz)
        return out

    def _sink_speed(self, c, q, qd, p, reg):
        """How fast (m/s) the sole moves into the region (along -N) at its origin; 0 without velocities or when it rises."""
        if qd is None or self.body_com is None:
            return 0.0
        Rb = Rotation.from_quat(q[c.body, 3:7]).as_matrix()
        com = q[c.body, :3] + Rb @ self.body_com[c.body]
        v = qd[c.body, :3] + np.cross(qd[c.body, 3:6], p - com)
        return max(0.0, -float(reg.N @ v))

    def _climb(self, c, zt):
        """How far (m) a loaded plate may rise in one tick: as fast as the sole climbs the paste coming under it, its
        advance times the rise of its underside along the advance (a pitched blade rides up its own ramp; a flat one
        cannot get paste above its level past its square leading edge and cuts it), and no faster than UP_RATE. Rising
        at UP_RATE (40 mm/s) whatever its pitch, a sliding blade was lifted onto every high in its path and carried by
        it: a darby rode on the struck surface's highest few per cent, held at grade it still rode 0.6 mm high, and a
        margin trowel rode over the jointer's tags instead of cutting them (A2)."""
        v = c.v_adv
        speed = float(np.hypot(v[0], v[1]))
        if speed < CLIMB_FROM:
            # not sliding: a sole pressed in (a needle pushed into stiff paste) is met by the paste where it stands
            # (limited to the climb, the plate lagged a fast needle into 91 kPa paste and it read 0.7 tau_b, not 9)
            return UP_RATE
        if c.rides:
            return UP_RATE                        # paste too strong to plough lifts the sole onto it (_rides)
        climb = CLIMB_FLOOR
        if float(zt[2]) > 0.2:
            slope = -(float(zt[0])*v[0] + float(zt[1])*v[1])/(speed*float(zt[2]))   # rise of the underside per metre advanced
            climb += speed*max(slope, 0.0)
        return min(UP_RATE, climb*self.dt)

    def _width(self, c, ins, reach, top):
        """Narrow width B of the bearing patch (Prandtl's thin-layer squeeze reads it): the sole's width, or less when
        only a strip of it bears (a pitched blade bears along its trailing edge): area / length."""
        n = int(((reach > top - BEAR_BAND) & ins).sum())
        area = max(n, 1)*self._cell**2
        long_, short = 2*max(c.sole.hx, c.sole.hy), 2*min(c.sole.hx, c.sole.hy)
        return float(max(min(short, area/long_), self._cell))

    def _release(self, c):
        """The sole has left its region: a roll that does not stay on the blade is set down where it was."""
        reg = c.region
        c.ref_level = None
        if reg is not None and c.roll > 0 and not (c.tool is not None and c.tool.carry):
            self._set_down(c, reg)
        c.level.pop(reg.name if reg else None, None)

    # ---------------------------------------------------------------- after the step: paste
    def after_step(self, world, q, qd, forces, dt):
        """forces: {contactor name: plate normal force (N)} from the solver's contacts this tick."""
        self._last_q = q
        self._advance_falling(dt)
        for c in self.contactors:
            f = float(forces.get(c.name, 0.0))
            c.F_raw = f
            c.F += (f - c.F)*min(1.0, dt/F_TAU)
            c.peak_force = max(c.peak_force, f)
            c.wrench[:] = 0.0
            self._carry_update(c, q, qd, dt)
            reg = c.region
            if reg is None:
                continue
            p, R = c.sole_pose(q[c.body])
            reg = ContactWindow(reg, c, p, R)
            reg.name = c.region.name
            fp = footprint(reg, c.sole, p, R)
            if fp is None:
                continue
            sl, inside, s, x, y = fp
            ins = inside & np.isfinite(s)
            if not ins.any():
                continue
            # The paste meets the sole where the plate is: the soft plate contact lets the sole sit a fraction of a
            # millimetre into the plate, and paste the plate says is bearing must not read as standing above the sole
            # (it would be cut, the bearing level would drop and the sole sink after it, tick by tick). The sole's
            # surface is taken up by that penetration.
            L = c.level.get(reg.name)
            if L is not None:
                if c.bear is c.sole:
                    s_low = float(np.nanmin(np.where(ins, s, np.nan)))
                else:
                    fb = footprint(reg, c.bear, p, R)
                    s_low = float(np.nanmin(np.where(fb[1] & np.isfinite(fb[2]), fb[2], np.nan))) if fb is not None else L
                pen = L - s_low
                if pen > 0.0:
                    s = s + pen                     # the sole is pressed into its plate: the paste meets it at the plate
                c.penetration = pen
            # likewise into the region's rigid base: the solver's soft contact lets a sole sit a fraction of a millimetre
            # into a bare board or the slab, and the paste meets it at the base. Counted from the sole as it sat, a knife
            # pressed on bare board laid a 0.1-0.6 mm film behind its edge wherever it went (A2)
            support = reg.support_height(sl)
            over_b = np.where(ins, support - s, np.nan)
            kb = int(np.nanargmax(over_b))
            pen_b = float(over_b.flat[kb])
            c.base_level = float(support.flat[kb])          # the base where the sole meets it (board, slab, form top)
            if pen_b > BASE_PEN_MAX:
                # not a soft contact's sinking: a part of the sole beside a raised base over a lattice cell (an edger's
                # 12.7 mm lip 0.5 mm off the form, under a form-top cell on one side of the pad: shifted by it, the
                # whole sole was lifted 12.7 mm and edged nothing along that form, A2)
                pen_b = -1.0
            elif pen_b > 0.0:
                s = s + pen_b
            else:
                # the sole's lowest point is sampled at cell centres, up to a cell's run of its tilt above the real edge
                # (0.57 mm for a knife at 35 degrees on the 1 mm board): where the solver has the sole resting on the
                # rigid base (a contact with a static body other than its plate) and the samples sit within that of the
                # base, the sole is on the base. Read from the samples alone, a knife pressed on bare board was "off"
                # it most ticks, laid films up to the sample's height and left the anchor holes 0.49 mm proud (A2)
                zt = reg.R.T @ R[:, 2]
                slack = 5e-5 + reg.cell*math.sqrt(max(0.0, 1.0 - float(zt[2])**2))
                rests = float(getattr(world, "sole_force", {}).get(c.name, 0.0)) - f > 0.05
                if rests and pen_b > -slack:
                    s = s + pen_b
                    pen_b = 0.0
            c.on_base = pen_b > -5e-5
            # sole velocity at its centre, in the region frame (body_qd: velocity of the centre of mass, world frame)
            Rb = Rotation.from_quat(q[c.body, 3:7]).as_matrix()
            com = q[c.body, :3] + Rb @ self.body_com[c.body]
            v_w = qd[c.body, :3] + np.cross(qd[c.body, 3:6], p - com)
            v_r = reg.R.T @ v_w
            v_par = v_r[:2]
            speed = float(np.hypot(*v_par))
            k_adv = min(1.0, dt/ADVANCE_TAU)
            c.v_lp = c.v_lp + (v_par - c.v_lp)*k_adv
            c.v_adv = c.v_adv + (c.v_lp - c.v_adv)*k_adv
            touching = f > 0.05 or self._touching(reg, sl, ins, s)
            if touching:
                c.no_contact_s = 0.0
            else:
                c.no_contact_s += dt
            self._cut_fill(c, reg, sl, ins, s, x, y, v_par, speed, f, dt, touching, p, R)
            if not touching and c.no_contact_s > LIFT_OFF_S and c.roll > 0:
                if c.tool is not None and c.tool.carry:
                    pass  # retention is evaluated for every pose, including away from a region
                else:
                    self._set_down(c, reg)
            self._forces(c, reg, sl, ins, s, p, v_w, v_r, speed, f, dt)
            if hasattr(reg, "contact_window"):
                reg.contact_window.flush()

    def _touching(self, reg, sl, ins, s):
        """The sole meets paste, or bears on the region's rigid base (bare board, the slab beside a patch, block faces,
        form tops: MuJoCo's colliders, so no plate force). Counted only on paste, a loaded knife pressed on the bare
        board beside the anchor holes never touched anything and filled none of them (A2)."""
        h = reg.h_at(sl)
        if bool(((h - s) > -5e-5)[ins & (reg.t_at(sl) > 1e-5)].any()):
            return True
        return bool(((reg.support_height(sl) - s) > -5e-5)[ins].any())

    def _cut_fill(self, c, reg, sl, ins, s, x, y, v_par, speed, f, dt, touching, p, R):
        if c.roll > 1e-15 and c.roll_mat not in (None, reg.material.name):
            return  # this reduced model does not mix distinct species
        cell2 = reg.cell**2
        tb, ts = reg.tb[sl], reg.ts[sl]
        h = reg.b[sl] + tb + ts + reg.bl[sl]
        over = np.where(ins, h - s, -1.0)
        t = tb + ts
        # A blister is an envelope, not solid paste. Collapse its void before
        # sampling transferable material; never turn that lift into a load.
        collapse = np.where(ins, np.minimum(np.maximum(over, 0.), reg.bl[sl]), 0.)
        reg.bl[sl] -= collapse
        cut = np.where(ins, np.clip(np.minimum(over-collapse, t), 0., None), 0.)
        if c.roll > 1e-15 and c.roll_mat not in (None, reg.material.name):
            cut[:] = 0.  # species mixing is unsupported, never transmute a load
        c.rides = bool(cut.any()) and self._rides(c, reg, sl, cut, v_par, speed, f, R)
        if c.rides:
            cut[:] = 0.
        vol_cut = float(cut.sum())*cell2
        if vol_cut > 0:
            packet = remove(reg, sl, cut)
            self._ensure_packet(c, reg)
            if c.packet is None or c.roll <= 1e-18:
                c.packet = packet
            else:
                c.packet.volume = c.roll
                c.packet.mix(packet)
            c.roll += vol_cut
            c.roll_mat = reg.material.name
            c.source_region = getattr(reg, "catch_root", reg).name
            self.log["cut_m3"] += vol_cut
            # Consolidation changes work history, not an untracked loss of paste.
            if float(np.hypot(*c.v_adv)) < HEAVE_SPEED:
                before = c.roll
                laid = self._heave(c, reg, p, R, vol_cut)
                c.roll = before-laid
        c.ploughed = vol_cut
        if vol_cut > 0:
            reg.dirty = True
        # leading direction of the roll: the sole's advance while it slides
        adv = float(np.hypot(*c.v_adv))
        if adv > 2e-3:
            d = c.v_adv/adv
            if c.lead is not None and float(d @ c.lead) < -0.3 and c.roll > 0 and not (c.tool is not None and c.tool.carry):
                self._set_down(c, reg)                                   # reversed: the old roll stays where it was
            c.lead = d
        # fill what lies below the bearing part of the sole from the roll
        if c.roll > 1e-15 and touching and c.roll_mat == reg.material.name:
            s_low = float(np.nanmin(np.where(ins, s, np.nan)))
            bearing = ins & (s - s_low < BEAR_BAND + 1e-4) & (reg.paste_mask[sl] | (t > 1e-5))
            if reg.material.name == "sealant":
                # sealant pressed by a tooling nose is confined in the joint and fills every gap under the nose up to
                # both walls (the Sikaflex-1A and Tremco Dymonic 100 data sheets: tooling makes full contact with the
                # joint walls); filled only under the nose's lowest band, a stretch sagged off one wall was never
                # refilled there and never touched that wall (A2)
                bearing = ins & reg.paste_mask[sl]
            h = reg.b[sl] + tb + ts + reg.bl[sl]
            # a sole bearing on the rigid base fills only what lies below where it bears (a hole in bare board, the patch's
            # recess below the slab): up to its whole bearing band it laid a film over the bare board (A2). A sealant
            # nose fills to its own shape, up to both walls
            # a knife's load rides on its leading face and is struck off at its edge: it fills only what lies below the
            # edge's line. Filled up to the pitched blade behind the edge (the whole bearing band, 0.4 mm), every knife
            # stroke laid a 0.36-0.38 mm wedge of compound behind it and left filled holes 0.375 mm proud (A2)
            knife = c.tool is not None and c.tool.kind == "knife"
            # bearing on the base, fill only below where it bears: the base's own level there (a knife on the board, a
            # float on the slab, an edger's flat on the form top; capped at the sole's lowest point, the edger's lip,
            # it filled nothing of the arc above it: A2)
            if getattr(c, "on_base", False) and reg.material.name != "sealant":
                top = np.minimum(s, c.base_level)
            elif knife and reg.material.name != "sealant":
                top = np.minimum(s, s_low)
            else:
                top = s
            room = np.where(bearing, np.clip(top - h, 0.0, None), 0.0)
            want = float(room.sum())*cell2
            if want > 0:
                frac = min(1.0, c.roll/want)
                add = (room*frac)
                self._ensure_packet(c, reg)
                deposit(reg, sl, add, c.packet)
                reg.dirty = True
                vol = float(add.sum())*cell2
                c.roll -= vol
                self.log["filled_m3"] += vol
        # the roll's prism: what exceeds it runs to the ends of the blade's long side (a blade that carries holds up to
        # its carry_cap: a loaded sealant tool overflows over the joint's edges until it is wiped on the rag)
        carry = c.tool is not None and c.tool.carry
        if c.roll > 0 and (not carry or c.tool.carry_cap is not None):
            if carry:
                cap = c.tool.carry_cap
            else:
                if c.bear is not c.sole:
                    # the roll stands against the bearing flat (an edger's blade): its rise is the flat's tilt over its
                    # length and width, not the cutting sole's depth (an edger's 12.7 mm lip, a slot cutter, made its
                    # roll 16 mm high and its overflow ridge as tall at 140 min)
                    Rr = reg.R.T @ R
                    rise = abs(float(Rr[2, 0]))*2*c.bear.hx + abs(float(Rr[2, 1]))*2*c.bear.hy
                else:
                    s_in = np.where(ins, s, np.nan)
                    rise = float(np.nanmax(s_in) - np.nanmin(s_in)) if ins.any() else 0.0
                lift = c.sole.lift
                # the face that pushes the roll: the long one, or the end when the blade moves along its length (an
                # edger run along a form pushes its roll with its 76 mm end, not its 152 mm side)
                end, _ = self._end_leads(c, reg, R)
                Lface = 2*(min if end else max)(c.sole.hx, c.sole.hy)
                cap = 0.5*(rise + lift)**2*Lface
            if c.roll > cap and c.lead is not None:
                if self._squeezes_out(reg, sl, ins):
                    if self._squeeze(c, reg, p, R, c.roll - cap):
                        c.roll = cap
                else:
                    top = None
                    if not carry:
                        # the roll rides on the paste it is pushed through: its top is its height above the sole's
                        # lowest point, or above the paste's surface where the sole came onto it this time, if that
                        # stands higher (a sunk sole). A level taken from the sole alone put a buried blade's roll
                        # below the surface around it and sent its whole overflow to the widest band; one taken from
                        # the paste around the sole, or from the paste being cut, rose with the overflow's own ridge
                        # whenever the tool went over it (a 240 mm tower at 200 N; a 123 mm ridge where an
                        # edger, turned 17 degrees in the hand, swept its own ridge)
                        s_low = float(np.nanmin(np.where(ins, s, np.nan)))
                        base = max(s_low, c.ref_level) if c.ref_level is not None else s_low
                        top = base + rise + lift
                    c.roll -= self._spill(c, reg, p, R, c.roll - cap, top=top)
        # work on the surface under the sliding sole (a sole pressed in place, or one trembling in a hand, does not work it):
        # gated on the sole's advance, low-passed over ADVANCE_TAU, not its speed. Gated on the speed, the pocket
        # penetrometer's needle wobbling over 1 mm/s in 91 kPa paste "worked" it at 780 kPa per pass of its 6 mm tip, grew
        # the weak surface layer to 3 mm round it and read 0.7 tau_b instead of 9
        if touching and float(np.hypot(*c.v_adv)) > WORK_SPEED:
            self._work(c, reg, sl, ins, s, speed, f, dt, v_par)

    def _rides(self, c, reg, sl, cut, v_par, speed, f, R):
        """Whether the paste standing in a sliding sole's way is too strong for the sole to plough it. Shearing it off at
        the sole's level takes the plough's force, PLOUGH tau A over the frontal area A it presents (the drag's law, with
        the yield stress of what would be cut as it stiffens with age: its bulk's. ASTM C403's set is the penetration
        resistance of the paste itself, so the weak surface layer's working-time ratio R_SURF is not taken as the strength
        of set paste: with it, 7 h ridges (ts 3.8 mm, 1.1 MPa) were still shaved under a 6 N hold), and a square edge
        driven into it is thrust up about as hard; the sole holds its level against that only with what presses it on the
        paste (the plate's load, or at least its weight), and a fingertip's rubber pad presses no harder than its rubber
        bears (PAD_RUBBER_E). Where it cannot, the paste is not cut but lifts the sole: it rides up
        (_climb lets its plate rise) and comes over it. Cut by geometry alone, concrete 7 h old (7 MPa) was ploughed by a
        fingertip under 150 N exactly as 5-minute concrete is, hardened ridges cut and re-laid at the flatness criteria's
        millimetre scale. The frontal area is the cut's cross-section across the motion: its volume over
        the rows of cells it spans along the motion (cells enter a footprint a whole row at a time, so the volume cut in
        one tick over the tick's advance would count a row's every few ticks as its frontal area)."""
        if float(np.hypot(*c.v_adv)) < CLIMB_FROM or speed <= 1e-6:
            return False                          # not sliding: a pressed sole cuts only what its plate's bearing let go
        total = float(cut.sum())
        tau = float((cut*reg.tau_b(sl)).sum())/total
        if c.tool is None and c.side_of is None and P.PLOUGH*tau > P.PAD_RUBBER_E:
            return True                           # a fingertip's rubber pad gives before set paste does (PAD_RUBBER_E)
        if float((reg.R.T @ R)[2, 2]) < RIDE_FLAT:
            # a blade pitched edge-down shears the paste onto its face (a large rake angle: the cut pulls it into the
            # work, not out); the thrust that lifts a sole belongs to a sole lying flat on the paste (applied to the
            # knives at 25-35 degrees, it lifted them off the compound and left the wallboard patch 0.4 mm proud: A2)
            return False
        ii, jj = np.nonzero(cut > 0)
        along = (reg.u[sl[0]][ii]*float(v_par[0]) + reg.v[sl[1]][jj]*float(v_par[1]))/speed
        rows = float(along.max() - along.min())/reg.cell + 1.0
        need = P.PLOUGH*tau*total*reg.cell/rows
        hold = max(f, c.F, c.mass*9.81*max(float(reg.N[2]), 0.0))
        return need > hold

    def _lay(self, reg, rng, wgt, vol, rough=True, top=None, packet=None):
        """Lay `vol` with the weights `wgt` over the lattice cells rng = ((i0, i1), (j0, j1)), which may run past the
        region's edges: every cell gets its share of the whole footprint, so a footprint the edge cuts lays no thicker
        than a whole one. The share beyond the edge has left the modelled surface: it is counted (log off_region_m3,
        reg.off_m3), never piled onto the cells inside (it was, and a roll set down by a float most of whose length was
        past the patch's edge stood as fins up to the whole roll's volume on a few cells: R6). Returns the
        volume laid on the region, or None when the weights are all zero. With `top` (a level along the region's
        normal) no cell inside is raised above it; what did not fit is self.excess (m3), neither laid nor counted."""
        (i0, i1), (j0, j1) = rng
        if hasattr(reg, "catch_root") and not reg.catch_floor and not hasattr(reg, "contact_window"):
            oi, oj = reg.catch_offset
            return self._lay(reg.catch_root, ((i0+oi, i1+oi), (j0+oj, j1+oj)), wgt, vol,
                             rough=rough, top=top, packet=packet)
        if packet is None:
            packet = sample(reg, np.s_[:, :], reg.t)
            if packet.volume <= 0:
                packet = Packet(vol, reg.material, {"age": float(np.median(reg.age()))})
        c2 = reg.cell**2
        tot = float(wgt.sum())*c2
        if tot <= 0:
            return None
        a0, a1, b0, b1 = max(i0, 0), min(i1, reg.nu), max(j0, 0), min(j1, reg.nv)
        laid = 0.0
        self.excess = 0.0
        if a0 < a1 and b0 < b1:
            sl = (slice(a0, a1), slice(b0, b1))
            add = wgt[a0 - i0:a1 - i0, b0 - j0:b1 - j0]*(vol/tot)
            if top is not None:
                fit = np.minimum(add, np.clip(top - reg.h_at(sl), 0.0, None))
                self.excess = float((add - fit).sum())*c2
                add = fit
            if add.any():
                deposit(reg, sl, add, packet)
                if rough:
                    _rough(reg, sl, add)
                reg.dirty = True
                laid = float(add.sum())*c2
        off = vol - laid - self.excess
        if off > 1e-15:
            I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
            outside = (I < 0) | (I >= reg.nu) | (J < 0) | (J >= reg.nv)
            amounts = wgt[outside]*(vol/tot)
            if hasattr(reg, "contact_window") and not reg.catch_floor:
                oi,oj=reg.catch_offset
                extra=np.where(outside,wgt,0.)
                excess0=self.excess
                self._lay(reg.catch_root,((i0+oi,i1+oi),(j0+oj,j1+oj)),extra,off,
                          rough=rough,top=top,packet=packet)
                self.excess += excess0
                return laid
            root = getattr(reg, "catch_root", reg)
            oi, oj = getattr(reg, "catch_offset", (0, 0))
            catch_deposit(root, (I[outside]+oi, J[outside]+oj), amounts, packet,
                          floor=getattr(reg, "catch_floor", False))
            self.log["off_region_m3"] = self.log.get("off_region_m3", 0.0) + off
        return laid

    def _end_leads(self, c, reg, R):
        """(True, s) when the blade advances along its length, its end leading (s: +1 toward the long axis' +end), on an
        open surface or in a mortar joint; (False, 1) when its long face leads, or in the sealant joint (whose overflow
        squeezes out over the joint's edges). A jointer's overflow runs out beside its leading end, over the arrises
        onto the block faces (the tags a mason brushes off): laid at both ends of the rod, the trailing end's share
        refilled the joint just tooled, from 2.2 to 0.5 mm deep (A2)."""
        if c.lead is None or reg.name == "joint":
            return False, 1.0
        Rr = reg.R.T @ R
        ax = Rr[:2, 0] if c.sole.hx >= c.sole.hy else Rr[:2, 1]
        ax = ax/max(float(np.linalg.norm(ax)), 1e-9)
        along = float(c.lead @ ax)
        across = float(c.lead @ np.array([-ax[1], ax[0]]))
        return (abs(along) > abs(across)), (1.0 if along >= 0 else -1.0)

    def _heave(self, c, reg, p, R, vol):
        """Bearing failure of a pressed sole: the displaced volume rises in a band around the footprint as wide as the
        sole's narrow side (Prandtl's passive wedges reach about that far beyond a strip footing; between 6 and 60 mm),
        deepest at the sole's edge and fading out linearly."""
        sole = c.sole
        Bh = float(np.clip(2*min(sole.hx, sole.hy), 0.006, 0.060))
        fp = footprint(reg, Sole("blade", sole.hx + Bh, sole.hy + Bh), p, R, clip=False)
        if fp is None:
            return 0.
        rng, inside, _, x, y = fp
        (i0, i1), (j0, j1) = rng
        # the wedges rise only in paste: not through a form, a slab or a wall beside the sole (the ring was laid over the
        # forms' tops and past them: an edger pressed at a form heaved 0.7 ml past the pad's region)
        soft = np.zeros(inside.shape, bool)
        a0, a1, b0, b1 = max(i0, 0), min(i1, reg.nu), max(j0, 0), min(j1, reg.nv)
        if a0 < a1 and b0 < b1:
            sl = (slice(a0, a1), slice(b0, b1))
            soft[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = reg.paste_mask[sl] | (reg.t_at(sl) > 1e-5)
        dist = np.maximum(np.abs(x) - sole.hx, np.abs(y) - sole.hy)
        ring = inside & soft & (dist > 0) & (_cover(sole, x, y, reg.cell) <= 0.25)      # never onto a cell the sole cuts
        wgt = np.where(ring, 1.0 - np.clip(dist/Bh, 0.0, 1.0), 0.0)
        if self._lay(reg, rng, wgt, vol, packet=c.packet) is None:
            return 0.
        self.log["heaved_m3"] = self.log.get("heaved_m3", 0.0) + vol
        return vol

    def _spill(self, c, reg, p, R, vol, top=None):
        """Put `vol` of the roll where it runs off the roll's ends: for a blade moving across its length, at the two ends
        of its long axis (lap ridges across the blade's width, deepest at its end and fading out over SPILL_BAND); for one
        moving along its length on an open surface, beside its leading corners (the roll stands across its end). With
        `top`, the roll's top level, the bead stands no higher than the roll it runs off: what does not fit spreads over
        wider bands (SPILL_WIDEN doublings) and what is still left stays in the roll. Returns the volume that left the
        roll (laid, or past the region's edge)."""
        sole = c.sole
        long_x = sole.hx >= sole.hy
        half = sole.hx if long_x else sole.hy
        end, sgn = self._end_leads(c, reg, R)
        left, placed = vol, 0.0
        for k in range(SPILL_WIDEN + 1 if top is not None else 1):
            band = SPILL_BAND*2**k
            if end:
                # the end leads: the roll stands across the end and runs off its two ends, beside the blade's leading
                # corners (an edger's overflow, left beside its track; laid at the long axis' ends, the front band was
                # re-cut every tick and the back one laid on the worked track: 2 L churned in one 0.5 m pass)
                big = Sole("blade", sole.hx + band, sole.hy + band)
                fp = footprint(reg, big, p, R, clip=False)
                if fp is None:
                    break
                rng, inside, _, x, y = fp
                ax_c, side_c = (x, y) if long_x else (y, x)
                side_half = sole.hy if long_x else sole.hx
                side = np.abs(side_c)
                ahead = np.abs(sgn*ax_c - half) <= band
                # an edger's lip, down the form's face along one long side, closes the roll's end on that side: the
                # overflow runs out on the open side only (laid on both, the lip side's share went over the form and
                # past the pad's region)
                open_side = (side_c*sole.lip_side < 0) if sole.lip > 0 else np.ones(side_c.shape, bool)
                wgt = np.where(inside & ahead & open_side & (side > side_half) & (_cover(sole, x, y, reg.cell) <= 0.25),
                               1.0 - np.clip((side - side_half)/band, 0.0, 1.0), 0.0)
            else:
                big = Sole("blade", sole.hx + (band if long_x else 0.0), sole.hy + (0.0 if long_x else band))
                fp = footprint(reg, big, p, R, clip=False)
                if fp is None:
                    break
                rng, inside, _, x, y = fp
                coord = np.abs(x if long_x else y)
                wgt = np.where(inside & (coord > half) & (_cover(sole, x, y, reg.cell) <= 0.25),
                               1.0 - np.clip((coord - half)/band, 0.0, 1.0), 0.0)
            if self._lay(reg, rng, wgt, left, top=top, packet=c.packet) is None:
                break
            placed += left - self.excess
            left = self.excess
            if left <= 1e-15:
                break
        self.log["spilled_m3"] += placed
        return placed

    def _squeezes_out(self, reg, sl, ins):
        """Where a tool's overflow goes: in a joint (a region with finished faces beside the paste) sealant, and mortar
        softer than SMEAR_TAU, squeeze out sideways over the joint's edges onto the faces (BIA TN 7B: tooled too soft,
        mortar smears onto the units; sealant pressed into a full joint comes out at its edges); stiffer mortar is
        pushed along the joint, and on open surfaces the roll runs to the blade's ends."""
        name = reg.material.name
        if not reg.ref_mask[sl].any() or name not in ("mortar", "sealant"):
            return False
        if name == "sealant":
            return True
        tau = reg.tau_b(sl)[ins & (reg.t_at(sl) > 1e-5)]
        return bool(len(tau)) and float(np.median(tau)) < SMEAR_TAU

    def _squeeze(self, c, reg, p, R, vol):
        """Put `vol` on the finished faces beside the sole's long sides (the cells of ref_mask within SPILL_BAND of
        the sole's sides, thickest at the sole), as smear."""
        sole = c.sole
        long_x = sole.hx >= sole.hy
        big = Sole("blade", sole.hx + (0.0 if long_x else SPILL_BAND), sole.hy + (SPILL_BAND if long_x else 0.0))
        fp = footprint(reg, big, p, R, clip=False)
        if fp is None:
            return False
        rng, inside, _, x, y = fp
        (i0, i1), (j0, j1) = rng
        # the finished faces go on past the region's edge (a joint's units, a wall)
        ref = np.ones(inside.shape, bool)
        a0, a1, b0, b1 = max(i0, 0), min(i1, reg.nu), max(j0, 0), min(j1, reg.nv)
        if a0 < a1 and b0 < b1:
            ref[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = reg.ref_mask[a0:a1, b0:b1]
        side = np.abs(y if long_x else x)
        half = sole.hy if long_x else sole.hx
        wgt = np.where(inside & ref & (_cover(sole, x, y, reg.cell) <= 0.25),
                       1.0 - np.clip((side - half)/SPILL_BAND, 0.0, 1.0), 0.0)
        if self._lay(reg, rng, wgt, vol, rough=False, packet=c.packet) is None:
            return self._spill(c, reg, p, R, vol) > 0
        self.log["squeezed_m3"] = self.log.get("squeezed_m3", 0.0) + vol
        return True

    def _set_down(self, c, reg):
        """Leave the roll where it stood ahead of the blade's leading face, as the prism it was there (the coupler's
        roll: a right triangle of legs H against the face, A = H^2/2 of cross-section over the face's length; what the
        paste's yield stress cannot hold then slumps in relax). The leading face is the long one when the blade moves
        across its length and the end when it moves along it (an edger run along a form: the roll was laid along its
        long axis whatever the motion, and a blade moving lengthwise left a 12 mm band right across the region)."""
        if c.roll <= 0:
            return
        try:
            q = self._last_q
        except AttributeError:
            q = None
        if q is not None:
            p, R = c.sole_pose(q[c.body])
            pr = reg.to_region(p)
            Rr = reg.R.T @ R
            long_x = c.sole.hx >= c.sole.hy
            ax = Rr[:2, 0] if long_x else Rr[:2, 1]                      # the blade's long axis in the region plane
            ax = ax/max(np.linalg.norm(ax), 1e-9)
            nx = np.array([-ax[1], ax[0]])                               # across it
            half_l, half_w = max(c.sole.hx, c.sole.hy), min(c.sole.hx, c.sole.hy)
            lead = c.lead if c.lead is not None else nx
            if abs(float(lead @ nx)) >= abs(float(lead @ ax)):
                face, out, off = ax, nx*np.sign(float(lead @ nx)), half_w        # the long face leads
                length = 2*half_l
            else:
                face, out, off = nx, ax*np.sign(float(lead @ ax)), half_l        # the end leads
                length = 2*half_w
            H = math.sqrt(2.0*c.roll/max(length, reg.cell))
            H = max(H, 2.0*reg.cell)                                     # no thinner than the grid can carry
            base = pr[:2] + out*off                                      # the leading face's line
            ends = np.array([base + face*length/2, base - face*length/2, base + face*length/2 + out*H,
                             base - face*length/2 + out*H])
            cc = reg.cell
            i0, i1 = int(math.floor(ends[:, 0].min()/cc)) - 2, int(math.ceil(ends[:, 0].max()/cc)) + 2
            j0, j1 = int(math.floor(ends[:, 1].min()/cc)) - 2, int(math.ceil(ends[:, 1].max()/cc)) + 2
            U, V = np.meshgrid((np.arange(i0, i1) + 0.5)*cc, (np.arange(j0, j1) + 0.5)*cc, indexing="ij")
            dl = (U - base[0])*face[0] + (V - base[1])*face[1]
            dw = (U - base[0])*out[0] + (V - base[1])*out[1]
            shape = np.where((np.abs(dl) < length/2) & (dw >= 0.0) & (dw < H), 1.0 - dw/H, 0.0)
            if self._lay(reg, ((i0, i1), (j0, j1)), shape, c.roll, packet=c.packet) is not None:
                self.log["set_down_m3"] += c.roll
                c.roll = 0.0
                return
        # No pose: retain the load until a physical location is available.

    def _ensure_packet(self, c, reg=None):
        if c.roll <= 0:
            c.packet = None
            return
        if c.packet is None:
            # Compatibility for explicitly phantom-loaded historical probes. Real
            # cutting always supplies a packet before increasing the load.
            from .materials import MATERIALS
            mat = reg.material if reg is not None and reg.material.name == c.roll_mat else MATERIALS[c.roll_mat or "compound"]()
            age = float(np.median(reg.age())) if reg is not None else 0.
            c.packet = Packet(c.roll, mat, {"age": age})
        c.packet.volume = c.roll

    def _carry_update(self, c, q, qd, dt):
        if c.roll <= 0 or c.tool is None or not c.tool.carry:
            return
        self._ensure_packet(c, c.region)
        packet = c.packet
        packet.values["age"] = packet.values.get("age", 0.)+dt
        p, R = c.sole_pose(q[c.body])
        # Yield stress bounds a mound; tensile adhesion on an overturned blade is
        # a much smaller estimated fraction. Weight applies even far from a region.
        tau = float(packet.material.tau_bulk(packet.values["age"]))
        up = float(R[2, 2])
        area = 4*c.sole.hx*c.sole.hy
        cap = .5*area*tau/(packet.material.rho*9.81*P.DROP_STICK)*max(.02, .5*(1.+up))
        if c.tool.carry_cap is not None:
            cap = min(cap, c.tool.carry_cap)
        if c.F_raw < .05 and c.roll > cap+1e-9:
            excess = c.roll-cap
            self.falling.append(dict(p=p.copy()+R[:, 2]*.002, v=qd[c.body, :3].copy(),
                                     packet=packet.clone(excess), root=c.source_region))
            c.roll = cap
        c.wrench[:3] = (0., 0., -9.81*c.roll*packet.material.rho)
        c.point = p
        self.log["carried_m3"] = max(self.log["carried_m3"], c.roll)

    def _hold_or_drop(self, c, reg):
        # Retained for old probe callers; ordinary stepping uses pose-aware update.
        self._ensure_packet(c, reg)

    def _advance_falling(self, dt):
        keep = []
        for drop in self.falling:
            previous = drop["p"].copy()
            drop["v"][2] -= 9.81*dt
            drop["p"] += drop["v"]*dt
            drop["packet"].values["age"] = drop["packet"].values.get("age", 0.)+dt
            root = self.regions.get(drop["root"])
            if root is None:
                root = next((r for r in self.regions.values() if r.material.name == drop["packet"].material.name), None)
            landing = None
            # Receive on horizontal material work/support surfaces before the
            # room floor. A dropped load above the hawk cannot pass through it.
            for surface in all_regions(self.regions):
                if surface.N[2] < .99 or surface.material.name != drop['packet'].material.name:
                    continue
                pos = surface.to_region(drop['p']);i, j = int(np.floor(pos[0]/surface.cell)), int(np.floor(pos[1]/surface.cell))
                if not (0 <= i < surface.nu and 0 <= j < surface.nv and surface.active[i, j]):
                    continue
                z = surface.o[2]+surface.h[i, j]
                if drop['p'][2] <= z <= previous[2] and (landing is None or z > landing[1]):
                    landing = surface, z
            if (drop["p"][2] > 0 and landing is None) or root is None:
                keep.append(drop)
                continue
            # A falling gob lands on the room floor. Ballistic flight is explicit;
            # lateral impact spread is a bounded, volume-normalized 45-degree mound.
            packet = drop["packet"]
            if landing is not None:
                surface = landing[0]
                pos = surface.to_region(drop['p'])
                rad = max(2, int(np.ceil((3*packet.volume/np.pi)**(1/3)/surface.cell)))
                centre = pos[:2]/surface.cell
                i0, j0 = int(centre[0])-rad, int(centre[1])-rad
                ii, jj = np.meshgrid(np.arange(i0, i0+2*rad+1), np.arange(j0, j0+2*rad+1), indexing='ij')
                weight = np.maximum(1.-np.hypot(ii+.5-centre[0], jj+.5-centre[1])/rad, 0.)
                self._lay(surface, ((i0, i0+2*rad+1),(j0,j0+2*rad+1)), weight, packet.volume, packet=packet)
                continue
            radius = max((3*packet.volume/np.pi)**(1/3), 2*root.cell)
            centre = drop["p"][:2]/root.cell
            rad = int(np.ceil(radius/root.cell))
            ii, jj = np.meshgrid(np.arange(int(centre[0])-rad, int(centre[0])+rad+1),
                                 np.arange(int(centre[1])-rad, int(centre[1])+rad+1), indexing="ij")
            weight = np.maximum(1.-np.hypot(ii+.5-centre[0], jj+.5-centre[1])*root.cell/radius, 0.)
            depth = weight*packet.volume/(weight.sum()*root.cell**2)
            on = depth > 0
            catch_deposit(root, (ii[on], jj[on]), depth[on], packet, floor=True)
        self.falling = keep

    def _work(self, c, reg, sl, ins, s, speed, f, dt, v_par=None):
        tool = c.tool
        s_low = float(np.nanmin(np.where(ins, s, np.nan)))
        h = reg.h_at(sl)
        bear = ins & (h - s > -BEAR_BAND) & (reg.t_at(sl) > 1e-5)
        n = int(bear.sum())
        if n == 0:
            return
        area = n*reg.cell**2
        pres = f/area if area > 0 else 0.0
        ds = speed*dt
        tau_s = reg.tau_s(sl)
        c.work_diagnostic = dict(pressure_pa=pres, bearing_area_m2=area,
                                 tau_bulk_pa=float(np.median(reg.tau_b(sl)[bear])),
                                 tau_surface_pa=float(np.median(tau_s[bear])))
        # a cell slides under the bearing part of the sole, in one pass, the bearing patch's extent along the motion
        # (a pitched blade's strip, a flat sole's width even where only its crests bear): embedding and texture are
        # per pass of that patch. Taken per metre slid, a pitched blade bearing on a 3 mm strip did a thirtieth of a
        # flat float's pass, and the darby on its own weight needed some 50 passes to get the stone down (A2)
        if v_par is not None and speed > 1e-9:
            dx, dy = float(v_par[0])/speed, float(v_par[1])/speed
            ii, jj = np.nonzero(bear)
            along = reg.u[sl[0]][ii]*dx + reg.v[sl[1]][jj]*dy
            b_along = float(along.max() - along.min()) + reg.cell
        else:
            b_along = max(2*min(c.sole.hx, c.sole.hy), reg.cell)
        passes = ds/b_along
        # aggregate embedded, paste brought up (floats and darbies above all)
        k = passes*P.EMBED_PASS*float(np.clip(pres/1000.0, 0.1, 20.0))
        E = reg.E[sl]; E[bear] *= np.float32(math.exp(-k))
        ts = reg.ts[sl]; tb = reg.tb[sl]
        grow = np.where(bear, np.minimum(reg.reserve[sl]*(1 - math.exp(-k)), tb), 0.0).clip(0, None)
        ts += grow; tb -= grow; reg.reserve[sl] -= grow
        if grow.any():
            reg.dirty = True
        # texture toward the tool's own while the surface layer yields under the sole: in shear at the interface (the
        # drag's share of tau_s) or in bearing where the sole presses harder. Bearing alone gave a
        # float at its time (1-3 kPa over 80 mm of its width, tau_s 6.4 kPa) a share of 0.03-0.09: the edger's
        # smooth track stayed at 0.05 mm under every pass (A2)
        if tool is not None:
            yields = np.maximum(np.clip(pres/(P.NC*tau_s), 0.0, 1.0), self._m_interface(reg, sl))
            a = reg.a[sl]
            a[bear] = (tool.texture + (a[bear] - tool.texture)*np.exp(-passes*P.TEX_PASS*yields[bear])).astype(np.float32)
            kind = reg.kind[sl]; kind[bear & (yields > 0.3)] = {"float": 3, "trowel": 8, "edger": 4, "knife": 9, "rod": 10}.get(tool.kind, 3)
            # densification by interface work, sealing by a pitched steel blade
            m_i = self._m_interface(reg, sl)
            Wt = (m_i*tau_s*ds*tool.dens).astype(np.float32)
            reg.W[sl] += np.where(bear, Wt, 0.0).astype(np.float32)
            D = reg.D[sl]
            D[bear] = 1.0 - (1.0 - D[bear])*np.exp(-Wt[bear]/P.DENS_ENERGY)
            if tool.seal > 0:
                # pitch of the sole against the surface: the rise across the bearing direction
                s_in = np.where(ins, s, np.nan)
                rise = float(np.nanmax(s_in) - s_low)
                width = 2*min(c.sole.hx, c.sole.hy)
                pitch = rise/max(width, 1e-3)
                S = reg.S[sl]
                S[bear] = 1.0 - (1.0 - S[bear])*math.exp(-ds*tool.seal*min(pitch, 0.6)*max(pres/1000.0, 0.1)/0.10)
        # a film worked into the surface (CIP 1): the top 6 mm's w/c rises and the cell remembers it
        w = reg.w[sl]
        wet = bear & (w > P.SHEEN_W)
        if wet.any():
            reg.dwc[sl][wet] += (1000.0*w[wet]/(P.CEMENT*0.006)).astype(np.float32)
            reg.wet_worked[sl][wet] = 1.0
            w[wet] = 0.0
        # joints: compaction (mortar), wall contact (sealant)
        name = reg.material.name
        if name == "mortar":
            from .materials import consolidate_mortar
            comp = reg.compact[sl]
            comp[bear] = consolidate_mortar(comp[bear], pres, ds, reg.tau_b(sl)[bear], self._m_interface(reg, sl)[bear])
        if name == "sealant" and hasattr(reg, "edge_l"):
            wet_p = pres > 2.0*float(np.median(tau_s[bear]))
            if wet_p:
                for flag, edge in ((reg.contact_l, reg.edge_l), (reg.contact_r, reg.edge_r)):
                    e = edge[sl] & bear & (reg.t_at(sl) > 1e-4) & (reg.age(sl) < reg.material.skin)
                    flag[sl][e] = 1.0
        if tool is not None and tool.kind == "edger":
            # the cells the lip runs down (the sole there within half the lip's depth of its lowest point): the concrete
            # is cut from the form there
            lip = ins & (s - s_low < 0.5*tool.meta["lip"])
            reg.compact[sl][lip] = 1.0
        if name == "sealant":
            torn = bear & (reg.age(sl) >= reg.material.skin)
            reg.skin_torn[sl][torn] = 1.
            reg.a[sl][torn] = np.maximum(reg.a[sl][torn], 5e-4)
        reg.seal_age[sl][bear] = reg.age(sl)[bear]
        reg.seal_t[sl][bear] = np.float32(reg.clock)                     # when each cell was last worked ...
        reg.tau_w[sl][bear] = tau_s[bear].astype(np.float32) if name in ("concrete", "repair_mortar") else \
            reg.tau_b(sl)[bear].astype(np.float32)                       # ... at what yield stress ...
        reg.worked[sl][bear] += np.float32(ds)                           # ... how far a sole slid over it (m)
        late = getattr(reg.material, "initial_set", None)
        if late is not None:
            old = reg.age(sl) > late
            reg.late[sl][bear & old] = 1.0                               # ... and whether after initial set
        c.work += f*0.0 + float((self._m_interface(reg, sl)*tau_s)[bear].mean())*area*ds

    def _m_interface(self, reg, sl):
        """Interface shear / surface yield stress: 0.05 on a wet film, rising toward 0.30 as the surface stiffens
        (log of tau_s from 300 Pa to 30 kPa); an estimate."""
        lo, hi = P.M_INTERFACE
        ts = np.maximum(reg.tau_s(sl), 1.0)
        x = np.clip((np.log10(ts) - math.log10(300.0))/2.0, 0.0, 1.0)
        m = lo + (hi - lo)*x
        return np.where(reg.w[sl] > P.SHEEN_W, lo, m)

    def _forces(self, c, reg, sl, ins, s, p, v_w, v_r, speed, f, dt):
        """Drag (interface, ploughing, roll) against the sole's in-plane motion and tack against its lifting, applied
        at the sole's centre, regularised so that d(force)/d(velocity) dt / m <= 0.5 (explicit limit)."""
        h = reg.h_at(sl)
        bear = ins & (h - s > -BEAR_BAND) & (reg.t_at(sl) > 1e-5)
        n = int(bear.sum())
        F = np.zeros(3)
        if n:
            area = n*reg.cell**2
            tau_s = reg.tau_s(sl)
            m_i = self._m_interface(reg, sl)
            tau_i = float((m_i*tau_s)[bear].mean())
            drag = tau_i*area
            drag += P.PLOUGH*float(np.median(tau_s[bear]))*c.ploughed/max(speed*dt, 1e-6) if c.ploughed > 0 else 0.0
            drag += 3.0*float(np.median(tau_s[bear]))*math.sqrt(max(c.roll, 0.0)/max(2*max(c.sole.hx, c.sole.hy), 1e-3)) \
                * 2*max(c.sole.hx, c.sole.hy) if c.roll > 0 and not (c.tool is not None and c.tool.carry) else 0.0
            # Green: the interface cannot carry more than the bearing allows
            drag = min(drag, max(f, 0.0)*0.95 + tau_i*area)
            v_reg = max(V_REG, 2.0*drag*dt/max(c.mass, 1e-3))
            vp = np.array([v_r[0], v_r[1], 0.0])
            F_r = -drag*vp/max(speed, v_reg)
            F = reg.R @ F_r
        # tack: a sole that has left its support (the plate carries nothing) while wet paste still clings to it is held
        # back over the first TACK_GAP of separation; a sole that rises because the surface
        # under it rises is in compression and has none
        vn = float(v_r[2])
        if f > 0.05 and n:
            c.tack_area = n*reg.cell**2
            c.tack_sep = 0.0
        elif c.tack_sep is not None:
            c.tack_sep += max(vn, 0.0)*dt
            if c.tack_sep >= P.TACK_GAP:
                c.tack_sep = None
            elif vn > 0:
                lo, hi = P.TACK
                x = np.clip((math.log10(max(vn*1e6, 1.0)) - math.log10(50.0))/1.0, 0.0, 1.0)
                sig = (lo + (hi - lo)*x)*{"concrete": 0.5, "sealant": 2.0}.get(reg.material.name, 1.0)
                Ft = sig*c.tack_area*(1.0 - c.tack_sep/P.TACK_GAP)
                Ft = min(Ft, 0.5*c.mass/dt**2*P.TACK_GAP)               # explicit-stable bound (never reached by a tool)
                F = F - Ft*reg.N
                c.tack_force = Ft
        c.wrench[:3] += F
        c.point = p                                                       # applied at the sole centre (world.py adds the moment)


def _cover(sole, x, y, cell):
    """Share of each cell (a cell-sized square about its centre) under the sole's rectangle, in the sole's frame."""
    return np.clip((sole.hx - np.abs(x))/cell + 0.5, 0.0, 1.0)*np.clip((sole.hy - np.abs(y))/cell + 0.5, 0.0, 1.0)


def _rough(reg, sl, add):
    """Paste set down by a tool (a heave, a spill, a dropped roll) is rough and shows its stone where it lies thick."""
    thick = add > 5e-4
    if thick.any():
        a = reg.a[sl]; a[thick] = np.maximum(a[thick], 6.0e-4)
        if reg.material.name == "concrete":
            E = reg.E[sl]; E[thick] = np.maximum(E[thick], 0.5*P.AGG_EXPOSED)
