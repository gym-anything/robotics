"""The lockstitch machine (stitch type 301): shaft and control box, needle bar, four-motion feed dog, presser foot,
needle puncture, deflection and damage, loop pickup and stitch formation, thread elements, tether and trimmer.

Every effect on the cloth goes through the cloth solver's contacts and constraints:
  * the feed dog is a kinematic solver shape; the presser foot is a body in the cloth solve (mass 0.12 kg, free along
    z only, pushed down by the presser spring and gravity, stopped by the dog teeth / plate or the lifter) carrying
    the two soles and toes as shapes. The foot rests on whatever cloth is under it and its load reaches the cloth
    through those contacts; the cloth is fed only by friction whose normal forces are the contact forces,
    so the foot rides up over thick places and bounces at speed by itself;
  * a needle that crosses a ply creates a hole at the material point where the ply is, held to the needle axis by a
    needle constraint (series compliance: hole + needle bending) whose axial force is capped by the penetration law;
  * at loop pickup the stitch-formation law reads the gates (needle intact, deflection within the hook clearance and
    loop reach, no flagging) from the simulated state; a formed stitch binds the holes of every pierced ply
    (interlock) and adds needle- and bobbin-thread elements; a failed gate leaves a skipped stitch.
Frame: the origin is where the needle axis meets the throat-plate top (z = 0); +y is the feed direction (away from
the operator), +x to the operator's right (toward the arm and the handwheel). Shaft angle theta = 0 at needle-bar top
dead centre.
"""

import math
from dataclasses import dataclass, asdict

import numpy as np

from . import params as P
from .cloth import (CON_BIND, CON_NEEDLE, CON_THREAD, CON_TETHER, KIND_KINEMATIC, SHAPE_BOX, SHAPE_CAPSULE)
from .layout import BED as _BED

BED_XY = (_BED["x"], _BED["y"])

TWO_PI = 2*math.pi
EYE_ABOVE_POINT = 3.0e-3          # needle point to the top of its eye (estimate)
NEEDLE_OUT_OF_CLAMP = P.NEEDLE_L_BDC - P.POINT_BDC   # 27 mm: the bar 15 mm above the plate at BDC, the point 12 below (Pfaff)
FOOT_THICK = 3.0e-3               # sole thickness (Sofronova 1.5-6 mm)
HOOK_SIDE = +1.0                  # the hook point passes the scarf on the +x side (estimate; DB x 1 threaded left to right)


def _quat_matrix(q):
    """Rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def _segment_hits_box(a, b, half):
    """Does the segment a-b (box frame) meet the axis-aligned box |x_i| <= half_i? (slab test)"""
    t0, t1 = 0.0, 1.0
    d = b - a
    for k in range(3):
        if abs(d[k]) < 1e-15:
            if abs(a[k]) > half[k]:
                return False
            continue
        u, v = (-half[k] - a[k])/d[k], (half[k] - a[k])/d[k]
        t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
        if t0 > t1:
            return False
    return True


@dataclass
class MachineSettings:
    head: str = "juki_ddl9000c"          # 'juki_ddl9000c' (A, C presets); others are declared in variants
    stitch_length: float = 2.5e-3        # m
    presser_force: float = P.PRESSER_F    # N
    needle_nm: int = 80
    needle_tension: float = P.NEEDLE_TENSION_REF   # N, static (dial)
    bobbin_tension: float = P.BOBBIN_TENSION      # N
    max_spm: float = P.MAX_SPM
    dog_height: float = P.DOG_HEIGHT
    penetration_scale: float = 0.8        # s_f per fabric/needle; muslin + Nm 80 prior 0.6-1.0
    fabric_tension_factor: float = 1.0
    needle_alpha: float = P.NEEDLE_ALPHA

    def check(self):
        if not 0.0 <= self.stitch_length <= P.STITCH_MAX + 1e-12:
            raise ValueError("stitch length must be in [0, 5] mm")
        if not 10.0 <= self.presser_force <= 60.0:
            raise ValueError("presser force must be in [10, 60] N (Juki DDL-9000C panel range)")
        if self.needle_nm not in (65, 70, 75, 80, 90, 100, 110):
            raise ValueError("needle size must be one of Nm 65-110 (DB x 1 #9-18)")
        if not 0.0 < self.needle_tension <= 3.0 or not 0.0 < self.bobbin_tension <= 1.0:
            raise ValueError("thread tensions must be positive (needle <= 3 N, bobbin <= 1 N)")


@dataclass
class Hole:
    ply: int
    face: int
    bary: tuple
    uv: tuple
    tick: int
    theta: float
    cycle: int
    z0: float
    con: int = -1
    lift_max: float = 0.0
    edge: bool = False
    engaged: bool = True


class Machine:
    def __init__(self, settings=None):
        self.s = settings or MachineSettings()
        self.s.check()
        r = 0.5*P.STROKE
        self.r = r
        self.l = P.CRANK_RATIO*r
        self.A_z = self.s.dog_height + P.DOG_BELOW
        self.z_off = -P.DOG_BELOW
        th = np.linspace(0, math.pi, 20001)
        zeye = self.needle_tip(th) + EYE_ABOVE_POINT
        self.theta_eye_down = float(th[np.argmax(zeye <= 0.0)])
        psi2 = math.pi - math.asin(-self.z_off/self.A_z)
        self.phi = self.theta_eye_down - psi2
        th2 = np.linspace(math.pi, TWO_PI, 20001)
        ztip = self.needle_tip(th2)
        self.theta_pickup = float(th2[np.argmax(ztip >= P.POINT_BDC + P.PICKUP_RISE)])
        self.theta_down_stop = math.pi + math.radians(10.0)
        self.theta = 0.0
        self.theta_prev = 0.0
        self.omega = 0.0
        self.speed_cmd = 0.0
        self.speed_ramped = 0.0
        self.needle_stop = "up"
        self.needle_installed = True
        self.seam = 0
        self.seam_break = False
        self.presser = "down"
        self.reverse = False
        self.reverse_now = False
        self.trim_request = False
        self.mode = "idle"
        self.stop_target = None
        self.faults = []
        self.obstacles = []          # (cloth shape, label): rigid bodies that can stand in the needle's path (hands)
        self.needle_state = "ok"
        self.cycle = 0
        self.holes = []
        self.engaged = []
        self.stitches = []
        self.interlocks = []
        self.threads = []
        self.tether = None
        self.tails = dict(start=P.TRIM_TAIL, end=None)
        self.cloth = None
        self.last_defl = np.zeros(2)
        self.max_defl_cycle = np.zeros(2)
        self.peak_lateral_force = 0.0
        self.log = []
        # presser foot (a body in the cloth solve): sole-bottom height and the load the cloth carries (measured)
        self.foot = -1
        self.lifter_z = 0.0          # the knee/auto lifter's stop under the presser bar (rate-limited)
        self.foot_z = 0.0
        self.foot_N = 0.0
        self.foot_trace = []

    # ------------------------------------------------------------------------------------------- kinematics
    def needle_tip(self, theta):
        th = np.asarray(theta, float)
        zb = self.r*np.cos(th) + np.sqrt(self.l**2 - (self.r*np.sin(th))**2) - self.l
        return P.POINT_BDC + self.r + zb

    def needle_tip_velocity(self, theta, omega):
        eps = 1e-6
        return (self.needle_tip(theta + eps) - self.needle_tip(theta - eps))/(2*eps)*omega

    def dog(self, theta):
        """Feed-dog top height above the plate and its offset along the feed (m)."""
        psi = theta - self.phi
        # the dog's stroke: the dial's stitch length times the feed regulator's calibration (FEED_STROKE_PER_SETTING)
        L = self.s.stitch_length*P.FEED_STROKE_PER_SETTING*(-1.0 if self.reverse_now else 1.0)
        return self.A_z*math.sin(psi) + self.z_off, -0.5*L*math.cos(psi)

    def needle_stiffness(self, z_load):
        d = self.s.needle_nm*1e-5
        I = math.pi*d**4/64
        z_clamp = float(self.needle_tip(self.theta)) + NEEDLE_OUT_OF_CLAMP
        L = max(3e-3, z_clamp - z_load)
        return self.s.needle_alpha*3*P.NEEDLE_E*I/L**3, L

    def foot_rects(self):
        """Sole rectangles (centre xy, half extents) of the two toes, world frame (foot at x = 0, y = 0)."""
        L, back = P.FOOT_LEN, P.FOOT_BACK
        yc = back - 0.5*(L - 5e-3)
        out = []
        for s, wt in ((-1, P.FOOT_W_LEFT), (1, P.FOOT_W_RIGHT)):
            x = s*(0.5*P.FOOT_SLOT + 0.5*wt)
            out.append(((x, yc, 0.0), (0.5*wt, 0.5*(L - 5e-3), 0.0)))
        return out

    # ------------------------------------------------------------------------------------------ scene setup
    def add_to_cloth(self, cloth, fabric):
        """Register the machine's cloth-contact shapes (before cloth.build())."""
        self.cloth = cloth
        self.fabric = fabric
        mu_steel = fabric["mu_steel"]
        tx, ty, tz = P.TABLE_TOP
        x0, x1 = -P.NEEDLE_FROM_LEFT, -P.NEEDLE_FROM_LEFT + tx
        y0, y1 = -P.NEEDLE_FROM_FRONT, -P.NEEDLE_FROM_FRONT + ty
        (bx0, bx1), (by0, by1) = BED_XY
        re = P.TABLE_EDGE_RADIUS
        # the surfaces under the cloth, all with their top at z = 0: the machine's bed (painted cast iron, the steel
        # friction; the throat plate is set in it) and the laminate table top around it, whose front edge is rounded
        # (a capsule along the edge, the top slab starting behind it). One table box spanning the bed put laminate
        # friction under the foot beside the plate's, and its sharp front corner caught the coarse cloth mesh where
        # the stack hangs over the edge.
        def box(name, xa, xb, ya, yb, za, zb, mu):
            return cloth.add_shape(name, SHAPE_BOX, (0.5*(xb - xa), 0.5*(yb - ya), 0.5*(zb - za)),
                                   (0.5*(xa + xb), 0.5*(ya + yb), 0.5*(za + zb)), mu=mu)
        mt = fabric["mu_table"]
        self.sh_bed = [box("machine_bed", bx0, bx1, by0, by1, -tz, 0.0, mu_steel),
                       box("table_front", x0, x1, y0 + re, by0, -re, 0.0, mt),
                       box("table_front_low", x0, x1, y0, by0, -tz, -re, mt),
                       box("table_back", x0, x1, by1, y1, -tz, 0.0, mt),
                       box("table_left", x0, bx0, by0, by1, -tz, 0.0, mt),
                       box("table_right", bx1, x1, by0, by1, -tz, 0.0, mt),
                       cloth.add_shape("table_edge", SHAPE_CAPSULE, (re, 0.5*tx, 0.0),
                                       (0.5*(x0 + x1), y0 + re, -re), mu=mt)]
        self.sh_plate = cloth.add_shape("throat_plate", SHAPE_BOX, (0.06, 0.03, 0.001), (0.0, 0.0, -0.001),
                                        mu=mu_steel)
        hw, hl = 0.5*P.DOG_W, 0.5*P.DOG_L
        xc = P.DOG_GAP + hw
        self.sh_dog = [cloth.add_shape(f"dog_{s}", SHAPE_BOX, (hw, hl, 0.004), (s*xc, 0.0, -0.005),
                                       kind=KIND_KINEMATIC, mu=fabric["mu_dog"], grip=P.DOG_TOOTH_GRIP) for s in (-1, 1)]
        # the presser foot: a body free along z (the bar's guide), sole bottom at its z
        self.foot = cloth.add_body((0.0, 0.0, 0.0), P.PRESSER_MASS, free=(0, 0, 1), floor=0.0)
        self.sh_foot = []
        L, back = P.FOOT_LEN, P.FOOT_BACK
        for (c, hh) in self.foot_rects():
            self.sh_foot.append(cloth.add_shape("foot_sole", SHAPE_BOX, (hh[0], hh[1], 0.5*FOOT_THICK),
                                                (c[0], c[1], 0.5*FOOT_THICK), kind=KIND_KINEMATIC, mu=mu_steel,
                                                body=self.foot))
            # turned-up toe in front (25 deg) so the stack slides under the foot
            a = math.radians(25.0)
            q = (math.sin(-a/2), 0.0, 0.0, math.cos(-a/2))
            y0 = back - L + 5e-3
            self.sh_foot.append(cloth.add_shape("foot_toe", SHAPE_BOX, (hh[0], 2.6e-3, 0.5*FOOT_THICK),
                                                (c[0], y0 - 2.2e-3, 0.5*FOOT_THICK + 1.1e-3), quat=q,
                                                kind=KIND_KINEMATIC, mu=mu_steel, body=self.foot))
        self.sh_pillar = cloth.add_shape("pillar", SHAPE_BOX, (0.045, 0.06, 0.15), (P.THROAT + 0.045, 0.0, 0.15),
                                        mu=P.CASTING_MU)
        self.sh_head = cloth.add_shape("head_bottom", SHAPE_BOX, (0.04, 0.03, 0.02), (0.0, 0.0, 0.065), mu=P.CASTING_MU)
        self.p_eye = cloth.add_particle((0, 0, 0.02), 0.0, 1e-4)
        self.p_bobbin = cloth.add_particle((0, 0, -0.006), 0.0, 1e-4)

    def after_build(self):
        c = self.cloth
        self.ply_tris = [np.arange(p_["t0"], p_["t0"] + p_["nt"]) for p_ in c.plies]
        self.thickness = max(p_["thickness"] for p_ in c.plies) if c.plies else 0.3e-3
        c.set_particle_force(self.foot, (0.0, 0.0, -self.s.presser_force))
        # lower the foot onto what is under it: the highest cloth surface under the soles, else the dog teeth / plate
        x = c.positions()
        r = c.radius.numpy()
        top = max(0.0, float(self.dog(self.theta)[0]))
        for cc, hh in self.foot_rects():
            under = (np.abs(x[:, 0] - cc[0]) <= hh[0]) & (np.abs(x[:, 1] - cc[1]) <= hh[1]) & (c.is_cloth_host == 1)
            if under.any():
                top = max(top, float((x[under, 2] + r[under]).max()))
        lift = {"down": 0.0, "lift1": P.FOOT_LIFT_AUTO1, "lift2": P.FOOT_LIFT_AUTO2}[self.presser]
        c.set_particle_position(self.foot, (0.0, 0.0, max(top, lift)))
        self._place(first=True)

    def _place(self, first=False):
        """Set the end-of-tick poses of the dog and the foot and the sandwich parameters for the next tick."""
        c = self.cloth
        zd, yd = self.dog(self.theta)
        xc = P.DOG_GAP + 0.5*P.DOG_W
        for s, idx in zip((-1, 1), self.sh_dog):
            p = (s*xc, yd, zd - 0.004)
            c.set_shape_targets(idx, p)
            if first:
                c._sp_host[idx] = p
        # the presser bar's stop: the sole on the dog teeth (when they are above the plate) or on the plate, or the
        # lifter, which travels at a finite speed (an instant 8.5 mm stop teleported the foot and launched it)
        lift = {"down": 0.0, "lift1": P.FOOT_LIFT_AUTO1, "lift2": P.FOOT_LIFT_AUTO2}[self.presser]
        if first:
            self.lifter_z = lift
        step = P.FOOT_LIFT_SPEED*c.dt
        self.lifter_z += float(np.clip(lift - self.lifter_z, -step, step))
        c.set_body_floor(self.foot, max(self.lifter_z, zd, 0.0))
        tip = float(self.needle_tip(self.theta))
        axis = np.zeros((8, 3))
        # the undeflected needle axis: the hole constraint's compliance already holds the needle's bending in series
        # with the hole (1/PIN_STIFF + 1/k_n), so moving the target by the deflection counted it twice
        axis[0] = (0.0, 0.0, tip)
        vel = np.zeros((8, 3))
        vel[0, 2] = float(self.needle_tip_velocity(self.theta, self.omega))
        c.axis_p.assign(axis)
        c.axis_v.assign(vel)

    # --------------------------------------------------------------------------------------------- control
    def command(self, speed_spm=None, needle_stop=None, presser=None, reverse=None, trim=None):
        """The control box (what a pedal / knee lifter / touch-back / trimmer would set). Validated, never clamped."""
        if speed_spm is not None:
            if isinstance(speed_spm, bool) or not isinstance(speed_spm, (int, float)) or not math.isfinite(speed_spm) \
                    or speed_spm < 0 or speed_spm > self.s.max_spm:
                raise ValueError(f"speed_spm must be in [0, {self.s.max_spm}]")
            if self.s.stitch_length > 4e-3 and speed_spm > P.MAX_SPM_LONG:
                raise ValueError("above 4 mm stitch length the head is limited to 4,000 sti/min (Juki)")
        if needle_stop is not None and needle_stop not in ("up", "down"):
            raise ValueError("needle_stop must be 'up' or 'down'")
        if presser is not None and presser not in ("down", "lift1", "lift2"):
            raise ValueError("presser must be 'down', 'lift1' or 'lift2'")
        if speed_spm is not None:
            self.speed_cmd = float(speed_spm)
            if speed_spm > 0:
                self.mode = "run"
                self.stop_target = None
            elif self.mode == "run":
                self.mode = "stopping"
                self.stop_target = None
        if needle_stop is not None:
            self.needle_stop = needle_stop
        if presser is not None:
            self.presser = presser
        if reverse is not None:
            self.reverse = bool(reverse)
        if trim:
            self.trim_request = True
            if self.mode == "idle":
                self.mode = "stopping"
                self.stop_target = None

    # ---------------------------------------------------------------------------------------------- per tick
    def pre_tick(self, dt):
        """Shaft over the tick (control box), presser-foot dynamics, end-of-tick poses."""
        self.theta_prev = self.theta
        ramp = P.SPEED_RAMP*dt
        if self.mode == "run":
            self.speed_ramped += float(np.clip(self.speed_cmd - self.speed_ramped, -ramp, ramp))
            self.omega = self.speed_ramped/60.0*TWO_PI
            self.theta += self.omega*dt
        elif self.mode == "stopping":
            stop = 0.0 if (self.needle_stop == "up" or self.trim_request) else self.theta_down_stop
            if self.stop_target is None:
                k = math.floor(self.theta/TWO_PI)
                tgt = k*TWO_PI + stop
                while tgt <= self.theta + 1e-9:
                    tgt += TWO_PI
                self.stop_target = tgt
            creep = 200.0
            self.speed_ramped = max(creep, self.speed_ramped - ramp)
            om = self.speed_ramped/60.0*TWO_PI
            if self.theta + om*dt >= self.stop_target:
                self.theta = self.stop_target
                self.omega = 0.0
                self.speed_ramped = 0.0
                self.mode = "idle"
                self.stop_target = None
                if self.trim_request:
                    self._trim()
            else:
                self.theta += om*dt
                self.omega = om
        else:
            self.omega = 0.0
        # reverse feed switches at the start of a feed stroke (the dog below the plate)
        zd, _ = self.dog(self.theta)
        if zd < 0:
            self.reverse_now = self.reverse
        self._place()

    def _ply_surface_at_axis(self, x, ply, ax, ay):
        c = self.cloth
        tris = self.ply_tris[ply]
        T = c.tri_host[tris]
        P0, P1, P2 = x[T[:, 0]], x[T[:, 1]], x[T[:, 2]]
        lo = np.minimum(np.minimum(P0[:, :2], P1[:, :2]), P2[:, :2])
        hi = np.maximum(np.maximum(P0[:, :2], P1[:, :2]), P2[:, :2])
        near = np.nonzero((lo[:, 0] <= ax) & (hi[:, 0] >= ax) & (lo[:, 1] <= ay) & (hi[:, 1] >= ay))[0]
        best = None
        for k in near:
            a, b, cc = P0[k], P1[k], P2[k]
            v0, v1, v2 = b[:2] - a[:2], cc[:2] - a[:2], np.array([ax, ay]) - a[:2]
            den = v0[0]*v1[1] - v1[0]*v0[1]
            if abs(den) < 1e-18:
                continue
            l1 = (v2[0]*v1[1] - v1[0]*v2[1])/den
            l2 = (v0[0]*v2[1] - v2[0]*v0[1])/den
            l0 = 1 - l1 - l2
            if min(l0, l1, l2) >= -1e-9:
                z = l0*a[2] + l1*b[2] + l2*cc[2]
                if best is None or z > best[2]:
                    best = (int(tris[k]), (l0, l1, l2), z)
        return best

    def _needle_obstacles(self, tick, tip, defl):
        """A rigid body in the needle's path (a robot finger or hand under the bar, or pushed into a needle stopped
        down) is struck: a needle meeting hardened steel or aluminium breaks (it is the weakest part by design) and
        the stub forms no stitch until it is replaced. Tested every tick on the exposed needle, point to clamp, at
        its deflected position, against the boxes the cloth sees for each hand."""
        if not self.obstacles or self.needle_state == "broken":
            return
        c = self.cloth
        r = 0.5*self.s.needle_nm*1e-5
        a = np.array([defl[0], defl[1], tip])
        b = np.array([defl[0], defl[1], tip + NEEDLE_OUT_OF_CLAMP])
        xb = None
        for i, label in self.obstacles:
            if not c.shapes[i]["on"]:
                continue
            R = _quat_matrix(c._sq_host[i])
            p = c._sp_host[i]
            if c.shapes[i]["body"] >= 0:
                # a shape riding on a body (a compliant hand's pads) is posed as an offset from it: the
                # world position adds the body's (read as world, the offset put the fingers a hand's height too low
                # and the needle passed through them: E15 case a sewed 5 stitches through a fingertip)
                xb = c.x.numpy() if xb is None else xb
                p = p + xb[c.shapes[i]["body"]]
            if _segment_hits_box(R.T @ (a - p), R.T @ (b - p), np.asarray(c.shapes[i]["half"], float) + r):
                self.needle_state = "broken"
                self.faults.append(dict(tick=tick, fault="needle_struck_robot", obstacle=label, tip_z_m=float(tip),
                                        speed_spm=float(self.speed_ramped)))
                return

    def post_tick(self, tick):
        """After the cloth step: stack height, punctures, engagement, deflection, loop pickup, threads, tether."""
        c = self.cloth
        self.foot_z = float(c.x.numpy()[self.foot, 2])
        rf = c.all_reactions()[0]
        self.foot_N = float(sum(rf[i][2] for i in self.sh_foot))
        tip0 = float(self.needle_tip(self.theta_prev))
        tip1 = float(self.needle_tip(self.theta))
        near_cloth = min(tip0, tip1) < 0.01 and max(tip0, tip1) > -0.02
        x = c.positions() if (near_cloth or self.engaged or self.interlocks) else None
        forces = c.c_force[:c.n_con].numpy() if c.n_con else np.zeros((0, 3))
        if c.method != "implicit":
            forces = forces/c.substeps
        lat = np.zeros(2)
        for h in self.engaged:
            if h.con >= 0:
                lat -= forces[h.con, :2]
        k_n, L = self.needle_stiffness(min([h.z0 for h in self.engaged], default=0.0))
        defl = lat/k_n
        self.last_defl = defl
        self.peak_lateral_force = max(self.peak_lateral_force, float(np.linalg.norm(lat)))
        if self.engaged and np.linalg.norm(defl) > np.linalg.norm(self.max_defl_cycle):
            self.max_defl_cycle = defl.copy()
        d = self.s.needle_nm*1e-5
        My = P.NEEDLE_SIGMA_Y*math.pi*d**3/32
        if self.needle_state in ("ok", "damaged") and np.linalg.norm(lat)*L > My:
            self.needle_state = "bent"
            self.faults.append(dict(tick=tick, fault="needle_bent", lateral_N=float(np.linalg.norm(lat)), lever_m=L))
        self._needle_obstacles(tick, tip1, defl)
        gap = P.PLATE_GAP_EDGE - 0.5*d
        if self.needle_state == "ok" and tip1 < 0 and np.linalg.norm(defl) > gap:
            self.needle_state = "damaged"
            self.faults.append(dict(tick=tick, fault="needle_struck_plate", deflection_m=float(np.linalg.norm(defl))))
        if x is not None and self.needle_installed and self.needle_state in ("ok", "damaged") and tip1 < tip0:
            ax, ay = float(defl[0]), float(defl[1])
            for ply in range(len(c.plies)):
                if any(h.ply == ply and h.engaged for h in self.engaged):
                    continue
                hit = self._ply_surface_at_axis(x, ply, ax, ay)
                if hit is None:
                    continue
                face, bary, z = hit
                if tip0 >= z > tip1:
                    T = c.tri_host[face]
                    uv = tuple(float(v) for v in np.dot(np.array(bary), c.X[T]))
                    # the cloth is resolved for stitching only in its seam band (FINE_H); a hole in a coarser triangle is
                    # outside what the world can sew, and ends the episode as a world fault rather than going on to
                    # buckle numerically (the 5 mm stitch's failure at the band's graded edge)
                    edge = max(float(np.linalg.norm(c.X[T[a]] - c.X[T[b]])) for a, b in ((0, 1), (1, 2), (2, 0)))
                    if edge > 2.0*P.FINE_H:
                        raise RuntimeError(f"the needle pierced ply {ply} at material ({uv[0]*1e3:.1f}, {uv[1]*1e3:.1f}) "
                                           f"mm, outside its seam band (a {edge*1e3:.1f} mm triangle; the band resolves "
                                           f"{P.FINE_H*1e3:g} mm): the world cannot sew there")
                    hole = Hole(ply=ply, face=face, bary=tuple(map(float, bary)), uv=uv, tick=tick, theta=self.theta,
                                cycle=self.cycle, z0=float(z))
                    hole.edge = self._edge_distance(ply, uv) < P.EDGE_LIG
                    hole.con = c.add_constraint(CON_NEEDLE, fa=face, ba=bary, pa=0, comp=1.0/P.PIN_STIFF + 1.0/k_n,
                                                cap=self._pen_force(), meta=dict(hole=len(self.holes)))
                    self.holes.append(hole)
                    self.engaged.append(hole)
        vt = abs(float(self.needle_tip_velocity(self.theta, self.omega)))
        still = []
        for h in self.engaged:
            hz = float(np.dot(h.bary, x[c.tri_host[h.face], 2])) if x is not None else h.z0
            if tip1 > hz + 0.2e-3 or self.needle_state in ("bent", "broken"):
                h.engaged = False
                c.set_constraint(h.con, on=0)
                continue
            depth = hz - tip1
            if tip1 < tip0:
                cap = self._pen_force(vt)*(min(1.0, depth/P.PEN_POINT_LEN) if depth < P.PEN_POINT_LEN else P.PEN_SHAFT_FRAC)
            else:
                cap = P.PEN_UP_FRAC*P.PEN_F_QS*0.5*self.s.penetration_scale
            c.con_host[h.con]["cap"] = cap
            # the engaged holes share one needle: each hole's series compliance carries n/k_n, so that n equal holes
            # deflect the needle by n F / k_n <= their offset (with 1/k_n each, two plies acted as two needles and the
            # needle deflected 0.235 mm under a 0.154 mm hole offset)
            c.con_host[h.con]["comp"] = 1.0/P.PIN_STIFF + len(self.engaged)/k_n
            c._con_dirty = True
            if (self.theta % TWO_PI) > math.pi:
                h.lift_max = max(h.lift_max, hz - h.z0)
            still.append(h)
        self.engaged = still
        a0, a1 = self.theta_prev, self.theta
        k = math.floor(a0/TWO_PI)
        for tp in (k*TWO_PI + self.theta_pickup, (k + 1)*TWO_PI + self.theta_pickup):
            if a0 < tp <= a1:
                self._pickup(tick)
                self.max_defl_cycle = np.zeros(2)
        if math.floor(a1/TWO_PI) > math.floor(a0/TWO_PI):
            self.cycle += 1
        if not self.engaged:
            self._set_stitches(x)
        self._draw_interlocks(x, take_up=not self.engaged)
        self._threads_check(tick, forces)
        if x is not None:
            self._update_tether(x)
        self.foot_trace.append((tick, self.foot_z, self.foot_N))

    def _pen_force(self, v=0.0):
        """Penetration resistance per ply (N): s_f (F_qs + c_v v) / 2 (Chmielowiec fit is for two plies)."""
        return 0.5*self.s.penetration_scale*(P.PEN_F_QS + P.PEN_C_V*v)

    def _edge_distance(self, ply, uv):
        B = self.cloth.plies[ply]["boundary"]
        X = self.cloth.X
        a, b = X[B[:, 0]], X[B[:, 1]]
        p = np.asarray(uv)
        ab = b - a
        t = np.clip(np.sum((p - a)*ab, 1)/np.maximum(np.sum(ab*ab, 1), 1e-18), 0, 1)
        q = a + t[:, None]*ab
        return float(np.min(np.linalg.norm(p - q, axis=1)))

    def _pickup(self, tick):
        holes = [h for h in self.holes if h.cycle == self.cycle]
        # the hook's point meets the needle's scarf now, at the loop pickup: the needle's deflection at this instant
        # decides a strike, a missed loop or a late one. The cycle's largest deflection is the wrong reading: it comes
        # as the needle enters the work at the end of the feed stroke, 100 deg of shaft earlier, and has relaxed by the
        # time the hook arrives
        at = np.asarray(self.last_defl, float) if self.engaged else np.zeros(2)
        rec = dict(cycle=self.cycle, tick=tick, theta=self.theta, speed_spm=self.speed_ramped,
                   reverse=self.reverse_now, holes=[self.holes.index(h) for h in holes], formed=False, reason=None,
                   deflection_m=[float(v) for v in at], peak_deflection_m=[float(v) for v in self.max_defl_cycle],
                   needle=self.needle_state)
        reason = None
        if self.needle_state in ("bent", "broken"):
            reason = "needle_" + self.needle_state
        elif not holes:
            reason = "no_fabric"
        else:
            dx, dy = at
            if HOOK_SIDE*dx > P.HOOK_CLEARANCE:
                reason = "hook_strike"
                if self.needle_state == "ok":
                    self.needle_state = "damaged"
                self.faults.append(dict(tick=tick, fault="hook_struck_needle", deflection_m=float(dx)))
            elif -HOOK_SIDE*dx > P.MISS_LOOP:
                reason = "missed_loop"
            elif abs(dy) > P.TIMING_DEFL:
                reason = "loop_timing"
            elif any(h.lift_max > P.FLAG_LIFT for h in holes):
                reason = "flagging"
        if reason is None:
            self._form_stitch(holes, rec)
        else:
            rec["reason"] = reason
        self.stitches.append(rec)

    def _form_stitch(self, holes, rec):
        c = self.cloth
        order = sorted(holes, key=lambda h: -h.z0)
        bound = [h for h in order if not h.edge]
        con_ids, bind_rest = [], []
        xh = c.positions() if len(bound) > 1 else None
        for a, b in zip(bound[:-1], bound[1:]):
            rest = 0.5*(c.plies[a.ply]["thickness"] + c.plies[b.ply]["thickness"]) + P.BIND_MAX_GAP
            # the interlock is the thread loop around the stack: its stiffness is the thread's EA over that length. At
            # loop pickup the needle is still in the cloth and the loop is still being fed: it is created slack, at
            # least as long as the holes' present separation, and drawn in by the take-up once the needle is out
            # (_draw_interlocks), as the thread elements are. Created at the stack's rest length, holes
            # 0.13 mm (median) to 0.49 mm farther apart started it at 23 N (median) to 87 N, against a thread tension
            # of ~0.3 N and a thread that breaks at ~6 N
            pa = np.dot(a.bary, xh[c.tri_host[a.face]])
            pb = np.dot(b.bary, xh[c.tri_host[b.face]])
            r0 = max(rest, float(np.linalg.norm(pb - pa)))
            con_ids.append(c.add_constraint(CON_BIND, fa=a.face, ba=a.bary, fb=b.face, bb=b.bary, rest=r0,
                                            comp=r0/P.THREAD_EA, meta=dict(kind="bind", cycle=self.cycle)))
            bind_rest.append(rest)
        il = dict(cycle=self.cycle, top=(order[0].face, order[0].bary, order[0].ply),
                  bottom=(order[-1].face, order[-1].bary, order[-1].ply), binds=con_ids, bind_rest=bind_rest,
                  drawn=False, locked=False,
                  holes=[self.holes.index(h) for h in order], plies=sorted({h.ply for h in bound}),
                  reverse=self.reverse_now)
        T_n = P.thread_set_tension(self.s.needle_tension, self.speed_ramped, self.s.fabric_tension_factor)
        T_b = P.thread_set_tension(self.s.bobbin_tension, self.speed_ramped, self.s.fabric_tension_factor)
        # at the interlock the needle thread pulls with its set tension less the friction of its path (eye, hole); the
        # seam's needle-thread elements carry that pull, the bobbin's the bobbin tension. At the default dial pair the
        # two are equal, so a balanced seam has no tension couple across the stack
        T_ni = T_n/P.NEEDLE_PATH_LOSS
        ratio = T_ni/T_b
        band = math.exp(P.THREAD_CAPSTAN_MU*math.pi)
        il["balance"] = "balanced" if 1.0/band <= ratio <= band else ("top_tight" if ratio > band else "bottom_tight")
        il["set_tension_N"] = (T_n, T_b)
        il["seam_tension_N"] = (T_ni, T_b)
        prev = self.interlocks[-1] if self.interlocks else None
        if self.seam_break:                   # threads were trimmed since the last stitch: a new seam begins
            prev = None
            self.seam_break = False
            self.seam += 1
            self.tether = None
        il["seam"] = self.seam
        if prev is not None:
            x = c.positions()
            for kind, end_new, end_old, T in (("needle", il["top"], prev["top"], T_ni),
                                              ("bobbin", il["bottom"], prev["bottom"], T_b)):
                pa = np.dot(end_old[1], x[c.tri_host[end_old[0]]])
                pb = np.dot(end_new[1], x[c.tri_host[end_new[0]]])
                Lk = float(np.linalg.norm(pb - pa))
                # at loop pickup the thread is still being fed: the element starts slack and is set to its tension
                # by the take-up once the needle has left the cloth (_set_stitches)
                kc = c.add_constraint(CON_THREAD, fa=end_old[0], ba=end_old[1], fb=end_new[0], bb=end_new[1],
                                      rest=Lk, comp=Lk/P.THREAD_EA, meta=dict(kind=kind, cycle=self.cycle))
                self.threads.append(dict(con=kc, kind=kind, cycle=self.cycle, rest=Lk, broken=False, tension=0.0,
                                         set_tension=T, set=False))
        self.interlocks.append(il)
        rec["formed"] = True
        rec["interlock"] = len(self.interlocks) - 1
        rec["balance"] = il["balance"]

    def _set_stitches(self, x):
        """Take-up: with the needle out of the cloth the new thread elements are drawn to their set tension (the
        element's rest length becomes the current length shortened by T / EA)."""
        c = self.cloth
        pending = [t for t in self.threads if not t["set"] and not t["broken"]]
        if not pending:
            return
        if x is None:
            x = c.positions()
        for t in pending:
            r = c.con_host[t["con"]]
            pa = np.dot(r["ba"], x[c.tri_host[r["fa"]]])
            pb = np.dot(r["bb"], x[c.tri_host[r["fb"]]])
            L = float(np.linalg.norm(pb - pa))
            rest = L/(1.0 + t["set_tension"]/P.THREAD_EA)
            c.set_constraint(t["con"], rest=rest, comp=rest/P.THREAD_EA)
            t["rest"] = rest
            t["set"] = True

    def _draw_interlocks(self, x, take_up):
        """Take-up of the interlocks: with the needle out of the cloth each new interlock is drawn toward the stack's
        rest length pulling with at most the needle thread's tension at the interlock (CON_BIND with a cap: a Huber
        energy, so the solve sees a constant pull), and locked (the cap removed: the set stitch holds as the thread's
        EA) once its tension is within the cap at the present separation."""
        c = self.cloth
        for il in self.interlocks:
            if il.get("locked", True):           # locked, or made before interlocks were drawn in (old snapshots)
                continue
            if not il["drawn"]:
                if not take_up:
                    continue
                T = float(il["seam_tension_N"][0])
                for kc, rest in zip(il["binds"], il["bind_rest"]):
                    c.set_constraint(kc, rest=rest, comp=rest/P.THREAD_EA, cap=T)
                il["drawn"] = True
                continue
            if x is None:
                x = c.positions()
            done = True
            for kc in il["binds"]:
                r = c.con_host[kc]
                if r["cap"] <= 0.0:
                    continue
                pa = np.dot(r["ba"], x[c.tri_host[r["fa"]]])
                pb = np.dot(r["bb"], x[c.tri_host[r["fb"]]])
                if (float(np.linalg.norm(pb - pa)) - r["rest"])/r["comp"] <= r["cap"]:
                    c.set_constraint(kc, cap=0.0)
                else:
                    done = False
            il["locked"] = done

    def _threads_check(self, tick, forces):
        c = self.cloth
        brk = (1 - P.THREAD_SEW_LOSS)*P.THREAD_BREAK*P.THREAD_LOOP
        for t in self.threads:
            if t["broken"]:
                continue
            T = float(np.linalg.norm(forces[t["con"]])) if t["con"] < len(forces) else 0.0
            t["tension"] = T
            if T > brk:
                t["broken"] = True
                c.set_constraint(t["con"], on=0)
                self.faults.append(dict(tick=tick, fault="thread_break", kind=t["kind"], cycle=t["cycle"], tension_N=T))

    def _update_tether(self, x):
        """Thread from the last stitch up to the needle eye and down to the bobbin (CON_TETHER: a tension-only pull
        capped at the static tension). While the shaft turns, the take-up lever supplies and takes up this thread
        and sets each stitch (the thread elements' set tension); it does not pull the seam, so the tether is slack.
        When the shaft stops, the thread hangs as the take-up left it: the tether's rest length is its length at
        that moment, and pulling the work away pays thread out through the tension discs at the static tension.
        A constant pull at the static tension while sewing (rest length 0) drew the seam's needle-thread chain taut
        back to the first stitch and curled the stitched stack up behind the foot."""
        c = self.cloth
        if not self.interlocks or self.tether == "trimmed":
            return
        il = self.interlocks[-1]
        eye = np.array([0.0, 0.0, float(self.needle_tip(self.theta)) + EYE_ABOVE_POINT])
        c.set_particle_target(self.p_eye, eye)
        if self.tether is None or self.tether[2] != il["cycle"]:
            if isinstance(self.tether, tuple):
                c.set_constraint(self.tether[0], on=0)
                c.set_constraint(self.tether[1], on=0)
            kn = c.add_constraint(CON_TETHER, fa=il["top"][0], ba=il["top"][1], pb=self.p_eye, rest=0.0, comp=1e-4,
                                  cap=self.s.needle_tension, meta=dict(kind="tether_needle"))
            kb = c.add_constraint(CON_TETHER, fa=il["bottom"][0], ba=il["bottom"][1], pb=self.p_bobbin, rest=0.0,
                                  comp=1e-4, cap=self.s.bobbin_tension, meta=dict(kind="tether_bobbin"))
            c.set_constraint(kn, on=0)
            c.set_constraint(kb, on=0)
            self.tether = (kn, kb, il["cycle"], False)
        live = self.mode == "idle"
        if live != self.tether[3]:
            kn, kb = self.tether[0], self.tether[1]
            if live:
                for k, p in ((kn, eye), (kb, x[self.p_bobbin])):
                    r = c.con_host[k]
                    pa = np.dot(r["ba"], x[c.tri_host[r["fa"]]])
                    c.set_constraint(k, on=1, rest=float(np.linalg.norm(p - pa)))
            else:
                c.set_constraint(kn, on=0)
                c.set_constraint(kb, on=0)
            self.tether = (kn, kb, self.tether[2], live)

    def _trim(self):
        c = self.cloth
        self.trim_request = False
        if isinstance(self.tether, tuple):
            c.set_constraint(self.tether[0], on=0)
            c.set_constraint(self.tether[1], on=0)
            self.tether = "trimmed"
        self.seam_break = True
        self.tails["end"] = P.TRIM_TAIL
        self.log.append(dict(event="trim", cycle=self.cycle))

    # --------------------------------------------------------------------------------------------- snapshots
    _SHARED = ("cloth", "fabric", "ply_tris", "s")

    def snapshot(self):
        import copy
        return copy.deepcopy({k: v for k, v in vars(self).items() if k not in self._SHARED})

    def restore(self, snap):
        import copy
        for k, v in copy.deepcopy(snap).items():
            setattr(self, k, v)

    # ----------------------------------------------------------------------------------------------- readout
    def status(self):
        """What a cell controller reads from the head (policy proprioception)."""
        return dict(shaft_angle_rad=float(self.theta % TWO_PI), speed_spm=float(self.speed_ramped),
                    needle="down" if float(self.needle_tip(self.theta)) < 0.0 else "up", presser=self.presser,
                    reverse=bool(self.reverse), mode=self.mode,
                    fault=next((f["fault"] for f in reversed(self.faults) if "fault" in f), None),
                    stitch_counter=int(sum(1 for s in self.stitches if s["formed"])))

    def ledger(self):
        return dict(settings=asdict(self.s), holes=[asdict(h) for h in self.holes], stitches=self.stitches,
                    interlocks=self.interlocks, threads=self.threads, faults=self.faults, tails=self.tails,
                    needle_state=self.needle_state, log=self.log)
