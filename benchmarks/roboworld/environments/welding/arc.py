"""The GMAW arc as an electrical circuit and a heat source.

Constant-voltage source, cycle-averaged:
    V_set - S I = V0 + Ra I + Ea L_a + I (R_cable + rho_w L_s / A_w)          (loop, free flight; linear in I)
    dL_s/dt = w_f - (a I + b L_s I^2)                                          (Lesnewich-form wire melting)
    L_a = d_tw - L_s                                                           (d_tw: contact tip to the arc root)
Short circuit (dip): the arc cannot be longer than the droplet that bridges it, so L_a = L_dip; the stick-out is
d_tw - L_dip, the current is the one that melts the wire as fast as it is fed (a I + b L_s I^2 = w_f), and the source's
voltage sets the fraction f of the time the wire is shorted: V_load = (1 - f)(V0 + Ra I + Ea L_dip) (the short itself
drops nothing beyond the loop's resistances). At the free-flight arc length L_dip both laws give the same point (f = 0),
so the current is continuous across the boundary.
The arc exists only while the source is commanded on, the wire's path to the work clamp is closed through touching
metal (`circuit_closed`), and the loop has an operating point (I above I_MIN).
Striking: with the arc commanded on and no arc, the wire inches toward the work at the run-in speed; it strikes when
it touches grounded metal (short circuit), as a robot GMAW start does.
Stray arcs: the grounded metal need not be the coupon. A wire that touches any conductor at work potential -- the
steel fixture table the ground clamp sits on, the zinc-plated toggle clamps bolted to it, the ground clamp itself
(world.py decides which bodies are connected) -- strikes there with the output on, and the source runs exactly as on the
coupon: the same loop, melting law, transfer mode and limits (burnback, stubbing, the current ceiling), with the arc
as long as the distance from the contact tip to that body's surface along the wire. Such a body lies outside the
thermal grid, so `ArcState.stray_node` names it and the source books what the arc does there instead of passing it to
the coupon: the wire metal melted onto it (`stray_m3`, at the transfer mode's efficiency; the rest is spatter as
always) and the arc's heat eta U I (`stray_j`); `heat()` gives the coupon nothing. The deposit's shape and the heated
body's temperature are not modelled (no grid there). Painted or insulated bodies, ungrounded steel and free air close
no circuit: the wire stalls against them or runs to its bound and the source reports an arc start failure.
Wire is conserved: what melts is what was fed less the change in stick-out, never more. Limits of a real source and
wire:
  burnback   -- the arc melts the wire faster than it is fed until the stick-out is gone: the arc climbs to the contact
                tip and fuses the wire to it; the arc goes out with a 'burnback' fault that switching the output off and
                on does not clear (the wire is welded to the tip; clearing it is a maintenance job the cell cannot do);
  ceiling    -- above SOURCE_I_MAX the source holds the current and its voltage falls below the set point;
  stubbing   -- in dip transfer, shorted more than STUB_SHORT_FRAC of the time, or with no current within the ceiling
                that melts the wire as fast as it is fed, the wire drives into the work: the arc goes out with a
                'stubbing' fault and the drive's push (the stick-out's buckling load, at most WIRE_PUSH_MAX, for
                STUB_PUSH_S) reacts on the gun.
A long arc is bounded by the contact tip, not by a length of its own: raising the voltage lengthens the arc until the
stick-out is gone (burnback). The arc goes out only below the source's least current (I_MIN). Touch sensing: with the arc off and sensing on,
the source reports contact when the stick-out wire touches grounded metal.

Transfer mode: not a setting. It follows from the operating point (`transfer_mode`): with the commanded wire feed and
voltage at the present contact-tip-to-work distance, a free-flight arc shorter than a droplet (about one wire diameter)
cannot stand -- the drops bridge it and the transfer is short circuit (dip); a free-flight arc above the transition
current in an argon-rich atmosphere is spray; otherwise globular. Air under the arc dilutes the argon, and below the
argon fraction spray needs there is no spray.

Heat: the work receives eta (V_terminal I). The droplets carry the deposited wire's enthalpy at the droplet
temperature (into a cylinder below the arc root for spray, into the surface under the wire for short circuit and
globular); the rest is a Gaussian surface flux about the arc root. The arc attaches where the wire axis meets metal.

Shielding (phenomenological, anchored to TWI's 1 % / 1.5 % air thresholds and the trade's draft and angle limits):
air fraction at the pool from the nozzle-to-work distance, the travel angle and the cross draft.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from . import params as PR

I_MIN = 20.          # A: the source's lowest output current (Auto-Continuum 350: 20-400 A [read: Miller AU/10.0]);
                     # below it no operating point
RUN_IN = 2.5/60.     # m/s: wire run-in (inching) speed while striking (robot GMAW 'creep' start; estimate)
MAX_STICKOUT = .100  # m: inching stops at this free stick-out. A model bound, not a device: the wire's sag is not
                     # modelled [authored]
STRIKE_REACH = .040  # m: farthest metal the inching wire can reach before the source gives up (estimate)
STRIKE_TIMEOUT = 2.0  # s: no arc after this long -> 'arc start failure' (source fault). Hirebotics' UR + Miller
                      # integration waits 2 s for the arc current [read: 'Miller - Arc Not Started', 2026-10-10]
LA_MIN = .5e-3       # m: shortest cycle-averaged arc (short circuits are averaged)
MIN_STICKOUT = .2e-3 # m: the shortest stick-out the model keeps (the wire's end at the contact tip's bore)
CONE_DEG = 8.        # fallback search cone half angle about the wire axis (estimate), used only when the axis itself
                     # meets no metal. Taking the nearest hit of the whole cone puts the root alternately on the bracket
                     # and on the base 3-5 mm from the corner, and the corner never fuses
STICKOUT_NOMINAL_FRAC = .7   # initial electrical stickout as a fraction of the commanded CTWD at arc start


@dataclass
class Settings:
    wfs_m_min: float = 12.7
    voltage_v: float = 29.
    gas_l_min: float = 19.
    gas_ar: float = .90             # argon fraction of the cell's shielding gas (90 % Ar / 10 % CO2, the procedure's)


@dataclass
class ArcState:
    on_cmd: bool = False            # source output commanded (robot I/O)
    sensing: bool = False           # touch sensing mode commanded
    gas_on: bool = False            # the valve is closed at rest
    arc: bool = False               # arc established
    current: float = 0.
    v_term: float = 0.
    v_arc: float = 0.
    l_s: float = .010               # m, electrical stickout
    l_a: float = 0.
    d_tw: float = float("inf")      # contact tip to arc root (m)
    root: np.ndarray = field(default_factory=lambda: np.zeros(3))
    root_body: int = 0
    wfs_actual: float = 0.          # m/s
    striking_s: float = 0.
    touch: bool = False
    fault: str = ""
    deposited_m3: float = 0.
    fed_m3: float = 0.
    spatter_m3: float = 0.
    arc_on_s: float = 0.
    stubbing: bool = False
    x_air: float = 0.
    inch: int = 0                   # wire inching with the output off: +1 forward, -1 backward, 0 stop (robot I/O)
    mode: str = ""                  # transfer mode while an arc burns: "spray" | "globular" | "short_circuit" | ""
    short_frac: float = 0.          # fraction of the time the wire is shorted (dip transfer; cycle average)
    l_s0: float = .010              # stick-out when this state began: the wire balance is fed = deposited + spatter
                                    # + (l_s - l_s0) A_w (Process.ledger)
    current_limited: bool = False   # the source holds its output at its current ceiling (SOURCE_I_MAX)
    push_s: float = 0.              # s left of the wire drive pushing a stubbed wire into the work
    stub_push_n: float = 0.         # the push the stubbed wire carries (N)
    wire_push_n: float = 0.         # that push's force on the gun along -wire axis now (N; the world applies it)
    stray_node: object = None       # the grounded body outside the coupon's grid the wire faces now (set by the world
                                    # each tick: -2 the fixture table and what is bolted to it, else a body id); None: the
                                    # coupon (or nothing grounded)
    stray_m3: float = 0.            # wire metal melted onto such bodies (never reaches the coupon's pool); the wire
                                    # balance is fed = deposited + spatter + stray + (l_s - l_s0) A_w
    stray_j: float = 0.             # arc heat delivered to such bodies, eta U I dt (J; their temperature is not modelled)


class CVSource:
    """Robot GMAW power source (constant voltage) with the cycle-averaged wire/arc dynamics."""

    def __init__(self, settings=None):
        self.s = settings or Settings()
        self.st = ArcState()
        self.a = PR.v(PR.LESNEWICH_A)
        self.b = PR.v(PR.LESNEWICH_B)
        self.ra, self.ea = PR.v(PR.ARC_RA), PR.v(PR.ARC_EA)
        self.slope, self.rc = PR.v(PR.CV_SLOPE), PR.v(PR.CABLE_R)
        self.rho_w = PR.v(PR.WIRE_RESISTIVITY)
        d = PR.v(PR.WIRE_D)
        self.aw = math.pi*d*d/4
        self.ocv = PR.v(PR.OCV)

    @property
    def v0(self):
        return PR.v(PR.ARC_V0)

    def melt_current(self, l_s):
        """Current that melts the wire as fast as it is fed at stick-out l_s (a I + b l_s I^2 = w_f)."""
        wf = self.s.wfs_m_min/60.
        q = self.b*max(l_s, 0.)
        if q <= 0.:
            return wf/self.a
        return (-self.a + math.sqrt(self.a*self.a + 4.*q*wf))/(2.*q)

    def short_fraction(self, I, l_s):
        """Fraction of the time the wire is shorted in dip transfer at current I and stick-out l_s (cycle average)."""
        v_load = self.s.voltage_v - self.slope*I - I*(self.rc + self.rho_w*l_s/self.aw)
        v_arc = self.v0 + self.ra*I + self.ea*PR.v(PR.DIP_ARC_LENGTH)
        return 1. - v_load/v_arc

    # ---------------------------------------------------------------- transfer mode
    def _steady_free_flight(self, ctwd):
        """Steady free-flight operating point (I, L_s, L_a) at a contact-tip-to-work distance, cached per 0.05 mm and
        set point (the mode decision runs every tick)."""
        key = (round(ctwd/5e-5), self.s.wfs_m_min, self.s.voltage_v)
        cache = self.__dict__.setdefault("_ff_cache", {})
        if key not in cache:
            if len(cache) > 4096:
                cache.clear()
            cache[key] = self.steady(ctwd, regime="free_flight")
        return cache[key]

    def transfer_mode(self, d_tw, x_air=0.):
        """Transfer mode at the commanded set points, the contact tip `d_tw` from the arc root and air fraction
        `x_air` under the arc, with a small hysteresis about the present mode (no chattering at a boundary)."""
        prev = self.st.mode
        ff = self._steady_free_flight(d_tw) if math.isfinite(d_tw) else None
        l_dip = PR.v(PR.DIP_ARC_LENGTH) + (.1e-3 if prev == "short_circuit" else -.1e-3 if prev else 0.)
        if ff is None or ff[2] < l_dip:
            return "short_circuit"
        ar = self.s.gas_ar*(1. - x_air)
        i_t = PR.v(PR.TRANSITION_CURRENT) + (-3. if prev == "spray" else 3. if prev else 0.)
        if ff[0] >= i_t and ar >= PR.v(PR.SPRAY_MIN_AR):
            return "spray"
        return "globular"

    # ---------------------------------------------------------------- circuit
    def loop_current(self, l_a, l_s):
        r = self.slope + self.ra + self.rc + self.rho_w*l_s/self.aw
        return (self.s.voltage_v - self.v0 - self.ea*l_a)/r

    def steady(self, ctwd, regime=None):
        """Steady operating point at a given contact-tip-to-work distance: (I, L_s, L_a) or None. `regime`
        ("free_flight" | "short_circuit") fixes the arc law; by default the law of the transfer mode the operating
        point itself selects (`transfer_mode`)."""
        if regime is None:
            mode = self.transfer_mode(ctwd)
            regime = "short_circuit" if mode == "short_circuit" else "free_flight"
        if regime == "short_circuit":
            l_dip = PR.v(PR.DIP_ARC_LENGTH)
            l_s = max(MIN_STICKOUT, ctwd - l_dip)
            I = self.melt_current(l_s)
            return (float(I), float(l_s), float(ctwd - l_s)) if self.short_fraction(I, l_s) < 1. else None
        v0 = self.v0
        wf = self.s.wfs_m_min/60.
        l_s = np.linspace(.2e-3, ctwd - LA_MIN, 400)
        l_a = ctwd - l_s
        I = (self.s.voltage_v - v0 - self.ea*l_a)/(self.slope + self.ra + self.rc + self.rho_w*l_s/self.aw)
        I = np.minimum(I, PR.v(PR.SOURCE_I_MAX))
        ok = I > I_MIN
        if not ok.any():
            return None
        res = np.where(ok, np.abs(wf - (self.a*I + self.b*l_s*I*I)), np.inf)
        k = int(np.argmin(res))
        return float(I[k]), float(l_s[k]), float(l_a[k])

    @staticmethod
    def _book(st, melted_m3, eta):
        """Book wire melted this step: the weld metal (eta of it) onto the coupon -- or onto the stray body the arc
        burns on -- and the rest as spatter."""
        if st.stray_node is None:
            st.deposited_m3 += melted_m3*eta
        else:
            st.stray_m3 += melted_m3*eta
        st.spatter_m3 += melted_m3*(1 - eta)

    def _stray_heat(self, st, dt):
        """The arc's heat on a stray body: what heat() would give the coupon, booked instead (the body has no grid)."""
        if st.stray_node is not None and st.arc:
            st.stray_j += PR.v(PR.ARC_ETA)*st.v_term*st.current*dt

    def _stub(self, st):
        """The wire drives into the work (dip transfer with too little voltage, or more wire than the current ceiling
        melts): the arc goes out, the source faults, and the feeder's push reacts on the gun until it stops."""
        st.arc = False
        st.current = 0.
        st.stubbing = True
        st.fault = "stubbing"
        st.push_s = PR.v(PR.STUB_PUSH_S)
        # the stubbed stick-out is a column fixed in the contact tip and free on the plate: it carries at most its Euler
        # buckling load, and the drive at most what its rolls give before they slip
        d = PR.v(PR.WIRE_D)
        ei = PR.v(PR.WIRE_E)*math.pi*d**4/64.
        st.stub_push_n = float(min(math.pi**2*ei/(4.*max(st.l_s, 1e-3)**2), PR.v(PR.WIRE_PUSH_MAX)))

    def update(self, dt, d_tw, circuit_closed, tip_touching, x_air=0.):
        """Advance the source one step. d_tw: contact tip to the nearest grounded metal along the attachment search
        (m, inf if none in reach); tip_touching: the stick-out wire (at its current length) touches metal; x_air: air
        fraction under the arc (shielding), which decides whether spray transfer is possible."""
        st, s = self.st, self.s
        if not st.arc:
            st.mode = ""
        st.d_tw = d_tw
        wf = s.wfs_m_min/60.
        i_max = PR.v(PR.SOURCE_I_MAX)
        st.current_limited = False
        # a stubbed wire is pushed into the work by the drive until the source stops it (the world applies the force)
        st.wire_push_n = st.stub_push_n if st.push_s > 0 else 0.
        st.push_s = max(0., st.push_s - dt)
        st.touch = bool(st.sensing and not st.on_cmd and circuit_closed and tip_touching)
        if not st.on_cmd and st.arc:
            st.arc = False
        if not st.arc and math.isfinite(d_tw) and st.l_s > d_tw and st.fault != "burnback":
            # the torch was brought closer than the stick-out with no arc to burn it back (output off, striking into an
            # open circuit, or faulted): the wire is pushed back into the contact tip (a stiff wire's buckling is not
            # modelled). Applied with the output off only, the wire stood 0.4 mm inside the plate when the torch settled
            # after it had stalled against an open circuit. The wire moves back up
            # the liner: negative feed
            back = st.l_s - max(.2e-3, d_tw)
            st.l_s -= back
            st.fed_m3 -= back*self.aw
        if st.fault == "burnback":
            # the wire is welded to the contact tip: nothing feeds, inches or arcs until the cell is reset
            st.current = st.v_arc = st.wfs_actual = 0.
            st.v_term = 0. if not st.on_cmd else self.ocv
            st.striking_s = 0.
            return
        if not st.on_cmd:
            # wire inching (robot I/O, output off): the feeder drives the wire at the run-in speed; forward it stops
            # against metal (the drive slips), backward it retracts into the contact tip
            run = 0.
            if st.inch > 0:
                run = min(RUN_IN*dt, max(0., MAX_STICKOUT - st.l_s))
                if math.isfinite(d_tw) and st.l_s + run > d_tw:
                    run = max(0., d_tw - st.l_s)
                st.l_s += run
                st.fed_m3 += run*self.aw
            elif st.inch < 0:
                run = -min(RUN_IN*dt, max(0., st.l_s - .2e-3))
                st.l_s += run
                st.fed_m3 += run*self.aw
            st.current = st.v_arc = 0.
            # output off; in touch-sensing mode the source puts its low sensing voltage on the wire and reads it
            # collapse on contact
            st.v_term = (0. if st.touch else PR.v(PR.TOUCH_SENSE_V)) if st.sensing else 0.
            st.wfs_actual = run/dt
            st.striking_s = 0.
            st.stubbing = False
            return
        if st.fault:
            st.current, st.wfs_actual = 0., 0.
            return
        if not st.arc:
            # striking: the wire inches toward the work and arcs on touching grounded metal
            st.striking_s += dt
            # the wire inches toward the work; against metal it stops and the feeder's drive slips (with no circuit
            # it would otherwise run on through the plate)
            # the run-in stops at the model's free stick-out bound as inching does (the wire's sag is not modelled;
            # repeated strikes with nothing in reach would stack it to 129 mm: E20)
            run = min(RUN_IN*dt, max(0., MAX_STICKOUT - st.l_s))
            if math.isfinite(d_tw) and st.l_s + run > d_tw:
                run = max(0., d_tw - st.l_s)
                st.stubbing = True
            st.wfs_actual = run/dt
            st.l_s += run
            st.fed_m3 += run*self.aw
            if circuit_closed and math.isfinite(d_tw) and d_tw >= LA_MIN + MIN_STICKOUT and st.l_s >= d_tw - LA_MIN:
                st.arc = True
                # the short that starts the arc burns the wire's end off: it leaves as strike spatter
                burn = max(0., st.l_s - max(.2e-3, d_tw - 1.5e-3))
                st.l_s -= burn
                st.spatter_m3 += burn*self.aw
                st.striking_s = 0.
                st.mode = self.transfer_mode(d_tw, x_air)
            elif st.striking_s > STRIKE_TIMEOUT:
                st.fault = "arc_start_failure"
            st.current, st.v_term = 0., self.ocv
            return
        # arc established: loop + stickout dynamics. With the contact tip closer to metal than the shortest arc on the
        # shortest stick-out there is no room for an arc: the tip shorts to the work and the arc goes out (the
        # stick-out's clamp would have no lower bound left and go to -0.4 mm: E20)
        if not circuit_closed or not math.isfinite(d_tw) or d_tw < LA_MIN + MIN_STICKOUT:
            st.arc = False
            st.current = 0.
            return
        st.mode = self.transfer_mode(d_tw, x_air)
        st.wfs_actual = wf
        st.fed_m3 += wf*dt*self.aw
        l_old = st.l_s
        if st.mode == "short_circuit":
            # dip: the bridging drop holds the arc at L_dip; the current melts the wire as fast as it is fed; the
            # voltage sets the shorted fraction. Shorted too long (too little voltage), or needing more current than the
            # source gives, the wire stubs into the work
            l_s = max(MIN_STICKOUT, d_tw - PR.v(PR.DIP_ARC_LENGTH))
            l_s = min(l_s, l_old + wf*dt)                 # the wire cannot outgrow its own feed
            I = self.melt_current(l_s)
            f = self.short_fraction(I, l_s)
            if I > i_max or f >= PR.v(PR.STUB_SHORT_FRAC):
                consumed = max(0., wf*dt + l_old - l_s)   # the stub's wire goes into the work
                st.l_s = l_s
                self._book(st, consumed*self.aw, 1.)
                self._stub(st)
                return
            st.stubbing = False
            if I <= I_MIN:
                st.arc = False
                st.current = 0.
                return
            f = max(0., f)
            st.short_frac = f
            st.l_s, st.l_a = l_s, d_tw - l_s
            st.current = I
            st.v_term = s.voltage_v - self.slope*I
            st.v_arc = (1. - f)*(self.v0 + self.ra*I + self.ea*st.l_a)
            consumed = max(0., wf*dt + l_old - l_s)
            self._book(st, consumed*self.aw, PR.v(PR.DEPOSITION_EFF)[st.mode])
            st.arc_on_s += dt
            self._stray_heat(st, dt)
            return
        st.short_frac = 0.
        l_s = min(st.l_s, d_tw - LA_MIN)
        st.stubbing = st.l_s > d_tw - LA_MIN
        l_a = d_tw - l_s
        I = self.loop_current(l_a, l_s)
        if I <= I_MIN:
            # no operating point: the arc goes out (arc too long for the open-circuit voltage)
            st.l_s = l_old + wf*dt
            st.arc = False
            st.current = 0.
            st.fault = "" if st.striking_s == 0 else st.fault
            return
        if I > i_max:
            I = i_max
            st.current_limited = True
        melt = self.a*I + self.b*l_s*I*I
        l_new = l_s + (wf - melt)*dt
        if l_new < MIN_STICKOUT:
            # burnback: the arc melts the wire faster than it is fed and has burnt the stick-out away; it climbs to the
            # contact tip and welds the wire to it. What melted this step is the stick-out that was left plus the feed
            consumed = wf*dt + l_old - MIN_STICKOUT
            self._book(st, consumed*self.aw, PR.v(PR.DEPOSITION_EFF)[st.mode])
            st.l_s = MIN_STICKOUT
            st.arc = False
            st.current = 0.
            st.fault = "burnback"
            return
        st.l_s = float(min(l_new, d_tw - LA_MIN))
        st.l_a = d_tw - st.l_s
        st.current = I
        if st.current_limited:
            # at its ceiling the source supplies the voltage the load needs at that current
            st.v_term = self.v0 + self.ea*st.l_a + I*(self.ra + self.rc + self.rho_w*st.l_s/self.aw)
        else:
            st.v_term = s.voltage_v - self.slope*I
        st.v_arc = self.v0 + self.ra*I + self.ea*st.l_a
        consumed = max(0., wf*dt + l_old - st.l_s)       # wire is conserved: fed less the stick-out's growth
        self._book(st, consumed*self.aw, PR.v(PR.DEPOSITION_EFF)[st.mode])
        st.arc_on_s += dt
        self._stray_heat(st, dt)

    def heat(self):
        """(total heat to the coupon W, droplet mass rate kg/s) for the current step; nothing while the arc burns on a
        stray body (its heat and metal are booked in the source state: update)."""
        st = self.st
        if not st.arc or st.stray_node is not None:
            return 0., 0.
        I = st.current
        melt = self.a*I + self.b*st.l_s*I*I
        if st.mode == "short_circuit":
            melt = min(melt, self.s.wfs_m_min/60.)          # dip: the current melts the wire as it is fed
        mdot = PR.v(PR.STEEL_RHO)*self.aw*melt*PR.v(PR.DEPOSITION_EFF)[st.mode]
        return PR.v(PR.ARC_ETA)*st.v_term*I, mdot


# -------------------------------------------------------------------------------------------- arc attachment
def cone_dirs(axis, half_deg=CONE_DEG, rings=(0., 4., 8.), per_ring=(1, 6, 10)):
    axis = np.asarray(axis, float)
    axis = axis/np.linalg.norm(axis)
    tmp = np.array([1., 0, 0]) if abs(axis[0]) < .9 else np.array([0, 1., 0])
    u = np.cross(axis, tmp)
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    out = []
    for ang, n in zip(rings, per_ring):
        if ang > half_deg:
            continue
        a = math.radians(ang)
        for k in range(n):
            phi = 2*math.pi*k/n
            out.append(math.cos(a)*axis + math.sin(a)*(math.cos(phi)*u + math.sin(phi)*w))
    return np.asarray(out)


def attach(grid, mat_host, body_host, p_tip, axis, reach=STRIKE_REACH, step=2e-4):
    """Arc root (point, distance from the contact tip, body id) for a wire from p_tip along `axis` (weld frame):
    where the wire axis meets metal (the spray column is stiff along the wire: the bead goes where the wire is aimed,
    and the Gaussian heat spread about that point covers both members of a fillet); if the axis meets nothing within
    `reach`, the nearest metal within the cone (the arc jumps to an edge it passes). (None, inf, 0) if nothing."""
    dirs = cone_dirs(axis)
    s = np.arange(step, reach + step, step)
    P = p_tip[None, None, :] + dirs[:, None, :]*s[None, :, None]          # [rays, steps, 3]
    flat = P.reshape(-1, 3)
    inside = ((flat[:, 0] >= grid.xe[0]) & (flat[:, 0] < grid.xe[-1]) & (flat[:, 1] >= grid.ye[0]) &
              (flat[:, 1] < grid.ye[-1]) & (flat[:, 2] >= grid.ze[0]) & (flat[:, 2] < grid.ze[-1]))
    i = np.clip(np.searchsorted(grid.xe, flat[:, 0]) - 1, 0, grid.nx - 1)
    j = np.clip(np.searchsorted(grid.ye, flat[:, 1]) - 1, 0, grid.ny - 1)
    k = np.clip(np.searchsorted(grid.ze, flat[:, 2]) - 1, 0, grid.nz - 1)
    idx = (i*grid.ny + j)*grid.nz + k
    metal = inside & (mat_host[idx] != 0)
    metal = metal.reshape(len(dirs), len(s))
    if not metal.any():
        return None, float("inf"), 0
    first = np.where(metal.any(1), metal.argmax(1), len(s))
    r = 0 if metal[0].any() else int(np.argmin(first))          # ray 0 is the wire axis
    hit_k = int(first[r])
    point = P[r, hit_k] - dirs[r]*step*.5
    return point, float(s[hit_k] - step*.5), int(body_host[idx.reshape(len(dirs), len(s))[r, hit_k]])

# -------------------------------------------------------------------------------------------- shielding
def air_fraction(ntwd, travel_angle_deg, draft_m_s, gas_on=True):
    """Air fraction in the shielding gas over the pool (phenomenological). Each deficit adds: the
    stand-off beyond the jet's core and the travel angle beyond its published limit (authored slopes: each limit at zero
    air, the 1 % porosity onset one nozzle diameter / 12.5 deg past it), and the cross draft (anchored to Kobelco's
    blowhole chart read at the cell's 19 L/min gas flow: 1 % air at 2.4 m/s, 1.5 % at 2.9 m/s, AIR_DRAFT). With the
    valve closed the arc burns in air (1.0)."""
    if not gas_on:
        return 1.
    D = PR.v(PR.NOZZLE_ID)
    core = PR.v(PR.SHIELD_CORE_D)
    x_d = PR.v(PR.AIR_PER_NOZZLE_D)*max(0., ntwd/D - core)
    x_a = PR.v(PR.AIR_PER_TRAVEL_DEG)*max(0., abs(travel_angle_deg) - PR.v(PR.TRAVEL_ANGLE_LIMIT))
    w0, w1 = PR.v(PR.AIR_DRAFT)
    x_w = w1*max(0., draft_m_s - w0)
    return min(1., x_d + x_a + x_w)


def porosity_fraction(x_air):
    """Pore volume fraction frozen into solidifying weld metal at air fraction x_air (phenomenological): none below
    TWI's 1 %, growing to 5 % at 2 % air and capped at 15 %."""
    lo, hi = PR.v(PR.AIR_POROSITY)
    if x_air <= lo:
        return 0.
    at2, cap = PR.v(PR.POROSITY_CURVE)
    return float(min(cap, at2*(x_air - lo)/(2*lo - lo)))
