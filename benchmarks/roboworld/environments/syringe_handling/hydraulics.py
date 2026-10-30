"""Liquid and gas inside the equipment: chambers, vessels, lines, depots, and the open air.

Everything here is CPU arithmetic on a handful of numbers per tick. A syringe's chamber has one state, its absolute
pressure; the path from its nozzle to wherever that ends has one state, its flow; the place it ends has at most one,
its pressure. One tick of the three is solved together and implicitly, with the plunger's speed left open, so that the
rigid solver receives the chamber's force as `F_bias + c_h v`:

    chamber   C_c dp_c/dt = A v - Q
    path      L dQ/dt = p_c - p_t - (R + 2 k |Q0|) Q - s          s: static head and the linearised kinetic term
    terminal  C_t dp_t/dt = Q - (p_t - p_ext)/R_leak               or p_t fixed

Volumes are m3, pressures Pa absolute unless a name says gauge, parcels are arrays of the volumes of
(saline, medication, contrast, blood, air). Liquid is incompressible; gas in a chamber or a vial is isothermal.
"""

import math
import numpy as np

from . import params as P

N_SPECIES = len(P.SPECIES) + 1          # the liquids and air
AIR = N_SPECIES - 1
ML = 1e-6
P_VAPOUR = 2.3e3                        # Pa: a liquid pulled below this opens a void (water at 20 C; literature)
ARRIVAL_BIN_S = 0.1
_RHO = np.array([P.LIQUIDS[s]["rho"] for s in P.SPECIES] + [P.AIR["rho"]])
_MU = np.array([P.LIQUIDS[s]["mu"] for s in P.SPECIES] + [P.AIR["mu"]])


def parcel(volume_m3, species):
    v = np.zeros(N_SPECIES)
    v[P.SPECIES.index(species) if species != "air" else AIR] = float(volume_m3)
    return v


def density(mix):
    t = float(mix.sum())
    return float(mix @ _RHO)/t if t > 0 else P.LIQUIDS["saline"]["rho"]


def viscosity(mix):
    """Volume-weighted logarithmic mixing of the parcel's liquids (air excluded unless it is all air)."""
    liq = mix[:AIR]
    t = float(liq.sum())
    if t <= 0:
        return P.AIR["mu"] if mix[AIR] > 0 else P.LIQUIDS["saline"]["mu"]
    return float(math.exp(float(liq @ np.log(_MU[:AIR]))/t))


def poiseuille(mu, length, bore):
    return 128.0*mu*length/(math.pi*bore**4)


def inertance(rho, length, bore):
    return rho*length/(math.pi*bore*bore/4)


class Ledger:
    """Where every microlitre is, by place; and what arrived in the vein and the tissue, when."""

    def __init__(self):
        self.places = {}
        self.arrivals = {}                # place -> list of (time, parcel)
        self.events = []

    def add(self, place, mix, t=None):
        if float(np.abs(mix).sum()) == 0.0:
            return
        self.places[place] = self.places.get(place, np.zeros(N_SPECIES)) + mix
        if t is not None:
            self.note(place, mix, t)

    def note(self, tag, mix, t):
        """Record an arrival (in 0.1 s bins) without moving anything: `place` arrivals come through add(); other tags
        are flows past a point (what has entered a line)."""
        rows = self.arrivals.setdefault(tag, [])
        k = int(t/ARRIVAL_BIN_S)
        if rows and rows[-1][2] == k:
            rows[-1] = (rows[-1][0], rows[-1][1] + mix, k)
        else:
            rows.append((float(t), np.array(mix, float), k))

    def volume(self, place, species=None):
        v = self.places.get(place)
        if v is None:
            return 0.0
        return float(v[:AIR].sum()) if species is None else float(v[AIR if species == "air" else P.SPECIES.index(species)])

    def rate_window(self, place, species, window_s, t_end=None):
        """Largest volume of `species` that arrived at `place` in any window of `window_s` seconds."""
        rows = self.arrivals.get(place, [])
        if not rows:
            return 0.0
        k = AIR if species == "air" else P.SPECIES.index(species)
        t = np.array([r[0] for r in rows])
        v = np.array([r[1][k] for r in rows])
        c = np.concatenate([[0.0], np.cumsum(v)])
        j = np.searchsorted(t, t - window_s + 0.5*ARRIVAL_BIN_S, side="left")
        return float((c[1:] - c[j]).max())

    def span(self, place, species, lo=0.05, hi=0.95):
        """Times at which the cumulative arrival of `species` at `place` passed the fractions lo and hi of its total."""
        rows = self.arrivals.get(place, [])
        k = AIR if species == "air" else P.SPECIES.index(species)
        if not rows:
            return None
        t = np.array([r[0] for r in rows])
        c = np.cumsum([r[1][k] for r in rows])
        if c[-1] <= 0:
            return None
        return float(t[np.searchsorted(c, lo*c[-1])]), float(t[min(np.searchsorted(c, hi*c[-1]), len(t) - 1)])

    def snapshot(self):
        return dict(places={k: v.copy() for k, v in self.places.items()},
                    arrivals={k: [(t, p.copy(), b) for t, p, b in v] for k, v in self.arrivals.items()}, events=list(self.events))

    def restore(self, s):
        self.places = {k: v.copy() for k, v in s["places"].items()}
        self.arrivals = {k: [(t, p.copy(), b) for t, p, b in v] for k, v in s["arrivals"].items()}
        self.events = list(s["events"])


class Stack:
    """Ordered liquid parcels; index 0 is at the port. What comes in last goes out first (a barrel) or last (a tube)."""

    def __init__(self, parcels=()):
        self.items = [np.array(p, float) for p in parcels if float(np.sum(p)) > 0]

    @property
    def volume(self):
        return float(sum(p.sum() for p in self.items))

    def total(self):
        return sum(self.items, np.zeros(N_SPECIES))

    def push_front(self, mix):
        if float(mix.sum()) <= 0:
            return
        if self.items and _same(self.items[0], mix):
            self.items[0] = self.items[0] + mix
        else:
            self.items.insert(0, np.array(mix, float))

    def push_back(self, mix):
        if float(mix.sum()) <= 0:
            return
        if self.items and _same(self.items[-1], mix):
            self.items[-1] = self.items[-1] + mix
        else:
            self.items.append(np.array(mix, float))

    def _pop(self, volume, front):
        out = np.zeros(N_SPECIES)
        need = float(volume)
        while need > 1e-18 and self.items:
            i = 0 if front else -1
            p = self.items[i]
            v = float(p.sum())
            if v <= need + 1e-18:
                out += p
                need -= v
                self.items.pop(i)
            else:
                f = need/v
                out += p*f
                self.items[i] = p*(1 - f)
                need = 0.0
        return out

    def pop_front(self, volume):
        return self._pop(volume, True)

    def pop_back(self, volume):
        return self._pop(volume, False)

    def front_mix(self):
        return self.items[0] if self.items else np.zeros(N_SPECIES)

    def mean_viscosity(self):
        """Resistance-weighted viscosity of a tube that holds this queue (parcels in series)."""
        t = self.volume
        return sum(float(p.sum())*viscosity(p) for p in self.items)/t if t > 0 else P.LIQUIDS["saline"]["mu"]

    def mean_density(self):
        t = self.volume
        return float(self.total() @ _RHO)/t if t > 0 else P.LIQUIDS["saline"]["rho"]

    def copy(self):
        return Stack(self.items)


def _same(a, b, tol=1e-3):
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= 0 or sb <= 0:
        return False
    return float(np.abs(a/sa - b/sb).max()) < tol


# ====================================================================== terminals: where a path ends
class Terminal:
    """Fixed pressure by default. `take(volume)` hands over what flows from it into the path; `give(mix, t)` receives."""
    kind = "fixed"
    compliance = math.inf
    leak_r = math.inf
    p_ext = P.P_ATM

    def pressure(self):
        return P.P_ATM

    def set_pressure(self, p):
        pass

    def give(self, mix, gas_at_atm, t):
        pass

    def take(self, volume):
        """-> (liquid mix, gas volume at this terminal's pressure)"""
        return np.zeros(N_SPECIES), float(volume)


class Closed(Terminal):
    kind = "closed"


class Atmosphere(Terminal):
    """An outlet in the open air: liquid leaves as drops or a jet (Spray), air comes back in."""
    kind = "air"

    def __init__(self, spray, origin_fn, bore, body=-1):
        self.spray, self.origin_fn, self.bore, self.body = spray, origin_fn, float(bore), body

    def give(self, mix, gas_at_atm, t):
        if float(mix[:AIR].sum()) > 0:
            self.spray.emit(self, mix, t)


class Reservoir(Terminal):
    """A large volume at a set pressure that also takes and gives liquid (the trainer's vein and its blood bag)."""
    kind = "reservoir"

    def __init__(self, ledger, place, gauge_pa, species="blood"):
        self.ledger, self.place, self.gauge, self.species = ledger, place, float(gauge_pa), species

    def pressure(self):
        return P.P_ATM + self.gauge

    def give(self, mix, gas_at_atm, t):
        self.ledger.add(self.place, mix, t)
        if gas_at_atm > 0:
            self.ledger.add(self.place, parcel(gas_at_atm, "air"), t)

    def take(self, volume):
        return parcel(volume, self.species), 0.0


class Vessel(Terminal):
    """A rigid closed vessel with liquid and a gas headspace (a vial). Whether the port is under the liquid is set by
    whoever knows the geometry (`wet`), each tick."""
    kind = "vessel"

    def __init__(self, interior_m3, liquid_mix, gas_at_atm_m3=None):
        self.interior = float(interior_m3)
        self.liquid = np.array(liquid_mix, float)
        v = float(self.liquid.sum())
        self.gas0 = float(self.interior - v if gas_at_atm_m3 is None else gas_at_atm_m3)     # m3 at one atmosphere
        self.wet = False
        self.head_pa = 0.0
        self.vented = False

    @property
    def liquid_volume(self):
        return float(self.liquid.sum())

    def gas_pressure(self):
        if self.vented:
            return P.P_ATM
        vg = max(self.interior - self.liquid_volume, 1e-12)
        return max(P.P_ATM*self.gas0/vg, P_VAPOUR)

    def pressure(self):
        return self.gas_pressure() + (self.head_pa if self.wet else 0.0)

    @property
    def compliance(self):
        if self.vented:
            return math.inf
        vg = max(self.interior - self.liquid_volume, 1e-12)
        return vg/self.gas_pressure()

    def give(self, mix, gas_at_atm, t):
        self.liquid = self.liquid + np.concatenate([mix[:AIR], [0.0]])
        self.gas0 += float(gas_at_atm) + float(mix[AIR])

    def take(self, volume):
        if self.wet and self.liquid_volume > 0:
            v = min(float(volume), self.liquid_volume)
            out = self.liquid*(v/self.liquid_volume)
            self.liquid = self.liquid - out
            rest = float(volume) - v
            if rest > 0 and not self.vented:
                p = self.gas_pressure()
                self.gas0 = max(self.gas0 - rest*p/P.P_ATM, 0.0)
            return out, rest
        if not self.vented:
            p = self.gas_pressure()
            self.gas0 = max(self.gas0 - float(volume)*p/P.P_ATM, 0.0)
        return np.zeros(N_SPECIES), float(volume)

    def snapshot(self):
        return dict(liquid=self.liquid.copy(), gas0=self.gas0)

    def restore(self, s):
        self.liquid, self.gas0 = s["liquid"].copy(), float(s["gas0"])


class Depot(Terminal):
    """Liquid injected into a porous elastic layer: it is stored under pressure in a region round the needle's opening
    and relaxes into the layer (Thomsen et al. 2014 for the orders of magnitude)."""
    kind = "depot"
    p_ext = P.P_ATM

    def __init__(self, ledger, place, layer, bore):
        L = P.LAYERS[layer]
        self.ledger, self.place, self.layer = ledger, place, layer
        self.perm = float(L["perm"])
        region = 4/3*math.pi*P.TISSUE["region_r"]**3
        self.compliance = region/P.TISSUE["bulk"]
        self.leak_r = (P.TISSUE["region_r"]**2*P.LIQUIDS["saline"]["mu"]/(self.perm*P.TISSUE["bulk"]))/self.compliance
        self.series_r = P.LIQUIDS["saline"]["mu"]/(4*math.pi*self.perm*0.5*bore)
        self.p = P.P_ATM

    def pressure(self):
        return self.p

    def set_pressure(self, p):
        self.p = float(p)

    def give(self, mix, gas_at_atm, t):
        self.ledger.add(self.place, mix, t)
        if gas_at_atm > 0:
            self.ledger.add(self.place, parcel(gas_at_atm, "air"), t)

    def take(self, volume):
        return np.zeros(N_SPECIES), 0.0                 # nothing comes back out of tissue through the needle

    def stored(self):
        return max(self.p - P.P_ATM, 0.0)*self.compliance

    def relax(self, dt):
        """Called while no needle feeds it: the stored volume goes into the layer."""
        self.p = P.P_ATM + (self.p - P.P_ATM)*math.exp(-dt/(self.leak_r*self.compliance))


class Node(Terminal):
    """A small closed compliance (a clamped tube): pressure rises with what is pushed into it."""
    kind = "node"

    def __init__(self, compliance):
        self.compliance = float(compliance)
        self.p = P.P_ATM
        self.held = np.zeros(N_SPECIES)

    def pressure(self):
        return self.p

    def set_pressure(self, p):
        self.p = float(p)

    def give(self, mix, gas_at_atm, t):
        self.held = self.held + mix

    def take(self, volume):
        v = min(float(volume), float(self.held.sum()))
        if v <= 0:
            return np.zeros(N_SPECIES), 0.0
        out = self.held*(v/float(self.held.sum()))
        self.held = self.held - out
        return out, 0.0


# ====================================================================== a path from a nozzle to a terminal
class Path:
    """Series resistances and inertances, an optional queue of liquid in a tube, a static head, and a terminal."""

    def __init__(self, terminal, key):
        self.terminal = terminal
        self.key = key                    # identity of the topology: the flow state is kept while it is the same
        self.r_liquid = 0.0               # Pa s/m3 at the viscosity of each element's present contents
        self.r_gas = 0.0                  # the same elements with air in them (needles and bores only)
        self.kin = 0.0                    # sum of K rho/(2 A^2) per unit rho
        self.l_geom = 0.0                 # sum of length/area
        self.head = 0.0                   # Pa, static head opposing outward flow
        self.queue = None                 # a Stack shared with the device that owns the tube, or None
        self.tag = None                   # if set, what the chamber sends into this path is noted as '<tag>:in'
        self.closed = False

    def add_tube(self, length, bore, mu, kinetic=0.0):
        self.r_liquid += poiseuille(mu, length, bore)
        self.r_gas += poiseuille(P.AIR["mu"], length, bore)
        a = math.pi*bore*bore/4
        self.kin += kinetic/(2*a*a)
        self.l_geom += length/a


class Chamber:
    """The inside of a syringe: an ordered column of liquid and at most one body of gas behind a piston of area A."""

    def __init__(self, spec):
        self.A = math.pi*spec["bore"]**2/4
        self.dead = spec["dead_ml"]*ML
        self.Cs = spec["compliance"]
        self.bore = spec["bore"]
        self.stack = Stack()
        self.gas0 = 0.0                   # m3 of gas at one atmosphere
        self.gas_pos = 1.0                # 0: at the nozzle end, 1: at the stopper
        self.p = P.P_ATM
        self.Q = 0.0
        self.key = None
        self.void = 0.0                   # m3 of vapour void while the liquid is pulled below its vapour pressure
        self.peak_gauge = 0.0
        self._coef = None
        self.q_gas = False                # the path's flow state is the gas law's (converted when the liquid's takes over)
        self.nozzle_gas = False           # gas at the nozzle this tick: it leaves first, whichever law carries the flow
        self.after_gas = None             # the liquid law (beta, alpha) for liquid that follows a bubble out within a tick

    # ---- geometry and state
    def fill(self, mix, gas_ml=0.0):
        self.stack = Stack([mix]) if float(np.sum(mix)) > 0 else Stack()
        self.gas0 = gas_ml*ML

    def volume(self, x):
        return self.A*max(x, 0.0) + self.dead

    def stroke_for(self, liquid_m3, gas_m3=0.0):
        return (liquid_m3 + gas_m3 - self.dead)/self.A

    @property
    def liquid(self):
        return self.stack.volume

    def gas_volume(self):
        return self.gas0*P.P_ATM/self.p + self.void

    def settle_pressure(self, x):
        """The pressure at which liquid, gas and the elastic barrel fill the space the piston leaves (no flow)."""
        V, Vl, g = self.volume(x), self.liquid, self.gas0*P.P_ATM
        # Vl + g/p = V + Cs (p - P_ATM)  ->  Cs p^2 + (V - Vl - Cs P_ATM) p - g = 0
        b = V - Vl - self.Cs*P.P_ATM
        if g <= 0.0:
            p = -b/self.Cs
        else:
            p = (-b + math.sqrt(b*b + 4*self.Cs*g))/(2*self.Cs)
        if p < P_VAPOUR:
            self.void = max(V + self.Cs*(P_VAPOUR - P.P_ATM) - Vl - g/P_VAPOUR, 0.0)
            p = P_VAPOUR
        else:
            self.void = 0.0
        self.p = p
        return p

    def compliance(self):
        if self.void > 0.0:
            return 1e-6
        return self.Cs + self.gas0*P.P_ATM/(self.p*self.p)

    # ---- which phase is at the nozzle
    def move_gas(self, axis_up, g_eff, dt, length):
        """axis_up: component of the unit 'up' vector along the barrel's axis from nozzle to stopper (+1: stopper up).
        The gas goes to the upper end at the tube drift speed if the Bond number allows."""
        if self.gas0 <= 0.0 or length <= 1e-6:
            return
        bond = P.LIQUIDS["saline"]["rho"]*g_eff*(0.5*self.bore)**2/P.SIGMA
        if bond < P.BOND_RISE or abs(axis_up) < 0.05:
            return
        speed = P.BUBBLE_DRIFT*math.sqrt(g_eff*self.bore)*abs(axis_up)
        self.gas_pos = min(1.0, max(0.0, self.gas_pos + math.copysign(speed*dt/length, axis_up)))

    def gas_at_nozzle(self, axis_up):
        if self.gas0 <= 0.0:
            return False
        if self.liquid <= 1e-12:
            return True
        frac = self.gas_volume()/max(self.gas_volume() + self.liquid, 1e-15)
        tip_up = -axis_up                 # +1: nozzle straight up
        return self.gas_pos < 0.05 and (tip_up > math.sin(math.radians(P.BUBBLE_TILT_DEG)) or frac > 0.5)

    # ---- one tick with a path
    def coefficients(self, path, dt, gas_first):
        """-> (c_h, F_bias): the piston feels A (p' - P_ATM) = F_bias + c_h v', v' the pressing speed."""
        if path.key != self.key:
            self.key, self.Q = path.key, 0.0
            self.q_gas = False
        T = path.terminal
        C = self.compliance()
        gas = gas_first if self.Q >= 0 else False
        open_path = not (path.closed or T.kind == "closed")
        # The gas law is used for a tick only if the bubble at the nozzle can carry that tick's flow. A syringe landing
        # thumb rest first with 0.1 microlitres of air at the nozzle took the gas law (205 times the liquid's flow for a
        # pressure), and the 1.84 microlitres of liquid behind the bubble left at the gas's rate as a 20 m/s jet that
        # reached the ceiling, with 27 times the work the plunger had done on the liquid (traced tick by tick). A
        # smaller bubble goes out ahead of the liquid at the liquid law's rate (advance).
        if gas and open_path:
            k = path.kin*P.AIR["rho"]
            beta = 1.0/(path.l_geom*P.AIR["rho"]/dt + path.r_gas + getattr(T, "series_r", 0.0) + 2*k*abs(self.Q))
            q_gas = beta*((path.l_geom*P.AIR["rho"]/dt)*self.Q + self.p - T.pressure() - (path.head - k*abs(self.Q)*self.Q))
            if q_gas*dt > self.gas0*P.P_ATM/max(self.p, P_VAPOUR):
                gas = False
        # where the liquid's law takes over from the gas's, the column keeps its momentum (L Q), not the gas's flow:
        # handed on as it was, the gas's flow drove the liquid out at 1314, 921, 660 microlitres a second with the
        # chamber under ambient and the plunger stopped
        if open_path and self.q_gas and not gas and self.Q > 0:
            self.Q *= P.AIR["rho"]/(path.queue.mean_density() if path.queue else density(self.stack.front_mix()))
        self.q_gas = bool(gas and open_path)
        self.nozzle_gas = bool(gas_first)
        self.after_gas = None
        if not open_path:
            beta = alpha_q = t0 = t1 = 0.0
        else:
            rho = P.AIR["rho"] if gas else (path.queue.mean_density() if path.queue else density(self.stack.front_mix()))
            R = (path.r_gas if gas else path.r_liquid) + getattr(T, "series_r", 0.0)
            k = path.kin*rho
            Lq = path.l_geom*rho
            Ct, Rl = T.compliance, T.leak_r
            pt = T.pressure()
            if math.isinf(Ct):
                t0, t1 = pt, 0.0
            else:
                g = dt/(Ct*Rl) if math.isfinite(Rl) else 0.0
                t0 = (pt + g*T.p_ext)/(1 + g)
                t1 = (dt/Ct)/(1 + g)
            s = path.head - k*abs(self.Q)*self.Q
            beta = 1.0/(Lq/dt + R + 2*k*abs(self.Q) + t1)
            alpha_q = beta*((Lq/dt)*self.Q - t0 - s)
            if gas:
                # the liquid's own law for whatever follows the bubble out within this tick, its column starting with
                # the gas's momentum (advance)
                rho_l = path.queue.mean_density() if path.queue else density(self.stack.front_mix())
                k_l, L_l, Q_l = path.kin*rho_l, path.l_geom*rho_l, self.Q*P.AIR["rho"]/rho_l
                beta_l = 1.0/(L_l/dt + path.r_liquid + getattr(T, "series_r", 0.0) + 2*k_l*abs(Q_l) + t1)
                self.after_gas = (beta_l, beta_l*((L_l/dt)*Q_l - t0 - (path.head - k_l*abs(Q_l)*Q_l)))
        den = 1.0 + dt*beta/C
        a = (self.p - (dt/C)*alpha_q)/den
        b = (dt/C)*self.A/den
        self._coef = (a, b, beta, alpha_q, t0, t1, gas)
        return self.A*b, self.A*(a - P.P_ATM)

    def advance(self, path, v_press, x_new, dt, t, ledger):
        """After the rigid step: the flow that goes with the piston's speed; move it; set the new pressure."""
        a, b, beta, alpha_q, t0, t1, gas = self._coef
        p_new = a + b*v_press
        Q = beta*p_new + alpha_q
        T = path.terminal
        self.Q = Q
        dV = Q*dt
        moved = dict(out=np.zeros(N_SPECIES), gas_out=0.0)
        if dV > 0:
            if gas or (self.nozzle_gas and self.gas0 > 0.0):     # what is at the nozzle leaves first
                g_avail = self.gas0*P.P_ATM/max(p_new, P_VAPOUR)
                vg = min(dV, g_avail)
                self.gas0 = max(self.gas0 - vg*max(p_new, P_VAPOUR)/P.P_ATM, 0.0)
                v_liq = dV - vg
                if gas and v_liq > 0.0 and self.after_gas is not None:
                    # The bubble ran out within the tick: the liquid behind it leaves at its own law's rate, not at the
                    # gas's. A push begun inside the tick (a crosshead from rest, a floor under a thumb rest) is not in
                    # the pressure the law was chosen with: with 0.1 microlitres of air at a 23 G needle's nozzle the gas
                    # law let the plunger through 2.2 microlitres in a tick, and 2.1 of liquid left at 23.7 m/s (bench).
                    # What does not leave stays in the chamber; its pressure is the next tick's.
                    beta_l, alpha_l = self.after_gas
                    v_liq = min(v_liq, max(beta_l*p_new + alpha_l, 0.0)*dt)
                    self.Q, self.q_gas = v_liq/dt, False
                    Q = (vg + v_liq)/dt
                liq = self.stack.pop_front(v_liq)
                self._send(path, liq, vg*max(p_new, P_VAPOUR)/P.P_ATM, t, ledger)
            else:
                liq = self.stack.pop_front(dV)
                short = dV - float(liq.sum())
                vg0 = 0.0
                if short > 1e-18:           # the liquid ran out: what follows is the gas
                    vg0 = min(short*max(p_new, P_VAPOUR)/P.P_ATM, self.gas0)
                    self.gas0 -= vg0
                self._send(path, liq, vg0, t, ledger)
        elif dV < 0:
            liq, gas_in = self._draw(path, -dV)
            self.stack.push_front(liq)
            if gas_in > 0:
                if self.gas0 <= 0.0:
                    self.gas_pos = 0.0      # a bubble that has just come in is at the nozzle
                self.gas0 += gas_in*T.pressure()/P.P_ATM if T.kind != "air" else gas_in
        if not (path.closed or T.kind == "closed") and math.isfinite(T.compliance):
            T.set_pressure(t0 + t1*Q)
        self.settle_pressure(x_new)
        self.peak_gauge = max(self.peak_gauge, self.p - P.P_ATM)
        return Q

    def _send(self, path, liq, gas_at_atm, t, ledger):
        T = path.terminal
        if path.tag is not None and (float(liq.sum()) > 0 or gas_at_atm > 0):
            ledger.note(path.tag + ":in", liq + parcel(gas_at_atm, "air"), t)
        if path.queue is not None:
            vin = float(liq.sum()) + gas_at_atm
            if vin <= 0:
                return
            mix = liq.copy()
            mix[AIR] += gas_at_atm          # a slug of air in a tube travels with the liquid
            path.queue.push_front(mix)
            out = path.queue.pop_back(vin)
            T.give(out, 0.0, t)
        else:
            T.give(liq, gas_at_atm, t)

    def _draw(self, path, volume):
        T = path.terminal
        if path.queue is not None:
            back, gas = T.take(volume)
            path.queue.push_back(back if gas <= 0 else back + parcel(gas, "air"))
            got = path.queue.pop_front(volume)
            air = float(got[AIR])
            got[AIR] = 0.0
            return got, air
        return T.take(volume)

    def snapshot(self):
        return dict(stack=[p.copy() for p in self.stack.items], gas0=self.gas0, gas_pos=self.gas_pos, p=self.p, Q=self.Q,
                    key=self.key, void=self.void, peak=self.peak_gauge, q_gas=self.q_gas, nozzle_gas=self.nozzle_gas)

    def restore(self, s):
        self.stack = Stack(s["stack"])
        self.gas0, self.gas_pos, self.p, self.Q = s["gas0"], s["gas_pos"], s["p"], s["Q"]
        self.key, self.void, self.peak_gauge = s["key"], s["void"], s["peak"]
        self.q_gas, self.nozzle_gas = s.get("q_gas", False), s.get("nozzle_gas", False)


# ====================================================================== liquid in the open
class Spray:
    """Drops and jets as ballistic parcels; where each lands is recorded."""

    def __init__(self, ledger):
        self.ledger = ledger
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.mix = np.zeros((0, N_SPECIES))
        self.src = np.zeros(0, int)
        self.pending = {}                 # outlet's body -> accumulated mix waiting to become a drop
        self.deposits = []                # (place, body, local xyz, mix)
        self.rng = np.random.default_rng(7)
        self.locate = None                # callable(p0, p1, src_body) -> (hit point, place name, body, local xyz) or None
        self.emitted = 0.0
        self.launches = []                # the last parcels let go: (time, origin, velocity, microlitres), for records

    def emit(self, term, mix, t):
        self._t = t
        # keyed by the outlet's body, not by the outlet object's id: a checkpoint restored in another process kept the
        # old ids, and the liquid hanging at a point was never joined by what the point gave next
        key = int(term.body) if term.body >= 0 else id(term)
        acc, last = self.pending.get(key, (np.zeros(N_SPECIES), None))
        acc = acc + mix
        origin, axis, vel_body, dt = term.origin_fn()
        vol = float(mix[:AIR].sum())
        a = math.pi*term.bore**2/4
        u = vol/dt/a if dt > 0 else 0.0
        rho = density(mix)
        weber = rho*u*u*term.bore/P.SIGMA
        if weber >= P.JET_WEBER:
            size = P.JET_PARCEL_ML*ML
            while float(acc[:AIR].sum()) >= size:
                f = size/float(acc[:AIR].sum())
                self._launch(origin, vel_body + u*self._cone(axis), acc*f, term.body)
                acc = acc*(1 - f)
        else:
            od = getattr(term, "outer", term.bore)
            drop = P.DROP_FACTOR*math.pi*od*P.SIGMA/(rho*P.GRAVITY)
            while float(acc[:AIR].sum()) >= drop:
                f = drop/float(acc[:AIR].sum())
                self._launch(origin, vel_body.copy(), acc*f, term.body)
                acc = acc*(1 - f)
        self.pending[key] = (acc, (origin, term.body))

    def hanging(self):
        """Liquid that has left an outlet and still hangs from it (less than one drop)."""
        return sum(float(a[:AIR].sum()) for a, _ in self.pending.values())

    def _cone(self, axis):
        a = np.asarray(axis, float)
        n = self.rng.normal(size=3)
        n -= (n @ a)*a
        n /= max(np.linalg.norm(n), 1e-9)
        ang = math.radians(P.JET_CONE_DEG)*0.5*abs(self.rng.normal())
        d = math.cos(ang)*a + math.sin(ang)*n
        return d/np.linalg.norm(d)

    def _launch(self, origin, vel, mix, body):
        self.pos = np.vstack([self.pos, origin])
        self.vel = np.vstack([self.vel, vel])
        self.mix = np.vstack([self.mix, mix])
        self.src = np.append(self.src, int(body))
        self.emitted += float(mix[:AIR].sum())
        self.launches.append((float(getattr(self, "_t", 0.0) or 0.0), np.array(origin, float), np.array(vel, float), 1e9*float(mix[:AIR].sum())))
        if len(self.launches) > 4000:
            del self.launches[:2000]

    def step(self, dt):
        if not len(self.pos):
            return
        g = np.array([0.0, 0.0, -P.GRAVITY])
        # air drag on each parcel as on a sphere of its volume (P.DRAG: Schiller and Naumann's coefficient, Stokes' at
        # low Reynolds number, 0.44 above 1000), implicit in the drag so that it is stable for any parcel at any tick.
        # Without it a 2 uL drop spat by a landing syringe reached the ceiling. A jet's parcels are each a sphere
        # here: the slipstream that carries a coherent jet further is left out (they fall short of a real jet).
        vol = np.maximum(self.mix[:, :AIR].sum(1), 1e-15)
        dia = (6.0*vol/math.pi)**(1.0/3.0)
        rho_l = np.array([density(m) for m in self.mix])
        sp = np.linalg.norm(self.vel, axis=1)
        re = np.maximum(P.AIR["rho"]*sp*dia/P.AIR["mu"], 1e-9)
        cd = np.where(re < 1000.0, 24.0/re*(1.0 + 0.15*re**0.687), 0.44)
        k = 0.75*(P.AIR["rho"]/rho_l)*cd*sp/dia          # 1/s: the drag's deceleration per unit of speed
        v1 = (self.vel + g*dt)/(1.0 + k*dt)[:, None]
        p1 = self.pos + 0.5*(self.vel + v1)*dt
        keep = np.ones(len(self.pos), bool)
        for i in range(len(self.pos)):
            hit = self.locate(self.pos[i], p1[i], int(self.src[i])) if self.locate else None
            if hit is None and p1[i, 2] <= 0.0:
                hit = (np.array([p1[i, 0], p1[i, 1], 0.0]), "floor", -1, p1[i].copy())
            if hit is not None:
                point, place, body, local = hit
                self.ledger.add(place, self.mix[i])
                self.deposits.append((place, int(body), np.asarray(local, float), self.mix[i].copy()))
                keep[i] = False
        self.pos, self.vel, self.mix, self.src = p1[keep], v1[keep], self.mix[keep], self.src[keep]

    def airborne(self):
        return float(self.mix[:, :AIR].sum()) if len(self.mix) else 0.0

    def snapshot(self):
        return dict(pos=self.pos.copy(), vel=self.vel.copy(), mix=self.mix.copy(), src=self.src.copy(),
                    deposits=[(a, b, c.copy(), d.copy()) for a, b, c, d in self.deposits],
                    pending={k: (a.copy(), o) for k, (a, o) in self.pending.items()}, emitted=self.emitted,
                    rng=self.rng.bit_generator.state)

    def restore(self, s):
        self.pos, self.vel, self.mix, self.src = s["pos"].copy(), s["vel"].copy(), s["mix"].copy(), s["src"].copy()
        self.deposits = [(a, b, c.copy(), d.copy()) for a, b, c, d in s["deposits"]]
        self.pending = {k: (a.copy(), o) for k, (a, o) in s["pending"].items()}
        self.emitted = s["emitted"]
        self.rng.bit_generator.state = s["rng"]
