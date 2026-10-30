"""Mast hydraulics: engine-driven gear pump -> priority valve (steering first) -> three-spool open-centre
valve worked by the cowl levers -> lift circuit (two main lift cylinders on the intermediate stage, the free-lift cylinder
on the carriage 2:1), two double-acting tilt cylinders, the side-shift cylinder; main relief; lowering regulator.

Each cylinder is a compressible oil column: its force on its joint is k (x_oil - x) with k = beta A_eff^2 / V (V the
chamber volume at the current stroke), where x_oil is the joint position implied by the oil in the chamber. The chamber
pressure is the column force over its area. A lift stage also carries the mast's friction (cylinder seals, rollers
under the load moment, chains): a Coulomb force of MAST_FRICTION times the force the column transmits, against the
stage's motion. The tilt and side-shift columns keep a viscous term (ZETA). Flows change the oil volumes:
  lift circuit: each lift cylinder reaches a common node through its line (hose, tube and fittings: a pressure drop of
  Darcy friction, Hagen-Poiseuille below Re ~1,190 and Blasius above, plus the fittings' minor losses, from the line's
  bore and length; HYD_LINES); the spool connects the node to the pump (raise) or to tank through the lowering regulator
  and the spool's own orifice (lower); the node pressure p_n solves the flow balance sum_i q_line_i(p_i - p_n) =
  Q_node(p_n) by bisection, so the cylinder needing the lower pressure moves first on raising (free lift, then the main
  stage) and the higher-pressure one first on lowering (main stage, then free lift), and a load whose pressure exceeds
  the relief never lifts.
  lowering regulator (LOWERING_REGULATOR): a restrictor-type pressure-compensated flow control valve, a fixed metering
  orifice in series with a compensator spool. The spool, held open by its spring, closes when the metering orifice's
  drop exceeds the spring's setting, so the flow out of the lift cylinders is held at the setting (the fork speed cap,
  LOWERING_SPEED_MAX) whatever the load's pressure. The spool does not move instantly: it is moved by the force
  imbalance against the oil it displaces through its damping orifice, a first-order lag of time constant tau on the
  compensator's conductance. A load bouncing on its oil faster than that sees the regulator as a fixed orifice, whose
  flow rises and falls with the pressure: that, the lines' losses and the mast friction are what damp a flowing column
  (an instantaneous cap on the flow would let a column bounce undamped).
  tilt and side-shift: spool flow (pump flow x opening, while below the relief) moves x_oil; closed spool = locked column.
The main relief sits at the pump: it bounds the node while raising and nothing else. A lift cylinder that is not being
fed is closed by its load-check valve, so a load bouncing on its oil can exceed the relief pressure, as on a truck (a
force cap on the lift columns would dump oil whenever a lowering load overshot, and the load would fall through it: a
self-sustained 7-19 MPa oscillation that keeps the rear wheels off the floor).
Lift-cylinder leakage (pressure-proportional) lets a raised load creep down slowly. With the engine off there is no flow;
gravity lowering still works through the valve (a real truck lowers with the engine off); tilt and shift lock.
Operator-presence switch open: all spools blocked (Toyota SAS read).
"""

import bisect
import math

import numpy as np

from . import params as P

A_MAIN = 2 * math.pi * P.LIFT_BORE_MAIN ** 2 / 4          # two main cylinders
A_FREE = math.pi * P.LIFT_BORE_FREE ** 2 / 4
A_TILT = 2 * math.pi * P.TILT_BORE ** 2 / 4 * 0.75         # rod-side annulus ~ 75 % (estimate) used for both directions
A_SHIFT = math.pi * P.SHIFT_BORE ** 2 / 4 * 0.75
DEAD_LEN = P.HYD_DEAD_LEN                                            # m of oil in hoses/dead volume per circuit (estimate)


def _line_table(line):
    """(dp, q) arrays, both increasing: the pressure drop of a lift line at flow q (m^3/s). Darcy friction over the
    line's length (f = 64/Re laminar, 0.316 Re^-1/4 Blasius above: the larger of the two, continuous at Re ~1,190) plus
    its fittings' minor losses K rho v^2 / 2 (HYD_LINES; viscosity OIL_VISCOSITY at the working temperature)."""
    d, L, K = line["d"], line["L"], line["K"]
    A = math.pi * d * d / 4
    q = np.concatenate([[0.0], np.geomspace(1e-7, 2e-2, 600)])
    v = q / A
    re = np.maximum(v * d / P.OIL_VISCOSITY, 1e-9)
    f = np.maximum(64.0 / re, 0.316 * re ** -0.25)
    dp = (f * L / d + K) * P.OIL_DENSITY * v * v / 2
    dp[0] = 0.0
    return dp.tolist(), q.tolist()


LINES = {k: _line_table(P.HYD_LINES[k]) for k in ("main", "free")}


def line_flow(name, dp):
    """Flow through lift line `name` (m^3/s, signed like dp) for a pressure drop dp (Pa) along it (table inverse)."""
    DP, Q = LINES[name]
    a = abs(dp)
    i = bisect.bisect_left(DP, a)
    if i <= 0:
        return 0.0
    if i >= len(DP):                                         # beyond the table: turbulent, q ~ sqrt(dp)
        qq = Q[-1] * math.sqrt(a / DP[-1])
    else:
        qq = Q[i - 1] + (Q[i] - Q[i - 1]) * (a - DP[i - 1]) / (DP[i] - DP[i - 1])
    return qq if dp >= 0 else -qq


# lowering regulator (pressure compensator): conductances K (q = K sqrt(dp)). sense "spool" (the default, load-compensated
# lowering): the compensator holds the drop across the lift spool's own lowering orifice at dp_set, so the flow follows
# the spool's opening whatever the load; there is no separate metering orifice (1 / K_MET^2 = 0) and the compensator's
# full opening is open_ratio x the spool's full-stroke orifice. sense "metering" (a restrictor regulator in the lift
# line): the compensator holds the drop across its own metering orifice, sized so dp_set passes the fork speed cap
_KS = P.VALVE_CD * math.sqrt(2.0 / P.OIL_DENSITY)            # m^3/s per (m^2 sqrt(Pa)) of orifice area
Q_REG = P.LOWERING_SPEED_MAX * (2 * math.pi * P.LIFT_BORE_MAIN ** 2 / 4) / 2.0   # main-stage oil flow at the cap (fork = 2 x main)
SPOOL_SENSED = P.LOWERING_REGULATOR["sense"] == "spool"
K_MET = math.inf if SPOOL_SENSED else Q_REG / math.sqrt(P.LOWERING_REGULATOR["dp_set"])
K_COMP_MAX = P.LOWERING_REGULATOR["open_ratio"] * (_KS * P.VALVE_AREA_MAX if SPOOL_SENSED else K_MET)
ZETA = P.HYD_ZETA                                                # tilt and side-shift column damping ratio (line losses; estimate)
FRICTION_V = 0.005                                         # m/s: speed over which the mast friction reverses (regularisation)
# lift-cylinder end-of-stroke cushions (a spear entering a bore throttles the port progressively): length 50 mm, the
# port's conductance falling as the cube of the remaining distance to 1 % of it, so that against the relief pressure the
# stage arrives at about 0.03 m/s instead of the full lift speed (estimates of the class; without them a stage
# strikes the stops at 0.3 m/s and rocks a rated-load truck onto its front wheels)
CUSHION_LEN = P.LIFT_CUSHION["length"]
CUSHION_MIN = P.LIFT_CUSHION["min_open"]


class Hydraulics:
    def __init__(self, dofs, masses, dt):
        """dofs: {'tilt','main','free','shift'} -> dof index (main = mast_int joint, free = carriage joint)."""
        self.dofs = dofs
        self.dt = dt
        self.m = masses                                     # rough reflected masses for damping
        self.x_oil = dict(main=0.0, free=0.0, tilt=0.0, shift=0.0)
        self.energy = dict(pump=0.0, relief_loss=0.0, throttle_loss=0.0, leak_loss=0.0, line_loss=0.0)
        self.k_comp = K_COMP_MAX                            # lowering compensator conductance (spring holds it open)
        self.tele = {}
        self.initialised = False

    def init_from(self, q, static=None):
        """Oil volumes for the joint positions q; `static` {name: generalized static force} pre-compresses each column
        by F/k so a held load starts at rest instead of dropping onto its oil."""
        for k, d in self.dofs.items():
            self.x_oil[k] = float(q[d])
            if static and k in static:
                kk, _, _ = self._k(k, float(q[d]))
                self.x_oil[k] += static[k] / kk
        self.initialised = True

    # chamber stiffness in joint coordinates
    def _k(self, name, x):
        if name == "main":
            V = A_MAIN * (DEAD_LEN + max(x, 0.0))
            return P.BULK_MODULUS * A_MAIN ** 2 / V, A_MAIN, 1.0
        if name == "free":                                   # carriage moves 2:1 relative to the free-lift rod
            y = max(x - P.CARRIAGE_LOWER, 0.0) / 2
            V = A_FREE * (DEAD_LEN + y)
            return P.BULK_MODULUS * A_FREE ** 2 / V / 4.0, A_FREE, 0.5
        if name == "tilt":
            V = A_TILT * (DEAD_LEN + P.HYD_COLUMN_LEN["tilt"])
            return P.BULK_MODULUS * A_TILT ** 2 / V * P.TILT_ARM ** 2, A_TILT, P.TILT_ARM
        V = A_SHIFT * (DEAD_LEN + P.HYD_COLUMN_LEN["shift"])
        return P.BULK_MODULUS * A_SHIFT ** 2 / V, A_SHIFT, 1.0

    def step(self, sig, q, qd, engine_w, steer_flow, seat_switch):
        """Returns ({dof: tau}, telemetry). q, qd: joint coordinate arrays (dof-indexed)."""
        if not self.initialised:
            self.init_from(q)
        dt = self.dt
        Q_pump = P.PUMP_DISPLACEMENT * max(engine_w, 0.0) / (2 * math.pi) * P.PUMP_VOL_EFF
        Q_avail = max(0.0, Q_pump - steer_flow)
        ok = bool(seat_switch)
        forces, press = {}, {}
        for name, d in self.dofs.items():
            x, v = float(q[d]), float(qd[d])
            k, A, lever = self._k(name, x)
            if name in ("main", "free"):
                F = max(k * (self.x_oil[name] - x), 0.0)     # single-acting: oil pushes, never pulls
            else:
                F = k * (self.x_oil[name] - x) - 2 * ZETA * math.sqrt(k * self.m[name]) * v
            forces[name] = F
            # pressure = column force over area (joint force -> cylinder force through the lever/2:1)
            if name == "free":
                press[name] = (2 * F) / A
            elif name == "tilt":
                press[name] = F / lever / A
            else:
                press[name] = F / A
        # ---------------- lift circuit (main + free in parallel on a common node)
        s_lift = sig["lever_lift"] if ok else 0.0             # +1 = pull = raise
        flows = dict(main=0.0, free=0.0)
        p_node = 0.0
        q_out, q_set = 0.0, Q_REG
        names = ("main", "free")
        if abs(s_lift) > 0.0:
            p_i = (press["main"], press["free"])
            # end-of-stroke cushions: in its last CUSHION_LEN a lift cylinder's port is throttled down to CUSHION_MIN of
            # its conductance, so a stage driven into its stop arrives slowly instead of striking it at full speed
            xm, xf = float(q[self.dofs["main"]]), float(q[self.dofs["free"]])
            to_top = (P.MAST_STAGE_STROKE - xm, P.CARRIAGE_FREE_LIFT - xf)
            to_bottom = (xm, xf - P.CARRIAGE_LOWER)
            cush = lambda r: CUSHION_MIN + (1.0 - CUSHION_MIN) * min(max(r / CUSHION_LEN, 0.0), 1.0) ** 3
            g_up = [cush(r) for r in to_top]
            g_down = [cush(r) for r in to_bottom]
            if s_lift > 0:
                Q_in = Q_avail * s_lift

                def balance(pn):                              # load-check valve: no back-flow while raising
                    return Q_in - sum(g_up[k] * line_flow(names[k], max(pn - p_i[k], 0.0)) for k in range(2))
                lo, hi = 0.0, P.RELIEF_PRESSURE
                if balance(hi) > 0:                           # relief: node at relief, surplus to tank
                    p_node = hi
                else:
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        if balance(mid) > 0:
                            lo = mid
                        else:
                            hi = mid
                    p_node = 0.5 * (lo + hi)
                into = [g_up[k] * line_flow(names[k], max(p_node - p_i[k], 0.0)) for k in range(2)]
                self.energy["relief_loss"] += max(0.0, Q_in - sum(into)) * p_node * dt
            else:
                # lowering: node to tank through the regulator (metering orifice + compensator) and the spool's orifice
                k_sp = _KS * P.VALVE_AREA_MAX * (-s_lift) ** P.LIFT_SPOOL_METERING     # the spool's lowering notch
                k_tot = (1.0 / k_sp ** 2 + 1.0 / K_MET ** 2 + 1.0 / self.k_comp ** 2) ** -0.5

                def out_flow(pn):
                    return k_tot * math.sqrt(max(pn, 0.0))

                def balance(pn):                              # flow from the cylinders into the node minus out
                    return -sum(g_down[k] * line_flow(names[k], pn - p_i[k]) for k in range(2)) - out_flow(pn)
                lo, hi = 0.0, float(max(max(p_i), 1.0))
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if balance(mid) > 0:
                        lo = mid
                    else:
                        hi = mid
                p_node = 0.5 * (lo + hi)
                into = [g_down[k] * line_flow(names[k], p_node - p_i[k]) for k in range(2)]
                q_out = out_flow(p_node)
                self.energy["throttle_loss"] += q_out * p_node * dt
                # the compensator's spring setting (drop across the metering orifice) rises as the spool closes (droop)
                set_ = math.sqrt(1.0 + P.LOWERING_REGULATOR["droop"] * (1.0 - self.k_comp / K_COMP_MAX))
                q_set = (k_sp * math.sqrt(P.LOWERING_REGULATOR["dp_set"]) if SPOOL_SENSED else Q_REG) * set_
                rest = p_node / q_set ** 2 - 1.0 / k_sp ** 2 - 1.0 / K_MET ** 2
                k_target = K_COMP_MAX if rest <= 1.0 / K_COMP_MAX ** 2 else rest ** -0.5
                self._comp_step(k_target, dt)
            for k in range(2):
                self.energy["line_loss"] += abs(into[k] * (p_node - p_i[k])) * dt
            flows["main"], flows["free"] = float(into[0]), float(into[1])
        if s_lift >= 0.0:
            self._comp_step(K_COMP_MAX, dt)                   # no lowering flow: the spring reopens the spool
        # leakage of the lift cylinders (creep down under load)
        for name in ("main", "free"):
            flows[name] -= P.LIFT_LEAK * press[name]
            self.energy["leak_loss"] += P.LIFT_LEAK * press[name] ** 2 * dt
        self.x_oil["main"] += flows["main"] / A_MAIN * dt
        self.x_oil["free"] += flows["free"] / A_FREE * 2.0 * dt
        # an emptied chamber cannot go below the stage's bottom stop
        self.x_oil["main"] = max(self.x_oil["main"], -0.005)
        self.x_oil["free"] = max(self.x_oil["free"], P.CARRIAGE_LOWER - 0.005)
        # no clamp at the top stops: oil pushed into a cylinder whose stage sits on its stop compresses the column, so
        # the chamber pressure rises until the other stage (or the relief) takes the flow (the relief cap below bounds it)
        # ---------------- tilt and side-shift (double acting): spool flow while the pump can push past the load
        for name, lever_sig, lim in (("tilt", "lever_tilt", (-P.MAST_TILT_BACK, P.MAST_TILT_FWD)),
                                     ("shift", "lever_shift", (-P.SIDESHIFT_TRAVEL, P.SIDESHIFT_TRAVEL))):
            s = sig[lever_sig] if ok else 0.0
            if s == 0.0 or Q_avail <= 0.0:
                continue
            k, A, lever = self._k(name, self.x_oil[name])
            # tilt: pull = tilt back (negative joint angle: forks up), push = forward; shift: pull = left (+y)
            direction = -1.0 if name == "tilt" else 1.0
            dx = direction * s * Q_avail / A * dt / (lever if name == "tilt" else 1.0)
            # blocked when the column is at the relief in the direction of motion
            if abs(press[name]) >= P.RELIEF_PRESSURE and np.sign(dx) == np.sign(forces[name]):
                self.energy["relief_loss"] += Q_avail * abs(s) * P.RELIEF_PRESSURE * dt
                continue
            self.x_oil[name] = float(np.clip(self.x_oil[name] + dx, lim[0] - 0.003, lim[1] + 0.003))
        tau = {}
        for name, d in self.dofs.items():
            F = forces[name]
            if name in ("main", "free"):
                # mast friction: Coulomb, in proportion to the force the column transmits, against the stage's motion
                tau[d] = F - P.MAST_FRICTION * F * math.tanh(float(qd[d]) / FRICTION_V)
                continue
            # tilt and side-shift: port relief (oil escapes over it rather than exceed it)
            k, A, lever = self._k(name, float(q[d]))
            Fmax = P.RELIEF_PRESSURE * A * (lever if name == "tilt" else 1.0) * P.PORT_RELIEF_RATIO
            if abs(F) > Fmax:
                F = math.copysign(Fmax, F)
                c = 2 * ZETA * math.sqrt(k * self.m[name])
                self.x_oil[name] = float(q[d]) + (F + c * float(qd[d])) / k
            tau[d] = F
        self.energy["pump"] += Q_pump * p_node * dt if s_lift > 0 else 0.0
        # the pump's outlet pressure for the engine's load: the lift node while raising; a tilt or side-shift column
        # being driven (its pressure, at most the relief); otherwise the open centre (standby, added by the world)
        p_pump = p_node if s_lift > 0 else 0.0
        for name, lever_sig in (("tilt", "lever_tilt"), ("shift", "lever_shift")):
            if ok and sig[lever_sig] != 0.0 and Q_avail > 0.0:
                p_pump = max(p_pump, min(abs(press[name]), P.RELIEF_PRESSURE))
        self.tele = dict(p_main=press["main"], p_free=press["free"], p_tilt=press["tilt"], p_shift=press["shift"],
                         p_node=p_node, p_pump=p_pump, q_pump=Q_pump, q_avail=Q_avail, x_oil=dict(self.x_oil), flows=flows,
                         q_lower=q_out, q_set=q_set, comp_open=self.k_comp / K_COMP_MAX)
        return tau

    def _comp_step(self, k_target, dt):
        """The compensator spool's first-order lag (time constant LOWERING_REGULATOR tau; 0 = an ideal regulator)."""
        tau = P.LOWERING_REGULATOR["tau"]
        a = 1.0 if tau <= dt else dt / tau
        self.k_comp += a * (k_target - self.k_comp)

    def checkpoint(self):
        return dict(x=dict(self.x_oil), e=dict(self.energy), init=self.initialised, k_comp=self.k_comp)

    def restore(self, c):
        self.x_oil, self.energy, self.initialised = dict(c["x"]), dict(c["e"]), c["init"]
        self.k_comp = c.get("k_comp", K_COMP_MAX)
