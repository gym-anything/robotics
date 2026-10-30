"""LPG powershift drivetrain and brakes.

engine (governed, torque curve, starter, friction) -> torque converter (K-factor capacity, torque ratio falling to the
coupling point) -> turbine (inertia) -> forward/reverse clutch packs (capacity ramped by engagement, lowered by the
inching pedal) -> reduction DRIVE_RATIO x efficiency -> open differential (equal torque to both front wheels) -> wheels.
Service and parking brakes: a dry-friction stick/slip element on each front-wheel joint (a stiff spring anchored where the
wheel stuck, capped at the brake torque; the anchor slides when the cap is exceeded), so a braked wheel holds at rest
without the creep of a regularised friction, and every joule of slip is booked.
Service brake: hydraulic, worked by force. The brake pedal pushes a master cylinder through its lever ratio; the line
pressure is the master cylinder's oil volume through the circuit's pressure-volume curve (shoe takeup against the return
springs, then the linings', drums' and hoses' compliance, stiffening with pressure: BRAKE_LINE, BRAKE_PV), and the pedal feels that pressure back through
the same ratio (controls.py), so the pressure follows the force on the pad and the travel follows from the oil the
brakes take. Each front drum's torque is BRAKE_FACTOR x drum radius x the wheel cylinder's force above the shoe return
springs. The inching pedal brakes only by pushing the brake pedal with its lug (controls.py).

Integration: engine and turbine speeds are scalar states advanced once per tick; the clutch is a smooth Coulomb coupling
whose slip band is set from the explicit-coupling stability bound on the lightest wheel-side inertia (a wheel in the
air); the converter couples engine and turbine implicitly (linearised backward Euler on the two speeds).
Inputs are the decoded control signals (controls.py); outputs are joint torques on the two front-wheel dofs.
"""

import math

import numpy as np

from . import params as P

RPM = 2 * math.pi / 60.0
TURBINE_J = P.TURBINE_J                     # kg m^2, turbine + input shafts (estimate)
STARTER_TORQUE = P.STARTER["torque"]                # N m at the crank (estimate: 1.4 kW starter through the ring gear)
START_SPEED = P.STARTER["fire_rpm"] * RPM            # rad/s at which the engine fires and runs
STALL_SPEED = P.STARTER["stall_rpm"] * RPM            # below this a running engine stalls
BRAKE_K = P.SERVICE_BRAKE_STICK_K                      # N m/rad, stick spring of the brake element (stiff vs the tyre)
WHEEL_J_MIN = P.WHEEL_J_MIN                    # kg m^2, lightest wheel-side inertia (wheel + armature, wheel in the air)

# service brake hydraulics (BRAKE_HARDWARE, BRAKE_LINE, BRAKE_FACTOR)
_HW, _LN = P.BRAKE_HARDWARE, P.BRAKE_LINE
A_MC = math.pi * P.BRAKE_HARDWARE["mc_bore"] ** 2 / 4       # master cylinder area
A_WC = math.pi * P.BRAKE_HARDWARE["wc_bore"] ** 2 / 4       # wheel cylinder area
PUSHROD_ARM = P.BRAKE_PAD_ARM / _HW["pedal_ratio"]          # m: pushrod travel per radian of pedal (beyond the play)
SERVICE_TORQUE_PER_PA = P.BRAKE_FACTOR * _HW["drum_d"] / 2 * A_WC     # N m per Pa above the shoe springs, each wheel
# the fluid's laminar flow through the brake tube to the drums (Hagen-Poiseuille) as a damper on the pedal (N m s/rad)
BRAKE_LINE_DAMPING = ((A_MC * PUSHROD_ARM) ** 2 * 128 * P.BRAKE_LINE["fluid_nu"] * P.BRAKE_LINE["fluid_rho"]
                      * P.BRAKE_LINE["tube_L"] / (math.pi * P.BRAKE_LINE["tube_d"] ** 4))


def brake_pressure(x):
    """Line pressure (Pa) with the brake pedal pressed x rad: the master cylinder's displaced oil (pushrod travel beyond
    the pedal's free play) through the circuit's pressure-volume curve: the shoes' takeup against their return springs,
    then V - takeup = a ln(1 + (p - p_shoe) / p_c), a circuit that stiffens with pressure (BRAKE_PV)."""
    x = min(x, P.PEDAL_BRAKE["travel"])          # on its floor stop the pedal pushes the pushrod no further
    u = max(0.0, x * P.BRAKE_PAD_ARM - _HW["play"]) / _HW["pedal_ratio"]
    v = A_MC * u
    if v <= _LN["takeup"]:
        return _LN["p_shoe"] * v / _LN["takeup"]
    return _LN["p_shoe"] + P.BRAKE_PV["p_c"] * math.expm1((v - _LN["takeup"]) / P.BRAKE_PV["a"])


def brake_pedal_torque(x):
    """The line pressure's torque against the pressed brake pedal (N m, opposing the press)."""
    return brake_pressure(x) * A_MC * PUSHROD_ARM if x * P.BRAKE_PAD_ARM > _HW["play"] else 0.0


def service_brake_torque(p):
    """Torque capacity of each front drum (N m) at line pressure p."""
    return SERVICE_TORQUE_PER_PA * max(0.0, p - _LN["p_shoe"])



def _interp_curve(n_rpm):
    c = P.ENGINE_TORQUE_CURVE
    xs = [a for a, _ in c]; ys = [b for _, b in c]
    return float(np.interp(n_rpm, xs, ys, left=ys[0], right=ys[-1]))


K_FACTOR = P.TC_STALL_RPM / math.sqrt(_interp_curve(P.TC_STALL_RPM))     # rpm / sqrt(N m): stall at TC_STALL_RPM


def pedal_force_at(x):
    """Pad force (N) that holds the brake pedal pressed x rad at rest: return spring plus line pressure."""
    spring = P.PEDAL_BRAKE["preload"] + P.PEDAL_BRAKE["k"] * x
    return (spring + brake_pedal_torque(x)) / P.BRAKE_PAD_ARM


def pedal_travel_for(force):
    """Pressed angle (rad) at which the pad force is `force` (bisection; the pedal's floor stop caps it)."""
    lo, hi = 0.0, P.PEDAL_BRAKE["travel"]
    if pedal_force_at(hi) <= force:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if pedal_force_at(mid) < force else (lo, mid)
    return 0.5 * (lo + hi)


# each drum's torque at BRAKE_STANDARD's pedal force: the scale of the telemetry's brake_frac (1 = the standard's force)
SERVICE_TORQUE_AT_STANDARD = service_brake_torque(brake_pressure(pedal_travel_for(P.BRAKE_STANDARD["pedal_force"])))


class Drivetrain:
    def __init__(self, wheel_dofs, dt):
        self.wheel_dofs = tuple(wheel_dofs)          # (left, right) front wheel dofs
        self.dt = dt
        self.w_e = 0.0                               # engine speed rad/s
        self.w_t = 0.0                               # turbine speed rad/s (forward gear sign convention: wheel side)
        self.running = False
        self.engage = 0.0                            # clutch engagement 0..1 (pressure ramp)
        self.gear = 0                                # +1 forward, -1 reverse, 0 neutral (the engaged pack)
        self.brake_anchor = [None, None]
        self.energy = dict(fuel=0.0, converter_loss=0.0, clutch_loss=0.0, gear_loss=0.0, brake_loss=0.0,
                           engine_friction=0.0, starter=0.0)
        # explicit-coupling bound: dt * c_clutch * R^2 / J_min < 0.5 with c = cap/slip_band
        self.slip_band = max(1.0, 2.0 * dt * P.CLUTCH_TORQUE * P.DRIVE_RATIO ** 2 / (2 * WHEEL_J_MIN))
        self.tele = {}

    # ------------------------------------------------------------------ converter
    @staticmethod
    def converter(w_i, w_t):
        """(impeller torque, turbine torque) for impeller/turbine speeds (rad/s)."""
        n_i = w_i / RPM
        if abs(n_i) < 1e-6:
            return 0.0, 0.0
        sr = w_t / w_i
        cap = (n_i / K_FACTOR) ** 2 * math.copysign(1.0, n_i)
        if sr < P.TC_COUPLING_SR:
            tr = P.TC_STALL_RATIO - (P.TC_STALL_RATIO - 1.0) * max(sr, 0.0) / P.TC_COUPLING_SR
            if sr < 0:                                # turbine driven backwards (reversal at speed): stall ratio, more capacity
                cap *= 1.0 + 0.5 * min(-sr, 1.0)
            ti = cap
        else:                                         # coupling: fluid coupling torque vanishing at sr = 1, reversing beyond
            ti = cap * (1.0 - sr) / (1.0 - P.TC_COUPLING_SR)
            tr = 1.0
        return ti, tr * ti

    # ------------------------------------------------------------------ step
    def step(self, sig, wheel_q, wheel_qd, seat_switch):
        """sig: decoded control signals. wheel_q/qd: (left, right) front wheel joint angle/speed. Returns {dof: tau}."""
        dt = self.dt
        key = sig["key"]
        # ---- engine running state
        if key == "off":
            self.running = False
        starter = 0.0
        if key == "start" and not self.running:
            starter = STARTER_TORQUE
            if self.w_e > START_SPEED:
                self.running = True
        if self.running and self.w_e < STALL_SPEED and key != "start":
            self.running = False
        # ---- governor: accelerator sets the target speed
        target = (P.ENGINE_IDLE + sig["pedal_accel"] * (P.ENGINE_MAX - P.ENGINE_IDLE)) * RPM
        throttle = float(np.clip(P.GOVERNOR_GAIN * (target - self.w_e) / RPM + P.GOVERNOR_BIAS, 0.0, 1.0)) if self.running else 0.0
        T_full = _interp_curve(self.w_e / RPM)
        T_eng = throttle * T_full if self.running else 0.0
        E = P.ENGINE_FRICTION_SHAPE
        T_fric = P.ENGINE_FRICTION * (E["at_rest"] + E["per_idle"] * self.w_e / (P.ENGINE_IDLE * RPM)) * math.tanh(self.w_e / E["w_reg"])
        # ---- direction, clutch engagement (presence switch open -> neutral, Toyota SAS read)
        want = sig["direction"] if seat_switch and self.running else 0
        if want != self.gear:
            self.engage = max(0.0, self.engage - dt / P.CLUTCH_RELEASE)          # release the engaged pack
            if self.engage <= 0.0:
                self.gear = want
        elif self.gear != 0:
            self.engage = min(1.0, self.engage + dt / P.CLUTCH_RAMP)
        inch = sig["pedal_inch"]
        clutch_frac = max(0.0, 1.0 - inch / P.INCHING_SLIP_FRACTION)
        # service brake: line pressure from the brake pedal (the inching pedal reaches it only through its lug)
        T_service = service_brake_torque(sig.get("brake_pressure", 0.0))
        brake_frac = T_service / SERVICE_TORQUE_AT_STANDARD
        cap = P.CLUTCH_TORQUE * self.engage * clutch_frac if self.gear != 0 else 0.0
        # ---- output shaft (turbine units): differential average, gear sign
        w_out = 0.5 * (wheel_qd[0] + wheel_qd[1]) * P.DRIVE_RATIO * self.gear if self.gear else 0.0
        # ---- converter + turbine + engine (linearised implicit)
        ti, tt = self.converter(self.w_e, self.w_t)
        slip = self.w_t - w_out
        T_cl = cap * math.tanh(slip / self.slip_band) if cap > 0 else 0.0
        # implicit update of the turbine with the clutch's slope (stiff when locked)
        dcl = cap / self.slip_band * (1.0 - math.tanh(slip / self.slip_band) ** 2) if cap > 0 else 0.0
        eps = 1e-3
        _, tt2 = self.converter(self.w_e, self.w_t + eps)
        dtt = (tt2 - tt) / eps
        A = TURBINE_J / dt - dtt + dcl
        self.w_t += (tt - T_cl) / A
        ti_e, _ = self.converter(self.w_e + eps, self.w_t)
        dti = (ti_e - ti) / eps
        Ae = P.ENGINE_J / dt + max(dti, 0.0)
        # the hydraulic pump on the engine's shaft: outlet pressure x displacement / (2 pi x mechanical efficiency); the
        # world sets pump_pressure from the circuits' pressures each tick (a weightless pump would let
        # the truck lift 2.5 t without slowing the engine)
        T_pump = (P.PUMP_DISPLACEMENT * getattr(self, "pump_pressure", 0.0) / (2 * math.pi * P.PUMP_HM_EFF)
                  if self.w_e > 1.0 else 0.0)
        self.tele_pump_torque = T_pump
        w_e_new = self.w_e + (T_eng + starter - T_fric - ti - T_pump) / Ae
        self.w_e = max(0.0, w_e_new)
        # ---- wheel torques (open differential; efficiency as a loss in the direction of power flow)
        tq_out = T_cl * P.DRIVE_RATIO * self.gear if self.gear else 0.0
        mean_w = 0.5 * (wheel_qd[0] + wheel_qd[1])
        # gear losses in the direction of power flow: driving delivers eta of the turbine torque, back-driving needs 1/eta
        eta = P.DRIVE_EFFICIENCY if tq_out * mean_w >= 0 else 1.0 / P.DRIVE_EFFICIENCY
        tq_w = 0.5 * tq_out * eta
        tau = {self.wheel_dofs[0]: tq_w, self.wheel_dofs[1]: tq_w}
        # ---- brakes: stick/slip element per front wheel
        Tb = T_service + P.PARK_BRAKE_TORQUE * sig.get("park_brake", sig["park"])
        for i in range(2):
            d = self.wheel_dofs[i]
            if Tb <= 1.0:
                self.brake_anchor[i] = None
                continue
            if self.brake_anchor[i] is None:
                self.brake_anchor[i] = wheel_q[i]
            c = 2.0 * math.sqrt(BRAKE_K * WHEEL_J_MIN) * P.BRAKE_STICK_ZETA
            tb = -BRAKE_K * (wheel_q[i] - self.brake_anchor[i]) - c * wheel_qd[i]
            if abs(tb) > Tb:
                tb = math.copysign(Tb, tb)
                self.brake_anchor[i] = wheel_q[i] + tb / BRAKE_K       # anchor slides: slipping brake
                self.energy["brake_loss"] += abs(Tb * wheel_qd[i]) * dt
            tau[d] += tb
        # ---- energy bookkeeping
        self.energy["fuel"] += max(0.0, T_eng * self.w_e) * dt
        self.energy["pump"] = self.energy.get("pump", 0.0) + T_pump * self.w_e * dt
        self.energy["starter"] += starter * self.w_e * dt
        self.energy["engine_friction"] += T_fric * self.w_e * dt
        self.energy["converter_loss"] += (ti * self.w_e - tt * self.w_t) * dt
        self.energy["clutch_loss"] += T_cl * slip * dt
        self.energy["gear_loss"] += (T_cl * w_out - 2 * tq_w * mean_w) * dt
        self.tele = dict(pump_torque=getattr(self, 'tele_pump_torque', 0.0), engine_rpm=self.w_e / RPM, turbine_rpm=self.w_t / RPM, running=self.running, gear=self.gear,
                         engage=self.engage, throttle=throttle, clutch_torque=T_cl, wheel_torque=tq_w, brake_cap=Tb,
                         brake_frac=brake_frac, clutch_frac=clutch_frac, service_brake_torque=T_service,
                         brake_pressure=sig.get("brake_pressure", 0.0))
        return tau

    def checkpoint(self):
        return dict(w_e=self.w_e, w_t=self.w_t, running=self.running, engage=self.engage, gear=self.gear,
                    anchor=list(self.brake_anchor), energy=dict(self.energy))

    def restore(self, c):
        self.w_e, self.w_t, self.running = c["w_e"], c["w_t"], c["running"]
        self.engage, self.gear, self.brake_anchor = c["engage"], c["gear"], list(c["anchor"])
        self.energy = dict(c["energy"])

    def set_running(self, rpm=None):
        """Start state helper (task construction, not a policy action): engine idling."""
        self.running = True
        self.w_e = (rpm or P.ENGINE_IDLE) * RPM
