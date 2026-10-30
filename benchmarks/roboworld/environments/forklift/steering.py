"""Hydrostatic (orbitrol) steering: steering wheel -> spool and meter -> cylinder -> left knuckle; right knuckle by the
Ackermann equality (truck.py).

A non-reaction unit (Danfoss OSPB/OSPC "ON"/"CN": "there is no corresponding movement of the steering wheel when the
driver is not steering", Technical Information p. 14): the cylinder's pressure never turns the wheel; only the hand pumps.

State: the meter (gerotor) angle th_m and the oil volume in the steer cylinder expressed as the left-knuckle angle it
implies, th_c. The wheel turns the spool against the preloaded neutral setting springs (STEER_TORQUE_SPRING, Danfoss
standard springs 0.8-3 N m, p. 12): below the 0.8 N m preload the spool does not deflect and the wheel is held to the meter
(stiff and damped, STEER_STOP); beyond it the spool deflects, the centring torque rising from 0.8 to 3.0 N m over its
travel. The spool deflection d = th_w - th_m is limited to +/-STEER_SPOOL_DEFLECTION, beyond which the wheel drives the
meter mechanically (a stiff stop). With pump flow, an open spool meters flow in proportion to its opening (the deflection
past the held region over the rest of the travel), up to the priority flow, D per revolution of the meter into the
cylinder (minus leakage proportional to the cylinder pressure: the reason knob and wheel angles drift apart); with the
engine off, the meter only turns when the wheel drives it through the stop, as a hand pump, and the wheel feels the
cylinder pressure as torque dp D / 2 pi. The cylinder is a stiff oil column about th_c acting on the left knuckle,
capped by the steering relief (at the stops or against a scrubbing tyre the pressure rises to the relief and the oil
does not move).

Kinematics simplification: cylinder volume is taken linear in the left knuckle angle over its
range (the real arm/tie-rod geometry makes it mildly nonlinear); lock-to-lock = STEER_TURNS_LOCK turns of the wheel.
"""

import math

import numpy as np

from . import params as P

STOP_K = P.STEER_STOP["k"]                        # N m/rad, wheel-to-meter stop beyond the spool travel (rigid feel; estimate)
STOP_C = P.STEER_STOP["c"]                           # N m s/rad


class Steering:
    def __init__(self, wheel_dof, knuckle_dof, knuckle_range, dt):
        self.wheel_dof, self.knuckle_dof = wheel_dof, knuckle_dof
        self.lo, self.hi = knuckle_range
        self.dt = dt
        span = self.hi - self.lo
        self.vol_per_rad = P.ORBITROL_DISPLACEMENT * P.STEER_TURNS_LOCK / span      # m^3 per knuckle radian
        # knuckle angle per meter radian (wheel turns positive = counter-clockwise seen by the driver = truck turns left;
        # a left turn swings the rear wheels to negative knuckle angles, truck.ackermann_fit)
        self.ratio = -span / (P.STEER_TURNS_LOCK * 2 * math.pi)
        # neutral setting springs: preloaded (a linear spring from zero lets any torque, the spinner knob's own weight
        # included, 0.26-0.35 N m, open the spool: a released wheel walks to where the knob hangs lowest)
        sp = P.STEER_TORQUE_SPRING
        self.pre = sp["preload"]
        self.d_hold = sp["preload"] / STOP_K                # deflection of the held wheel at the preload (0.015 deg)
        self.k_spool = (sp["full"] - sp["preload"]) / (P.STEER_SPOOL_DEFLECTION - self.d_hold)
        self.th_m = 0.0
        self.th_c = 0.0
        self.th_w0 = 0.0                                   # wheel angle at which the meter reference was set
        self.energy = dict(relief_loss=0.0, leak_loss=0.0, throttle_loss=0.0, manual_work=0.0)
        self.tele = {}

    def reset_to(self, th_w, th_k):
        self.th_m = th_w
        self.th_c = th_k

    def step(self, th_w, om_w, th_k, om_k, pump_flow, seat_switch):
        """th_w/om_w: steering wheel joint angle/speed; th_k/om_k: left knuckle. pump_flow: flow available to the
        priority valve (m^3/s). Returns ({dof: tau}, flow used m^3/s)."""
        dt = self.dt
        dmax = P.STEER_SPOOL_DEFLECTION
        d = th_w - self.th_m
        # cylinder: stiff oil column about th_c (the knuckle angle its oil volume implies), damped by line losses
        k = P.STEER_KNUCKLE_STIFFNESS
        c = P.STEER_CYL_DAMPING["zeta"] * 2 * math.sqrt(k * P.STEER_CYL_DAMPING["inertia"])   # line losses (estimate)
        tau_relief = P.STEER_RELIEF * self.vol_per_rad
        pumped = pump_flow > 1e-9
        # 1. metered motion: the spool opening (deflection past the held region / the rest of the travel) meters up to
        #    the available flow
        w_max = pump_flow / P.ORBITROL_DISPLACEMENT * 2 * math.pi if pumped else 0.0
        opening = float(np.clip((abs(d) - self.d_hold) / (dmax - self.d_hold), 0.0, 1.0))
        dth_m = w_max * math.copysign(opening, d) * dt
        # 2. the wheel ahead of the spool travel drives the meter mechanically (the gerotor becomes a hand pump)
        d_new = th_w - (self.th_m + dth_m)
        forced = d_new - math.copysign(dmax, d_new) if abs(d_new) > dmax else 0.0
        # 3. the oil cannot raise the column above the relief: metering stops there (pump flow goes over the relief) and
        #    the wheel meets the spool stop, turning further only as fast as leakage lets the cylinder yield
        move = dth_m + forced
        if move != 0.0:
            dc = self.ratio * move
            tau_after = k * (self.th_c + dc - th_k) - c * om_k
            if abs(tau_after) > tau_relief and dc * tau_after > 0:
                room = (math.copysign(tau_relief, tau_after) + c * om_k) / k + th_k - self.th_c
                frac = float(np.clip(room / dc, 0.0, 1.0))
                dth_m, forced = dth_m * frac, forced * frac
                move = dth_m + forced
        self.th_m += move
        dth_c = self.ratio * move
        flow_used = abs(dth_m) / dt / (2 * math.pi) * P.ORBITROL_DISPLACEMENT
        tau_k = k * (self.th_c + dth_c - th_k) - c * om_k
        at_relief = abs(tau_k) >= tau_relief * 0.999
        if abs(tau_k) > tau_relief:                         # an external load beyond the relief makes the column yield
            tau_k = math.copysign(tau_relief, tau_k)
            dth_c = th_k + (tau_k + c * om_k) / k - self.th_c
        dp = tau_k / self.vol_per_rad                      # cylinder pressure difference (Pa), sign = steering side
        # leakage across the unit and cylinder seals (Q = c_leak dp) moves oil toward relieving the pressure
        leak = P.STEER_LEAK * dp
        dth_c -= leak / self.vol_per_rad * dt
        self.th_c = float(self.th_c + dth_c)
        # reaction torque on the steering wheel: the preloaded centring springs within the spool travel (held to the
        # meter below the preload, stiff and damped; past it preload + k_spool x the further deflection); beyond the
        # travel the stop, plus the torque of pumping oil by hand against the cylinder pressure (dp D / 2 pi) while the
        # wheel outruns the flow
        d = th_w - self.th_m
        if abs(d) <= self.d_hold:
            tau_w = -STOP_K * d - STOP_C * om_w
        else:
            tau_w = -math.copysign(self.pre + self.k_spool * (min(abs(d), dmax) - self.d_hold), d)
        if abs(d) > dmax:
            tau_w -= STOP_K * (d - math.copysign(dmax, d)) + STOP_C * om_w
        tau_hand = 0.0
        if forced != 0.0:
            tau_hand = abs(dp) * P.ORBITROL_DISPLACEMENT / (2 * math.pi)
            tau_w -= tau_hand * math.tanh(om_w / 0.2)
            self.energy["manual_work"] += abs(tau_hand * om_w) * dt
        self.energy["leak_loss"] += abs(leak * dp) * dt
        if at_relief:
            self.energy["relief_loss"] += abs(tau_relief * om_k) * dt
        self.tele = dict(meter=self.th_m, cyl_target=self.th_c, knuckle=th_k, pressure_Pa=dp, spool=d,
                         wheel_torque=tau_w, flow=flow_used, relief=at_relief, hand_pump_torque=tau_hand)
        return {self.wheel_dof: tau_w, self.knuckle_dof: tau_k}, flow_used

    def checkpoint(self):
        return dict(m=self.th_m, c=self.th_c, e=dict(self.energy))

    def restore(self, c):
        self.th_m, self.th_c, self.energy = c["m"], c["c"], dict(c["e"])
