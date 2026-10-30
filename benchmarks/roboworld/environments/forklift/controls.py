"""The truck's controls as mechanisms: return springs, detents and the sensors the truck reads.

Every pedal, lever, the direction lever, the parking-brake lever, the key, the horn button and the seat are rigid bodies on
joints (truck.py). This module computes the spring and detent torques on those joints from their angles, and reads the
joint angles the way the truck's own linkages and switches do: throttle, inching fraction, brake fraction and the service
brake's line pressure (its master cylinder; the pressure also pushes back on the pedal, so it follows the force on the
pad), spool fractions, direction selected, parking brake applied, key position, horn switch, seat (operator-presence)
switch. The inching pedal pushes the brake pedal through a lug past its clutch range. Nothing else in the
code can set those signals: the only way to change them is to move the control bodies (the robot's hands and feet, or a
labelled phantom force at the pad in an assisted probe).
"""

import math

import numpy as np

from . import params as P
from .truck import PEDAL_PRESS_SIGN
from .drivetrain import BRAKE_LINE_DAMPING, brake_pedal_torque, brake_pressure

LEVERS = ("lever_lift", "lever_tilt", "lever_shift")
PEDALS = ("pedal_inch", "pedal_brake", "pedal_accel")
SPOOL_DEADBAND = P.SPOOL_DEADBAND          # fraction of lever travel before a spool opens (open-centre valve overlap; estimate)
SWITCH_FILTER_S = P.SEAT_SWITCH_FILTER         # seat-switch debounce (s)
PRELOAD_ZONE = 0.002           # rad over which a preloaded centring spring builds its preload (spool spring seat)
LEVER_DAMPING = P.LEVER_DAMPING            # N m s/rad, spool drag in oil referred to the lever (estimate)
DIR_DAMPING = P.DIR_DAMPING              # N m s/rad, direction-lever detent friction (estimate, ~0.7 critical)
KEY_DAMPING = P.KEY_DAMPING            # N m s/rad, key cylinder (estimate, ~0.7 critical)


def _smooth_sign(x, eps):
    return math.tanh(x / eps)


class Controls:
    def __init__(self, truck, dof_of, operator_mass):
        self.truck = truck
        self.dof = {n: dof_of[n] for n in (*PEDALS, *LEVERS, "dir_lever", "park_lever", "key", "seat", "horn")
                    if n in dof_of}
        s = P.SEAT_SUSPENSION
        # the seat's weight adjustment is set for the robot sitting on it (a real seat's weight dial), so the loaded seat
        # rests at mid-travel: preload balances seat + the share of the operator's weight on the cushion
        self.operator_mass = float(operator_mass)
        self.seat_preload = (P.M_SEAT + P.SEAT_LOAD_SHARE * self.operator_mass) * P.G + P.SEAT_SUSPENSION["k"] * P.SEAT_RIDE
        self.seat_switch = False
        self._seat_load_f = 0.0
        self.signals = {}

    def torques(self, q, qd, dt):
        """Spring/detent generalized forces {dof: tau} and the decoded signals."""
        tau = {}
        sig = {}
        x_of = {}
        for n in PEDALS:
            d = self.dof[n]
            spec = {"pedal_accel": P.PEDAL_ACCEL, "pedal_brake": P.PEDAL_BRAKE, "pedal_inch": P.PEDAL_INCH}[n]
            sgn = PEDAL_PRESS_SIGN[n]
            x = sgn * q[d]                                   # pressed travel >= 0
            x_of[n] = x
            tau[d] = -sgn * (spec["preload"] + spec["k"] * max(x, 0.0))
            sig[n] = float(np.clip(x / spec["travel"], 0.0, 1.0))
        # ---- service brake pedal: its master cylinder. The line pressure pushes back on the pedal through the pushrod
        # (drivetrain.brake_pressure: the displaced oil through the circuit's pressure-volume curve), so the pedal settles
        # where the pad force balances return spring and pressure: pressure follows force, travel follows the oil taken.
        db, sb = self.dof["pedal_brake"], PEDAL_PRESS_SIGN["pedal_brake"]
        tau[db] += -sb * brake_pedal_torque(max(x_of["pedal_brake"], 0.0))
        if x_of["pedal_brake"] * P.BRAKE_PAD_ARM > P.BRAKE_HARDWARE["play"]:     # the pushrod moves oil: the tube's drag
            tau[db] -= BRAKE_LINE_DAMPING * qd[db]
        sig["brake_pressure"] = brake_pressure(max(x_of["pedal_brake"], 0.0))
        # ---- inching pedal: past INCHING_SLIP_FRACTION of its travel (the clutch fully released) its lug meets the brake
        # pedal's arm and pushes it down, so the inching pedal brakes through the same master cylinder and feels its
        # pressure (single-pedal inch-brake practice); a contact force element between the two pedal arms (coaxial pivots)
        di, si = self.dof["pedal_inch"], PEDAL_PRESS_SIGN["pedal_inch"]
        lug = P.INCH_BRAKE_LUG
        gap = x_of["pedal_inch"] - x_of["pedal_brake"] - P.INCHING_SLIP_FRACTION * P.PEDAL_INCH["travel"]
        if gap > 0.0:
            rate = si * qd[di] - sb * qd[db]                 # closing speed of the lug (rad/s)
            f = max(0.0, lug["k"] * gap + lug["c"] * rate)
            tau[db] += sb * f                                 # pushes the brake pedal down
            tau[di] += -si * f                                # and the inching pedal back
        sig["inch_lug_engaged"] = bool(gap > 0.0)
        for n in LEVERS:
            d = self.dof[n]
            L = P.LEVER
            # preloaded centring spring: the lever does not leave centre until the preload is overcome (stiff zone of
            # PRELOAD_ZONE rad), plus the spool's viscous drag in oil
            tau[d] = -(L["preload"] * _smooth_sign(q[d], PRELOAD_ZONE) + L["k"] * q[d]) - LEVER_DAMPING * qd[d]
            f = -q[d] / L["travel"]                          # pull toward the operator = negative angle = +1
            sig[n] = float(np.sign(f) * max(0.0, (abs(f) - SPOOL_DEADBAND) / (1 - SPOOL_DEADBAND))) if abs(f) > SPOOL_DEADBAND else 0.0
        # direction lever: detents at F (-travel), N (0), R (+travel)
        d = self.dof["dir_lever"]
        T = P.DIR_LEVER["travel"]
        x = float(np.clip(q[d], -T, T))
        tau[d] = -P.DIR_LEVER["detent"] * math.sin(2 * math.pi * x / T) - DIR_DAMPING * qd[d]
        sig["direction"] = 1 if q[d] < -0.5 * T else (-1 if q[d] > 0.5 * T else 0)
        # parking brake: over-centre hand lever, applied = negative angle. Released side: preloaded spring and cable
        # tension rising toward the over-centre point; past it the toggle snaps the lever into the applied stop.
        d = self.dof["park_lever"]
        pk = P.PARK_LEVER
        if q[d] > -pk["overcentre"]:
            tau[d] = pk["preload"] - pk["k"] * q[d]
        else:
            tau[d] = -(pk["preload"] + pk["k"] * (q[d] + pk["travel"]))
        tau[d] -= P.PARK_LEVER_DAMPING * qd[d]
        sig["park"] = 1.0 if q[d] < -pk["overcentre"] else 0.0          # set: the toggle is over its centre
        # the brake's share of PARK_BRAKE_TORQUE: the cable stretches with the pull beyond its slack, full clamp at centre
        sig["park_brake"] = float(np.clip((-q[d] - pk["slack"]) / (pk["overcentre"] - pk["slack"]), 0.0, 1.0))
        # key: OFF (0) and ON detents, spring return from START
        d = self.dof["key"]
        K = P.KEY
        if q[d] > K["on"]:
            tau[d] = -K["k"] * (q[d] - K["on"])            # spring return from START to ON (continuous at ON)
        else:
            tau[d] = -P.KEY_DETENT * math.sin(2 * math.pi * q[d] / K["on"])
        tau[d] -= KEY_DAMPING * qd[d]
        KS = P.KEY_SWITCH
        sig["key"] = "start" if q[d] > K["on"] + KS["start"] * (K["start"] - K["on"]) else ("on" if q[d] > KS["on"] * K["on"] else "off")
        # horn button (prismatic, pressed = negative travel)
        if "horn" in self.dof:
            d = self.dof["horn"]
            H = P.HORN_SPRING                               # 3 N preload up, 1.5 N/mm; the switch closes at 2 mm
            tau[d] = H["preload"] - H["k"] * q[d] - H["c"] * qd[d]
            sig["horn"] = bool(q[d] < -H["switch"])
        # seat suspension: preloaded spring and damper about mid-travel
        d = self.dof["seat"]
        s = P.SEAT_SUSPENSION
        spring = self.seat_preload - s["k"] * q[d]
        tau[d] = spring - s["c"] * qd[d]
        load = spring - P.M_SEAT * P.G                       # quasi-static cushion load
        a = min(1.0, dt / SWITCH_FILTER_S)
        self._seat_load_f += a * (load - self._seat_load_f)
        self.seat_switch = self._seat_load_f > P.SEAT_SWITCH_N
        sig["seat_switch"] = self.seat_switch
        sig["seat_load_N"] = self._seat_load_f
        self.signals = sig
        return tau, sig

    def checkpoint(self):
        return dict(seat_load=self._seat_load_f, switch=self.seat_switch)

    def restore(self, c):
        self._seat_load_f = c["seat_load"]; self.seat_switch = c["switch"]
