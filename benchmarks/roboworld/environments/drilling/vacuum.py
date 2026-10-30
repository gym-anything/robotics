"""The shop vacuum of the anchor stations: a physical unit that must be switched on.

A wet/dry vacuum with a certified HEPA filter of the RIDGID RV2400HF class (14 US gal; owner's manual SP6775, p. 5:
"supplied with a certified HEPA dust filter and meets the EPA definition of a HEPA vacuum"; p. 10: "turn the unit on by
pushing the switch from position "O" to position "I""; repair parts p. 16: 830564 "Actuator, Switch", 73180 "Switch,
Toggle"). It stands on the floor beside the robot cart: a static body (its tank, power head and hose port are colliders),
with its on/off rocker on top of the power head. The rocker is a real joint: a two-position actuator held in either
position by a detent (a position drive towards the nearer stop, VACUUM_ROCKER_* in params.py); a fingertip push on its
"I" half that exceeds the detent turns it over to ON, a push on its "O" half turns it back. Nothing switches it but
contact.

Interface used by the other work streams (fixed):
  layout["vacuum"] = {"xyz": [x, y, z], "yaw": rad, "port_local": [x, y, z], "switch_local": [x, y, z],
                      "switch_normal_local": [x, y, z]}         (unit frame: origin on the floor under the tank axis)
  scene body labels "vacuum_unit" (the static unit) and "vacuum_switch" (the rocker)
  scene.tools["vacuum"].running(world) -> bool                  (the rocker on its "I" side)
switch_local is the centre of the rocker's "I" half (where a fingertip turns the unit on); port_local the centre of the
hose inlet on the tank's front.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from .tools import JointRef, Parts, Tool, link, rel_xform, set_initial_q, wq

# unit geometry (unit frame: z up, x towards the inlet; origin on the floor under the tank's axis)
TANK_R = 0.5*P.VACUUM_TANK_D_M
TANK_H = P.VACUUM_TANK_H_M
HEAD_H = P.VACUUM_HEAD_H_M
ROCKER_L, ROCKER_W, ROCKER_T = P.VACUUM_ROCKER_SIZE_M   # along its pivot's normal (the two halves), across, thickness
THROW = P.VACUUM_ROCKER_THROW_RAD


def geometry():
    """Points of the unit in its own frame (used by layout.py for the interface fields). switch_local: the top face of
    the rocker's "I" half, mid-half, with the unit OFF (that half raised by the throw), where a fingertip turns it on;
    switch_off_local: the "O" half's top face with the unit ON."""
    top = TANK_H + HEAD_H
    pivot = np.array([0.5*TANK_R, 0., top + 0.002])
    a, h = 0.25*ROCKER_L, ROCKER_T
    c, s = math.cos(THROW), math.sin(THROW)
    raised = np.array([a*c - h*s, 0., a*s + h*c])            # (a, h) turned so its half rises by the throw
    return dict(top=top, pivot=pivot,
                switch_local=pivot + raised,
                switch_off_local=pivot + raised*np.array([-1., 1., 1.]),
                port_local=np.array([TANK_R + 0.03, 0., 0.75*TANK_H]))


class VacuumUnit(Tool):
    """Static shop vacuum with an on/off rocker (two stable positions on a revolute joint)."""

    def __init__(self, sc, spec):
        self.name = "vacuum"
        pr = Parts(sc)
        b = sc.b
        p0 = np.asarray(spec["xyz"], float)
        Rz = Rotation.from_euler("z", spec["yaw"])
        pose = (p0, Rz)
        g = geometry()
        U = link(sc, "vacuum_unit", pose)
        # tank (on its caster base), power head, hose inlet: colliders of a static unit
        pr.cyl(U, "vacuum_tank", (0., 0., 0.5*TANK_H), TANK_R, 0.5*TANK_H, "z", "plastic_grey", density=200.,
               mu=P.MU["default"])
        pr.cyl(U, "vacuum_head", (0., 0., TANK_H + 0.5*HEAD_H), TANK_R - 0.01, 0.5*HEAD_H, "z", "controller_black",
               density=200., mu=P.MU["default"])
        pr.cyl(U, "vacuum_inlet", tuple(g["port_local"] - np.array([0.015, 0., 0.])), 0.032, 0.03, "x", "controller_black",
               density=200.)
        # the rocker's bezel: a raised housing on the head top with the rocker in it (a notch the fingertip must find)
        sc.box("vacuum_switch_bezel_front", tuple(g["pivot"] + np.array([0.5*ROCKER_L + 0.004, 0., -0.004])),
               (0.003, 0.5*ROCKER_W + 0.006, 0.006), "controller_black", body=U, density=200.)
        sc.box("vacuum_switch_bezel_back", tuple(g["pivot"] + np.array([-0.5*ROCKER_L - 0.004, 0., -0.004])),
               (0.003, 0.5*ROCKER_W + 0.006, 0.006), "controller_black", body=U, density=200.)
        Sp = (p0 + Rz.apply(g["pivot"]), Rz)
        S = link(sc, "vacuum_switch", Sp)
        sc.box("vacuum_switch_rocker", (0., 0., 0.5*ROCKER_T), (0.5*ROCKER_L, 0.5*ROCKER_W, 0.5*ROCKER_T),
               "switch_red", body=S, density=1100., mu=P.MU["default"])
        I = sc.wp.transform_identity()
        ju = b.add_joint_fixed(-1, U, parent_xform=sc.wp.transform(tuple(map(float, p0)), wq(Rz)), child_xform=I,
                               label="vacuum_unit_fixed")
        # rocker about the unit's y axis: positive turns the "I" half (+x) down = ON
        k = P.VACUUM_ROCKER_DETENT_NM/THROW
        js = b.add_joint_revolute(U, S, parent_xform=rel_xform(sc, pose, Sp), child_xform=I, axis=(0., 1., 0.),
                                  limit_lower=-THROW, limit_upper=THROW, target_pos=-THROW, target_ke=k,
                                  target_kd=P.VACUUM_ROCKER_DAMPING, armature=P.VACUUM_ROCKER_ARMATURE,
                                  actuator_mode=sc.newton.JointTargetMode.POSITION, label="vacuum_switch")
        b.add_articulation([ju, js], label="vacuum_assembly")
        set_initial_q(b, js, -THROW)
        self.unit, self.switch = U, S
        self.body = U
        self.bodies = [U, S]
        self.j = dict(rocker=JointRef(js, "rocker"))
        self.spring_targets = dict(rocker=-THROW)
        self.positions = (-THROW, THROW)
        sc.mutual_filter(self.bodies)
        sc.tool_bodies.update(self.bodies)

    def joints(self):
        return list(self.j.values())

    def pre_step(self, w):
        # the detent: a drive towards the nearer of the two stops (snaps over at the middle)
        q = self.j["rocker"].pos(w)
        w.tq[self.j["rocker"].q] = self.positions[1] if q > 0. else self.positions[0]

    def running(self, world):
        return bool(self.j["rocker"].pos(world) > 0.)

    def state(self):
        return dict(running=self.running(self.w), rocker_rad=round(self.j["rocker"].pos(self.w), 5))


def build(sc, spec):
    return sc.add_tool(VacuumUnit(sc, spec))
