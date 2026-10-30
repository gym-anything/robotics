"""MIG guns on the torch arm's flange: one spec per arm model.

BA1 (UR10e, default): Tregaskiss BA1 class cobot gun. Read: 22 deg neck, TCP 350 mm, 2.76 kg with its mounting arm,
  centre of mass (0.67, -147.9, 35.65) mm in the maker's frame, 5/8 in (15.9 mm) nozzle bore, contact tip recessed
  1/8 in (3.2 mm) for spray; a tapered nozzle (Tregaskiss 401-6-62, a BA1 nozzle: 1.062 in outside diameter, 73 mm long, read; tip 19 mm, estimate).
  Authored geometry (no drawing was available): mounting insulator, straight body along the
  flange axis, then the neck and nozzle bent 22 deg about the tool x axis; lengths put the work point at the 19 mm
  contact-tip-to-work distance 350 mm out along the flange axis. Part masses sum to the read 2.76 kg (centre of mass 143
  mm out along the axis, read 148 mm; lateral offset 8 mm, read 36 mm: the mounting arm's offset is not modelled).
RM2 (FR3, light torch): TBi RoboMIG RM2 class neck (0.7 kg, read) on a KS-2 class collision sensor (0.85 kg with its
  mount, read; the sensor returns to +-0.03 mm at 300 mm, so the work point is ~300 mm out), 22 deg neck; authored
  geometry to a 300 mm work point, ~1.6 kg in all (the FR3 alternative inside its 3 kg payload; the hose package hangs
  from a balancer and is not carried).
All in the flange frame with z out of the flange (UR10e tool0, FR3 link8).
"""

import math
from dataclasses import dataclass, field

import numpy as np

TIP_RECESS = 3.2e-3
CTWD_NOMINAL = .019


@dataclass(frozen=True)
class GunSpec:
    name: str
    neck_deg: float
    mount: tuple               # (r, z0, z1, mass)
    body: tuple                # (r, z0, z1, mass)
    neck: tuple                # (r, length, mass)
    nozzle: tuple              # (r_base, r_tip, length, mass): a tapered nozzle
    nozzle_sides: int = 32     # the collision cone's polygon (circumscribed: it never undercuts the cone)
    source: str = ""

    @property
    def axis(self):
        a = math.radians(self.neck_deg)
        return np.array([0., math.sin(a), math.cos(a)])

    @property
    def neck_start(self):
        return np.array([0., 0., self.body[2]])

    @property
    def nozzle_start(self):
        return self.neck_start + self.axis*self.neck[1]

    @property
    def nozzle_end(self):
        return self.nozzle_start + self.axis*self.nozzle[2]

    @property
    def contact_tip(self):
        return self.nozzle_end - self.axis*TIP_RECESS

    @property
    def tcp(self):
        return self.contact_tip + self.axis*CTWD_NOMINAL

    def rotation(self):
        """Local z -> the wire axis: rotation about the flange x axis by -neck_deg (asserted)."""
        a = math.radians(self.neck_deg)
        R = np.array([[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])
        assert np.allclose(R @ np.array([0., 0., 1.]), self.axis)
        return R

    def parts(self):
        """(name, kind, center (flange frame), rotation (local z = shape axis), dims, mass)."""
        Rz, Rn = np.eye(3), self.rotation()
        m, b, n, z = self.mount, self.body, self.neck, self.nozzle
        out = [("gun_mount", "cylinder", np.array([0, 0, (m[1] + m[2])/2]), Rz, (m[0], (m[2] - m[1])/2), m[3]),
               ("gun_body", "cylinder", np.array([0, 0, (b[1] + b[2])/2]), Rz, (b[0], (b[2] - b[1])/2), b[3]),
               ("gun_neck", "capsule", self.neck_start + self.axis*n[1]/2, Rn, (n[0], n[1]/2), n[2])]
        # the tapered nozzle as one convex cone (local +z toward the work, dims (r_base, r_tip, half length)). A straight
        # 11.5 mm cylinder could not reach the procedure's 19 mm contact tip to work in a 90 deg corner at 45 deg (its tip
        # 0.3 mm into both walls; a phantom until MuJoCo's midphase was switched off); four stacked cylinders at their
        # wider ends' radii stood 1 mm proud of the cone at the tip and stopped a pressed torch early
        r_base, r_tip, L, mass = z
        out.append(("gun_nozzle", "cone", self.nozzle_start + self.axis*L/2, Rn, (r_base, r_tip, L/2), mass))
        return out

    def mass_properties(self):
        ps = self.parts()
        mass = sum(p[5] for p in ps)
        return mass, sum(p[5]*p[2] for p in ps)/mass


BA1 = GunSpec("BA1 cobot MIG gun (Tregaskiss class)", 22., mount=(.035, 0., .045, .60), body=(.022, .045, .200, 1.30),
              neck=(.013, .072, .55), nozzle=(.0135, .0095, .073, .31),
              source="Tregaskiss BA1 / CA3, read; nozzle the 401-6-62 tapered 5/8 in bore nozzle, listed for the BA1 "
                     "with AccuLock R diffusers (Tregaskiss SP-BA1 p. 5, read): 1.062 in outside diameter, 2.88 in "
                     "(73 mm) long (distributor listing, read); 19 mm tip outside diameter, estimate: bore + 1.5 mm "
                     "wall; the neck was shortened 3 mm when the nozzle took its read length so the work "
                     "point stays at the read 350 mm; the other lengths authored")
RM2 = GunSpec("RM2 light torch on a KS-2 collision sensor (TBi class)", 22., mount=(.031, 0., .062, .85),
              body=(.017, .062, .150, .17), neck=(.012, .075, .43), nozzle=(.0105, .0085, .060, .15),
              source="TBi RoboMIG RM2 + KS-2, read (masses); geometry authored to a 300 mm work point")
GUNS = {"ur10e": BA1, "fr3": RM2,
        # the Panda: the same light torch on link8 (the same ISO 9409-1-31.5-4-M5 flange as the FR3, 3 kg payload [read:
        # Franka Emika Panda datasheet]); its Franka Hand removed (robots/models/panda welding_torch profile)
        "panda": RM2,
        # the G1: the light torch clamped to the back of the right Dex3 palm (G1_TORCH_MOUNT below)
        "g1": RM2}


@dataclass(frozen=True)
class WristMount:
    """A torch clamped to a robot link that has no tool flange: the flange frame (the gun's mounting face; z along the
    gun, the neck bending toward +y) in the link frame, and the clamp that holds it there (a collider with mass)."""
    link: str
    origin: tuple              # flange frame origin, link frame (m)
    axes: tuple                # flange frame x, y, z axes in the link frame
    clamp_center: tuple        # clamp box centre, link frame (m)
    clamp_half: tuple          # clamp box half extents (m)
    clamp_mass: float          # kg
    source: str = ""

    def rotation(self):
        R = np.array(self.axes, float).T
        assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) > 0
        return R


# The G1 (29 joints, Dex3-1 hands) has no tool flange: the RM2 + KS-2 light torch is clamped to the back of the right
# Dex3 palm with its axis along the hand (the palm is part of the URDF's right_wrist_yaw_link: its collision mesh spans
# x 41.5-128.2 mm, y -24.4-17.3 mm, z +-43.9 mm; the fingers flex toward +y, so the back of the hand is -y). The KS-2's
# 62 mm body sits 1.6 mm off the palm's back, its axis 57 mm behind the link axis, from x = 45 mm; the neck bends 22 deg
# away from the fingers (toward -y). An aluminium saddle clamp (62 x 15 x 60 mm, 0.15 kg) bridges palm and sensor.
# Chosen by the reach study E23: bent toward the fingers instead, the torch reached 1 of 9
# seam poses from a stance where the dorsal bend reached 8 [authored placement, estimate masses]
G1_TORCH_MOUNT = WristMount("right_wrist_yaw_link", origin=(.045, -.057, 0.), axes=((0., 0., 1.), (0., -1., 0.), (1., 0., 0.)),
                            clamp_center=(.076, -.0327, 0.), clamp_half=(.031, .0073, .030), clamp_mass=.15,
                            source="authored: saddle clamp on the Dex3 palm's back; 6061 aluminium, 2700 kg/m3")
MOUNTS = {"g1": G1_TORCH_MOUNT}


def for_model(model):
    if model not in GUNS:
        raise ValueError(f"no MIG gun is defined for arm model {model!r}")
    return GUNS[model]


# the default gun's constants (UR10e), kept for modules that use them directly
NECK_DEG = BA1.neck_deg
AXIS_TOOL = BA1.axis
CONTACT_TIP = BA1.contact_tip
TCP_TOOL = BA1.tcp


def parts():
    return BA1.parts()


def mass_properties():
    return BA1.mass_properties()
