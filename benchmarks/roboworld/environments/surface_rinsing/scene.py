"""Physics scene (Newton 1.5 ModelBuilder): the dish room's installed parts as static colliders, the articulated
fixtures that control water (wrist-blade valve handles, swing spout, spring-balanced pre-rinse spray valves with their
squeeze levers and hoses, the hose reel's hose and valve, lever wastes), the Mobile FR3 Duo (classroom_arrangement's
robot stack, unchanged) and the preset's ware, produce and racks as free rigid bodies.

Valve springs are explicit forces applied by the water system each tick (water.network), never position targets with
references beyond a stop (as in painting and soil_excavation). Hoses are capsule chains with two bending axes per joint
(joint stiffness EI / l, light damping). Revolved ware collides through its convex sector hulls; its mass and inertia are
the revolved polygon's integrals at the catalogue mass.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement.scene import Scene as ClassroomScene, volume
from classroom_arrangement.furniture import Part
from wine_pouring import geometry as G
from . import params as P
from . import layout as L
from . import ware as W


# hose bending: stiffness EI / l (EI 0.05 N m2) and damping 0.03 N m s/rad per joint (estimate: a rubber hose's hysteresis;
# with 0.004 the reel hose was still swinging at 0.14 m/s at the end of the 5 s reset settle)
HOSE_KD = 0.03


def quat_from_R(R):
    return tuple(Rotation.from_matrix(R).as_quat())


def frame_z_to(d, up_hint=(0., 0., 1.)):
    """Rotation whose local z is the unit vector d (x chosen orthogonal, as close to up_hint x d as possible)."""
    d = np.asarray(d, float); d = d/np.linalg.norm(d)
    x = np.cross(up_hint, d)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross((1., 0., 0.), d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    return np.stack([x, y, d], 1)


def lever_damping(spec):
    """Critical damping of a squeeze lever's hinge against its return spring (N m s/rad); params.LEVER."""
    arm = spec["lever_arm"]
    k = (spec["full_n"] - spec["crack_n"])*arm/((spec["travel"] - spec["crack_travel"])/arm)
    return 2.0*math.sqrt(k*P.LEVER["armature"])


class RinsingScene(ClassroomScene):
    ENV_ID = "surface-rinsing"

    def __init__(self, config, newton, wp, spec):
        self.c, self.newton, self.wp, self.spec = config, newton, wp, spec
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=(0., 0., -P.GRAVITY))
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.parts = {}, {}
        self.bodies, self.items = {}, {}
        self.glide_shapes, self.wheel_shapes, self.pad_shapes = [], [], {}
        self.cameras = {}
        self.static_shapes = {}
        self.static_role = {}            # shape index -> station / pool role of an installed part
        self.fixtures = {}               # name -> dict(kind, bodies, joints, ...): water-controlling mechanisms
        self.fixture_bodies = set()
        self.ware = {}                   # item name -> ware.Ware
        self.visual_only = set()         # collider shapes never drawn (the render uses the revolved polygon)
        self.room()
        self.robot()
        self.fixtures_build()
        self.items_build()
        self.views()

    # ------------------------------------------------------------------ helpers
    def set_mass(self, body, mass, com=None, inertia=None):
        b, wp = self.b, self.wp
        b.body_mass[body] = float(mass)
        if com is not None:
            b.body_com[body] = wp.vec3(*map(float, com))
        if inertia is not None:
            b.body_inertia[body] = wp.mat33(*np.diag(inertia).ravel().tolist())
            b.body_inv_inertia[body] = wp.mat33(*np.diag(1.0/np.asarray(inertia)).ravel().tolist())
        b.body_inv_mass[body] = 1.0/float(mass)

    def add_part(self, body, p, prefix):
        if not p.collide and body >= 0 and p.mass > 0 and p.kind in ("box", "cylinder", "capsule"):
            # drawn-only part with mass (a swivel collar): its mass and inertia at its place, no collision, not drawn
            wp = self.wp
            hx, hy, hz = (p.size if p.kind == "box" else (p.size[0], p.size[0], p.size[1]))
            i = self.b.add_shape_box(body, xform=wp.transform(p.pos, wp.quat(*p.quat)), hx=hx, hy=hy, hz=hz,
                                     cfg=self.cfg(density=p.mass/(8*hx*hy*hz), has_shape_collision=False,
                                                  has_particle_collision=False, is_visible=False),
                                     label=f"{prefix}/{p.name}_mass")
            self.materials[i] = "mass_only"
            return None
        if p.kind == "sphere":
            if not p.collide:
                return None
            wp = self.wp
            dens = p.mass/(4/3*math.pi*p.size[0]**3) if body >= 0 else 0.
            i = self.b.add_shape_sphere(body, xform=wp.transform(p.pos, wp.quat(*p.quat)), radius=p.size[0],
                                        cfg=self.cfg(mu=p.mu if p.mu is not None else 0.4, density=dens), label=f"{prefix}/{p.name}")
            self.materials[i] = p.material
            return i
        return super().add_part(body, p, prefix)

    # ------------------------------------------------------------------ room
    def room(self):
        b = self.b
        b.add_ground_plane(cfg=self.cfg(mu=0.6), label="floor")
        self.materials[b.shape_count - 1] = "quarry_tile"
        self.static_shapes["floor"] = b.shape_count - 1
        self.static_role[b.shape_count - 1] = "floor"
        for p in L.static_parts():
            p.mu = 0.35 if p.material in ("steel", "chrome") else 0.5
            i = self.add_part(-1, p, "room")
            self.static_shapes[p.name] = i
            self.static_role[i] = p.role

    # ------------------------------------------------------------------ fixtures
    def fixtures_build(self):
        st = self.spec["station"]
        if st == "pot-sink":
            hz = L.SPLASH_TOP
            self.blade("hot", (L.RISER_X - 0.10, L.Y - 0.075, hz), (0., -1., 0.), "mix", +1)
            self.blade("cold", (L.RISER_X + 0.10, L.Y - 0.075, hz), (0., -1., 0.), "mix", -1)
            self.blade("faucet_valve", (L.RISER_X, L.Y - 0.090, hz + 0.16), (0., -1., 0.), "faucet", -1)
            self.swing_spout("spout", (L.RISER_X, L.RISER_Y - 0.03, hz + 0.12), 0.30, math.radians(40.0))
            self.spray_valve("prsv", (L.RISER_X, L.Y - 0.30, hz + 0.47 + L.PRSV_RISE), P.PRSV, links=P.PRSV["hose_links"])
            for n, (x0, x1) in L.BOWLS.items():
                self.lever_waste(f"waste_{n}", (0.5*(x0 + x1), L.SINK_Y0 - 0.012, L.RIM - 0.055), n)
        elif st == "pre-rinse":
            rx, ry = L.R_RISER
            hz = L.TABLE_Z + P.DISH_TABLE["splash_h"] + 0.08
            self.blade("hot", (0.06, ry - 0.10, hz), (1., 0., 0.), "mix", -1)
            self.blade("cold", (0.06, ry + 0.10, hz), (1., 0., 0.), "mix", +1)
            self.spray_valve("prsv", (rx + 0.28, ry, L.TABLE_Z + P.DISH_TABLE["splash_h"] + 0.67 + L.PRSV_RISE), P.PRSV,
                             links=P.PRSV["hose_links"])
        elif st == "prep-sink":
            fx, fy = L.V_FAUCET
            hz = L.V_RIM + 0.21
            self.blade("hot", (L.X - 0.065, fy + 0.10, hz), (-1., 0., 0.), "mix", -1)
            self.blade("cold", (L.X - 0.065, fy - 0.10, hz), (-1., 0., 0.), "mix", +1)
            self.fixtures["gooseneck"] = dict(kind="gooseneck", outlet=(L.VBOWL_X0 + 0.5*L.VB_D + 0.02, fy, L.V_RIM + 0.36),
                                               direction=(0., 0., -1.), bodies=[], joints=[])
        elif st == "can-wash":
            self.reel_hose("reel", P.REEL)

    def blade(self, name, pivot, axis, group, sign, up=(0., 0., 1.)):
        """Wrist-blade quarter-turn handle (4 in blade): revolute about its stem axis (`axis`, pointing at the user), blade
        up when closed; it opens by turning sign x 90 deg about the stem (sign -1: the other way, for the cold side). Any
        link or object pushing the blade turns it (Coulomb friction, end stops)."""
        b, wp = self.b, self.wp
        F = P.FAUCET
        axis = np.asarray(axis, float)
        Rb = np.stack([np.cross(axis, up), axis, np.asarray(up, float)], 1)      # local x across, y stem, z blade
        body = b.add_link(xform=wp.transform(tuple(pivot), wp.quat(*quat_from_R(Rb))), label=f"fx_{name}")
        parts = [Part("hub", "cylinder", (0.014, 0.018), (0., 0.018, 0.), (0.7071067811865476, 0., 0., 0.7071067811865476),
                      mass=0.04, material="chrome"),
                 Part("blade", "box", (0.007, 0.004, F["blade_len"]/2), (0., 0.030, F["blade_len"]/2), mass=F["blade_mass"],
                      material="chrome")]
        shapes = [self.add_part(body, p, f"fx_{name}") for p in parts]
        j = b.add_joint_revolute(-1, body, parent_xform=wp.transform(tuple(pivot), wp.quat(*quat_from_R(Rb))),
                                 child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(0., float(sign), 0.),
                                 limit_lower=0.0, limit_upper=math.radians(F["turn_deg"]), friction=F["friction_nm"],
                                 armature=2e-3, target_ke=0.0, target_kd=0.0, label=f"fx_{name}_joint")
        b.add_articulation([j], label=f"fx_{name}_articulation")
        self.bodies[f"fx_{name}"] = body
        self.parts[f"fx_{name}"] = parts
        self.fixture_bodies.add(body)
        self.fixtures[name] = dict(kind="blade", group=group, body=body, joint=f"fx_{name}_joint", shapes=shapes, sign=sign,
                                   pivot=tuple(pivot), axis=tuple(axis))

    def swing_spout(self, name, pivot, length, yaw0):
        """12 in add-on swing spout: a tube on a vertical swivel (O-ring friction) with a laminar outlet at its tip."""
        b, wp = self.b, self.wp
        Rz = Rotation.from_euler("z", yaw0).as_matrix()
        body = b.add_link(xform=wp.transform(tuple(pivot), wp.quat(*quat_from_R(Rz))), label=f"fx_{name}")
        # the swivel collar turns round the riser (coaxial): drawn and weighed, not a collider; the tube starts outside it
        parts = [Part("swivel", "cylinder", (0.016, 0.025), (0., 0., 0.), mass=0.15, material="chrome", collide=False),
                 Part("tube", "capsule", (0.0095, (length - 0.03)/2 - 0.0095), (0., -0.03 - (length - 0.03)/2, 0.),
                      (0.7071067811865476, 0., 0., 0.7071067811865476), mass=0.25, material="chrome"),
                 Part("nose", "capsule", (0.011, 0.02), (0., -length, -0.025), mass=0.05, material="chrome")]
        shapes = [self.add_part(body, p, f"fx_{name}") for p in parts]
        j = b.add_joint_revolute(-1, body, parent_xform=wp.transform(tuple(pivot), wp.quat(*quat_from_R(Rz))),
                                 child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(0., 0., 1.),
                                 limit_lower=-math.radians(130.0) - yaw0, limit_upper=math.radians(130.0) - yaw0,
                                 friction=0.15, armature=5e-3, label=f"fx_{name}_joint")
        b.add_articulation([j], label=f"fx_{name}_articulation")
        self.bodies[f"fx_{name}"] = body
        self.parts[f"fx_{name}"] = parts
        self.fixture_bodies.add(body)
        self.fixtures[name] = dict(kind="spout", body=body, joint=f"fx_{name}_joint", shapes=shapes,
                                   outlet=(0., -length, -0.047), direction=(0., 0., -1.))

    def spray_valve(self, name, anchor, spec, links=6, link_len=0.07):
        """Pre-rinse spray valve on a spring balancer: balancer slide (prismatic, explicit spring force), a hose of
        capsule links with two bending axes per joint, the valve body and its squeeze lever (revolute, explicit
        preloaded return spring, hard stops). The unit's overhead spring and 44 in hose are this: a 0.63 m free hose
        (params.PRSV hose_links x 7 cm) hanging from a slide with 0.6 m of travel. (With 0.42 m of hose the valve
        cannot be brought in front of a racked plate and pointed back at it.)"""
        b, wp = self.b, self.wp
        joints = []
        bal = b.add_link(xform=wp.transform(tuple(anchor), wp.quat_identity()), label=f"fx_{name}_balancer")
        self.add_part(bal, Part("clamp", "cylinder", (0.012, 0.015), (0., 0., 0.), mass=0.05, material="chrome"), f"fx_{name}_bal")
        # the balancer's slide: damping 8 N s/m, near-critical for its 30 N/m spring and ~0.7 kg (estimate: a reel-type
        # balancer's friction; at 2 N s/m the hanging valve still swung at 5 mm/s after a 5 s reset settle)
        js = b.add_joint_prismatic(-1, bal, parent_xform=wp.transform(tuple(anchor), wp.quat_identity()),
                                   child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(0., 0., -1.),
                                   limit_lower=0.0, limit_upper=0.60, armature=0.05, damping=8.0, label=f"fx_{name}_balancer_joint")
        joints.append(js)
        parent, ppos = bal, np.asarray(anchor, float) + np.array([0., 0., -0.015])
        chain = []
        r_h, m_link = 0.008, P.PRSV["mass"]*0 + 0.32*link_len
        k_bend = 0.05/link_len                              # EI / l (EI 0.05 N m2, estimate)
        for k in range(links):
            top = ppos.copy()
            c = top + np.array([0., 0., -link_len/2])
            lk = b.add_link(xform=wp.transform(tuple(c), wp.quat_identity()), label=f"fx_{name}_hose{k}")
            self.add_part(lk, Part("tube", "capsule", (r_h, link_len/2 - r_h), (0., 0., 0.), mass=m_link, material="hose_steel"),
                          f"fx_{name}_hose{k}")
            pj = (0., 0., -0.015) if k == 0 else (0., 0., -link_len/2)
            jd = b.add_joint_d6(parent, lk, parent_xform=wp.transform(pj, wp.quat_identity()),
                                child_xform=wp.transform((0., 0., link_len/2), wp.quat_identity()),
                                angular_axes=[b.JointDofConfig(axis=(1., 0., 0.), target_ke=k_bend, target_kd=HOSE_KD, armature=2e-4,
                                                               limit_lower=-1.2, limit_upper=1.2),
                                              b.JointDofConfig(axis=(0., 1., 0.), target_ke=k_bend, target_kd=HOSE_KD, armature=2e-4,
                                                               limit_lower=-1.2, limit_upper=1.2)],
                                label=f"fx_{name}_hose{k}_joint")
            joints.append(jd)
            chain.append(lk)
            self.fixture_bodies.add(lk)
            parent, ppos = lk, top + np.array([0., 0., -link_len])
        # valve body: z along the body axis (hose inlet up), spray face at z = -0.135 facing -z
        top = ppos
        vc = top + np.array([0., 0., -0.045])
        valve = b.add_link(xform=wp.transform(tuple(vc), wp.quat_identity()), label=f"fx_{name}_valve")
        vparts = [Part("body", "cylinder", (0.016, 0.06), (0., 0., -0.015), mass=0.30, material="chrome"),
                  Part("head", "cylinder", (spec.get("face_r", 0.027), 0.012), (0., 0., -0.083), mass=0.18, material="chrome"),
                  Part("bumper", "cylinder", (spec.get("face_r", 0.027) + 0.003, 0.006), (0., 0., -0.098), mass=0.04,
                       material="rubber_grey"),
                  Part("inlet", "cylinder", (0.010, 0.012), (0., 0., 0.050), mass=0.03, material="chrome")]
        vshapes = [self.add_part(valve, p, f"fx_{name}_valve") for p in vparts]
        jv = b.add_joint_d6(parent, valve, parent_xform=wp.transform((0., 0., -link_len/2), wp.quat_identity()),
                            child_xform=wp.transform((0., 0., 0.060), wp.quat_identity()),
                            angular_axes=[b.JointDofConfig(axis=(1., 0., 0.), target_ke=k_bend, target_kd=HOSE_KD, armature=2e-4,
                                                           limit_lower=-1.2, limit_upper=1.2),
                                          b.JointDofConfig(axis=(0., 1., 0.), target_ke=k_bend, target_kd=HOSE_KD, armature=2e-4,
                                                           limit_lower=-1.2, limit_upper=1.2)],
                            label=f"fx_{name}_valve_joint")
        joints.append(jv)
        # squeeze lever on the +x side: pivot at z = +0.03 about local y; positive angle swings its pad toward the body
        piv = (0.030, 0., 0.030)
        lever = b.add_link(xform=wp.transform(tuple(vc + np.array(piv)), wp.quat_identity()), label=f"fx_{name}_lever")
        lparts = [Part("lever", "box", (0.004, 0.010, 0.045), (0.0, 0., -0.045), mass=spec["lever_mass"], material="chrome")]
        lshapes = [self.add_part(lever, p, f"fx_{name}_lever") for p in lparts]
        jl = b.add_joint_revolute(valve, lever, parent_xform=wp.transform(piv, wp.quat_identity()),
                                  child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(0., 1., 0.),
                                  limit_lower=0.0, limit_upper=spec["travel"]/spec["lever_arm"],
                                  armature=P.LEVER["armature"], damping=lever_damping(spec), label=f"fx_{name}_lever_joint")
        joints.append(jl)
        b.add_articulation(joints, label=f"fx_{name}_articulation")
        for x in (valve, lever, bal):
            self.fixture_bodies.add(x)
        self.bodies.update({f"fx_{name}_valve": valve, f"fx_{name}_lever": lever, f"fx_{name}_balancer": bal})
        self.parts[f"fx_{name}_valve"], self.parts[f"fx_{name}_lever"] = vparts, lparts
        self.fixtures[name] = dict(kind="spray_valve", spec=spec, valve=valve, lever=lever, balancer=bal, chain=chain,
                                   lever_joint=f"fx_{name}_lever_joint", balancer_joint=f"fx_{name}_balancer_joint",
                                   face=(0., 0., -0.104), direction=(0., 0., -1.), shapes=vshapes + lshapes,
                                   lever_pad=(0.0, 0.0, -0.06), anchor=tuple(anchor), park=tuple(vc))

    def reel_hose(self, name, spec, link_len=0.125):
        """Open hose reel (static drum) paying out its last ~3 m of hose as a capsule chain that hangs from the drum, lies
        on the floor and rises to an EB-0107 class squeeze valve docked head-down in its wall holster."""
        b, wp = self.b, self.wp
        r, h = L.REEL, L.HOLSTER
        anchor = np.array([r["x"], r["y"] + 0.19, r["z"] - 0.23])          # just under the drum
        # valve docked: bumper face on the holster floor, inlet (hose) up
        vc = np.array([h["x"], h["y"], h["z0"] + 0.104 + 0.001])
        inlet = vc + np.array([0., 0., 0.060])
        zf = spec["hose_r"] + 0.0015
        top = inlet + np.array([0.0, 0.0, 0.16])
        side = np.array([inlet[0] + 0.11, inlet[1] + 0.06, 0.0])

        def polyline(e):
            """drum -> floor -> (a loop out into the bay, e longer) -> up beside the holster -> over -> down into the inlet"""
            loop = np.array([anchor[0] - 0.20, anchor[1] + 0.30 + e, zf])
            return [anchor, np.array([anchor[0], anchor[1], zf]), loop, np.array([side[0], side[1], zf]),
                    np.array([side[0], side[1], top[2]]), top, inlet]

        def walk(pts, n):
            """n links of exactly link_len between consecutive joint points (chords, so a link that spans a corner is
            straight and rigid: walking the arc length instead left links 1-4 cm short and the chain's FK put the
            valve 4 cm off its holster, where the hose's first step threw it at 14 m/s; rest test)"""
            seg = [np.asarray(b_, float) - np.asarray(a, float) for a, b_ in zip(pts[:-1], pts[1:])]
            out = [np.asarray(pts[0], float)]
            k, u = 0, 0.0                                 # current segment and position along it (0..1)
            for _ in range(n):
                cur = out[-1]
                found = False
                while k < len(seg):
                    a = np.asarray(pts[k], float)
                    d = seg[k]
                    # solve |a + d t - cur| = L for the smallest t >= u
                    A = d @ d; B = 2*d @ (a - cur); C = (a - cur) @ (a - cur) - link_len**2
                    disc = B*B - 4*A*C
                    if disc >= 0:
                        t = (-B + math.sqrt(disc))/(2*A)
                        if t >= u - 1e-12 and t <= 1.0 + 1e-12:
                            out.append(a + d*t); u = t; found = True
                            break
                    k += 1; u = 0.0
                if not found:
                    return out, None
            # arc position of the last point: segments before k plus u of segment k
            arc = sum(float(np.linalg.norm(x)) for x in seg[:k]) + u*float(np.linalg.norm(seg[k]))
            return out, arc - sum(float(np.linalg.norm(x)) for x in seg)

        need = sum(float(np.linalg.norm(np.asarray(b_) - np.asarray(a))) for a, b_ in zip(polyline(0.0)[:-1], polyline(0.0)[1:]))
        gap = None
        for n in range(int(need/link_len) - 2, int(need/link_len) + 6):
            # the n-th joint's arc position minus the path length falls as the floor loop lengthens (e): bisect for 0
            _, g0 = walk(polyline(0.0), n)
            if g0 is not None and g0 < 0:
                continue                         # even the shortest loop is too long for n links
            lo, hi = 0.0, 1.5
            for _ in range(80):
                mid_e = 0.5*(lo + hi)
                _, g = walk(polyline(mid_e), n)
                lo, hi = (mid_e, hi) if (g is None or g > 0) else (lo, mid_e)
            pts_, gap = walk(polyline(hi), n)
            if gap is not None and abs(gap) < 1e-6:
                break
        if gap is None or abs(gap) > 1e-6:
            raise RuntimeError(f"hose layout did not close on the valve inlet (gap {gap})")
        joint_pts = pts_
        joint_pts[-1] = inlet
        joints, chain = [], []
        # bend limits +-2.6 rad (a hose kinks past ~150 deg): the laid-out path turns 90 deg at the floor, so +-1.5 rad
        # started those joints beyond their limits and the stops threw the last link at 14 m/s at tick 3 (rest test)
        parent, k_bend = -1, 0.05/link_len
        m_link = spec["hose_lin_mass"]*link_len
        prev_R = None
        for k in range(n):
            a, c = joint_pts[k], joint_pts[k + 1]
            d = c - a
            Rl = frame_z_to(-d/np.linalg.norm(d))          # link local z points back toward its parent end
            lk = b.add_link(xform=wp.transform(tuple(0.5*(a + c)), wp.quat(*quat_from_R(Rl))), label=f"fx_{name}_hose{k}")
            self.add_part(lk, Part("tube", "capsule", (spec["hose_r"], link_len/2 - spec["hose_r"]), (0., 0., 0.),
                                   mass=m_link, material="hose_green"), f"fx_{name}_hose{k}")
            pxf = wp.transform(tuple(anchor), wp.quat(*quat_from_R(Rl))) if parent < 0 else \
                wp.transform((0., 0., -link_len/2), wp.quat(*quat_from_R(prev_R.T @ Rl)))
            axes = [b.JointDofConfig(axis=(1., 0., 0.), target_ke=k_bend, target_kd=HOSE_KD, armature=2e-4, limit_lower=-2.6,
                                     limit_upper=2.6),
                    b.JointDofConfig(axis=(0., 1., 0.), target_ke=k_bend, target_kd=HOSE_KD, armature=2e-4, limit_lower=-2.6,
                                     limit_upper=2.6)]
            jd = b.add_joint_d6(parent, lk, parent_xform=pxf, child_xform=wp.transform((0., 0., link_len/2), wp.quat_identity()),
                                angular_axes=axes, label=f"fx_{name}_hose{k}_joint")
            joints.append(jd)
            chain.append(lk)
            self.fixture_bodies.add(lk)
            parent, prev_R = lk, Rl
        Rv = np.eye(3)
        valve = b.add_link(xform=wp.transform(tuple(vc), wp.quat_identity()), label=f"fx_{name}_valve")
        vparts = [Part("body", "cylinder", (0.018, 0.06), (0., 0., -0.015), mass=0.32, material="chrome"),
                  Part("head", "cylinder", (0.030, 0.012), (0., 0., -0.083), mass=0.22, material="chrome"),
                  Part("bumper", "cylinder", (0.033, 0.006), (0., 0., -0.098), mass=0.05, material="rubber_grey"),
                  Part("inlet", "cylinder", (0.011, 0.012), (0., 0., 0.050), mass=0.04, material="chrome")]
        vshapes = [self.add_part(valve, p, f"fx_{name}_valve") for p in vparts]
        jv = b.add_joint_d6(parent, valve, parent_xform=wp.transform((0., 0., -link_len/2), wp.quat(*quat_from_R(prev_R.T @ Rv))),
                            child_xform=wp.transform((0., 0., 0.060), wp.quat_identity()),
                            angular_axes=[b.JointDofConfig(axis=(1., 0., 0.), target_ke=k_bend, target_kd=HOSE_KD, armature=2e-4,
                                                           limit_lower=-2.6, limit_upper=2.6),
                                          b.JointDofConfig(axis=(0., 1., 0.), target_ke=k_bend, target_kd=HOSE_KD, armature=2e-4,
                                                           limit_lower=-2.6, limit_upper=2.6)],
                            label=f"fx_{name}_valve_joint")
        joints.append(jv)
        piv = (0.032, 0., 0.030)
        lever = b.add_link(xform=wp.transform(tuple(vc + np.array(piv)), wp.quat_identity()), label=f"fx_{name}_lever")
        lparts = [Part("lever", "box", (0.004, 0.011, 0.045), (0.0, 0., -0.045), mass=P.PRSV["lever_mass"], material="chrome")]
        lshapes = [self.add_part(lever, p, f"fx_{name}_lever") for p in lparts]
        jl = b.add_joint_revolute(valve, lever, parent_xform=wp.transform(piv, wp.quat_identity()),
                                  child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(0., 1., 0.),
                                  limit_lower=0.0, limit_upper=P.PRSV["travel"]/P.PRSV["lever_arm"],
                                  armature=P.LEVER["armature"], damping=lever_damping(P.PRSV), label=f"fx_{name}_lever_joint")
        joints.append(jl)
        b.add_articulation(joints, label=f"fx_{name}_articulation")
        self.fixture_bodies.update({valve, lever})
        self.bodies.update({f"fx_{name}_valve": valve, f"fx_{name}_lever": lever})
        self.parts[f"fx_{name}_valve"], self.parts[f"fx_{name}_lever"] = vparts, lparts
        spec2 = dict(P.PRSV, **{k: v for k, v in spec.items() if k in ("Q_rated", "p_rated", "n_orifice", "cone_deg", "ring_r", "Cv")})
        self.fixtures[name] = dict(kind="spray_valve", spec=spec2, valve=valve, lever=lever, balancer=None, chain=chain,
                                   lever_joint=f"fx_{name}_lever_joint", balancer_joint=None, face=(0., 0., -0.104),
                                   direction=(0., 0., -1.), shapes=vshapes + lshapes, lever_pad=(0.0, 0.0, -0.06),
                                   anchor=tuple(anchor), park=tuple(vc), hose=dict(L=spec["hose_L"], D=spec["hose_D"]))

    def lever_waste(self, name, pivot, pool):
        """Lever waste on a bowl's front apron: a 90 deg twist lever (axis front-to-back) opening the basket drain."""
        b, wp = self.b, self.wp
        R = frame_z_to((0., -1., 0.), up_hint=(0., 0., 1.))
        body = b.add_link(xform=wp.transform(tuple(pivot), wp.quat(*quat_from_R(R))), label=f"fx_{name}")
        parts = [Part("knob", "cylinder", (0.012, 0.010), (0., 0., 0.010), mass=0.04, material="chrome"),
                 Part("lever", "box", (0.045, 0.006, 0.006), (0.045, 0., 0.022), mass=0.05, material="chrome")]
        shapes = [self.add_part(body, p, f"fx_{name}") for p in parts]
        j = b.add_joint_revolute(-1, body, parent_xform=wp.transform(tuple(pivot), wp.quat(*quat_from_R(R))),
                                 child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(0., 0., 1.),
                                 limit_lower=0.0, limit_upper=math.pi/2, friction=0.30, armature=2e-3, label=f"fx_{name}_joint")
        b.add_articulation([j], label=f"fx_{name}_articulation")
        self.bodies[f"fx_{name}"] = body
        self.parts[f"fx_{name}"] = parts
        self.fixture_bodies.add(body)
        self.fixtures[name] = dict(kind="waste", pool=pool, body=body, joint=f"fx_{name}_joint", shapes=shapes)

    # ------------------------------------------------------------------ items
    def items_build(self):
        sp = self.spec
        for name, it in sp["items"].items():
            if "pose" not in it:
                continue
            self._item(name, it["kind"], it["pose"])
        # items placed relative to others (plates in a peg rack, tomatoes in the colander)
        for name, it in sp["items"].items():
            if "pose" in it:
                continue
            if "in_rack" in it:
                rack, k = it["in_rack"]
                self._item(name, it["kind"], self.plate_in_rack_pose(sp["items"][rack]["pose"], k))
            elif "in_colander" in it:
                self._item(name, it["kind"], self.tomato_in_colander_pose(sp["items"]["colander"]["pose"], it["in_colander"]))

    def plate_in_rack_pose(self, rack_pose, k):
        """Plate k standing on its rim in slot k of a peg rack, face toward the rack's -y side, leaning back against the
        posts: the slot offset and lean (6-12 deg) are chosen from the plate's own hull vertices for the largest clearance
        to both rows of posts (a fixed 8 deg lean puts the rim 6 mm into a post), its lowest point 0.5 mm
        above the rack floor."""
        wr = W.build("peg_rack")
        wp_ = W.build("plate")
        x, y, z, *q = rack_pose
        Rr = Rotation.from_quat(q)
        pts = np.concatenate(wp_.hulls)
        posts = [(p.pos[0], p.pos[1], p.size[0], p.pos[2] - p.size[1], p.pos[2] + p.size[1]) for p in wr.prims
                 if p.name.startswith("post_")]
        best = None
        for lean in np.radians([6.0, 8.0, 10.0, 12.0]):
            Rl = Rotation.from_euler("x", math.pi/2 - lean)
            P_ = Rl.apply(pts)
            zlow = P_[:, 2].min()
            for dy in np.linspace(-0.012, 0.012, 25):
                Q = P_ + np.array([0.0, wr.slots[k] + dy - P_[:, 1].mean(), wr.floor_z + 0.0005 - zlow])
                clear = 1.0
                for (px, py, r, z0, z1) in posts:
                    m = (Q[:, 2] > z0) & (Q[:, 2] < z1)
                    if m.any():
                        clear = min(clear, float(np.hypot(Q[m, 0] - px, Q[m, 1] - py).min() - r))
                if best is None or clear > best[0]:
                    best = (clear, lean, Q.mean(axis=0) - Rl.apply(pts).mean(axis=0) + np.array([0., 0., 0.]), Rl)
        clear, lean, off, Rl = best
        self.plate_clearance = getattr(self, "plate_clearance", {})
        self.plate_clearance[k] = round(clear*1e3, 2)
        R = Rr*Rl
        p = np.array([x, y, z]) + Rr.apply(off)
        return (*p, *R.as_quat())

    def tomato_in_colander_pose(self, col_pose, k):
        x, y, z, *q = col_pose
        d = P.ITEMS["tomato"]
        r = 0.5*(d["r"] + d["c"])
        foot = 0.025
        ring = [(0., 0.)] + [(0.032*math.cos(a), 0.032*math.sin(a)) for a in np.linspace(0, 2*math.pi, 7)[:-1]] + \
               [(0.064*math.cos(a), 0.064*math.sin(a)) for a in np.linspace(0, 2*math.pi, 9)[:-1]]
        px, py = ring[k % len(ring)]
        layer = k//len(ring)
        R = Rotation.from_quat(q)
        p = np.array([x, y, z]) + R.apply([px, py, foot + 0.0015 + r + layer*2*r + 0.001])
        return (*p, 0., 0., 0., 1.)

    def sponge_item(self, name, pose):
        """Two-body scrub sponge: the base (scour pad + lower cellulose, free) and the top (upper cellulose) on a
        compression slide along the base's -z (0 .. 10.5 mm of the 12 mm cellulose) with the pore water's damping as the
        joint's implicit damping; its foam spring is an explicit force (water.sponge) so a pinch across the thickness
        compresses it and squeezes water out."""
        b, wp = self.b, self.wp
        w = W.build("sponge")
        x, y, z, *q = pose
        base = b.add_link(xform=wp.transform((x, y, z), wp.quat(*q)), label=name)
        jf = b.add_joint_free(base, label=f"{name}_free_joint")
        shapes = [self._sponge_core(base, p, name) for p in w.prims]
        R = Rotation.from_quat(q)
        tp = np.array([x, y, z]) + R.apply([0., 0., w.top_z])
        top = b.add_link(xform=wp.transform(tuple(tp), wp.quat(*q)), label=f"{name}_top")
        tshapes = [self._sponge_core(top, p, f"{name}_top") for p in w.top]
        js = b.add_joint_prismatic(base, top, parent_xform=wp.transform((0., 0., w.top_z), wp.quat_identity()),
                                   child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(0., 0., -1.),
                                   limit_lower=0.0, limit_upper=P.SPONGE["q_stop"], armature=0.02, damping=P.SPONGE["damping"],
                                   label=f"{name}_squeeze_joint")
        b.add_articulation([jf, js], label=f"{name}_articulation")
        self.bodies[name], self.bodies[f"{name}_top"] = base, top
        self.parts[name], self.parts[f"{name}_top"] = w.prims, w.top
        self.ware[name] = w
        self.items[name] = dict(kind="sponge", body=base, top=top, shapes=[s_ for s_ in shapes + tshapes if s_ is not None], glides=[],
                                joint=f"{name}_squeeze_joint")
        return base

    def _sponge_core(self, body, p, prefix):
        """One layer of the sponge's firm core as a convex prism: the box p (half-sizes hx, hy, hz at p.pos) with both
        ends rounded in plan to arcs of params.SPONGE end_r, so that an end pressed into a curved wall meets it at its
        middle (a square-cornered core wedged into the pan's wall hulls, A5 scrub). Its mass is p's."""
        import trimesh
        wp = self.wp
        hx, hy, hz = p.size
        rr = P.SPONGE["end_r"]
        a = math.asin(min(1.0, hy/rr))
        xc = hx - rr                                         # the arc's centre on the long axis: the tip is at hx
        ang = np.linspace(-a, a, 9)
        end = [(xc + rr*math.cos(t), rr*math.sin(t)) for t in ang]
        plan = end + [(-x_, y_) for x_, y_ in end[::-1]]
        pts = np.array([(x_ + p.pos[0], y_ + p.pos[1], p.pos[2] + s_*hz) for x_, y_ in plan for s_ in (-1.0, 1.0)])
        hull = trimesh.convex.convex_hull(pts)
        mesh = self.newton.Mesh(np.asarray(hull.vertices, np.float32), np.asarray(hull.faces, np.int32).ravel())
        i = self.b.add_shape_convex_hull(body, mesh=mesh, xform=wp.transform((0., 0., 0.), wp.quat_identity()),
                                         cfg=self.cfg(gap=P.CP.MESH_CONTACT_GAP_M, density=p.mass/float(hull.volume),
                                                      mu=p.mu if p.mu is not None else 0.5),
                                         label=f"{prefix}/{p.name}")
        self.materials[i] = "collision_hidden"
        self.visual_only.add(i)
        return i

    def _item(self, name, kind, pose):
        b, wp = self.b, self.wp
        if kind == "sponge":
            return self.sponge_item(name, pose)
        w = W.build(kind)
        x, y, z, *q = pose
        body = b.add_body(xform=wp.transform((x, y, z), wp.quat(*q)), label=name)
        shapes = []
        import trimesh
        for k, pts in enumerate(w.hulls):
            hull = trimesh.convex.convex_hull(pts)
            mesh = self.newton.Mesh(np.asarray(hull.vertices, np.float32), np.asarray(hull.faces, np.int32).ravel())
            i = b.add_shape_convex_hull(body, mesh=mesh, xform=wp.transform((0., 0., 0.), wp.quat_identity()),
                                        cfg=self.cfg(gap=P.CP.MESH_CONTACT_GAP_M, density=0.,
                                                     mu=w.friction), label=f"{name}/hull{k}")
            self.materials[i] = "collision_hidden"
            self.visual_only.add(i)
            shapes.append(i)
        for p in w.prims:
            p.mu = w.friction if p.mu is None else p.mu
            i = self.add_part(body, p, name)
            if i is not None:
                shapes.append(i)
        if w.solid is not None:
            poly = G.ccw(w.solid)
            vol = G.revolved_volume(poly)
            c = G.Container(kind, poly, w.cavity["polygon"] if w.cavity else poly, 0.0, 0.0, w.mass/vol, w.material, [])
            m, com, inertia = c.mass_properties()
            hm = getattr(w, "handle_mass", 0.0)
            if hm:
                # a riveted handle: its own mass at its centre, combined (parallel axis, diagonal approximation)
                hp = next(p for p in w.prims if p.name == "handle")
                ph, sz = np.asarray(hp.pos, float), 2*np.asarray(hp.size, float)
                Ih = hm/12*np.array([sz[1]**2 + sz[2]**2, sz[0]**2 + sz[2]**2, sz[0]**2 + sz[1]**2])
                mt = w.mass + hm
                ct = (w.mass*np.asarray(com) + hm*ph)/mt
                dm_, dh_ = np.asarray(com) - ct, ph - ct
                par = lambda d_: np.array([d_[1]**2 + d_[2]**2, d_[0]**2 + d_[2]**2, d_[0]**2 + d_[1]**2])
                inertia = np.asarray(inertia) + w.mass*par(dm_) + Ih + hm*par(dh_)
                m, com = mt, ct
            self.set_mass(body, m if hm else w.mass, com, inertia)
        self.bodies[name] = body
        self.parts[name] = w.prims
        self.ware[name] = w
        self.items[name] = dict(kind=kind, body=body, shapes=shapes, glides=[])
        return body

    # ------------------------------------------------------------------ cameras
    def views(self):
        """Diagnostic views of each station (privileged; never policy observations) and a fixed overview."""
        st = self.spec["station"]
        v = {
            "pot-sink": dict(room=([3.3, 2.2, 2.2], [3.2, 4.9, 0.95]), station=([2.35, 3.45, 1.75], [3.15, 4.95, 0.95]),
                             sink=([3.35, 4.25, 1.75], [3.3, 5.0, 0.90]), side=([4.95, 4.05, 1.45], [3.4, 4.95, 1.0]),
                             wash=([2.86, 4.38, 1.50], [2.84, 4.98, 0.62])),
            "pre-rinse": dict(room=([2.6, 1.1, 2.1], [0.35, 2.9, 0.95]), station=([1.55, 2.0, 1.75], [0.35, 3.1, 0.95]),
                              sink=([1.0, 3.1, 1.65], [0.37, 3.1, 0.90]), side=([1.35, 4.4, 1.5], [0.3, 3.0, 1.0])),
            "prep-sink": dict(room=([4.6, 1.9, 2.1], [6.8, 3.6, 0.95]), station=([5.5, 2.7, 1.75], [6.8, 3.7, 0.95]),
                              sink=([6.1, 3.9, 1.65], [6.85, 3.9, 0.90]), side=([5.6, 5.0, 1.5], [6.8, 3.5, 1.0])),
            "can-wash": dict(room=([4.2, 3.1, 2.2], [6.2, 0.9, 0.5]), station=([4.9, 2.4, 1.7], [6.1, 1.0, 0.5]),
                             sink=([6.1, 2.0, 1.6], [6.1, 1.1, 0.4]), side=([4.6, 0.4, 1.4], [6.2, 1.0, 0.5])),
        }[st]
        for name, (eye, tgt) in v.items():
            self.cameras[name] = dict(body=-1, eye=eye, target=tgt, up=[0, 0, 1], fov=60. if name != "room" else 70.,
                                      role="diagnostic")
