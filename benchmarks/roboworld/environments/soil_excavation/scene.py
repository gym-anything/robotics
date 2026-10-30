"""Physics scene (Newton 1.5 ModelBuilder, MuJoCo CPU): the front yard's static ground, the Mobile FR3 Duo (the
classroom_arrangement robot stack, unchanged) and the episode's tools and props.

MuJoCo sees: the slab, the ground outside the soil block (its faces are the block's far field), the block floor, the
tarp, the irrigation lateral, the props, the tools and inclusions (stones, root ball, post) as rigid bodies. It does
*not* see soil: inside the block footprint there is no MuJoCo floor between grade and the block floor, and everything
there is carried by the soil through the MPM coupling (world.py).

Mechanisms: the post-hole digger's two halves turn on a hinge pin (revolute, pin friction); the drill's chuck and the
auger turn on the spindle (revolute, motor torque from world.py) and its trigger slides on a sprung prismatic joint the
gripper squeezes (painting's aerosol button pattern). Round grips are octagonal prisms in MuJoCo (as in painting: a
cylinder-pad contact is a single point with no torsional hold); the soil sees the round grip.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement.scene import Scene as ClassroomScene
from classroom_arrangement.furniture import Part
from classroom_arrangement import params as CP
from . import params as P
from . import layout as L
from . import tools as T

BUILDERS = dict(spade=T.spade, trench_shovel=T.trench_shovel, trowel=T.trowel, mattock=T.mattock, tamper=T.tamper,
                post=T.post, rootball=T.rootball)
IFACE = dict(spade="stainless", trench_shovel="steel", trowel="aluminium", mattock="steel", tamper="wood", post="wood",
             rootball="rootball", stone="stone", digger_a="steel", digger_b="steel", drill="robot", auger="steel")
MU_TOOL = dict(spade=0.40, trench_shovel=0.40, trowel=0.40, mattock=0.40, tamper=0.45, post=0.45, rootball=0.6, stone=0.55,
               digger_a=0.40, digger_b=0.40, drill=0.45, auger=0.40, trigger=0.6)
PARKED = (0.0, -2.9, 0.4)                   # held tools are built here and placed into the hands by the environment
SURFACE_MU = dict(concrete=0.70, ground=0.60, bed=0.60, lawn=0.55, far_field=0.60, wood=0.50, hedge=0.80)


def octagon_prism(name, r, half_len, centre, mass, material, mu=None, axis_quat=(0., 0., 0., 1.), n=6):
    """A round grip of radius r along local z as the union of n thin boxes 180/n deg apart, each a x b with
    b = a tan(90/n deg) and corners on the circle (a = r cos(90/n deg)): the outline stays between 0.966 r and r for
    n = 6. MuJoCo's cylinder-box pair yields a single contact point (no torsional resistance at a Franka pad), box-box
    yields a contact patch, as the slightly compliant real grip does. (Frozen copy of painting.tools.octagon_prism: two
    square boxes 45 deg apart would make an eight-pointed star whose points stand 1.31 r out; see holding.py for why
    the soil world does not import painting's physics.)"""
    th = math.pi/(2*n)
    a = r*math.cos(th)
    b = a*math.tan(th)
    base = Rotation.from_quat(axis_quat)
    return [Part(f"{name}_{k}", "box", (a, b, half_len), centre,
                 tuple(float(x) for x in (base*Rotation.from_euler("z", k*math.pi/n)).as_quat()), mass=mass/n,
                 material=material, role="grip", mu=mu) for k in range(n)]


def mujoco_parts(parts):
    """Collision parts for MuJoCo: round grips become six-slab prisms (octagon_prism)."""
    out = []
    for p in parts:
        if "mpm_only" in p.role or p.kind == "mesh":
            continue
        if "grip" in p.role and p.kind in ("cylinder", "capsule"):
            r, hh = p.size
            for q in octagon_prism(p.name, r, hh + (r if p.kind == "capsule" else 0.), p.pos, p.mass, p.material,
                                   axis_quat=p.quat):
                q.role = p.role
                out.append(q)
            continue
        out.append(p)
    return out


def cut_shapes(parts):
    """The cutting regions of a tool (parts tagged "cut": blade plates, the mattock's pick and adze) in the body frame,
    as boxes or cylinders, with the direction the cutting edge faces in the region's frame, for the soil's cut closure
    (soil.k_cut). Plates are built with their length along local z and the edge at the low end (spade, trenching
    shovel, trowel, digger blades, adze: edge -z); a pick or spike is a segment from its root to its tip (+z)."""
    out = []
    for p in parts:
        if "cut" not in p.role.split() or p.kind not in ("box", "cylinder", "capsule"):
            continue
        out.append(dict(kind="box" if p.kind == "box" else "cylinder", size=tuple(float(x) for x in p.size),
                        pos=tuple(float(x) for x in p.pos), quat=tuple(float(x) for x in p.quat),
                        edge=(0., 0., -1.) if p.kind == "box" else (0., 0., 1.)))
    return out


def soil_shapes(parts):
    """The parts the soil sees (MPM collider shapes in the body frame)."""
    out = []
    for p in parts:
        if "no_mpm" in p.role or "visual" in p.role:
            continue
        if not p.collide and "mpm_only" not in p.role:
            continue
        s = dict(kind=p.kind, size=tuple(p.size), pos=tuple(p.pos), quat=tuple(p.quat))
        if p.kind == "mesh":
            s["mesh"] = p.mesh
        out.append(s)
    return out


class SoilScene(ClassroomScene):
    ENV_ID = "soil-excavation"

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
        self.joints_extra = {}
        self.coupled = {}                 # name -> dict(body, shapes (soil collider, body frame), iface)
        self.statics_soil = []            # static colliders the soil sees: dict(name, iface, shapes)
        self.yard()
        self.robot()
        self.tools()

    # ------------------------------------------------------------------ yard
    def _static_box(self, name, lo, hi, mu, material):
        b, wp = self.b, self.wp
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c, h = 0.5*(lo + hi), 0.5*(hi - lo)
        i = b.add_shape_box(-1, xform=wp.transform(tuple(c), wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2],
                            cfg=self.cfg(mu=mu, density=0.), label=name)
        self.materials[i] = material
        self.static_shapes[name] = i
        return dict(kind="box", size=tuple(h), pos=tuple(c), quat=(0., 0., 0., 1.))

    def yard(self):
        sp = self.spec
        preset = sp["preset"]
        mats = {m: P.MATERIALS[m] for m in P.MATERIALS}
        layers = sp["layers"]
        soil_far = []
        # the soil sees two static colliders (each collider costs rasterisation work at every soil node): the far
        # field (the faces around and under the block, rough and adhesive) and every other surface (slab, ground,
        # foundation, tarp, props: per-shape friction, one low adhesion)
        ground = []
        for g in L.ground_boxes(preset):
            shape = self._static_box(g["name"], g["lo"], g["hi"], SURFACE_MU[g["surface"]], g["surface"])
            if L.adjacent_to_block(g, preset) and g["surface"] != "concrete":
                soil_far.append((g, shape))
            else:
                shape["mu"] = P.INTERFACE["concrete"][0] if g["surface"] == "concrete" else 0.60
                ground.append(shape)
        # far field: the faces adjacent to the block continue its soil (mu = tan phi, adhesion = c of the layer at
        # mid-depth of the block)
        # the property beyond the yard: rigid world only (the soil never reaches it), so a base driven anywhere the
        # renders show ground stays on ground and stops at a fence, the house or the hedge
        for g in L.context_boxes():
            self._static_box(g["name"], g["lo"], g["hi"], SURFACE_MU[g["surface"]], g["surface"])
        mid = layers[-1][1]
        m = mats[mid]
        far = (math.tan(math.radians(m["phi"])), m["c"])
        self.statics_soil.append(dict(name="far_field", iface=far, shapes=[s for _, s in soil_far]))
        # tarp: a thin static sheet (not a cloth) on the walkway or lawn
        (x0, y0), (x1, y1) = sp["tarp"]
        z0 = 0.0 if y1 <= 0.0 and y0 >= L.SLAB["y"][0] else L.GRADE
        tarp = self._static_box("tarp", (x0, y0, z0), (x1, y1, z0 + 0.001), 0.40, "tarp")
        tarp["mu"] = P.INTERFACE["tarp"][0]
        ground.append(tarp)
        self._ground = dict(name="ground", iface=(0.60, P.INTERFACE["tarp"][1]), shapes=ground)
        self.statics_soil.append(self._ground)
        if "pipe" in sp:
            pp = sp["pipe"]
            i = self.b.add_shape_capsule(-1, xform=self.wp.transform((0.0, pp["y"], pp["z"]),
                                                                     self.wp.quat(*Rotation.from_euler("y", 90, degrees=True).as_quat())),
                                         radius=pp["r"], half_height=3.0, cfg=self.cfg(mu=0.35, density=0.), label="pipe")
            self.materials[i] = "pvc"
            self.static_shapes["pipe"] = i
            self.statics_soil.append(dict(name="pipe", iface=P.INTERFACE["pvc"], shapes=[dict(
                kind="capsule", size=(pp["r"], 3.0), pos=(0.0, pp["y"], pp["z"]),
                quat=tuple(Rotation.from_euler("y", 90, degrees=True).as_quat()), mu=P.INTERFACE["pvc"][0])]))
        for name, pr in sp["props"].items():
            self._prop(name, pr)

    def _prop(self, name, pr):
        x, y = pr["pos"]
        if pr["kind"] == "path_light":          # aluminium bollard on a ground stake, just off the trench line
            s = self._static_box(name, (x - 0.03, y - 0.03 - 0.10, L.GRADE - 0.15), (x + 0.03, y + 0.03 - 0.10, 0.40), 0.4,
                                 "path_light")
        elif pr["kind"] == "stake":             # 1x2 wooden survey stake marking the post centre... set 0.25 m off it
            s = self._static_box(name, (x - 0.02, y - 0.30, L.GRADE - 0.25), (x + 0.02, y - 0.27, 0.30), 0.5, "stake")
        elif pr["kind"] == "pail":                # 2-gallon pail on the walkway: a ring of 16 wall boxes and a bottom
            Pa = P.PAIL
            shapes = [self._static_box(f"{name}_bottom", (x - Pa["r"], y - Pa["r"], 0.0), (x + Pa["r"], y + Pa["r"], 0.006),
                                       0.4, "pail")]
            for k in range(16):
                a = 2*math.pi*k/16
                c = (x + (Pa["r"] + Pa["wall"]/2)*math.cos(a), y + (Pa["r"] + Pa["wall"]/2)*math.sin(a), Pa["height"]/2)
                hw = Pa["r"]*math.tan(math.pi/16) + 0.002
                i = self.b.add_shape_box(-1, xform=self.wp.transform(c, self.wp.quat(*Rotation.from_euler("z", a).as_quat())),
                                         hx=Pa["wall"]/2, hy=hw, hz=Pa["height"]/2, cfg=self.cfg(mu=0.4, density=0.), label=f"{name}_wall{k}")
                self.materials[i] = "pail"
                shapes.append(dict(kind="box", size=(Pa["wall"]/2, hw, Pa["height"]/2), pos=c,
                                   quat=tuple(Rotation.from_euler("z", a).as_quat())))
            for sh in shapes:
                sh["mu"] = P.INTERFACE["plastic"][0]
                self._ground["shapes"].append(sh)
            return
        elif pr["kind"] == "valve_box":
            i = self.b.add_shape_cylinder(-1, xform=self.wp.transform((x, y, L.GRADE - 0.125), self.wp.quat_identity()),
                                          radius=0.127, half_height=0.125, cfg=self.cfg(mu=0.4, density=0.), label=name)
            self.materials[i] = "valve_box"
            self.static_shapes[name] = i
            s = dict(kind="cylinder", size=(0.127, 0.125), pos=(x, y, L.GRADE - 0.125), quat=(0., 0., 0., 1.))
        else:
            raise ValueError(pr["kind"])
        s["mu"] = P.INTERFACE["plastic"][0]
        self._ground["shapes"].append(s)

    # ------------------------------------------------------------------ tools and inclusions
    def _add_parts(self, body, parts, name, mu):
        from classroom_arrangement.scene import volume
        wp = self.wp
        shapes = []
        for p in mujoco_parts(parts):
            # numpy scalars reach Warp's vector maths inside Newton's shape builders and are rejected (vec3f * float64:
            # the trench and hardpan presets did not build); geometry as plain floats
            p.size, p.pos, p.quat = (tuple(float(x) for x in v) for v in (p.size, p.pos, p.quat))
            p.mass = float(p.mass)
            if p.mu is None:
                p.mu = mu
            if p.kind == "ellipsoid":
                a, bb, c = p.size
                i = self.b.add_shape_ellipsoid(body, xform=wp.transform(p.pos, wp.quat(*p.quat)), rx=a, ry=bb, rz=c,
                                               cfg=self.cfg(mu=p.mu, density=p.mass/(4/3*math.pi*a*bb*c)), label=f"{name}/{p.name}")
                self.materials[i] = p.material
            elif not p.collide:
                if p.mass <= 0:
                    continue
                # drawn-only part with mass: its mass and inertia at its place (painting scene pattern)
                hx = p.size[0]; hy = p.size[1] if p.kind == "box" else p.size[0]; hz = p.size[2] if p.kind == "box" else p.size[1]
                i = self.b.add_shape_box(body, xform=wp.transform(p.pos, wp.quat(*p.quat)), hx=hx, hy=hy, hz=hz,
                                         cfg=self.cfg(density=p.mass/(8*hx*hy*hz), has_shape_collision=False,
                                                      has_particle_collision=False, is_visible=False), label=f"{name}/{p.name}_mass")
                self.materials[i] = "mass_only"
            elif p.mass <= 0:
                # massless collision envelope (auger flight's swept cylinder)
                i = self.add_part(body, Part(p.name, p.kind, p.size, p.pos, p.quat, 1e-6, p.material, p.role, True, p.mu), name)
            else:
                i = self.add_part(body, p, name)
            if i is not None:
                shapes.append(i)
        return shapes

    def _free(self, name, kind, pos, quat, parts, coupled=True):
        wp = self.wp
        body = self.b.add_body(xform=wp.transform(tuple(pos), wp.quat(*quat)), label=name)
        shapes = self._add_parts(body, parts, name, MU_TOOL.get(kind, 0.45))
        self.bodies[name] = body
        self.parts[name] = parts
        self.items[name] = dict(kind=kind, body=body, shapes=shapes, glides=[])
        if coupled:
            self.coupled[name] = dict(body=body, shapes=soil_shapes(parts), iface=P.INTERFACE[IFACE[kind]], cuts=cut_shapes(parts))
        return body

    def tools(self):
        sp = self.spec
        held = sorted(set(sp["held"].values()))
        kit = [t for t in sp.get("kit", []) if t not in held]     # a held tool is not also in the kit (one body per name)
        for k, tool in enumerate(held + kit):
            pos = (PARKED[0] + 0.5*k, PARKED[1], PARKED[2])
            if tool in sp.get("kit", []) and tool not in held:
                pos, quat = self.kit_pose(k - len(held), tool)
            else:
                quat = (0., 0., 0., 1.)
            if tool == "digger":
                self.digger(pos)
            elif tool == "drill":
                self.drill(pos)
            else:
                self._free(tool, tool, pos, quat, BUILDERS[tool]())
        for name, it in sp["items"].items():
            if it["kind"] == "stone":
                x, y, z, q = it["pose"]
                self._free(name, "stone", (x, y, z), q, T.stone(it["radii"], name))
            elif it["kind"] in ("rootball", "post"):
                x, y, z, q = it["pose"]
                self._free(name, it["kind"], (x, y, z), q, BUILDERS[it["kind"]]())

    def kit_pose(self, k, tool):
        """Tools of the kit lie on the walkway behind the robot, blade flat, their lowest point 2 mm above the slab
        (the first version put every tool's centre at 3 cm: the trenching shovel's blade started inside the slab and
        was thrown out at 32 m/s)."""
        rx, ry, yaw = self.spec["robot"]
        side = -1 if yaw > 0 else 1
        # lying along -x from its grip end, i.e. away from the robot (along +x a 1.2 m tamper lay under the base)
        Rk = Rotation.from_euler("z", 0.)*Rotation.from_euler("y", -90, degrees=True)
        zmin = 0.0
        for p in BUILDERS[tool]():
            if not p.collide or p.kind == "mesh":
                continue
            Rp = (Rk*Rotation.from_quat(p.quat)).as_matrix()
            if p.kind == "box":
                ext = float(np.abs(Rp[2]) @ np.asarray(p.size, float))
            elif p.kind in ("cylinder", "capsule"):
                az = abs(Rp[2, 2])
                ext = az*p.size[1] + p.size[0]*math.sqrt(max(0., 1 - az*az)) + (p.size[0]*az if p.kind == "capsule" else 0.)
            else:
                ext = float(max(p.size))
            zmin = min(zmin, float(Rk.apply(np.asarray(p.pos, float))[2]) - ext)
        pos = (float(rx - 0.9 + 0.25*k), float(ry + side*0.25), float(0.002 - zmin))
        return pos, tuple(Rk.as_quat())

    def digger(self, pos):
        """Two halves on the hinge pin (revolute about the pin's +y axis, pin friction); open/close by spreading the
        handles. Half a is the articulation root (free joint)."""
        b, wp = self.b, self.wp
        S = P.DIGGER
        pa, pb = T.digger_half("a"), T.digger_half("b")
        la = b.add_link(xform=wp.transform(tuple(pos), wp.quat_identity()), label="digger_a")
        jf = b.add_joint_free(la, label="digger_free_joint")
        sa = self._add_parts(la, pa, "digger_a", MU_TOOL["digger_a"])
        lb = b.add_link(xform=wp.transform(tuple(pos), wp.quat_identity()), label="digger_b")
        sb = self._add_parts(lb, pb, "digger_b", MU_TOOL["digger_b"])
        jr = b.add_joint_revolute(la, lb, parent_xform=wp.transform((0., 0., 0.), wp.quat_identity()),
                                  child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(0., 1., 0.),
                                  target_ke=0.0, target_kd=0.0, friction=S["pin_friction"], armature=0.0,
                                  limit_lower=-0.02, limit_upper=0.60, label="digger_hinge")
        # q > 0 brings the handles together and spreads the blades (open); at q = 0 the blades meet (the halves do not
        # collide with each other, so the stop is the joint limit: at -0.35 squeezed blades passed through each other)
        b.add_articulation([jf, jr], label="digger_articulation")
        for s0 in sa:
            for s1 in sb:
                b.add_shape_collision_filter_pair(s0, s1)
        for nm, link, parts, shp in (("digger_a", la, pa, sa), ("digger_b", lb, pb, sb)):
            self.bodies[nm] = link
            self.parts[nm] = parts
            self.items[nm] = dict(kind=nm, body=link, shapes=shp, glides=[])
            self.coupled[nm] = dict(body=link, shapes=soil_shapes(parts), iface=P.INTERFACE[IFACE[nm]], cuts=cut_shapes(parts))
        self.bodies["digger"] = la
        self.joints_extra["digger_hinge"] = jr

    def drill(self, pos):
        """Drill body (free) + chuck/auger on the spindle (revolute about the drill's +x motor axis) + trigger
        (prismatic along +x into the grip, sprung outward with the trigger preload)."""
        b, wp = self.b, self.wp
        D = P.DRILL
        body = b.add_link(xform=wp.transform(tuple(pos), wp.quat_identity()), label="drill")
        jf = b.add_joint_free(body, label="drill_free_joint")
        sd = self._add_parts(body, T.drill_body(), "drill", MU_TOOL["drill"])
        # spindle: chuck + auger, axis +x at z = 0.04 (motor axis); the auger's -z axis is mapped onto +x
        sp_pos = np.asarray(pos) + np.array([0.075, 0., 0.04])
        q_aug = tuple(Rotation.from_euler("y", -90, degrees=True).as_quat())    # auger -z -> world +x (drill frame)
        spindle = b.add_link(xform=wp.transform(tuple(sp_pos), wp.quat_identity()), label="auger")
        chuck = [Part("chuck", "cylinder", (0.022, 0.035), (-0.035, 0., 0.), tuple(Rotation.from_euler("y", 90, degrees=True).as_quat()),
                      0.35, "steel_black")]
        aug = T.auger()
        Ra = Rotation.from_quat(q_aug)
        aug_parts = []
        for p in aug:
            q = Part(p.name, p.kind, p.size, tuple(Ra.apply(p.pos)), tuple((Ra*Rotation.from_quat(p.quat)).as_quat()), p.mass,
                     p.material, p.role, p.collide, p.mu)
            if hasattr(p, "mesh"):
                # mesh vertices stay in the part frame: the part's rotation (Ra) is applied by its transform. The first
                # build also rotated the vertices, which turned the flight a further 90 deg into the drill body: the
                # auger went 16 cm into the bed without touching the soil (D9 first run)
                q.mesh = p.mesh
            aug_parts.append(q)
        ss = self._add_parts(spindle, chuck + aug_parts, "auger", MU_TOOL["auger"])
        js = b.add_joint_revolute(body, spindle, parent_xform=wp.transform((0.075, 0., 0.04), wp.quat_identity()),
                                  child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(1., 0., 0.),
                                  target_ke=0.0, target_kd=0.0, friction=0.02, armature=2.0e-4, label="drill_spindle")
        # trigger: slides into the grip front (-x of the grip), sprung out
        tr_pos = np.asarray(pos) + np.array([-0.112, 0., -0.005])
        trig = b.add_link(xform=wp.transform(tuple(tr_pos), wp.quat_identity()), label="drill_trigger")
        st = self._add_parts(trig, T.drill_trigger(), "drill_trigger", MU_TOOL["trigger"])
        k_spring, pre = D["trigger_spring"][1], D["trigger_spring"][0]
        # the return spring (preload + rate) is applied by world._drill_motor as a force: a joint position drive's
        # target outside the travel was replaced by mid-travel (4 mm) at load, which held the trigger half pressed and
        # ran the motor from tick 0 (D9 v2-v4)
        jt = b.add_joint_prismatic(body, trig, parent_xform=wp.transform((-0.112, 0., -0.005), wp.quat_identity()),
                                   child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(-1., 0., 0.),
                                   target_pos=0.0, target_ke=0.0, target_kd=8.0, limit_lower=0.0,
                                   limit_upper=D["trigger_travel"], armature=0.05, label="drill_trigger_joint")
        b.add_articulation([jf, js, jt], label="drill_articulation")
        for s0 in sd:
            for s1 in ss + st:
                b.add_shape_collision_filter_pair(s0, s1)
        self.bodies.update(drill=body, auger=spindle, drill_trigger=trig)
        self.parts.update(drill=T.drill_body(), auger=chuck + aug_parts, drill_trigger=T.drill_trigger())
        self.items["drill"] = dict(kind="drill", body=body, shapes=sd, glides=[])
        self.items["auger"] = dict(kind="auger", body=spindle, shapes=ss, glides=[])
        self.items["drill_trigger"] = dict(kind="drill_trigger", body=trig, shapes=st, glides=[])
        self.coupled["auger"] = dict(body=spindle, shapes=soil_shapes(chuck + aug_parts), iface=P.INTERFACE["steel"],
                                     cuts=cut_shapes(aug_parts))
        self.coupled["drill"] = dict(body=body, shapes=soil_shapes(T.drill_body()), iface=P.INTERFACE["robot"])
        self.joints_extra.update(drill_spindle=js, drill_trigger=jt)
