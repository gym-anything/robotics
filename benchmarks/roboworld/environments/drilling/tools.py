"""Tool mechanisms: real moving parts, operated only by contact.

Every tool is a set of Newton links and joints built into the scene; nothing about a tool is commanded by an action.
Per tick, before the solver step, each tool reads its own joint states (trigger travel, selector detents, switch
position, depth-stop nut, SDS bit slide) and sets its drives: a spindle is a MuJoCo velocity actuator whose damping
is the motor's torque-speed slope (M = c (s w0 - w), integrated implicitly; E1a), capped by the motor's
stall torque, the clutch or the belt; springs are position drives; detents are position drives to the nearest detent.

  CordlessDrill  DeWalt DCD800 class: trigger (8 mm, 4-10 N), two-speed gearbox slider, FWD/lock/REV slider,
                 clutch collar (15 settings 1.7-8.5 N m + drill mode), keyless chuck + bit
  RotaryHammer   DeWalt DCH273 class: trigger, mode knob (rotary / hammer-drill / hammer) locked unless its release
                 button is held in, side handle, depth rod
                 (friction slide), SDS carrier with the bit sliding 12 mm against a 5 N spring; blows only in the
                 idle-stroke window, each removing concrete (holes.blow) and kicking the housing back with the striker's
                 momentum; electronic overload clutch (params.HAMMER_CLUTCH_NM, 12.5 N m held for 50 ms)
  DrillPress     Grizzly G7943 class: feed shaft with three spokes, rack-and-pinion quill (joint equality), return
                 spring, depth-stop nut, START and STOP push buttons, induction motor through a belt (fixed speed per
                 task)
  Vise           4 in drill-press vise: crank screw coupled to the moving jaw (2.5 mm pitch) with load-dependent
                 thread and screw-collar friction (self-locking), bolted to the table or loose
  CentrePunch    General #78 class automatic punch: a sharp conical point sliding on the main spring; at the release
                 force the internal hammer strikes with the spring's energy and leaves a dimple sized by the share of
                 that energy the struck part's support does not absorb and by the material's hardness
  PhantomSpindle ASSISTED (probes only): a drilling-dynamometer rig, a bit on a carriage driven at set speed with a
                 set feed or a set force. Never part of a task scene.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P

RPM = 2*math.pi/60


def wq(R):
    """scipy Rotation -> warp quaternion tuple (x, y, z, w)."""
    q = R.as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


class JointRef:
    """A Newton joint with its coordinate, dof, MuJoCo joint and (optional) actuator indices, bound after finalize."""

    def __init__(self, index, name):
        self.index, self.name = int(index), name
        self.q = self.dof = self.mj = self.act = None

    def bind(self, world):
        m = world.model
        self.q = int(m.joint_q_start.numpy()[self.index])
        self.dof = int(m.joint_qd_start.numpy()[self.index])
        self.mj = world.mj_joint_of_dof[self.dof]
        mjm = world.solver.mj_model
        acts = [a for a in range(mjm.nu) if int(mjm.actuator_trnid[a][0]) == self.mj]
        self.acts = acts
        self.act = acts[0] if acts else None
        return self

    def pos(self, w):
        return float(w.jq[self.q])

    def vel(self, w):
        return float(w.jqd[self.dof])


class Spindle:
    """A revolute spindle with a DC/induction-motor velocity drive. kd = motor torque-speed slope; target speed = the
    command; torque cap = jnt_actfrcrange (stall, clutch or belt)."""

    def __init__(self, joint, bodies, axis_body_frame, walking_bodies, percussive=False):
        self.joint = joint
        self.bodies = list(bodies)          # bodies turning with the spindle (chuck, bit)
        self.axis = np.asarray(axis_body_frame, float)
        self.walking_bodies = list(walking_bodies)
        self.percussive = percussive
        self.mode = "drill"
        self.c = 0.
        self.cap = 0.
        self.cmd = 0.

    def bind(self, world):
        self.w = world
        self.joint.bind(world)
        mjm = world.solver.mj_model
        I = world.model.body_inertia.numpy()
        self._I_axis = float(mjm.dof_armature[self.joint.dof]) + sum(
            float(self.axis @ np.asarray(I[b], float) @ self.axis) for b in self.bodies)
        mass = world.model.body_mass.numpy()
        self._m_walk = float(sum(mass[b] for b in self.walking_bodies))

    def set_drive(self, target_speed, c, cap):
        w, mjm = self.w, self.w.solver.mj_model
        self.cmd, self.c, self.cap = float(target_speed), float(c), float(cap)
        a = self.joint.act
        mjm.actuator_gainprm[a][0] = self.c
        mjm.actuator_biasprm[a][2] = -self.c
        mjm.jnt_actfrcrange[self.joint.mj] = (-self.cap, self.cap)
        w.tqd[self.joint.dof] = self.cmd

    def angle(self):
        return self.joint.pos(self.w)

    def speed(self):
        return self.joint.vel(self.w)

    def drive_torque(self):
        return float(self.w.solver.mj_data.qfrc_actuator[self.w.solver.mj_model.jnt_dofadr[self.joint.mj]])

    def inertia_about_axis(self):
        return self._I_axis

    def drive_damping(self):
        return 0. if abs(self.drive_torque()) >= 0.99*self.cap else self.c

    def walking_mass(self):
        return self._m_walk


def set_initial_q(b, joint, value):
    """Start a 1-dof joint at `value` (Newton starts every joint at 0, whatever its drive target)."""
    b.joint_q[b.joint_q_start[joint]] = float(value)


def detent_target(q, positions):
    positions = np.asarray(positions, float)
    return float(positions[int(np.argmin(np.abs(positions - q)))]), int(np.argmin(np.abs(positions - q)))


class Tool:
    name = "tool"
    assisted = False

    def bind(self, world):
        self.w = world
        for j in self.joints():
            j.bind(world)
        # preloaded springs: Newton replaces a position target outside the joint's limits with the limits' midpoint
        # (every trigger would rest at half travel), so the intended targets are written here
        for key, value in getattr(self, "spring_targets", {}).items():
            world.tq[self.j[key].q] = value

    def joints(self):
        return []

    def pre_step(self, w):
        pass

    def post_step(self, w):
        pass

    def state(self):
        return {}

    def snapshot(self):
        return {k: v for k, v in self.__dict__.items() if k.startswith("s_")}

    def restore(self, snap):
        self.__dict__.update(snap)


# ====================================================================== builder helpers
class Parts:
    """Shape helpers bound to a scene builder: every shape registers its render material in scene.materials."""

    def __init__(self, scene):
        self.sc = scene
        self.b = scene.b
        self.wp = scene.wp

    def cfg(self, **kw):
        return self.sc.cfg(**kw)

    def box(self, body, name, pos, half, material, rot=None, collide=True, **kw):
        return self.sc.box(name, pos, half, material, body=body, rot=rot, collide=collide, **kw)

    def cyl(self, body, name, pos, radius, half_h, axis, material, collide=True, roll=False, **kw):
        """A cylinder along `axis` (x, y or z, or a vector) in the body frame. roll=True: a round part's rim that can
        roll on a bench or floor gets rolling resistance (MuJoCo condim 6, lever arm params.ROLLING_B_M)."""
        R = _rot_z_to(axis)
        extra = {}
        if roll and collide and P.ROLLING_B_M > 0:
            # torsional coefficient at MuJoCo's own floor (mjMINMU = 1e-5, what the CPU solver clamps a zero to: the
            # same physics, and the model converter does not warn about a zero in a condim-6 geom in every session)
            kw = dict(kw, mu_rolling=P.ROLLING_B_M, mu_torsional=1.001e-5)   # (1e-5 lands under mjMINMU in single precision)
            extra = dict(custom_attributes={"mujoco:condim": 6})
        cfg = self.cfg(**kw) if collide else self.sc.visual_cfg()
        i = self.b.add_shape_cylinder(body, xform=self.wp.transform(pos, wq(R)), radius=radius, half_height=half_h,
                                      cfg=cfg, label=name, **extra)
        self.sc.materials[i] = material
        if not collide:
            self.sc.visual.add(i)
        return i

    def capsule(self, body, name, pos, radius, half_h, axis, material, collide=True, **kw):
        R = _rot_z_to(axis)
        cfg = self.cfg(**kw) if collide else self.sc.visual_cfg()
        i = self.b.add_shape_capsule(body, xform=self.wp.transform(pos, wq(R)), radius=radius, half_height=half_h,
                                     cfg=cfg, label=name)
        self.sc.materials[i] = material
        if not collide:
            self.sc.visual.add(i)
        return i

    def bit(self, body, name, start, axis, d_mm, length, material, point_angle_deg=118., **kw):
        """A drill bit on `body`: a cylindrical body from `start` along `axis` and a conical point (convex hull) whose
        apex is the tip at start + length*axis. A flat-ended cylinder touched a face with its rim when tilted, holding
        the point 0.8 mm off a 15-degree entry (build session); a real twist drill touches with its point."""
        sc = self.sc
        a = np.asarray(axis, float)/np.linalg.norm(axis)
        r = 0.5e-3*d_mm
        h = r/math.tan(math.radians(0.5*point_angle_deg))
        L = max(length - h, 1e-3)
        start = np.asarray(start, float)
        self.cyl(body, f"{name}_body", tuple(start + a*0.5*L), r, 0.5*L, a, material, **kw)
        n = 24
        ang = 2*math.pi*np.arange(n)/n
        t1 = np.cross(a, [0., 0., 1.]) if abs(a[2]) < .9 else np.cross(a, [1., 0., 0.])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(a, t1)
        base = start + a*(L - 0.0002)
        verts = [base + r*(math.cos(q)*t1 + math.sin(q)*t2) for q in ang] + [start + a*length]
        verts = np.asarray(verts, np.float32)
        faces = []
        for k in range(n):
            faces += [k, (k + 1) % n, n]                       # sides to the apex
        for k in range(1, n - 1):
            faces += [0, k + 1, k]                             # base cap
        mesh = sc.newton.Mesh(verts, np.asarray(faces, np.int32))
        cfg = self.cfg(**kw)
        i = sc.b.add_shape_convex_hull(body, mesh=mesh, xform=sc.wp.transform_identity(), cfg=cfg, label=f"{name}_point")
        sc.materials[i] = material
        return i

    def hex_prism(self, body, name, center, axis, across_flats, half_len, material, **kw):
        """A hexagonal prism along the body's x axis: a hex nut or bolt head with its real across-flats width, which a
        cylinder or a square stands in for badly (a round nut rolls; a square head is the wrong shape). Built as the
        union of three boxes turned 60 degrees apart (each the hexagon less two opposite corner triangles), so its
        contacts are the boxes' face contacts at the world's box gap; flats normal to z, as the bolt rests. (A convex
        hull of the prism, stiff, threw an M8 bolt dropped on the floor at 17 m/s: E14.)
        The three boxes cover the hexagon twice: give them half the material's density. Returns the three shape ids."""
        assert axis == "x"
        rc = across_flats/math.sqrt(3.)                              # circumradius
        ids = []
        for k in range(3):
            R = Rotation.from_euler("x", 60.*k, degrees=True)
            ids.append(self.sc.box(f"{name}{k}" if k else name, center, (half_len, 0.5*rc, 0.5*across_flats), material,
                                   body=body, rot=wq(R), **kw))
        return ids

    def sphere(self, body, name, pos, radius, material, collide=True, **kw):
        cfg = self.cfg(**kw) if collide else self.sc.visual_cfg()
        i = self.b.add_shape_sphere(body, xform=self.wp.transform(pos, self.wp.quat_identity()), radius=radius,
                                    cfg=cfg, label=name)
        self.sc.materials[i] = material
        if not collide:
            self.sc.visual.add(i)
        return i


def bit_volume(r, length, point_angle_deg):
    """Volume of Parts.bit's shapes: the cylindrical body and the conical point (its base 0.2 mm inside the body)."""
    h = r/math.tan(math.radians(0.5*point_angle_deg))
    L = max(length - h, 1e-3)
    return math.pi*r*r*L + math.pi*r*r*(length - L + 0.0002)/3.


def _rot_z_to(axis):
    if isinstance(axis, str):
        axis = {"x": (1., 0., 0.), "y": (0., 1., 0.), "z": (0., 0., 1.), "-x": (-1., 0., 0.), "-y": (0., -1., 0.),
                "-z": (0., 0., -1.)}[axis]
    a = np.asarray(axis, float)/np.linalg.norm(axis)
    z = np.array([0., 0., 1.])
    c = float(z @ a)
    if c > 1 - 1e-12:
        return Rotation.identity()
    if c < -1 + 1e-12:
        return Rotation.from_rotvec([math.pi, 0., 0.])
    v = np.cross(z, a)
    return Rotation.from_rotvec(v/np.linalg.norm(v)*math.acos(c))


def link(sc, name, pose, mass=None, com=None, inertia=None):
    """A link at world pose (p, scipy Rotation). mass/inertia: None = from shape densities."""
    p, R = pose
    kw = {}
    if mass is not None:
        kw.update(mass=float(mass))
    if com is not None:
        kw.update(com=sc.wp.vec3(*map(float, com)))
    if inertia is not None:
        kw.update(inertia=sc.wp.mat33(*np.diag(inertia).ravel().tolist()), lock_inertia=True)
    return sc.b.add_link(xform=sc.wp.transform(tuple(map(float, p)), wq(R)), label=name, **kw)


def rel_xform(sc, parent_pose, child_pose):
    """Transform of the child's frame in the parent's frame (both world poses (p, R))."""
    pp, Rp = parent_pose
    pc, Rc = child_pose
    return sc.wp.transform(tuple(map(float, Rp.inv().apply(np.asarray(pc) - np.asarray(pp)))), wq(Rp.inv()*Rc))


def compose(pose, p_local, R_local=None):
    p, R = pose
    return (np.asarray(p) + R.apply(p_local), R*(R_local or Rotation.identity()))


# ====================================================================== cordless drill/driver
class CordlessDrill(Tool):
    """DCD800 class. Frame: x along the spindle towards the bit, z up (grip below), y left; origin at the chuck's rear
    face. Masses from the ledger: the 2.0 Ah battery (BATTERY_2AH_KG) and the bare tool (DRILL_BARE_KG) split over the
    gear/motor head, grip and chuck in the authored proportions 0.80 : 0.20 : 0.26 after the selectors' own masses
    (total DRILL_MASS_KG = 1.68 kg, printed by E1a) plus the bit."""
    SPLIT = dict(head=0.80, grip=0.20, chuck=0.26)
    V = dict(head=math.pi*0.026**2*0.125, grip=0.034*0.022*0.080, chuck=math.pi*0.0165**2*0.053)
    # trigger, direction button, gear slider (1200 kg/m3) and clutch collar (1100 kg/m3) at their materials' densities
    SMALL_KG = P.TRIGGER_MASS_KG + 1200.*(0.008*0.048*0.008 + 0.016*0.012*0.008) + 1100.*math.pi*0.028**2*0.014

    @classmethod
    def part_density(cls, part):
        f = (P.DRILL_BARE_KG - cls.SMALL_KG)/sum(cls.SPLIT.values())
        return f*cls.SPLIT[part]/cls.V[part]
    CLUTCH_SETTINGS = list(np.linspace(P.DRILL_CLUTCH_NM[0], P.DRILL_CLUTCH_NM[1], 15)) + [math.inf]

    def __init__(self, sc, name, pose, bit_d_mm, bit_len_m, bit_type="hss", gear=1, clutch=15, wear=1.0,
                 free=True, layout_note=""):
        self.name = name
        self.d_mm = float(bit_d_mm)
        pr = Parts(sc)
        b = sc.b
        self.pose0 = pose
        H = link(sc, f"{name}_housing", pose)
        # Colliders fitted to the Poly Haven Drill_01 model that draws the tool (183 x 53 x 185 mm; profile measured in
        # this build): mesh -> drill frame x = -m_x - 0.075, y = -m_y + 0.0025, z = m_z - 0.157.
        # Masses: the ledger's (class docstring)
        pr.cyl(H, f"{name}_head", (-0.0625, 0., -0.002), 0.026, 0.0625, "x", "drill_yellow",
               density=self.part_density("head"), mu=P.MU["default"])
        sc.box(f"{name}_grip", (-0.092, 0., -0.070), (0.017, 0.011, 0.040), "drill_grip", body=H,
               rot=wq(Rotation.from_euler("y", 12, degrees=True)), density=self.part_density("grip"), mu=P.MU["rubber"])
        sc.box(f"{name}_battery", (-0.090, 0., -0.132), (0.040, 0.0245, 0.026), "drill_battery", body=H,
               density=P.BATTERY_2AH_KG/(0.080*0.049*0.052))
        self.housing = H
        Hpose = pose
        # chuck + bit on the spindle (revolute about x)
        Cp = compose(Hpose, (0.0, 0., 0.))
        C = link(sc, f"{name}_chuck", Cp)
        pr.cyl(C, f"{name}_chuck_body", (0.0265, 0., 0.), 0.0165, 0.0265, "x", "chuck_black",
               density=self.part_density("chuck"))
        # the bit: a separate welded link (hole-capable item), shank excluded from drillable solids
        r_bit = 0.5e-3*bit_d_mm
        stick = bit_len_m
        Bp = compose(Hpose, (0.053, 0., 0.))
        Bk = link(sc, f"{name}_bit", Bp)
        pr.bit(Bk, f"{name}_bit", (0., 0., 0.), (1., 0., 0.), bit_d_mm, stick, "hss_bit", density=7850.,
               mu=P.MU["bit_wall"])
        self.bit = Bk
        self.chuck = C
        # trigger at the top front of the grip: slides into the grip along -x
        Tp = compose(Hpose, (-0.055, 0., -0.045))
        T = link(sc, f"{name}_trigger", Tp)
        sc.box(f"{name}_trigger_blade", (0.0, 0., 0.), (0.006, 0.006, 0.011), "drill_trigger", body=T,
               density=P.TRIGGER_MASS_KG/(0.012*0.012*0.022), mu=P.MU["default"])
        # FWD / lock / REV button through the housing above the trigger (along y; its ends stand 5 mm proud)
        Fp = compose(Hpose, (-0.062, 0., -0.020))
        F = link(sc, f"{name}_direction", Fp)
        sc.box(f"{name}_direction_knob", (0., 0., 0.), (0.004, 0.024, 0.004), "drill_trigger", body=F, density=1200.)
        # gear slider on top of the head (along x)
        Gp = compose(Hpose, (-0.070, 0., 0.026))
        G = link(sc, f"{name}_gear", Gp)
        sc.box(f"{name}_gear_knob", (0., 0., 0.), (0.008, 0.006, 0.004), "drill_trigger", body=G, density=1200.)
        # clutch collar (ring around the head front)
        Kp = compose(Hpose, (-0.008, 0., 0.))
        K = link(sc, f"{name}_clutch", Kp)
        pr.cyl(K, f"{name}_clutch_ring", (0., 0., -0.002), 0.028, 0.007, "x", "drill_collar", density=1100.)
        # appearance: the Drill_01 model on the housing (its chuck drawn static); our own chuck primitive hidden
        R_md = Rotation.from_euler("z", math.pi)
        self.visual = [dict(asset="Drill_01", body=H, prims=None, scale=1.0,
                            pos=(-0.075, 0.0025, -0.157),            # d = R_z(pi) m + pos
                            quat=wq(R_md), hide_shapes_of=[H, C, K])]
        I = sc.wp.transform_identity()
        joints = []
        if free:
            joints.append(b.add_joint_free(H, label=f"{name}_free"))
        else:
            joints.append(b.add_joint_fixed(-1, H, parent_xform=sc.wp.transform(tuple(pose[0]), wq(pose[1])),
                                            child_xform=I, label=f"{name}_fixed"))
        jsp = b.add_joint_revolute(H, C, parent_xform=rel_xform(sc, Hpose, Cp), child_xform=I, axis=(1., 0., 0.),
                                   target_kd=0., target_ke=0., effort_limit=P.DRILL_STALL_NM[0],
                                   actuator_mode=sc.newton.JointTargetMode.VELOCITY,
                                   armature=P.ROTOR_INERTIA_CHUCK, label=f"{name}_spindle")
        joints.append(jsp)
        joints.append(b.add_joint_fixed(C, Bk, parent_xform=rel_xform(sc, Cp, Bp), child_xform=I, label=f"{name}_bit_weld"))
        k_tr = (P.TRIGGER_FULL_N - P.TRIGGER_PRELOAD_N)/P.TRIGGER_TRAVEL_M
        jtr = b.add_joint_prismatic(H, T, parent_xform=rel_xform(sc, Hpose, Tp), child_xform=I, axis=(-1., 0., 0.),
                                    limit_lower=0., limit_upper=P.TRIGGER_TRAVEL_M, target_pos=-P.TRIGGER_PRELOAD_N/k_tr,
                                    target_ke=k_tr, target_kd=2*0.4*math.sqrt(k_tr*(P.TRIGGER_MASS_KG + P.TRIGGER_ARMATURE_KG)),
                                    actuator_mode=sc.newton.JointTargetMode.POSITION, armature=P.TRIGGER_ARMATURE_KG,
                                    label=f"{name}_trigger")
        joints.append(jtr)
        self.dir_positions = (-0.004, 0., 0.004)          # FWD (pushed in from the left), lock, REV
        jdir = b.add_joint_prismatic(H, F, parent_xform=rel_xform(sc, Hpose, Fp), child_xform=I, axis=(0., 1., 0.),
                                     limit_lower=-0.0045, limit_upper=0.0045, target_pos=-0.004,
                                     target_ke=2*3.0/0.004, target_kd=2.0, armature=P.SELECTOR_ARMATURE_KG,
                                     actuator_mode=sc.newton.JointTargetMode.POSITION, label=f"{name}_direction")
        joints.append(jdir)
        self.gear_positions = (0.0, 0.010)                 # gear 1 (back), gear 2 (forward)
        jg = b.add_joint_prismatic(H, G, parent_xform=rel_xform(sc, Hpose, Gp), child_xform=I, axis=(1., 0., 0.),
                                   limit_lower=-0.001, limit_upper=0.011,
                                   target_pos=self.gear_positions[gear - 1], target_ke=2*4.0/0.010, target_kd=2.0,
                                   armature=P.SELECTOR_ARMATURE_KG, actuator_mode=sc.newton.JointTargetMode.POSITION, label=f"{name}_gear")
        joints.append(jg)
        pitch = math.radians(20.)
        self.clutch_positions = tuple(pitch*k for k in range(16))     # settings 1..15, then drill mode
        jk = b.add_joint_revolute(H, K, parent_xform=rel_xform(sc, Hpose, Kp), child_xform=I, axis=(1., 0., 0.),
                                  limit_lower=-0.05, limit_upper=pitch*15 + 0.05,
                                  target_pos=self.clutch_positions[clutch - 1], target_ke=2*0.15/pitch, target_kd=0.002,
                                  armature=2e-4, actuator_mode=sc.newton.JointTargetMode.POSITION, label=f"{name}_clutch")
        joints.append(jk)
        b.add_articulation(joints, label=f"{name}_assembly")
        set_initial_q(b, jdir, self.dir_positions[0])
        set_initial_q(b, jg, self.gear_positions[gear - 1])
        set_initial_q(b, jk, self.clutch_positions[clutch - 1])
        self.bodies = [H, C, Bk, T, F, G, K]
        self.spring_targets = dict(trigger=-P.TRIGGER_PRELOAD_N/k_tr)
        self.j = dict(spindle=JointRef(jsp, "spindle"), trigger=JointRef(jtr, "trigger"),
                      direction=JointRef(jdir, "direction"), gear=JointRef(jg, "gear"), clutch=JointRef(jk, "clutch"))
        self.spindle = Spindle(self.j["spindle"], [C, Bk], (1., 0., 0.), [H, C, Bk, T, F, G, K])
        from .holes import Item
        self.item = Item(f"{name}_bit", "bit", Bk, axis=(1., 0., 0.), tip=(stick, 0., 0.), radius=r_bit,
                         length=stick, d_mm=bit_d_mm, bit_type=bit_type, spindle=self.spindle, stickout=stick,
                         wear0=wear)
        self.s_soft = 0.
        self.s_brake = 0.
        self.s_events = []
        sc.mutual_filter(self.bodies)
        sc.tool_bodies.update(self.bodies)

    def joints(self):
        return list(self.j.values())

    def bind(self, world):
        super().bind(world)
        self.spindle.bind(world)

    def readings(self, w):
        x = self.j["trigger"].pos(w)
        s = float(np.clip((x - P.TRIGGER_DEADBAND_M)/(P.TRIGGER_TRAVEL_M - 2*P.TRIGGER_DEADBAND_M), 0., 1.))
        _, d = detent_target(self.j["direction"].pos(w), self.dir_positions)
        _, g = detent_target(self.j["gear"].pos(w), self.gear_positions)
        _, k = detent_target(self.j["clutch"].pos(w), self.clutch_positions)
        return s, ("forward", "lock", "reverse")[d], g + 1, k + 1

    def pre_step(self, w):
        s, direction, gear, clutch = self.readings(w)
        for key, positions in (("direction", self.dir_positions), ("gear", self.gear_positions),
                               ("clutch", self.clutch_positions)):
            tgt, _ = detent_target(self.j[key].pos(w), positions)
            w.tq[self.j[key].q] = tgt
        if direction == "lock":
            s = 0.
        # soft start: the command ramps over DRILL_SOFT_START_S
        rate = w.dt/P.DRILL_SOFT_START_S
        self.s_soft = min(s, self.s_soft + rate) if s > self.s_soft else s
        w0 = P.DRILL_GEAR_RPM[gear - 1]*RPM
        M_stall = P.DRILL_STALL_NM[gear - 1]
        sign = 1. if direction == "forward" else -1.
        if self.s_soft > 1e-3:
            c = M_stall/w0
            cap = min(self.s_soft*M_stall, self.CLUTCH_SETTINGS[clutch - 1])
            self.spindle.set_drive(sign*self.s_soft*w0, c, max(cap, 1e-3))
        else:
            # released: electronic brake stopping the chuck within DRILL_BRAKE_S from full speed
            w_now = self.spindle.speed()
            brake = self.spindle.inertia_about_axis()*w0/P.DRILL_BRAKE_S
            self.spindle.set_drive(0., brake/max(abs(w_now), 1.), brake)

    def state(self):
        w = self.w
        s, direction, gear, clutch = self.readings(w)
        return dict(trigger_m=round(self.j["trigger"].pos(w), 6), speed_command=round(self.s_soft, 4),
                    direction=direction, gear=gear, clutch_setting=clutch if clutch <= 15 else "drill",
                    spindle_rpm=round(self.spindle.speed()/RPM, 2), drive_torque_nm=round(self.spindle.drive_torque(), 4))


# ====================================================================== SDS-plus rotary hammer
class RotaryHammer(Tool):
    """DCH273 class. Frame: x along the bit, z up (D-handle below/behind), y left; origin at the front of the chuck
    sleeve. Built mass 3.12 kg with the 5 Ah battery, without the bit (params.HAMMER_MASS_KG; E1a prints it); side
    handle on the collar 0.22 m ahead of the main grip; mode knob on the left side with its release button (the knob
    turns only while the button is held in: DCH273 manual); depth rod along the side."""
    MODES = ("rotary", "hammer_drill", "hammer")

    # housing part volumes and the authored split of the bare tool's mass over them (gear/percussion barrel 1.55,
    # motor 0.55, grip 0.25 and the handle, collar and side-handle parts at their materials' densities); the split is
    # scaled so that the built tool weighs HAMMER_BARE_KG + BATTERY_5AH_KG (it built 3.61 kg + a 0.25 kg bit before)
    V = dict(barrel=math.pi*0.036**2*0.20, motor=0.09*0.08*0.08, handle_top=0.09*0.032*0.028, grip=0.034*0.038*0.12,
             collar=math.pi*0.034**2*0.024, side_stem=math.pi*0.008**2*0.07, side_grip=math.pi*0.016**2*0.11)
    SPLIT = dict(barrel=1.55, motor=0.55, handle_top=600.*V["handle_top"], grip=0.25, collar=1500.*V["collar"],
                 side_stem=7850.*V["side_stem"], side_grip=600.*V["side_grip"])
    # the other links at their materials' densities: chuck sleeve, trigger, mode dial + pointer, button, depth rod
    MOVING_KG = (2500.*math.pi*0.022**2*0.06 + P.TRIGGER_MASS_KG
                 + 1100.*(math.pi*0.018**2*0.012 + 0.016*0.006*0.004 + math.pi*0.0035**2*0.002)
                 + 7850.*math.pi*0.003**2*0.22)

    @classmethod
    def housing_density(cls, part):
        f = (P.HAMMER_BARE_KG - cls.MOVING_KG)/sum(cls.SPLIT.values())
        return f*cls.SPLIT[part]/cls.V[part]

    def __init__(self, sc, name, pose, bit_d_mm, bit_len_m, mode=1, wear=1.0, free=True):
        self.name = name
        self.d_mm = float(bit_d_mm)
        pr = Parts(sc)
        b = sc.b
        rho = self.housing_density
        H = link(sc, f"{name}_housing", pose)
        # gear/percussion body along x behind the chuck, motor below, D-handle at the rear, battery under the handle
        pr.cyl(H, f"{name}_barrel", (-0.13, 0., 0.), 0.036, 0.10, "x", "hammer_yellow", density=rho("barrel"))
        sc.box(f"{name}_motor", (-0.12, 0., -0.075), (0.045, 0.040, 0.040), "hammer_black", body=H,
               density=rho("motor"))
        sc.box(f"{name}_handle_top", (-0.27, 0., 0.0), (0.045, 0.016, 0.014), "hammer_black", body=H,
               density=rho("handle_top"))
        grip_c = (-0.30, 0., -0.075)
        sc.box(f"{name}_grip", grip_c, (0.017, 0.019, 0.060), "hammer_grip", body=H,
               rot=wq(Rotation.from_euler("y", -12, degrees=True)), density=rho("grip"), mu=P.MU["rubber"])
        sc.box(f"{name}_battery", (-0.27, 0., -0.165), (0.045, 0.038, 0.035), "drill_battery", body=H,
               density=P.BATTERY_5AH_KG/(0.09*0.076*0.07))
        pr.cyl(H, f"{name}_collar", (-0.022, 0., 0.), 0.034, 0.012, "x", "hammer_black", density=rho("collar"))
        # side handle: a rubber grip on a stem below the collar, 0.22 m ahead of the main grip
        pr.cyl(H, f"{name}_side_stem", (-0.022, 0., -0.06), 0.008, 0.035, "z", "steel_dark", density=rho("side_stem"))
        pr.cyl(H, f"{name}_side_grip", (-0.022, 0., -0.145), 0.016, 0.055, "z", "hammer_grip", density=rho("side_grip"),
               mu=P.MU["rubber"])
        # dust shroud ring (render) on the depth-rod clamp
        self.housing = H
        Cp = compose(pose, (0., 0., 0.))
        C = link(sc, f"{name}_chuck", Cp)
        pr.cyl(C, f"{name}_chuck_sleeve", (-0.03, 0., 0.), 0.022, 0.03, "x", "chuck_black", density=2500.)
        Bp = compose(pose, (0., 0., 0.))
        Bk = link(sc, f"{name}_bit", Bp)
        r_bit = 0.5e-3*bit_d_mm
        self.bit_kg = P.SDS_BIT_KG_PER_M*(bit_d_mm/10.)**2*bit_len_m
        pr.bit(Bk, f"{name}_bit", (0., 0., 0.), (1., 0., 0.), bit_d_mm, bit_len_m, "sds_bit", point_angle_deg=130.,
               density=self.bit_kg/bit_volume(r_bit, bit_len_m, 130.), mu=P.MU["bit_wall"])
        self.bit, self.chuck = Bk, C
        Tp = compose(pose, (-0.282, 0., -0.030))
        T = link(sc, f"{name}_trigger", Tp)
        sc.box(f"{name}_trigger_blade", (0., 0., 0.), (0.006, 0.006, 0.014), "drill_trigger", body=T,
               density=P.TRIGGER_MASS_KG/(0.012*0.012*0.028))
        Kp = compose(pose, (-0.13, 0.040, 0.010), Rotation.from_euler("x", -90, degrees=True))
        K = link(sc, f"{name}_mode_knob", Kp)
        pr.cyl(K, f"{name}_mode_dial", (0., 0., 0.006), 0.018, 0.006, "z", "hammer_black", density=1100.)
        sc.box(f"{name}_mode_pointer", (0.012, 0., 0.013), (0.008, 0.003, 0.002), "drill_trigger", body=K, density=1100.)
        # mode selector release button at the dial's centre, standing 1 mm proud of the pointer; it slides into the
        # dial along the knob axis against its spring and, held in, lifts the lock pin out of the selector's notch
        Up = compose(Kp, (0., 0., 0.015))
        U = link(sc, f"{name}_mode_button", Up)
        pr.cyl(U, f"{name}_mode_button_cap", (0., 0., 0.), 0.0035, 0.001, "z", "drill_trigger", density=1100.)
        Dp = compose(pose, (-0.05, 0.045, -0.02))
        D = link(sc, f"{name}_depth_rod", Dp)
        pr.cyl(D, f"{name}_depth_rod_bar", (0., 0., 0.), 0.003, 0.11, "x", "steel_bright", density=7850.)
        I = sc.wp.transform_identity()
        joints = [b.add_joint_free(H, label=f"{name}_free") if free else
                  b.add_joint_fixed(-1, H, parent_xform=sc.wp.transform(tuple(pose[0]), wq(pose[1])), child_xform=I)]
        jsp = b.add_joint_revolute(H, C, parent_xform=rel_xform(sc, pose, Cp), child_xform=I, axis=(1., 0., 0.),
                                   target_kd=0., target_ke=0., effort_limit=P.HAMMER_STALL_NM,
                                   actuator_mode=sc.newton.JointTargetMode.VELOCITY, armature=4e-4,
                                   label=f"{name}_spindle")
        joints.append(jsp)
        k_sds = P.SDS_SPRING_K_N_M
        jsds = b.add_joint_prismatic(C, Bk, parent_xform=rel_xform(sc, Cp, Bp), child_xform=I, axis=(-1., 0., 0.),
                                     limit_lower=0., limit_upper=P.SDS_PLAY_M, target_pos=-P.SDS_SPRING_N/k_sds,
                                     target_ke=k_sds, target_kd=2.0, armature=0.,
                                     actuator_mode=sc.newton.JointTargetMode.POSITION, label=f"{name}_sds")
        joints.append(jsds)
        k_tr = (P.TRIGGER_FULL_N - P.TRIGGER_PRELOAD_N)/P.TRIGGER_TRAVEL_M
        jtr = b.add_joint_prismatic(H, T, parent_xform=rel_xform(sc, pose, Tp), child_xform=I, axis=(-1., 0., 0.),
                                    limit_lower=0., limit_upper=P.TRIGGER_TRAVEL_M, target_pos=-P.TRIGGER_PRELOAD_N/k_tr,
                                    target_ke=k_tr, target_kd=2*0.4*math.sqrt(k_tr*(P.TRIGGER_MASS_KG + P.TRIGGER_ARMATURE_KG)),
                                    actuator_mode=sc.newton.JointTargetMode.POSITION, armature=P.TRIGGER_ARMATURE_KG,
                                    label=f"{name}_trigger")
        joints.append(jtr)
        self.mode_positions = (0., math.radians(60.), math.radians(120.))
        jm = b.add_joint_revolute(H, K, parent_xform=rel_xform(sc, pose, Kp), child_xform=I, axis=(0., 0., 1.),
                                  limit_lower=-0.05, limit_upper=math.radians(120.) + 0.05,
                                  target_pos=self.mode_positions[mode], target_ke=2*0.25/math.radians(60.),
                                  target_kd=0.003, armature=3e-4, actuator_mode=sc.newton.JointTargetMode.POSITION,
                                  label=f"{name}_mode")
        joints.append(jm)
        # the release button: a push button like the press's (half preloaded, PRESS_BUTTON_N-class force), its linkage
        # inertia the drill selectors' (E14-sized)
        k_mb = 0.5*P.MODE_BUTTON_N/(P.MODE_UNLOCK_FRACTION*P.MODE_BUTTON_TRAVEL_M)   # MODE_BUTTON_N at the unlock point
        jmb = b.add_joint_prismatic(K, U, parent_xform=rel_xform(sc, Kp, Up), child_xform=I, axis=(0., 0., -1.),
                                    limit_lower=0., limit_upper=P.MODE_BUTTON_TRAVEL_M, target_pos=-0.5*P.MODE_BUTTON_N/k_mb,
                                    target_ke=k_mb, target_kd=2*0.7*math.sqrt(k_mb*P.SELECTOR_ARMATURE_KG),
                                    armature=P.SELECTOR_ARMATURE_KG, actuator_mode=sc.newton.JointTargetMode.POSITION,
                                    label=f"{name}_mode_button")
        joints.append(jmb)
        # the depth rod last: joints in the tree's depth-first order (the button hangs on the knob)
        jd = b.add_joint_prismatic(H, D, parent_xform=rel_xform(sc, pose, Dp), child_xform=I, axis=(1., 0., 0.),
                                   limit_lower=-0.06, limit_upper=0.14, friction=6.0, armature=0.01,
                                   label=f"{name}_depth_rod")
        joints.append(jd)
        b.add_articulation(joints, label=f"{name}_assembly")
        set_initial_q(b, jm, self.mode_positions[mode])
        self.mode_range = (-0.05, math.radians(120.) + 0.05)
        self.bodies = [H, C, Bk, T, K, D, U]
        self.spring_targets = dict(trigger=-P.TRIGGER_PRELOAD_N/k_tr, sds=-P.SDS_SPRING_N/k_sds,
                                   mode_button=-0.5*P.MODE_BUTTON_N/k_mb)
        # appearance: a smooth procedural housing following these colliders (visual_models.rotary_hammer)
        self.visual = [dict(procedural="rotary_hammer", body=H, hide_shapes_of=[H])]
        self.j = dict(spindle=JointRef(jsp, "spindle"), sds=JointRef(jsds, "sds"), trigger=JointRef(jtr, "trigger"),
                      mode=JointRef(jm, "mode"), depth_rod=JointRef(jd, "depth_rod"),
                      mode_button=JointRef(jmb, "mode_button"))
        self.spindle = Spindle(self.j["spindle"], [C, Bk], (1., 0., 0.), self.bodies, percussive=True)
        from .holes import Item
        self.item = Item(f"{name}_bit", "bit", Bk, axis=(1., 0., 0.), tip=(bit_len_m, 0., 0.), radius=r_bit,
                         length=bit_len_m, d_mm=bit_d_mm, bit_type="sds", spindle=self.spindle, stickout=bit_len_m,
                         wear0=wear)
        self.s_phase = 0.
        self.s_soft = 0.
        self.s_tripped = False
        self.s_overload_s = 0.
        self.s_recoil = []          # [ticks left, amplitude N, direction (world), point (world)]
        self.s_blows = 0
        self.s_blows_delivered = 0
        self.s_removed_mm3 = 0.
        self.s_mode_lock = self.mode_positions[mode]       # the selector's lock pin starts in its notch
        sc.mutual_filter(self.bodies)
        sc.tool_bodies.update(self.bodies)

    def joints(self):
        return list(self.j.values())

    def bind(self, world):
        super().bind(world)
        self.spindle.bind(world)
        self.p_s_full = math.sqrt(2*P.STRIKER_MASS_KG*P.HAMMER_BLOW_J)

    def readings(self, w):
        x = self.j["trigger"].pos(w)
        s = float(np.clip((x - P.TRIGGER_DEADBAND_M)/(P.TRIGGER_TRAVEL_M - 2*P.TRIGGER_DEADBAND_M), 0., 1.))
        _, k = detent_target(self.j["mode"].pos(w), self.mode_positions)
        return s, self.MODES[k]

    def mode_engaged(self, w):
        """The selector is in an operable position: its lock pin in a notch, or within the pin's play of a detent ('no
        operable positions in between', DCH273 manual)."""
        q = self.j["mode"].pos(w)
        tgt, _ = detent_target(q, self.mode_positions)
        return self.s_mode_lock is not None or abs(q - tgt) <= P.MODE_LOCK_PLAY_RAD

    def mode_unlocked(self, w):
        return self.j["mode_button"].pos(w) >= P.MODE_UNLOCK_FRACTION*P.MODE_BUTTON_TRAVEL_M

    def in_window(self, w):
        return self.j["sds"].pos(w) >= P.SDS_PLAY_M - P.IDLE_WINDOW_M

    def pre_step(self, w):
        s, mode = self.readings(w)
        self.spindle.mode = "rotary" if mode == "rotary" else "hammer"
        q_mode = self.j["mode"].pos(w)
        tgt, _ = detent_target(q_mode, self.mode_positions)
        w.tq[self.j["mode"].q] = tgt
        # selector lock: unless the release button is held in, the lock pin sits in the notch of the detent the knob is
        # at (the knob's joint range closes to the pin's play); between detents with the button let go the pin rides
        # on the dial until the detent spring brings a notch under it
        if self.mode_unlocked(w):
            self.s_mode_lock = None
        elif self.s_mode_lock is None and abs(q_mode - tgt) <= P.MODE_LOCK_PLAY_RAD:
            self.s_mode_lock = tgt
        lo, hi = self.mode_range
        if self.s_mode_lock is not None:
            lo, hi = self.s_mode_lock - P.MODE_LOCK_PLAY_RAD, self.s_mode_lock + P.MODE_LOCK_PLAY_RAD
        w.solver.mj_model.jnt_range[self.j["mode"].mj] = (lo, hi)
        if not self.mode_engaged(w):
            s = 0.                               # gears not engaged between positions: neither rotation nor blows
        if s < 1e-3:
            self.s_tripped = False
        rate = w.dt/P.DRILL_SOFT_START_S
        self.s_soft = min(s, self.s_soft + rate) if s > self.s_soft else s
        w0 = P.HAMMER_RPM*RPM
        # motor: torque-speed line through the rotation gear, up to its own stall torque. The electronic overload
        # clutch interrupts the drive when the bit jams: the motor at HAMMER_CLUTCH_FRACTION of the stall torque its
        # trigger allows (the spindle slowed to a tenth of its commanded speed) for HAMMER_CLUTCH_S; releasing the
        # trigger re-arms it (DCH273 manual, Overload Clutch)
        c = P.HAMMER_STALL_NM/w0
        if self.s_soft > 1e-3 and mode != "hammer" and not self.s_tripped:
            self.spindle.set_drive(self.s_soft*w0, c, self.s_soft*P.HAMMER_STALL_NM)
            if abs(self.spindle.drive_torque()) >= self.s_soft*P.HAMMER_CLUTCH_NM:
                self.s_overload_s += w.dt
            else:
                self.s_overload_s = 0.
            if self.s_overload_s >= P.HAMMER_CLUTCH_S:
                self.s_tripped = True
                self.s_overload_s = 0.
                w.log_event(dict(type="hammer_clutch_trip", tool=self.name, tick=w.tick))
        else:
            self.s_overload_s = 0.
            # rotation off: the gear train holds the chuck (spindle lock in hammer-only mode, brake otherwise)
            self.spindle.set_drive(0., 0.5, 5. if mode == "hammer" else 2.)
        # blows: f_b = s x 76.7 Hz, E_b = 2.1 J x s^2.4, only inside the idle-stroke window, not after a clutch trip
        if self.s_soft > 1e-3 and mode != "rotary" and not self.s_tripped:
            self.s_phase += self.s_soft*P.HAMMER_BPM/60.*w.dt
            if self.s_phase >= 1.:
                self.s_phase -= 1.
                self.s_blows += 1
                if self.in_window(w):
                    E = P.HAMMER_BLOW_J*self.s_soft**P.HAMMER_ENERGY_EXP
                    dv = w.hm.blow(self.item, E)
                    self.s_removed_mm3 += dv
                    self.s_blows_delivered += 1
                    p_s = math.sqrt(2*P.STRIKER_MASS_KG*E)
                    # the air spring's reaction: a half-sine RECOIL_DUTY of the blow period wide (Wang 2014)
                    n = max(1, int(round(P.RECOIL_DUTY/(self.s_soft*P.HAMMER_BPM/60.)/w.dt)))
                    # amplitude such that the n sampled ticks carry exactly p_s (the continuous half-sine's amplitude
                    # pi p_s/(2 n dt) sampled at 4 ticks gave 2.6 % too much impulse)
                    amp = p_s/(w.dt*sum(math.sin(math.pi*(j + 0.5)/n) for j in range(n)))
                    self.s_recoil.append([n, n, amp])
        else:
            self.s_phase = 0.
        # recoil: half-sine pulse of impulse p_s along -bit axis on the housing, through the spindle axis
        if self.s_recoil:
            p, R = w.body_pose(self.housing)
            axis_w = R @ np.array([1., 0., 0.])
            F = 0.
            keep = []
            for left, n, amp in self.s_recoil:
                k = n - left
                F += amp*math.sin(math.pi*(k + 0.5)/n)
                if left > 1:
                    keep.append([left - 1, n, amp])
            self.s_recoil = keep
            w.add_force(self.housing, -F*axis_w, p)

    def state(self):
        w = self.w
        s, mode = self.readings(w)
        return dict(trigger_m=round(self.j["trigger"].pos(w), 6), speed_command=round(self.s_soft, 4), mode=mode,
                    sds_slide_m=round(self.j["sds"].pos(w), 5), in_idle_window=bool(self.in_window(w)),
                    spindle_rpm=round(self.spindle.speed()/RPM, 2), drive_torque_nm=round(self.spindle.drive_torque(), 3),
                    clutch_tripped=self.s_tripped, blows_scheduled=self.s_blows, blows_delivered=self.s_blows_delivered,
                    removed_mm3=round(self.s_removed_mm3, 2), depth_rod_m=round(self.j["depth_rod"].pos(w), 4),
                    mode_knob_deg=round(math.degrees(self.j["mode"].pos(w)), 3), mode_engaged=bool(self.mode_engaged(w)),
                    mode_button_m=round(self.j["mode_button"].pos(w), 5), mode_unlocked=bool(self.mode_unlocked(w)))


# ====================================================================== drill press
class DrillPress(Tool):
    """Benchtop drill press (G7943 class) on a steel stand, drawn by the Poly Haven drill_press_01 model whose parts give
    the colliders (profiles measured in this build). Frame: the press's own - x towards the operator (the
    table front), z up, origin on the floor under the column. Model -> press: x = 0.06 - m_y, y = m_x, z = m_z + stand.
    The spindle axis is vertical at x = SPINDLE_X; the feed shaft axis is along +y on the right of the head; positive
    shaft rotation lowers the quill (x_quill = -r_p phi). Start and stop push-buttons on the head's left side."""
    SPINDLE_X = 0.19
    HEAD_BOTTOM = 0.500          # above the stand top
    CHUCK = (0.395, 0.491, 0.025)
    HUB = (0.124, 0.1085, 0.546)

    def __init__(self, sc, name, origin_xy, yaw, belt_rpm, bit_d_mm, bit_len_m, depth_stop_m=None):
        self.name = name
        self.d_mm = float(bit_d_mm)
        pr = Parts(sc)
        b = sc.b
        Rz = Rotation.from_euler("z", yaw)
        base = (np.array([origin_xy[0], origin_xy[1], 0.]), Rz)
        self.base_pose = base
        self.belt_rpm = float(belt_rpm)
        S = P.PRESS_STAND_M
        P_ = lambda *v: tuple(map(float, base[0] + Rz.apply(v)))
        rot = wq(Rz)
        # stand, base casting, column, table and arm, head (static colliders under the model)
        sc.box(f"{name}_stand", P_(0.09, 0., 0.5*S), (0.21, 0.15, 0.5*S), "stand_grey", rot=rot)
        sc.box(f"{name}_base", P_(0.0905, 0., S + 0.03), (0.2045, 0.13, 0.03), "press_hidden", rot=rot)
        pr.cyl(-1, f"{name}_column", P_(0., 0., S + 0.28), 0.035, 0.22, "z", "press_hidden")
        pr.cyl(-1, f"{name}_table", P_(self.SPINDLE_X, 0.003, S + 0.194), 0.095, 0.010, "z", "press_hidden",
               mu=P.MU["steel_steel"])
        sc.box(f"{name}_table_arm", P_(0.06, 0.003, S + 0.16), (0.045, 0.03, 0.028), "press_hidden", rot=rot)
        sc.box(f"{name}_head", P_(0.063, -0.0175, S + 0.58), (0.163, 0.0995, 0.08), "press_hidden", rot=rot)
        sc.box(f"{name}_belt_housing", P_(-0.019, 0., S + 0.69), (0.274, 0.10, 0.03), "press_hidden", rot=rot)
        self.table_top = S + 0.204
        # quill (origin at the head bottom) + spindle with its chuck + bit
        hb = S + self.HEAD_BOTTOM - 0.001
        q_top = (np.array(P_(self.SPINDLE_X, 0., hb)), Rz)
        Q = link(sc, f"{name}_quill", q_top)
        # the quill's mass (PRESS_QUILL_KG less the turning chuck) on its collider stub; the upper tube is drawn only
        pr.cyl(Q, f"{name}_quill_tube", (0., 0., -0.005), 0.024, 0.004, "z", "steel_bright",
               density=(P.PRESS_QUILL_KG - P.PRESS_CHUCK_KG)/(math.pi*0.024**2*0.008))
        pr.cyl(Q, f"{name}_quill_upper", (0., 0., 0.045), 0.024, 0.045, "z", "steel_bright", collide=False)
        S_ = q_top
        Sb = link(sc, f"{name}_spindle", S_)
        c0, c1, cr = self.CHUCK
        pr.cyl(Sb, f"{name}_chuck_body", (0., 0., 0.5*(c0 + c1) - self.HEAD_BOTTOM), cr, 0.5*(c1 - c0), "z",
               "chuck_black", density=P.PRESS_CHUCK_KG/(math.pi*cr**2*(c1 - c0)))
        Bp = compose(S_, (0., 0., c0 - self.HEAD_BOTTOM))
        Bk = link(sc, f"{name}_bit", Bp)
        r_bit = 0.5e-3*bit_d_mm
        pr.bit(Bk, f"{name}_bit", (0., 0., 0.), (0., 0., -1.), bit_d_mm, bit_len_m, "hss_bit", density=7850.,
               mu=P.MU["bit_wall"])
        self.quill, self.spindle_body, self.bit = Q, Sb, Bk
        # feed shaft (axis along the press y) with three spokes and knobs at R = 0.21 m (the model's handle)
        hx, hy, hzz = self.HUB
        Fp = (np.array(P_(hx, hy, S + hzz)), Rz)
        F = link(sc, f"{name}_feed_shaft", Fp)
        pr.cyl(F, f"{name}_hub", (0., 0., 0.), 0.024, 0.0265, "y", "press_hidden", density=7850.)
        self.spoke_angles = (math.radians(-31.), math.radians(90.), math.radians(211.))
        for k, a0 in enumerate(self.spoke_angles):
            d = np.array([math.cos(a0), 0., math.sin(a0)])
            pr.capsule(F, f"{name}_spoke{k}", tuple(d*0.5*P.PRESS_HANDLE_R + np.array([0., 0.035, 0.])), 0.0055,
                       0.5*P.PRESS_HANDLE_R - 0.03, d, "press_hidden", density=7850.)
            pr.sphere(F, f"{name}_knob{k}", tuple(d*P.PRESS_HANDLE_R + np.array([0., 0.04, 0.])), 0.014, "press_hidden",
                      density=1200., mu=P.MU["rubber"])
        # depth-stop nut on its rod beside the quill (front left), and the start / stop push-buttons
        Np = (np.array(P_(self.SPINDLE_X, -0.068, S + 0.475)), Rz)
        N = link(sc, f"{name}_depth_nut", Np)
        pr.cyl(N, f"{name}_depth_nut_body", (0., 0., 0.), 0.011, 0.006, "z", "steel_bright", density=7850.)
        buttons = []
        # a push button stands proud of its bezel by more than its travel: the cap's back face is 3.3 mm in front of the
        # head's face (3 mm travel). At 1.5 mm the head stops the cap short of the 2.1 mm that switches
        for k, (bx, label) in enumerate(((0.047, "start"), (0.079, "stop"))):
            Wp = (np.array(P_(bx, -0.117 - P.PRESS_BUTTON_TRAVEL_M - 0.0003, S + 0.6285)), Rz)
            W = link(sc, f"{name}_{label}_button", Wp)
            sc.box(f"{name}_{label}_button_cap", (0., -0.002, 0.), (0.012, 0.002, 0.012), "press_hidden", body=W,
                   density=1200.)
            buttons.append((W, Wp, label))
        I = sc.wp.transform_identity()
        self.r_p = P.PRESS_PINION_R
        travel = P.PRESS_TRAVEL_M
        jq = b.add_joint_prismatic(-1, Q, parent_xform=sc.wp.transform(tuple(q_top[0]), rot), child_xform=I,
                                   axis=(0., 0., 1.), limit_lower=-travel, limit_upper=0., armature=0.02,
                                   label=f"{name}_quill")
        jsp = b.add_joint_revolute(Q, Sb, parent_xform=I, child_xform=I, axis=(0., 0., -1.), target_kd=0., target_ke=0.,
                                   effort_limit=50., actuator_mode=sc.newton.JointTargetMode.VELOCITY,
                                   armature=P.PRESS_ROTOR_INERTIA, label=f"{name}_spindle")
        jb = b.add_joint_fixed(Sb, Bk, parent_xform=rel_xform(sc, S_, Bp), child_xform=I, label=f"{name}_bit_weld")
        b.add_articulation([jq, jsp, jb], label=f"{name}_quill_assembly")
        k_sp = P.PRESS_SPRING_RATE
        js = b.add_joint_revolute(-1, F, parent_xform=sc.wp.transform(tuple(Fp[0]), rot), child_xform=I, axis=(0., 1., 0.),
                                  target_pos=-P.PRESS_SPRING_PRE_NM/k_sp, target_ke=k_sp, target_kd=0.05,
                                  actuator_mode=sc.newton.JointTargetMode.POSITION, armature=0.01, friction=0.05,
                                  label=f"{name}_feed_shaft")
        b.add_articulation([js], label=f"{name}_feed")
        jn = b.add_joint_revolute(-1, N, parent_xform=sc.wp.transform(tuple(Np[0]), rot), child_xform=I, axis=(0., 0., 1.),
                                  armature=1e-4, friction=0.02, label=f"{name}_depth_nut")
        b.add_articulation([jn], label=f"{name}_nut")
        jbs = {}
        k_b = P.PRESS_BUTTON_N/P.PRESS_BUTTON_TRAVEL_M
        for W, Wp, label in buttons:
            jw = b.add_joint_prismatic(-1, W, parent_xform=sc.wp.transform(tuple(Wp[0]), rot), child_xform=I,
                                       axis=(0., 1., 0.), limit_lower=0., limit_upper=P.PRESS_BUTTON_TRAVEL_M,
                                       target_pos=-0.5*P.PRESS_BUTTON_N/k_b, target_ke=k_b, target_kd=0.5, armature=0.005,
                                       actuator_mode=sc.newton.JointTargetMode.POSITION, label=f"{name}_{label}_button")
            b.add_articulation([jw], label=f"{name}_{label}_button")
            jbs[label] = JointRef(jw, label)
        from newton._src.solvers.mujoco.equality import _add_equality_constraint
        _add_equality_constraint(b, constraint_type=sc.newton.solvers.SolverMuJoCo.EqType.JOINT, joint1=jq, joint2=js,
                                 polycoef=[0., -self.r_p, 0., 0., 0.], label=f"{name}_rack_pinion",
                                 custom_attributes={"mujoco:eq_solref": (2*P.DT, 1.),
                                                    "mujoco:eq_solimp": (.9999, .9999, .001, .5, 2.)})
        self.bodies = [Q, Sb, Bk, F, N] + [W for W, _, _ in buttons]
        self.spring_targets = dict(start=-0.5*P.PRESS_BUTTON_N/k_b, stop=-0.5*P.PRESS_BUTTON_N/k_b)
        self.j = dict(quill=JointRef(jq, "quill"), spindle=JointRef(jsp, "spindle"), shaft=JointRef(js, "shaft"),
                      nut=JointRef(jn, "nut"), start=jbs["start"], stop=jbs["stop"])
        self.spindle = Spindle(self.j["spindle"], [Sb, Bk], (0., 0., -1.), [Q, Sb, Bk])
        from .holes import Item
        self.item = Item(f"{name}_bit", "bit", Bk, axis=(0., 0., -1.), tip=(0., 0., -bit_len_m), radius=r_bit,
                         length=bit_len_m, d_mm=bit_d_mm, bit_type="hss", spindle=self.spindle, stickout=bit_len_m)
        self.depth_stop0 = -travel if depth_stop_m is None else -float(depth_stop_m)
        self.s_on = False
        sc.mutual_filter(self.bodies)
        sc.tool_bodies.update(self.bodies)
        # appearance: the model's static mesh on the world, its feed handle on the shaft (about the hub), the depth
        # limiter, belt cover and buttons; our chuck and bit drawn by their own shapes; the model's bit part omitted
        Rm = Rz*Rotation.from_euler("z", math.pi/2)
        model_origin = np.array(P_(0.06, 0., S))
        hub_m = np.array([0.092, -0.064, 0.546])                     # model coordinates of the hub centre
        hub_p = np.array([0.06 + 0.064, 0.092, S + 0.546])
        self.visual = [dict(asset="drill_press_01", body=-1, pos=tuple(model_origin), quat=wq(Rm), scale=1.0,
                            prims=["drill_press_01", "drill_press_01_lid", "drill_press_01_belt", "drill_press_01_gear_a",
                                   "drill_press_01_gear_b", "drill_press_01_plate", "drill_press_01_limiter",
                                   "drill_press_01_plug", "drill_press_01_catch", "drill_press_01_button_a",
                                   "drill_press_01_button_b"]),
                       dict(asset="drill_press_01", body=F, pos=tuple(Rotation.from_euler("z", math.pi/2).apply(-hub_m)),
                            quat=wq(Rotation.from_euler("z", math.pi/2)), scale=1.0, prims=["drill_press_01_handle"])]
        self.hidden_material = "press_hidden"

    def joints(self):
        return list(self.j.values())

    def bind(self, world):
        super().bind(world)
        self.spindle.bind(world)
        # induction motor through the belt: synchronous 1800 rpm motor, rated 1725 rpm
        n = self.belt_rpm
        self.w_rated = n*RPM
        self.w_sync = n*RPM*1800./1725.
        self.M_rated = P.PRESS_MOTOR_W/self.w_rated
        self.c = self.M_rated/(self.w_sync - self.w_rated)
        self.M_break = P.PRESS_BREAKDOWN*self.M_rated

    def depth_stop(self, w):
        """Quill lower limit (m, negative) set by the nut's turns on its M12 x 1.5 rod (1.5 mm per turn)."""
        return float(np.clip(self.depth_stop0 + self.j["nut"].pos(w)/(2*math.pi)*P.PRESS_STOP_PITCH_M,
                             -P.PRESS_TRAVEL_M, 0.))

    def pre_step(self, w):
        mjm = w.solver.mj_model
        mjm.jnt_range[self.j["quill"].mj][0] = self.depth_stop(w)
        # magnetic switch: the start button latches the motor on, the stop button releases it
        pressed = lambda k: self.j[k].pos(w) > P.PRESS_BUTTON_SWITCH_FRACTION*P.PRESS_BUTTON_TRAVEL_M
        if pressed("stop"):
            self.s_on = False
        elif pressed("start"):
            self.s_on = True
        if self.s_on:
            self.spindle.set_drive(self.w_sync, self.c, self.M_break)
        else:
            # coast-down: bearing and belt drag with a PRESS_COAST_S time constant
            self.spindle.set_drive(0., self.spindle.inertia_about_axis()/P.PRESS_COAST_S, 3.)

    def state(self):
        w = self.w
        return dict(switch_on=self.s_on, belt_rpm=self.belt_rpm, spindle_rpm=round(self.spindle.speed()/RPM, 2),
                    quill_m=round(self.j["quill"].pos(w), 5), depth_stop_m=round(self.depth_stop(w), 5),
                    shaft_rad=round(self.j["shaft"].pos(w), 4), drive_torque_nm=round(self.spindle.drive_torque(), 4),
                    start_button_m=round(self.j["start"].pos(w), 5), stop_button_m=round(self.j["stop"].pos(w), 5))


class Vise(Tool):
    """4 in drill-press vise. Frame: x along the screw (towards the crank = the operator), z up. The fixed jaw is at
    the far end (-x); the moving jaw closes towards -x when the crank turns clockwise seen from the crank (+x)."""

    def __init__(self, sc, name, pose, opening_m, bolted=True):
        self.name = name
        b = sc.b
        pr = Parts(sc)
        Vb = link(sc, f"{name}_body", pose)
        # the bed in two halves either side of a 30 mm through-slot under the jaws (a bit breaking through exits into it)
        for k, sgn in enumerate((-1, 1)):
            sc.box(f"{name}_bed{k}", (0., sgn*0.040, 0.015), (0.115, 0.025, 0.015), "vise_blue", body=Vb, density=7200.,
                   mu=P.MU["steel_steel"])
        sc.box(f"{name}_fixed_jaw", (-0.095, 0., 0.050), (0.020, 0.0575, 0.020), "vise_blue", body=Vb, density=7200.)
        fp = sc.box(f"{name}_fixed_plate", (-0.0725, 0., 0.052), (0.0025, 0.051, 0.018), "steel_dark", body=Vb,
                    density=7850., mu=P.VISE_JAW_MU)
        sc.stiff_shapes.add(fp)
        # a pair of 24 mm parallels on the bed either side of the slot: thin work sits near the top of the jaws, as a
        # machinist sets it (on the bare bed a 6 mm bar was hidden 28 mm down between the jaws)
        for k, sgn in enumerate((-1, 1)):
            sc.stiff_shapes.add(sc.box(f"{name}_parallel{k}", (-0.045, sgn*0.035, 0.044), (0.06, 0.012, 0.014),
                                       "steel_bright", body=Vb, density=7850., mu=P.MU["steel_steel"]))
        x_jaw = -0.07 + opening_m
        Jp = compose(pose, (x_jaw, 0., 0.))
        J = link(sc, f"{name}_jaw", Jp)
        sc.box(f"{name}_jaw_block", (0.0225, 0., 0.050), (0.020, 0.0575, 0.020), "vise_blue", body=J, density=7200.)
        jp = sc.box(f"{name}_jaw_plate", (0.0025, 0., 0.052), (0.0025, 0.051, 0.018), "steel_dark", body=J, density=7850.,
                    mu=P.VISE_JAW_MU)
        sc.stiff_shapes.add(jp)
        Sp = compose(pose, (0.13, 0., 0.035))
        S = link(sc, f"{name}_screw", Sp)
        pr.cyl(S, f"{name}_screw_hub", (0.005, 0., 0.), 0.012, 0.010, "x", "steel_dark", density=7850.)
        pr.capsule(S, f"{name}_crank_arm", (0.02, 0., 0.04), 0.005, 0.04, "z", "steel_bright", density=7850.)
        pr.cyl(S, f"{name}_crank_knob", (0.04, 0., P.VISE_CRANK_R), 0.010, 0.025, "x", "knob_black", density=1200.,
               mu=P.MU["rubber"])
        I = sc.wp.transform_identity()
        joints = [b.add_joint_fixed(-1, Vb, parent_xform=sc.wp.transform(tuple(pose[0]), wq(pose[1])), child_xform=I,
                                    label=f"{name}_bolts") if bolted else b.add_joint_free(Vb, label=f"{name}_free")]
        jj = b.add_joint_prismatic(Vb, J, parent_xform=rel_xform(sc, pose, Jp), child_xform=I, axis=(1., 0., 0.),
                                   limit_lower=-opening_m, limit_upper=0.102 - opening_m, armature=0.05,
                                   label=f"{name}_jaw")
        js = b.add_joint_revolute(Vb, S, parent_xform=rel_xform(sc, pose, Sp), child_xform=I, axis=(1., 0., 0.),
                                  armature=2e-3, friction=P.VISE_DRY_FRICTION_NM, label=f"{name}_screw")
        joints += [jj, js]
        b.add_articulation(joints, label=f"{name}_assembly")
        from newton._src.solvers.mujoco.equality import _add_equality_constraint
        # clockwise from the crank (rotation about -x, i.e. negative about +x) closes: x_jaw = (p/2 pi) theta
        self.eq_label = f"{name}_screw_thread"
        self.eq_newton = _add_equality_constraint(b, constraint_type=sc.newton.solvers.SolverMuJoCo.EqType.JOINT, joint1=jj, joint2=js,
                                 polycoef=[0., P.VISE_PITCH_M/(2*math.pi), 0., 0., 0.], label=self.eq_label,
                                 custom_attributes={"mujoco:eq_solref": (2*P.DT, 1.),
                                                    "mujoco:eq_solimp": (.9999, .9999, .001, .5, 2.)})
        self.bodies = [Vb, J, S]
        self.body = Vb
        self.jaw_body = J
        self.bolted = bolted
        self.j = dict(jaw=JointRef(jj, "jaw"), screw=JointRef(js, "screw"))
        self.s_clamp_n = 0.
        sc.mutual_filter(self.bodies)
        sc.tool_bodies.update(self.bodies)

    def joints(self):
        return list(self.j.values())

    def bind(self, world):
        super().bind(world)
        mjm = world.solver.mj_model
        e2n = np.asarray(world.solver.mjc_eq_to_newton_eq.numpy()).reshape(-1)
        self.eq_id = next(i for i in range(mjm.neq) if int(e2n[i]) == int(self.eq_newton))

    def pre_step(self, w):
        # thread friction (ISO 16047 form, flank angle folded into mu): T_f = mu F r_m, plus the screw shoulder's collar
        # friction on the body end, mu_c F r_c (Shigley's power-screw torque); F = the thread's axial load, read from
        # last tick's solved equality force (jaw axis force, N)
        import mujoco
        m, d = w.solver.mj_model, w.solver.mj_data
        F = 0.
        n = d.nefc
        if n:
            types = np.asarray(d.efc_type[:n])
            ids = np.asarray(d.efc_id[:n])
            rows = np.flatnonzero((types == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) & (ids == self.eq_id))
            if len(rows):
                F = abs(float(d.efc_force[rows[0]]))
        self.s_clamp_n = F
        m.dof_frictionloss[self.j["screw"].dof if False else m.jnt_dofadr[self.j["screw"].mj]] = \
            P.VISE_DRY_FRICTION_NM + (P.VISE_THREAD_MU*P.VISE_SCREW_RM + P.VISE_COLLAR_MU*P.VISE_COLLAR_R_M)*F

    def state(self):
        w = self.w
        return dict(opening_m=round(self.j["jaw"].pos(w), 5), screw_rad=round(self.j["screw"].pos(w), 4),
                    clamp_force_n=round(self.s_clamp_n, 1), bolted=self.bolted)


# ====================================================================== cordless impact wrench with adaptive torque
class ImpactWrench(Tool):
    """Hilti SIW 6AT-A22 class 1/2 in cordless impact wrench with the SI-AT-A22 Adaptive Torque module (the setting tool
    ESR-4266 4.2.3 permits for the 1/2 in KB-TZ2, except where Hilti's installation instructions 2274242-03.2025 exclude
    it: "AT-System not applicable for grout-filled CMU and wood fastening applications"; ledger-anchor uses it through a
    timber ledger, a disclosed deviation: layout.wall_layout), a deep 3/4 in impact socket on its anvil. Frame: x along
    the anvil towards the socket's mouth, z up (grip and battery below), y left; origin at the socket's mouth.

    Mechanism (Hilti US 10,589,407 B2: a hammer on the motor's drive shaft, pressed against the anvil by a spring through
    a spiral cam; when the anvil resists more than the cam releases, the hammer rides off the anvil's jaws, is flung
    forward and strikes them tangentially). The trigger runs the motor. While the socket meets less than the cam's
    release torque (the nut running down its thread) the anvil turns with the motor. Above it the hammer strikes at the
    blow rate. A blow is a hammer of inertia I_h moving at the drive shaft's speed w that meets the anvil's jaws through
    a one-way torsion spring k (the jaws, anvil and socket): it carries the hammer's kinetic energy I_h w^2/2 and no more,
    it cannot drive the anvil faster than about the hammer moves, and against a nut that does not turn the spring's
    torque peaks at w sqrt(k I_h). The product k I_h is fixed by the maker's largest torque at its largest speed
    (IMPACT_MAX_TORQUE_NM at IMPACT_MAX_RPM; the three gears' 90/135/213 N m at 1000/1500/2300 rpm are proportional to
    speed, as this law is), and I_h is an estimate (IMPACT_HAMMER_INERTIA). The torque a blow leaves on a nut is
    therefore a result: the nut turns on while the spring's torque exceeds its thread and bearing friction, the work
    done on it is the blow's energy less what the spring still holds when the hammer has stopped (that share rebounds),
    and the blows of a speed can never take a nut beyond that speed's peak torque.

    The Adaptive Torque module (a start state: its box scanned for the 1/2 in KB-TZ2) does what the patent's first phase
    describes: it runs the motor at a stored first speed (AT_FIRST_RPM: authored, the wrench's full speed; Hilti's
    table is not published) and senses the tightening torque on the anvil's shaft; when the torque a blow passed to
    the socket reaches the stored threshold (authored: the anchor's installation torque) it stops the motor (green
    light). The torque an anchor is left with is therefore where the module's reading stops the blows: the threshold,
    give or take what one blow adds and what the reading errs by (E20: 67.2-68.7 N m for 68). A run that
    does not get there within AT_TIMEOUT_S of impacting stops with a red light. The patent's later phases (a waiting
    time for the anchor to relax, a fixed number of further blows, a timed phase at a lower speed) answer a relaxation
    of the preload that the anchor model does not have; their numbers are unpublished and they are NOT modelled.
    Every further pull of the trigger on a set anchor runs the program again and adds a blow (E20 retrigger: 0.3-1.4
    N m a pull; past 1.7 times the installation torque at the 77th pull, the stud broken at the 95th). Without the
    module (`at_module=False`, a start state no task uses) the wrench runs in its gear for as long as the trigger is
    held: in gear 3 it takes the 1/2 in anchor past 1.7 times its installation torque in 2.6 s and breaks its stud in
    3.1 s (E20 no_module). Releasing the trigger re-arms the wrench.

    DECLARED DEVICES (what is not a body or not at its real scale):
    - Time dilation. A real blow lasts under a millisecond and the 1 ms step cannot resolve it. Every blow is stretched
      IMPACT_DILATION (s) times: hammer inertia s^2 I_h at speed w/s on the same spring. The blow's energy and its peak
      torque are unchanged; its duration and its angular impulse are s times the real ones and the anvil moves s times
      slower. The anvil and socket carry s^2 times their own inertia for the duration of a blow (the body's inertia about
      the anvil's axis, written when a blow begins and when the anvil has come to rest after it; an armature would put
      the reaction on the housing). Between blows they have their own inertia and turn freely.
    - The hammer is not a body. Its state is an energy (kinetic plus spring) and the spring's compression, stepped
      here; the spring's torque acts on the anvil body and its reaction on the hammer, inside this state. Each tick the
      energy falls by the torque times the rotation the solver gave the anvil, so the work a blow does on the nut can
      never exceed the hammer's energy; an overdraft within the last tick of a blow is taken from the next blow.
    - The spring's unloading after the hammer has stopped is not applied to the anvil (it does no work on a nut held
      by its friction); the energy it holds then is the rebound and is lost.
    - The housing feels the mean torque of re-accelerating the REAL hammer (I_h w x blow rate), not the blows.
    - The module's sensed torque is the torque the socket passed on in a tick, from the anvil's own equation of motion
      (the hammer's torque less the dilated anvil's inertia times its acceleration); a blow's reading is the largest
      value held in two of its ticks (authored signal processing)."""
    MODES = ("idle", "run", "impact", "done", "stopped")

    def __init__(self, sc, name, pose, free=True, target_nm=None, at_module=True, gear=3):
        self.name = name
        pr = Parts(sc)
        b = sc.b
        H = link(sc, f"{name}_housing", pose)
        # housing colliders on the SIW 6AT-A22's envelope (154 x 94 x 228 mm with its battery): the head with the
        # gear case and hammer along x behind the socket, the grip below its rear, the battery under the grip
        vol = dict(head=math.pi*0.030**2*0.120, grip=0.034*0.032*0.110, battery=0.120*0.080*0.070)
        mk = dict(head=0.62*P.IMPACT_TOOL_KG, grip=0.38*P.IMPACT_TOOL_KG,
                  battery=P.IMPACT_BATTERY_KG + (P.AT_MODULE_KG if at_module else 0.))        # the module sits on the pack
        pr.cyl(H, f"{name}_head", (-0.068 - 0.060, 0., 0.), 0.030, 0.060, "x", "anchor_box_red", density=mk["head"]/vol["head"],
               mu=P.MU["default"])
        sc.box(f"{name}_grip", (-0.150, 0., -0.085), (0.017, 0.016, 0.055), "drill_grip", body=H,
               density=mk["grip"]/vol["grip"], mu=P.MU["rubber"])
        sc.box(f"{name}_battery", (-0.140, 0., -0.175), (0.060, 0.040, 0.035), "drill_battery", body=H,
               density=mk["battery"]/vol["battery"])
        self.housing = H
        # anvil + deep socket: the socket's hex at its mouth, 6 steel walls whose inner faces stand SOCKET_CLEARANCE_M
        # off the nut's flats, 10 mm deep with the lead-in (shorter than the 11.1 mm nut: the socket's rim clears the
        # washer)
        A = link(sc, f"{name}_anvil", pose, mass=P.SOCKET_KG, com=(-0.035, 0., 0.),
                 inertia=(0.5*P.SOCKET_KG*(0.014**2 + 0.0105**2), 0.5*P.SOCKET_KG*0.063**2/3., 0.5*P.SOCKET_KG*0.063**2/3.))
        af, t, depth = 0.01905 + 2*P.SOCKET_CLEARANCE_M, 0.006, 0.010
        hw = 0.01905/(2*math.sqrt(3.)) + 0.0035
        c = P.SOCKET_LEAD_IN_M
        walls = []
        tilt = Rotation.from_euler("y", -45., degrees=True)      # a facet's x along the chamfer, its z into the steel
        for k in range(6):
            Rk = Rotation.from_euler("x", 60.*k, degrees=True)
            nrm = Rk.apply([0., 0., 1.])
            # the hex wall behind the lead-in
            ctr = np.array([-0.5*(depth + c), 0., 0.]) + nrm*(0.5*af + 0.5*t)
            walls.append(sc.box(f"{name}_socket_wall{k}", tuple(ctr), (0.5*(depth - c), hw, 0.5*t), "steel_dark", body=A,
                                rot=wq(Rk), density=0., mu=P.MU["steel_steel"]))
            # the mouth's lead-in: a 45 degree chamfer SOCKET_LEAD_IN_M deep and wide on each wall (it centres the socket
            # on a nut it meets off its axis and lets the nut's corners in before the hexes are in line), and the flat
            # rim outside it. Without it the socket's flat rim stood on the nut's face whenever the two were more than
            # the hex's 0.15 mm clearance out of line (an anchor lies 0.3 mm low in its 13.3 mm hole; E20 pushon)
            h_f = 0.0015
            mid = np.array([-0.5*c, 0., 0.5*af + 0.5*c]) + tilt.apply([0., 0., h_f])
            walls.append(sc.box(f"{name}_socket_lead{k}", tuple(Rk.apply(mid)), (c/math.sqrt(2.), hw, h_f), "steel_dark",
                                body=A, rot=wq(Rk*tilt), density=0., mu=P.MU["steel_steel"]))
            ctr = np.array([-0.5*c, 0., 0.]) + nrm*(0.5*af + c + 0.5*(t - c))
            walls.append(sc.box(f"{name}_socket_rim{k}", tuple(ctr), (0.5*c, hw, 0.5*(t - c)), "steel_dark", body=A,
                                rot=wq(Rk), density=0., mu=P.MU["steel_steel"]))
        # the hex's bottom: a ring of 6 segments (7.0-9.6 mm from the axis: the stud passes on into the deep socket's
        # bore without touching it) on which the nut's outer face bears when the socket is pushed home
        for k in range(6):
            Rk = Rotation.from_euler("x", 60.*k + 30., degrees=True)
            ctr = np.array([-depth - 0.001, 0., 0.]) + Rk.apply([0., 0., 1.])*0.0083
            walls.append(sc.box(f"{name}_socket_seat{k}", tuple(ctr), (0.001, 0.0045, 0.0013), "steel_dark", body=A, rot=wq(Rk),
                                density=0., mu=P.MU["steel_steel"]))
        sc.stiff_shapes.update(walls)
        pr.cyl(A, f"{name}_socket_barrel", (-0.0315, 0., 0.), 0.014, 0.0315, "x", "steel_dark", collide=False)
        self.anvil, self.socket_walls = A, walls
        # trigger at the top front of the grip (as the drill's)
        Tp = compose(pose, (-0.135, 0., -0.040))
        T = link(sc, f"{name}_trigger", Tp)
        sc.box(f"{name}_trigger_blade", (0., 0., 0.), (0.006, 0.006, 0.011), "drill_trigger", body=T,
               density=P.TRIGGER_MASS_KG/(0.012*0.012*0.022), mu=P.MU["default"])
        I = sc.wp.transform_identity()
        joints = [b.add_joint_free(H, label=f"{name}_free") if free else
                  b.add_joint_fixed(-1, H, parent_xform=sc.wp.transform(tuple(pose[0]), wq(pose[1])), child_xform=I,
                                    label=f"{name}_fixed")]
        jsp = b.add_joint_revolute(H, A, parent_xform=I, child_xform=I, axis=(1., 0., 0.), target_kd=0., target_ke=0.,
                                   effort_limit=P.IMPACT_CAM_NM, actuator_mode=sc.newton.JointTargetMode.VELOCITY,
                                   armature=P.IMPACT_ROTOR_INERTIA, label=f"{name}_anvil")
        joints.append(jsp)
        k_tr = (P.TRIGGER_FULL_N - P.TRIGGER_PRELOAD_N)/P.TRIGGER_TRAVEL_M
        jtr = b.add_joint_prismatic(H, T, parent_xform=rel_xform(sc, pose, Tp), child_xform=I, axis=(-1., 0., 0.),
                                    limit_lower=0., limit_upper=P.TRIGGER_TRAVEL_M, target_pos=-P.TRIGGER_PRELOAD_N/k_tr,
                                    target_ke=k_tr, target_kd=2*0.4*math.sqrt(k_tr*(P.TRIGGER_MASS_KG + P.TRIGGER_ARMATURE_KG)),
                                    actuator_mode=sc.newton.JointTargetMode.POSITION, armature=P.TRIGGER_ARMATURE_KG,
                                    label=f"{name}_trigger")
        joints.append(jtr)
        b.add_articulation(joints, label=f"{name}_assembly")
        self.bodies = [H, A, T]
        self.spring_targets = dict(trigger=-P.TRIGGER_PRELOAD_N/k_tr)
        self.j = dict(anvil=JointRef(jsp, "anvil"), trigger=JointRef(jtr, "trigger"))
        self.spindle = Spindle(self.j["anvil"], [A], (1., 0., 0.), self.bodies)
        # the wrench's program: with the module, its stored first speed for the scanned anchor and the torque threshold
        # at which it stops; without it, the speed of the gear the wrench stands in
        self.target_nm = float(target_nm if target_nm is not None else P.KBTZ2["torque"])
        self.at_module, self.gear = bool(at_module), int(gear)
        self.set_program(self.at_module, self.gear)
        self.s_mode = "idle"
        self.s_light = "off"
        self.s_phase = 0.
        self.s_blows = 0
        self.s_impact_s = 0.
        self.s_last_turns = []
        self.s_last_torques = []
        self.s_held = 0
        self.s_run_s = 0.
        # the blow in progress (dilated): energy left (hammer's kinetic + spring), spring compression, the torque applied
        # in the last tick, the anvil's speed and angle and the hammer's speed at the last tick
        self.s_engaged, self.s_coasting, self.s_coast_ticks = False, False, 0
        self.s_E, self.s_x, self.s_debt = 0., 0., 0.
        self.s_tau, self.s_wa, self.s_wh, self.s_angle = 0., 0., 0., 0.
        self.s_sensed, self.s_turn, self.s_wait, self.s_apart = 0., 0., 0, 0
        self.s_top = (0., 0.)
        self.s_dilated = False
        sc.mutual_filter(self.bodies)
        sc.tool_bodies.update(self.bodies)
        self.visual = []

    def set_program(self, at_module=True, gear=3):
        """The start state of the wrench's electronics: the Adaptive Torque module fitted and set for the scanned anchor
        (every task), or the plain wrench in one of its three gears (ASSISTED start state of a probe: E20)."""
        self.at_module, self.gear = bool(at_module), int(gear)
        if self.at_module:
            self.n_rpm = min(P.IMPACT_MAX_RPM, P.AT_FIRST_RPM*self.target_nm/P.KBTZ2["torque"])
        else:
            self.n_rpm = float(P.IMPACT_GEAR_RPM[self.gear - 1])
        self.blow_hz = P.IMPACT_MAX_IPM/60.*self.n_rpm/P.IMPACT_MAX_RPM
        w_h = self.n_rpm*RPM
        self.blow_energy = 0.5*P.IMPACT_HAMMER_INERTIA*w_h*w_h                              # J per blow
        self.blow_peak_nm = w_h*math.sqrt(P.IMPACT_JAW_STIFFNESS*P.IMPACT_HAMMER_INERTIA)   # against a nut that holds

    def joints(self):
        return list(self.j.values())

    def bind(self, world):
        super().bind(world)
        self.spindle.bind(world)
        mjm = world.solver.mj_model
        mj_of = {n: j for j, n in world._mj_to_newton.items()}
        self._mj_anvil = mj_of[self.anvil]
        # the anvil body's principal axis that lies along the anvil's axis (its x), and its own inertia about it
        q = mjm.body_iquat[self._mj_anvil]
        axes = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        self._axis_k = int(np.argmax(np.abs(axes[0])))
        self._anvil_inertia = float(mjm.body_inertia[self._mj_anvil][self._axis_k])
        self._armature = float(mjm.dof_armature[mjm.jnt_dofadr[self.j["anvil"].mj]])
        # ENGINE: the socket's hex on the nut's at MuJoCo's stiffest contact impedance (params.NUT_HEX_SOLIMP)
        walls = {int(x) for x in self.socket_walls}
        for g in range(mjm.ngeom):
            if g < len(world._geom_shape) and int(world._geom_shape[g]) in walls:
                mjm.geom_solimp[g][:3] = P.NUT_HEX_SOLIMP
        self._dilate(self.s_dilated)

    def _dilate(self, on):
        """The anvil's and socket's inertia about the anvil's axis: s^2 times their own for the duration of a blow."""
        f = P.IMPACT_DILATION**2 if on else 1.
        self.w.solver.mj_model.body_inertia[self._mj_anvil][self._axis_k] = self._anvil_inertia*f
        self.s_dilated = bool(on)

    def restore(self, snap):
        super().restore(snap)
        self._dilate(self.s_dilated)

    def trigger(self, w):
        x = self.j["trigger"].pos(w)
        return float(np.clip((x - P.TRIGGER_DEADBAND_M)/(P.TRIGGER_TRAVEL_M - 2*P.TRIGGER_DEADBAND_M), 0., 1.))

    def _end_impacting(self):
        self.s_engaged = self.s_coasting = False
        self.s_tau = 0.
        if self.s_dilated:
            self._dilate(False)

    def pre_step(self, w):
        s = self.trigger(w)
        w_cmd = self.n_rpm*RPM
        if s < 0.1:
            if self.s_mode != "idle":
                self._end_impacting()
                self.s_mode, self.s_phase = "idle", 0.
        elif self.s_mode == "idle" and s > 0.5:
            self.s_mode, self.s_light, self.s_impact_s, self.s_blows = "run", "off", 0., 0
            self.s_last_turns, self.s_last_torques, self.s_debt = [], [], 0.
            self.s_run_s = 0.
        if self.s_mode == "run":
            # the motor turns the anvil through the engaged hammer; the cam lets go above IMPACT_CAM_NM. The motor
            # comes up to speed over IMPACT_SPINUP_S: started at full speed, the 80 g cm2 anvil and socket reached
            # 77 rad/s within a tick, crossed the hex's clearance and struck the nut's corners 0.6 mm deep in one step
            # (a 147 N kick along the stud in the sound hole, a nut at 5400 rad/s in the ledger case, an anchor thrown
            # 28 mm out of its hole with the socket turned 20 degrees; E20)
            self.s_run_s += w.dt
            w_now = w_cmd*min(1., self.s_run_s/P.IMPACT_SPINUP_S)
            self.spindle.set_drive(w_now, P.IMPACT_CAM_NM/w_cmd, P.IMPACT_CAM_NM)
            # the cam lets go when the load holds the anvil below half speed against the full cam torque for 10 ms (not
            # while the motor merely spins the anvil and a free nut up to speed). Counted with a leak, not as an
            # unbroken run of ticks: on a nut that stands, the light anvil chatters on the hex's hard contact under the
            # 5 N m (every third tick above 2 % of the speed), and an unbroken run never came (E20 sound)
            held = abs(self.spindle.drive_torque()) >= 0.98*P.IMPACT_CAM_NM and abs(self.spindle.speed()) < 0.5*w_now
            self.s_held = self.s_held + 1 if held else max(0, self.s_held - 1)
            if self.s_held >= int(round(0.010/w.dt)):
                self.s_mode, self.s_phase, self.s_held = "impact", 1., 0
                self.s_engaged = self.s_coasting = False
                self.s_wait = self.s_apart = 0
                self.s_tau, self.s_wa, self.s_angle = 0., self.spindle.speed(), self.spindle.angle()
                w.log_event(dict(type="impact_wrench_impacting", tool=self.name, tick=w.tick))
        elif self.s_mode == "impact":
            self._impact(w, w_cmd)
        if self.s_mode in ("done", "stopped", "idle"):
            self.spindle.set_drive(0., 0.5, 0.2)          # motor off: the gear train's drag holds the anvil lightly
        if self.s_mode == "impact":
            # the housing takes the mean torque of re-accelerating the real hammer, not the blows
            p, R = w.body_pose(self.housing)
            w.add_torque(self.housing, -P.IMPACT_HAMMER_INERTIA*w_cmd*self.blow_hz*(R @ np.array([1., 0., 0.])))

    def _impact(self, w, w_cmd):
        """One tick of impacting (the class docstring's declared devices): account for what the last tick's torque did,
        end or begin a blow, and put this tick's spring torque on the anvil."""
        dt, sp = w.dt, self.spindle
        S = P.IMPACT_DILATION
        k, Ih = P.IMPACT_JAW_STIFFNESS, P.IMPACT_HAMMER_INERTIA*S*S
        Ia = self._anvil_inertia*(S*S if self.s_dilated else 1.) + self._armature
        wa, ang = sp.speed(), sp.angle()
        # the torque the socket passed on in the last tick (the module's transducer on the anvil's shaft)
        d = ang - self.s_angle
        sensed = self.s_tau + sp.drive_torque() - Ia*(wa - self.s_wa)/dt
        # the hammer is off the jaws between blows: the motor does not drive the anvil. (With the blow delivered by
        # the hammer the socket's hex stays on the nut's flats without a drag between blows: E20 sound, 0 corners
        # passed, hex at most 3.1 degrees out of line.)
        sp.set_drive(0., 0., 0.)
        self.s_impact_s += dt
        self.s_phase += self.blow_hz*dt
        if self.s_engaged or self.s_coasting:
            # the blow's reading: the largest torque passed on in at least two ticks of the blow (the second largest
            # tick value). A single tick can carry the dilated anvil's own momentum stopped on the nut's flats, 3 N m
            # above what the nut turns at (E20 sound: 68.0 in one tick, then 61.1, 64.4, 64.3 while the
            # nut turned at 64.5)
            a_, b_ = self.s_top
            self.s_top = (sensed, a_) if sensed > a_ else ((a_, sensed) if sensed > b_ else (a_, b_))
            self.s_sensed = self.s_top[1]
            self.s_turn += d
        if self.s_engaged:
            # the work the spring's torque did on the anvil comes out of the blow's energy; the spring's compression
            # follows the hammer's and the anvil's travel (negative: the anvil has run ahead and the jaws are apart)
            self.s_E -= self.s_tau*d
            self.s_x = min(self.s_x + self.s_wh*dt - d, math.sqrt(2.*max(self.s_E, 0.)/k))
        elif self.s_coasting:
            # the hammer has stopped; the anvil's own motion still turns the nut. The blow is read when the anvil rests
            self.s_coast_ticks += 1
            if abs(wa) < 0.02*w_cmd/S or self.s_coast_ticks >= int(round(0.5/(self.blow_hz*dt))):
                self.s_coasting = False
                self._dilate(False)
                self.s_last_torques = (self.s_last_torques + [round(self.s_sensed, 2)])[-8:]
                self.s_last_turns = (self.s_last_turns + [round(math.degrees(self.s_turn), 3)])[-8:]
                if self.at_module and self.s_sensed >= self.target_nm:
                    self.s_mode, self.s_light = "done", "green"
                    self._end_impacting()
                    w.log_event(dict(type="impact_wrench_done", tool=self.name, tick=w.tick, blows=self.s_blows,
                                     sensed_nm=round(self.s_sensed, 2)))
                    return
                if self.s_sensed < P.IMPACT_CAM_NM:
                    # the anvil gave way below the cam's release torque: the hammer stays on the jaws and the motor
                    # drives the anvil again (the nut ran on)
                    self.s_mode, self.s_held, self.s_run_s = "run", 0, 0.
                    self._end_impacting()
                    return
        if self.at_module and self.s_impact_s > P.AT_TIMEOUT_S:
            self.s_mode, self.s_light = "stopped", "red"
            self._end_impacting()
            w.log_event(dict(type="impact_wrench_timeout", tool=self.name, tick=w.tick, blows=self.s_blows))
            return
        if self.s_phase >= 1. and not self.s_engaged and not self.s_coasting:
            # the hammer comes round and strikes: its kinetic energy, the jaws just touching. The anvil's inertia is
            # dilated from here, which gives a moving anvil s^2 times its energy: the blow waits up to 10 ms for the
            # anvil to stand (below half the hammer's dilated speed: at most 5 % of the blow's energy)
            self.s_wait += 1
            if abs(wa) < 0.5*w_cmd/S or self.s_wait > int(round(0.010/dt)):
                self.s_phase = self.s_phase - 1. if self.s_phase < 2. else 0.
                self.s_engaged, self.s_x, self.s_E = True, 0., self.blow_energy - self.s_debt
                self.s_debt, self.s_sensed, self.s_turn, self.s_wait, self.s_apart = 0., 0., 0., 0, 0
                self.s_top = (0., 0.)
                self.s_blows += 1
                self._dilate(True)
        tau, wh = 0., 0.
        if self.s_engaged:
            ek = self.s_E - 0.5*k*max(self.s_x, 0.)**2          # the hammer's kinetic energy
            self.s_apart = self.s_apart + 1 if self.s_x < 0. else 0
            if ek <= 1e-3*self.blow_energy:
                # the hammer has stopped: what the spring holds rebounds; an overdraft is the next blow's debt
                self.s_engaged, self.s_coasting, self.s_coast_ticks = False, True, 0
                self.s_debt = max(0., -ek)
            elif self.s_apart >= int(round(0.010/dt)):
                # the anvil has run ahead of the hammer for 10 ms (nothing holds it): the blow is over, the hammer
                # keeps what it has left
                self.s_engaged, self.s_coasting, self.s_coast_ticks = False, True, 0
            else:
                wh = math.sqrt(2.*ek/Ih)
                x_end = min(self.s_x + (wh - wa)*dt, math.sqrt(2.*self.s_E/k))
                tau = k*max(0., x_end)
                p, R = w.body_pose(self.housing)
                w.add_torque(self.anvil, tau*(R @ np.array([1., 0., 0.])))
        self.s_tau, self.s_wa, self.s_wh, self.s_angle = tau, wa, wh, ang

    def state(self):
        w = self.w
        return dict(trigger_m=round(self.j["trigger"].pos(w), 6), mode=self.s_mode, light=self.s_light,
                    anvil_rpm=round(self.spindle.speed()/RPM, 2), blows=self.s_blows,
                    at_module=self.at_module, at_target_nm=round(self.target_nm, 1), speed_rpm=round(self.n_rpm, 1),
                    blow_energy_j=round(self.blow_energy, 3), blow_peak_nm=round(self.blow_peak_nm, 1),
                    last_blow_sensed_nm=list(self.s_last_torques[-4:]), last_blow_turns_deg=list(self.s_last_turns[-4:]))


# ====================================================================== automatic centre punch
class CentrePunch(Tool):
    """Automatic centre punch (Rennsteig 430 130: 125 mm long, 14 mm shaft, adjustable 60-130 N; params.PUNCH_BODY_M,
    PUNCH_RELEASE_RANGE_N). `release_n`: the cap's setting, a start state (None: PUNCH_RELEASE_N, the middle of the
    range). Frame: x along the punch towards the point. The point is a
    2.1 mm rod with a rounded contact tip (PUNCH_TIP_R_M; drawn as the 90-degree cone, PUNCH_TIP_HALF_ANGLE) sliding in
    its bore with dry friction (PUNCH_POINT_FRICTION_N): stood on it, the punch cannot balance. Cocked, the point slides into the body against the main spring (PUNCH_RELEASE_N at
    PUNCH_RELEASE_TRAVEL_M); there the internal hammer is released and strikes: the energy the spring stored
    (1/2 k x^2 = PUNCH_ENERGY_J) goes into the strike and the dimple (made where the point rests on a drillable face),
    so after the release only the light return spring (PUNCH_RETURN_N) holds the point out. When the point is back out
    (the punch lifted) the trip pin re-seats and the punch is cocked again."""

    # the body behind the point's 13 mm: barrel and cap make up the rest of PUNCH_BODY_M's length (the nose starts 5 mm
    # behind the frame's origin)
    BARREL_R = 0.5*P.PUNCH_BODY_M[1]
    CAP_R, CAP_LEN = 0.5*P.PUNCH_BODY_M[1] + 0.0005, 0.012
    BARREL_LEN = P.PUNCH_BODY_M[0] - 0.013 - 0.005 - CAP_LEN
    BARREL_X = -0.005 - 0.5*BARREL_LEN                     # barrel centre
    CAP_TOP_X = -0.005 - BARREL_LEN - CAP_LEN               # the cap's end face

    def __init__(self, sc, name, pose, release_n=None):
        self.name = name
        self.release_n = float(P.PUNCH_RELEASE_N if release_n is None else release_n)
        b = sc.b
        pr = Parts(sc)
        Bd = link(sc, f"{name}_body", pose)
        # a tube round its spring and hammer, not a solid bar: barrel and cap share the density that brings the whole
        # punch (with its solid point) to the maker's mass (params.PUNCH_MASS_KG)
        r_pt = P.PUNCH_TIP_R_M
        m_point = 7850.*(math.pi*r_pt**2*2*(0.0065 - r_pt) + 4./3.*math.pi*r_pt**3)
        rho_b = (P.PUNCH_MASS_KG - m_point)/(math.pi*(self.BARREL_R**2*self.BARREL_LEN + self.CAP_R**2*self.CAP_LEN))
        barrel = pr.cyl(Bd, f"{name}_barrel", (self.BARREL_X, 0., 0.), self.BARREL_R, 0.5*self.BARREL_LEN, "x",
                        "punch_knurl", density=rho_b, mu=P.MU["steel_steel"], roll=True)
        cap = pr.cyl(Bd, f"{name}_cap", (self.CAP_TOP_X + 0.5*self.CAP_LEN, 0., 0.), self.CAP_R, 0.5*self.CAP_LEN, "x",
                     "steel_dark", density=rho_b, roll=True)
        sc.loose_shapes.update([barrel, cap])
        Pp = compose(pose, (0., 0., 0.))
        Pt = link(sc, f"{name}_point", Pp)
        # the point: a 2.1 mm rod whose contact end is a hemisphere of its own radius (PUNCH_TIP_R_M) with the tip 13 mm
        # ahead of the body (a capsule; drawn as the 90-degree cone). Sharper contact tips failed in this engine: a
        # convex hull of the 90-degree cone and a 0.3 mm tip sphere were pushed sideways (bar thrown, no dimple) once
        # the soft contact let them sink under the 100 N push. A centre of mass ~70 mm above a 1.05 mm
        # rounded tip cannot balance on it, as the old flat 2.1 mm disc did
        r_pt = P.PUNCH_TIP_R_M
        pr.capsule(Pt, f"{name}_point_rod", (0.0065, 0., 0.), r_pt, 0.0065 - r_pt, "x", "steel_bright", density=7850.)
        I = sc.wp.transform_identity()
        k = self.release_n/P.PUNCH_RELEASE_TRAVEL_M
        j0 = b.add_joint_free(Bd, label=f"{name}_free")
        jp = b.add_joint_prismatic(Bd, Pt, parent_xform=rel_xform(sc, pose, Pp), child_xform=I, axis=(-1., 0., 0.),
                                   limit_lower=0., limit_upper=P.PUNCH_RELEASE_TRAVEL_M + 0.001, target_pos=0.,
                                   target_ke=k, target_kd=self.kd(k),
                                   armature=P.PUNCH_POINT_ARMATURE, actuator_mode=sc.newton.JointTargetMode.POSITION,
                                   friction=P.PUNCH_POINT_FRICTION_N, label=f"{name}_point")
        b.add_articulation([j0, jp], label=f"{name}_assembly")
        self.bodies = [Bd, Pt]
        self.body, self.point = Bd, Pt
        self.k = k
        # return spring after the release: half its force preloaded, PUNCH_RETURN_N at the end of the point's travel
        self.k_ret = 0.5*P.PUNCH_RETURN_N/(P.PUNCH_RELEASE_TRAVEL_M + 0.001)
        self.x_ret = -0.5*P.PUNCH_RETURN_N/self.k_ret
        self.j = dict(point=JointRef(jp, "point"))
        from .holes import Item
        self.item = Item(f"{name}_point", "punch", Pt, axis=(1., 0., 0.), tip=(0.013, 0., 0.), radius=0.00105,
                         length=0.013)
        self.s_cocked = True
        self.s_strikes = []
        sc.mutual_filter(self.bodies)
        sc.tool_bodies.update(self.bodies)

    BODY_KG = 7850.*math.pi*(BARREL_R**2*BARREL_LEN + CAP_R**2*CAP_LEN)      # barrel + cap, as built above

    @classmethod
    def kd(cls, k):
        """Damping of the point's spring: PUNCH_POINT_DAMPING of critical for the punch body bouncing on it."""
        return 2*P.PUNCH_POINT_DAMPING*math.sqrt(k*(cls.BODY_KG + P.PUNCH_POINT_ARMATURE))

    def joints(self):
        return list(self.j.values())

    def _spring(self, w, k, x0):
        """The point's spring (MuJoCo position actuator): stiffness k, rest position x0 (negative: preloaded)."""
        mjm = w.solver.mj_model
        a = self.j["point"].act
        mjm.actuator_gainprm[a][0] = k
        mjm.actuator_biasprm[a][1] = -k
        mjm.actuator_biasprm[a][2] = -self.kd(k)
        w.tq[self.j["point"].q] = x0

    def point_force(self, w):
        """Spring force on the point (N, positive pushing it out)."""
        x = self.j["point"].pos(w)
        return self.k*x if self.s_cocked else self.k_ret*(x - self.x_ret)

    def pre_step(self, w):
        x = self.j["point"].pos(w)
        if self.s_cocked and self.k*x >= self.release_n:
            self.s_cocked = False
            self._strike(w, self.k*x)
        elif not self.s_cocked and x <= P.PUNCH_RECOCK_M:
            self.s_cocked = True
        if self.s_cocked:
            self._spring(w, self.k, 0.)
        else:
            self._spring(w, self.k_ret, self.x_ret)

    def _struck_mass(self, w, d, tip_w, ax_w):
        """Effective mass (kg) of the struck part at the point along the strike: infinite for a fixed part or one
        backed under the point (the point inside the patch of its support contacts on the far side, normals along the
        strike: a bar on the bench); otherwise the free part's 1/(1/m + (r x n).I^-1.(r x n)) at the point."""
        import mujoco
        if d.static:
            return math.inf, "fixed"
        m, dat = w.solver.mj_model, w.solver.mj_data
        own = {d.body, getattr(d, "extra_body", None), d.proxy_body}
        mine = set(self.bodies)
        e1 = np.cross(ax_w, [0., 0., 1.] if abs(ax_w[2]) < 0.9 else [1., 0., 0.])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax_w, e1)
        pts, f6 = [], np.zeros(6)
        for i in range(dat.ncon):
            c = dat.contact[i]
            b = [w._mj_to_newton.get(int(m.geom_bodyid[g]), -1) for g in c.geom]
            if not ((b[0] in own) ^ (b[1] in own)) or b[0] in mine or b[1] in mine:
                continue
            n = np.asarray(c.frame[:3], float)
            mujoco.mj_contactForce(m, dat, i, f6)
            if f6[0] <= 0. or abs(float(n @ ax_w)) < 0.7 or float((np.asarray(c.pos) - tip_w) @ ax_w) <= 0.:
                continue
            q = np.asarray(c.pos, float) - tip_w
            pts.append((float(q @ e1), float(q @ e2)))
        if len(pts) >= 3:
            from scipy.spatial import Delaunay
            try:
                if Delaunay(np.asarray(pts)).find_simplex(np.zeros(2), tol=1e-3) >= 0:
                    return math.inf, "backed"
            except Exception:
                pass
        if pts and min(math.hypot(*q) for q in pts) < 0.002:
            return math.inf, "backed"
        mass = float(w.model.body_mass.numpy()[d.body])
        q = w.bq[d.body]
        R = Rotation.from_quat(q[3:]).as_matrix()
        com = q[:3] + R @ w.body_com[d.body]
        I_w = R @ np.asarray(w.model.body_inertia.numpy()[d.body], float) @ R.T
        rn = np.cross(tip_w - com, ax_w)
        return 1./(1./mass + float(rn @ np.linalg.solve(I_w, rn))), "free"

    def _strike(self, w, F):
        """The internal hammer, driven by the main spring's stored energy E = F^2/(2k) (PUNCH_ENERGY_J at the release
        force), strikes the point and through it the part: a plastic impact of the hammer (PUNCH_POINT_ARMATURE) on
        the part's effective mass M at the point leaves E M/(M + m_h) in the dimple (all of it on a fixed or backed
        part, less on a free light or overhanging one); the dimple depth follows from the cone indentation law. The
        strike's momentum J = m_h v M/(M + m_h) goes into the part at the point and the spring's reaction J back into
        the punch body (an equal and opposite pair)."""
        hm = w.hm
        it = self.item
        E_s = F*F/(2*self.k)
        m_h = P.PUNCH_POINT_ARMATURE
        tip_w, ax_w = hm.item_world(it)
        for dn, (k, mode, hid) in it.slots.items():
            d = hm.drillables[dn]
            if mode not in ("pad", "hole", "mouth"):
                continue
            tip_l, ax_l = hm.local(d, tip_w, ax_w)
            n = None
            f = hm.face_under(d, tip_l, ax_l)
            if f is not None:
                n = f[0]
            elif mode in ("hole", "mouth"):
                h = next(x for x in d.holes if x.id == hid)
                n = h.face_normal
            if n is None:
                continue
            Hn = P.HARDNESS_PA.get(d.material, 0.)
            if Hn <= 0:
                continue
            M, support = self._struck_mass(w, d, np.asarray(tip_w, float), np.asarray(ax_w, float))
            share = 1. if math.isinf(M) else M/(M + m_h)
            E_d = E_s*share
            # rigid-plastic cone indentation: E = (pi/3) tan^2(a) H h^3
            ta = math.tan(P.PUNCH_TIP_HALF_ANGLE)
            depth = (3*E_d/(math.pi*ta*ta*Hn))**(1/3)
            h = hm.make_dimple(d, tip_l, n, depth)
            J = m_h*math.sqrt(2*E_s/m_h)*share
            if not d.static:
                hm._force(d.body, np.asarray(ax_w, float)*J/w.dt, tip_w)
            hm._force(self.body, -np.asarray(ax_w, float)*J/w.dt, tip_w)
            self.s_strikes.append(dict(tick=w.tick, workpiece=dn, hole=h.id, depth_mm=round(depth*1e3, 4),
                                       release_force_n=round(F, 1), strike_energy_j=round(E_s, 4),
                                       dimple_energy_j=round(E_d, 4), support=support,
                                       struck_mass_kg=None if math.isinf(M) else round(M, 4)))
            w.log_event(dict(type="punch_strike", tool=self.name, workpiece=dn, hole=h.id, tick=w.tick))
            return
        self.s_strikes.append(dict(tick=w.tick, workpiece=None, release_force_n=round(F, 1),
                                   strike_energy_j=round(E_s, 4)))

    def state(self):
        return dict(point_travel_m=round(self.j["point"].pos(self.w), 6), cocked=self.s_cocked,
                    point_force_n=round(self.point_force(self.w), 3), strikes=list(self.s_strikes))


# ====================================================================== ASSISTED probe rig
class PhantomSpindle(Tool):
    """ASSISTED - a drilling dynamometer: a bit on a carriage sliding along `axis` (world) and turning at a set speed.
    feed mode 'force': a set axial force on the carriage; 'rate': the carriage driven at a set speed (velocity drive).
    Used only by probes; never built into a task scene."""
    assisted = True

    def __init__(self, sc, name, tip_start_w, axis_w, bit_d_mm, bit_len_m=0.06, carriage_kg=2.0, bit_type="hss",
                 lateral_free=False):
        self.name = name
        b = sc.b
        pr = Parts(sc)
        a = np.asarray(axis_w, float)/np.linalg.norm(axis_w)
        R = _rot_z_to(-a)                                 # local -z = feed direction... use local x = a
        Rx = Rotation.align_vectors([a], [[1., 0., 0.]])[0]
        root = np.asarray(tip_start_w, float) - a*(bit_len_m + 0.05)
        Cp = (root, Rx)
        Cr = link(sc, f"{name}_carriage", Cp, mass=carriage_kg, inertia=(0.01, 0.01, 0.01))
        Sp = (root, Rx)
        S = link(sc, f"{name}_spindle", Sp)
        pr.cyl(S, f"{name}_collet", (0.025, 0., 0.), 0.015, 0.025, "x", "chuck_black", density=4000.)
        Bp = compose(Sp, (0.05, 0., 0.))
        Bk = link(sc, f"{name}_bit", Bp)
        pr.bit(Bk, f"{name}_bit", (0., 0., 0.), (1., 0., 0.), bit_d_mm, bit_len_m, "hss_bit",
               density=7850., mu=P.MU["bit_wall"])
        I = sc.wp.transform_identity()
        joints = []
        if lateral_free:
            # compliant holder: two lateral slides with springs (E5: 24 N/mm lateral)
            L1 = link(sc, f"{name}_lat1", Cp, mass=0.05, inertia=(1e-5, 1e-5, 1e-5))
            L2 = link(sc, f"{name}_lat2", Cp, mass=0.05, inertia=(1e-5, 1e-5, 1e-5))
            joints.append(b.add_joint_prismatic(-1, L1, parent_xform=sc.wp.transform(tuple(root), wq(Rx)), child_xform=I,
                                                axis=(0., 1., 0.), target_ke=P.PHANTOM_LATERAL_K, target_kd=200.,
                                                actuator_mode=sc.newton.JointTargetMode.POSITION, label=f"{name}_lat1"))
            joints.append(b.add_joint_prismatic(L1, L2, parent_xform=I, child_xform=I, axis=(0., 0., 1.),
                                                target_ke=P.PHANTOM_LATERAL_K, target_kd=200.,
                                                actuator_mode=sc.newton.JointTargetMode.POSITION, label=f"{name}_lat2"))
            jf = b.add_joint_prismatic(L2, Cr, parent_xform=I, child_xform=I, axis=(1., 0., 0.), target_kd=0.,
                                       actuator_mode=sc.newton.JointTargetMode.VELOCITY, effort_limit=1e5,
                                       label=f"{name}_feed")
        else:
            jf = b.add_joint_prismatic(-1, Cr, parent_xform=sc.wp.transform(tuple(root), wq(Rx)), child_xform=I,
                                       axis=(1., 0., 0.), target_kd=0., actuator_mode=sc.newton.JointTargetMode.VELOCITY,
                                       effort_limit=1e5, label=f"{name}_feed")
        joints.append(jf)
        js = b.add_joint_revolute(Cr, S, parent_xform=I, child_xform=I, axis=(1., 0., 0.), target_kd=0.,
                                  actuator_mode=sc.newton.JointTargetMode.VELOCITY, effort_limit=1e4, armature=1e-3,
                                  label=f"{name}_spin")
        joints.append(js)
        joints.append(b.add_joint_fixed(S, Bk, parent_xform=rel_xform(sc, Sp, Bp), child_xform=I))
        b.add_articulation(joints, label=f"{name}_rig")
        self.bodies = [Cr, S, Bk] + ([L1, L2] if lateral_free else [])
        self.j = dict(feed=JointRef(jf, "feed"), spin=JointRef(js, "spin"))
        self.spindle = Spindle(self.j["spin"], [S, Bk], (1., 0., 0.), [Cr, S, Bk])
        from .holes import Item
        self.item = Item(f"{name}_bit", "bit", Bk, axis=(1., 0., 0.), tip=(bit_len_m, 0., 0.),
                         radius=0.5e-3*bit_d_mm, length=bit_len_m, d_mm=bit_d_mm, bit_type=bit_type,
                         spindle=self.spindle, stickout=bit_len_m)
        self.feed_mode, self.feed_value, self.rpm = "force", 0., 0.
        sc.mutual_filter(self.bodies)
        sc.tool_bodies.update(self.bodies)

    def joints(self):
        return list(self.j.values())

    def bind(self, world):
        super().bind(world)
        self.spindle.bind(world)

    def command(self, rpm, feed_mode, value):
        """ASSISTED: spin at `rpm`; feed_mode 'force' (value N along the axis) or 'rate' (value m/s)."""
        self.rpm, self.feed_mode, self.feed_value = float(rpm), feed_mode, float(value)

    def pre_step(self, w):
        # an ideal speed-controlled spindle; torque_cap (N m) makes it a motor of limited torque (E3b: stall)
        self.spindle.set_drive(self.rpm*RPM, 50., getattr(self, "torque_cap", None) or 1e4)
        mjm = w.solver.mj_model
        jf = self.j["feed"]
        if self.feed_mode == "force":
            mjm.actuator_gainprm[jf.act][0] = 0.
            mjm.actuator_biasprm[jf.act][2] = 0.
            w.tqd[jf.dof] = 0.
            w.jf[jf.dof] += self.feed_value
        else:
            mjm.actuator_gainprm[jf.act][0] = 2e5
            mjm.actuator_biasprm[jf.act][2] = -2e5
            w.tqd[jf.dof] = self.feed_value

    def state(self):
        return dict(ASSISTED=True, rpm=self.rpm, feed_mode=self.feed_mode, feed_value=self.feed_value,
                    feed_pos_m=round(self.j["feed"].pos(self.w), 6), spin_rad_s=round(self.spindle.speed(), 3))
