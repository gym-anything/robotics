"""The handled equipment: syringes, needles, vial, intravenous lock, injection pad, and the liquid that joins them.

Each device builds its rigid bodies in the scene (Newton builder), binds to the compiled MuJoCo model, and takes part
in every tick through the `Fluidics` manager:

    pre_step   geometry (what is connected to what, which phase is at each outlet, where each needle tip is), the
               implicit one-tick coefficients of each chamber, and the forces, friction limits and armatures they give
    (the rigid step)
    post_step  the flows that go with the speeds the solver returned; parcels moved; drops released; records kept

Fits are built one way: a light carrier on a joint of the part it belongs to, and an equality that is
switched on only when a geometric condition the robot has to produce is met. Nothing here sets a pose of a held
object or moves liquid by rule.

Frames. Syringe: +x from the nozzle toward the thumb rest, origin on the axis at the chamber's front wall. Needle
(hub frame): the same sense, origin at the hub's mouth plane; the tube points along -x. A female Luer mouth frame has
its origin on the mouth plane and +x pointing out of the bore, so a mated male's +x and the female's +x coincide.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import parts as T
from . import hydraulics as H

X = np.array([1.0, 0.0, 0.0])
I3 = Rotation.identity()


def _twist_about_x(R):
    """Angle of the rotation about x nearest to R (swing-twist)."""
    q = R.as_quat()
    return 2.0*math.atan2(q[0], q[3])


def set_body_mass(mjm, b, mass):
    """Change a body's mass at run time and keep the model's subtree masses with it. MuJoCo's centre-of-mass pass
    divides by body_subtreemass, which is only written at compile time and by mj_setConst: with 2 g of liquid lumped
    into a 2 g barrel and the old subtree mass, subtree_com for that barrel was 0.6 m from the barrel (found when a
    phantom hand that carried the weight 'at the subtree's centre' spun a filled syringe up)."""
    dm = float(mass) - float(mjm.body_mass[b])
    mjm.body_mass[b] = mass
    k = int(b)
    while True:
        mjm.body_subtreemass[k] += dm
        if k == 0:
            break
        k = int(mjm.body_parentid[k])


class Female:
    """A female Luer fitting: where its mouth is on its body, and what lies behind it."""

    def __init__(self, owner, name, body, mouth_p, mouth_R=None, spring=None):
        self.owner, self.name, self.body = owner, name, int(body)
        self.p, self.R = np.asarray(mouth_p, float), (mouth_R or I3)
        self.spring = spring                    # callable(depth) -> force resisting insertion (N), or None
        self.mate = None

    def mouth_world(self, w):
        if self.body < 0:
            return self.p.copy(), self.R
        p, R = T.world_pose(w, self.body)
        return p + R.apply(self.p), R*self.R


# ====================================================================== syringe
class Syringe:
    def __init__(self, sc, name, kind, pose, liquid=None, liquid_ml=0.0, gas_ml=None, plunger_ml=None, mount="free"):
        self.sc, self.name, self.kind = sc, name, kind
        self.s = s = P.SYRINGES[kind]
        self.chamber = ch = H.Chamber(s)
        b, wp = sc.b, sc.wp
        r = 0.5*s["od"]
        L = s["barrel_len"]
        fl = s["flange"]
        lu = P.LUER
        # contents: liquid column, and the gas that fills the rest of what the plunger leaves (a dry syringe holds
        # its dead space of air)
        liq = H.parcel(liquid_ml*H.ML, liquid) if liquid and liquid_ml > 0 else np.zeros(H.N_SPECIES)
        total_ml = plunger_ml if plunger_ml is not None else liquid_ml + (gas_ml or 0.0)
        x0 = max((total_ml*H.ML - ch.dead)/ch.A, 0.0)
        gas = max(ch.volume(x0) - float(liq.sum()), 0.0) if gas_ml is None else gas_ml*H.ML
        if x0 > s["stroke_max"] - 0.0005:
            raise ValueError(f"{name}: {total_ml:g} mL does not fit a {s['label']} syringe (the stopper would stand at "
                             f"{1e3*x0:.1f} mm of {1e3*s['stroke_max']:.1f})")
        ch.fill(liq, gas/H.ML)
        ch.gas_pos = 1.0
        self.x0 = x0
        self.first_break = float(s.get("seal_first", 0.0))
        self.pose0 = pose
        I = wp.transform_identity()
        # ---- barrel: tube, flange, collar and nozzle (all one body)
        self.barrel = B = T.link(sc, f"{name}_barrel", pose)
        rho_b = s["mass_barrel"]/(math.pi*(r*r - (0.5*s["bore"])**2)*L + fl[0]*fl[1]*fl[2])
        # The tube's collider is a capsule (its rounded ends lie inside the shoulder and under the flange). Against a
        # flat fingertip pad MuJoCo's capsule-box routine always gives the line of contact as its two ends; the convex
        # routine a cylinder goes through gives one point at the deeper end as soon as the barrel lies 0.2 degrees off
        # the pad's face, and a held syringe hums between the one-point and two-point states.
        v_cyl, v_cap = math.pi*r*r*(L + 0.001), math.pi*r*r*(L - 2*r) + 4.0/3.0*math.pi*r**3
        self.shape_barrel = T.capsule(sc, B, f"{name}_barrel_tube", (r, 0., 0.), (L - r, 0., 0.), r, "syringe_barrel",
                                      density=rho_b*(1 - (0.5*s["bore"]/r)**2)*v_cyl/v_cap, mu=P.MU["pp"])
        self.shape_flange = sc.box(f"{name}_flange", (L + 0.5*fl[2], 0., 0.), (0.5*fl[2], 0.5*fl[0], 0.5*fl[1]), "syringe_barrel",
                                   body=B, density=rho_b, mu=P.MU["pp"])
        col_r = min(0.5*lu["collar_od"], r)
        self.shape_collar = T.cylinder(sc, B, f"{name}_collar", (-0.5*(lu["nozzle_len"] - lu["nozzle_proj"]), 0., 0.), col_r,
                   0.5*(lu["nozzle_len"] - lu["nozzle_proj"]), "x", "syringe_barrel", density=300., mu=P.MU["pp"])
        self.shape_tip = T.cylinder(sc, B, f"{name}_nozzle_tip", (-lu["nozzle_len"] + 0.5*lu["nozzle_proj"], 0., 0.),
                                    0.5*lu["tip_d"], 0.5*lu["nozzle_proj"], "x", "syringe_barrel", density=900., mu=P.MU["pp"])
        # ---- plunger: rod, thumb rest (an octagon of two boxes: face contacts for a fingertip), stopper inside
        self.rod_len = L + fl[2] + s["proj"]            # stopper front to the thumb rest's underside
        pp = T.compose(pose, (x0, 0., 0.))
        self.plunger = Pl = T.link(sc, f"{name}_plunger", pp)
        rw = 0.5*s["rod_w"]
        vol_rod = (2*rw)**2*self.rod_len
        tr, tt = 0.5*s["thumb_d"], s["thumb_t"]
        rho_p = s["mass_plunger"]/(vol_rod + math.pi*tr*tr*tt)
        # the rod's collision box is slimmer than the rod (its mass is the rod's): inside the barrel nothing reaches it.
        # At the rod's own width its corners stood 0.5 mm under the barrel's surface, a gripping pad sank that far, held
        # the plunger through the wall and drew it out 1 mm when the barrel slid in the hand (first complete rollout).
        rc = min(rw, 0.42*0.5*s["bore"])
        self.shape_rod = sc.box(f"{name}_rod", (0.5*self.rod_len, 0., 0.), (0.5*self.rod_len, rc, rc), "syringe_rod", body=Pl,
                                density=rho_p*(rw/rc)**2, mu=P.MU["pp"])
        a8 = tr*math.cos(math.pi/8)
        self.shape_thumb = []
        for k in range(2):
            self.shape_thumb.append(sc.box(f"{name}_thumb{k}", (self.rod_len + 0.5*tt, 0., 0.), (0.5*tt, a8, a8*math.tan(math.pi/8)),
                                           "syringe_rod", body=Pl, rot=T.wq(Rotation.from_euler("x", 45.*k, degrees=True)),
                                           density=0.5*rho_p*math.pi*tr*tr/(4*a8*a8*math.tan(math.pi/8)), mu=P.MU["pp"]))
        sc.box(f"{name}_thumb_fill", (self.rod_len + 0.5*tt, 0., 0.), (0.5*tt, a8, a8*math.tan(math.pi/8)), "syringe_rod", body=Pl,
               rot=T.wq(Rotation.from_euler("x", 90., degrees=True)), density=0., mu=P.MU["pp"])
        T.cylinder(sc, Pl, f"{name}_stopper", (0.5*s["stopper_len"], 0., 0.), 0.5*s["bore"], 0.5*s["stopper_len"], "x",
                   "stopper_rubber", collide=False)
        # ---- the nozzle's carrier: a slide along the axis (depth of a mated fitting) and a turn about it
        tip = T.compose(pose, (-lu["nozzle_len"], 0., 0.))
        self.c1 = C1 = T.link(sc, f"{name}_luer_slide", tip, mass=P.CARRIER_MASS, inertia=(1e-9, 1e-9, 1e-9))
        self.c2 = C2 = T.link(sc, f"{name}_luer_turn", tip, mass=P.CARRIER_MASS, inertia=(1e-9, 1e-9, 1e-9))
        self.j_mount = None
        if mount == "fixed":        # a bench clamp (probes only): the barrel held in a test stand
            j_free = b.add_joint_fixed(-1, B, parent_xform=wp.transform(tuple(map(float, pose[0])), T.wq(pose[1])), child_xform=I,
                                       label=f"{name}_clamp")
        elif mount == "slide":      # a testing machine's carriage (probes only): the barrel on a driven slide along its axis
            nt = sc.newton
            j_free = b.add_joint_prismatic(-1, B, parent_xform=wp.transform(tuple(map(float, pose[0])), T.wq(pose[1])), child_xform=I,
                                           axis=(1., 0., 0.), limit_lower=-0.3, limit_upper=0.3, target_pos=0.0, target_ke=2.0e5,
                                           target_kd=2.0e3, armature=1.0, actuator_mode=nt.JointTargetMode.POSITION,
                                           label=f"{name}_carriage")
            self.j_mount = T.JointRef(j_free, "carriage")
        else:
            j_free = b.add_joint_free(B, label=f"{name}_free")
        j_pl = b.add_joint_prismatic(B, Pl, parent_xform=I, child_xform=I, axis=(1., 0., 0.), limit_lower=0.0,
                                     limit_upper=s["stroke_max"], friction=s["seal_glide"], armature=P.PLUNGER_BASE_ARMATURE,
                                     label=f"{name}_plunger_slide")
        T_tip = wp.transform((-lu["nozzle_len"], 0., 0.), wp.quat_identity())
        j_ls = b.add_joint_prismatic(B, C1, parent_xform=T_tip, child_xform=I, axis=(1., 0., 0.), limit_lower=-0.001,
                                     limit_upper=lu["d_seat"] + 0.0021, friction=P.LUER_FRICTION["slide_n"],
                                     armature=P.CARRIER_ARMATURE[0], damping=P.LUER_DAMPING[0], label=f"{name}_luer_depth")
        j_lt = b.add_joint_revolute(C1, C2, parent_xform=I, child_xform=I, axis=(1., 0., 0.), limit_lower=-1e4, limit_upper=1e4,
                                    friction=P.LUER_FRICTION["tau0"], armature=P.CARRIER_ARMATURE[1], damping=P.LUER_DAMPING[1],
                                    label=f"{name}_luer_turn")
        b.add_articulation([j_free, j_pl, j_ls, j_lt], label=f"{name}_assembly")
        T_set_q(b, j_pl, x0)
        self.j = dict(plunger=T.JointRef(j_pl, "plunger"), depth=T.JointRef(j_ls, "depth"), turn=T.JointRef(j_lt, "turn"))
        self.j_free = j_free
        from newton._src.solvers.mujoco.equality import _add_equality_constraint
        stiff = {"mujoco:eq_solref": P.FIT_SOLREF, "mujoco:eq_solimp": P.FIT_SOLIMP}
        self.eq_hydraulic = _add_equality_constraint(b, constraint_type=sc.newton.solvers.SolverMuJoCo.EqType.JOINT,
                                                     joint1=j_pl, joint2=j_ls, polycoef=[0., 0., 0., 0., 0.], enabled=True,
                                                     label=f"{name}_implicit_fluid", custom_attributes=stiff)
        self.eq_thread = _add_equality_constraint(b, constraint_type=sc.newton.solvers.SolverMuJoCo.EqType.JOINT, joint1=j_ls,
                                                  joint2=j_lt, polycoef=[0., lu["lead"]/(2*math.pi), 0., 0., 0.], enabled=False,
                                                  label=f"{name}_thread", custom_attributes=stiff)
        self._add_eq = lambda body2, label: _add_equality_constraint(
            b, constraint_type=sc.newton.solvers.SolverMuJoCo.EqType.WELD, body1=C2, body2=body2, enabled=False,
            label=label, custom_attributes=stiff)
        self.eq_weld = {}                               # female name -> equality index
        sc.mutual_filter([B, Pl, C1, C2])
        sc.bodies[name] = B
        sc.places[B] = sc.places[Pl] = name
        self.state = "free"
        self.mate = None
        self.start_mate = None
        self.flags = dict(extracted=False, overridden=False)
        self.touch = {}                                 # what the bare nozzle tip has touched: place -> first time
        self.rest_s = 1e9                               # time the stopper has stood still
        self.moving = False
        self.path = None
        self.atmos = None
        self._depth_now = 0.0
        self._turn_ref = 0.0
        self._arm = None
        self._g_lp = None
        self._v_prev = None
        self.last = dict(Q=0.0, c_h=0.0, F_bias=0.0, seal=0.0)

    # ---- construction-time links
    def declare_mates(self, females):
        for f in females:
            self.eq_weld[f.name] = self._add_eq(f.body, f"{self.name}_on_{f.name}")

    def start_on(self, female, depth=None):
        """Built already screwed onto this fitting (a needle as packed and fitted, a tip cap), made with 27.5 N."""
        self.start_mate = female
        self.start_depth = P.LUER["d_seat"] + 0.00055 if depth is None else depth
        T_set_q(self.sc.b, self.j["depth"].index, self.start_depth)

    def mated_pose(self, depth=None, turn=0.0):
        """World pose (p, R) a female body must have to sit on this syringe's nozzle at `depth`, given its mouth."""
        lu = P.LUER
        d = lu["d_seat"] + 0.00055 if depth is None else depth
        p, R = self.pose0
        return T.compose((p, R), (-lu["nozzle_len"] + d, 0., 0.), Rotation.from_euler("x", turn))

    # ---- binding to the compiled model
    def bind(self, w):
        mjm = w.solver.mj_model
        for jr in self.j.values():
            jr.bind(w)
        if self.j_mount is not None:
            self.j_mount.bind(w)
        self.d_pl = int(mjm.jnt_dofadr[self.j["plunger"].mj])
        self.d_ls = int(mjm.jnt_dofadr[self.j["depth"].mj])
        self.d_lt = int(mjm.jnt_dofadr[self.j["turn"].mj])
        self.a_pl = int(mjm.jnt_qposadr[self.j["plunger"].mj])
        self.a_ls = int(mjm.jnt_qposadr[self.j["depth"].mj])
        self.a_lt = int(mjm.jnt_qposadr[self.j["turn"].mj])
        for dof in (self.d_pl, self.d_ls, self.d_lt):
            mjm.dof_solref[dof] = P.FIT_SOLREF
            mjm.dof_solimp[dof] = P.FIT_SOLIMP
        # A joint equality's regularisation is scaled by the plain sum of its two dofs' invweight0, whatever the
        # gearing between them: with the turn's 5e6 /(kg m2) beside the slide's 20 /kg the thread holds 1 N of the seat's
        # 27.5 N and runs back 0.5 mm per tick. The turn's invweight0 is therefore kept in
        # the slide's units, (lead / 2 pi)^2 / inertia, which also hardens the turn's own friction row.
        c1 = P.LUER["lead"]/(2*math.pi)
        w.keep_invweight(self.d_ls, 1.0/(P.CARRIER_MASS + P.CARRIER_ARMATURE[0]))
        w.keep_invweight(self.d_lt, c1*c1/(1e-9 + P.CARRIER_ARMATURE[1]))
        self._q0 = (float(mjm.qpos0[self.a_ls]), float(mjm.qpos0[self.a_lt]))
        self.mj_barrel = w._newton_to_mj[self.barrel]
        self.mj_plunger = w._newton_to_mj[self.plunger]
        self.mass0 = float(mjm.body_mass[self.mj_barrel])
        self.ipos0 = mjm.body_ipos[self.mj_barrel].copy()
        self.inertia0 = mjm.body_inertia[self.mj_barrel].copy()
        self.m_plunger = float(mjm.body_mass[self.mj_plunger])
        self.g_tip = w.geom_of_shape(self.shape_tip)
        self.g_nozzle = [self.g_tip, w.geom_of_shape(self.shape_collar)]
        self._nozzle_masks = [(int(mjm.geom_contype[g]), int(mjm.geom_conaffinity[g])) for g in self.g_nozzle]
        spray = w.fluid.spray
        self.atmos = H.Atmosphere(spray, lambda: self._tip_origin(w), P.LUER["bore_d"], body=self.mj_barrel)
        self.atmos.outer = P.LUER["tip_d"]
        self.chamber.settle_pressure(self.x0)
        self._armature_set(w, P.PLUNGER_BASE_ARMATURE)
        self.lump(w)
        if self.start_mate is not None:
            f = self.start_mate
            d0 = self.start_depth
            d = w.solver.mj_data
            d.qpos[self.a_ls] = d0
            w.eq_row(self.eq_thread)[0] = self._thread_c0(d0, 0.0)
            w.eq_enable(self.eq_thread, True)
            self._weld_to(w, f, 0.0)
            self._nozzle_collide(w, False)
            self.state, self.mate, f.mate = "threaded", f, self

    def _thread_c0(self, depth, turn):
        """Offset of the thread's equality so that it holds at this depth and turn. MuJoCo's joint equality is written
        about the model's reference positions: (depth - depth0) = c0 + c1 (turn - turn0)."""
        return (depth - self._q0[0]) - (P.LUER["lead"]/(2*math.pi))*(turn - self._q0[1])

    def _armature_set(self, w, value):
        w.set_armature(self.d_pl, value, 1.0/(self.m_plunger + value))
        self._arm = value

    def _tip_origin(self, w):
        p, R = T.world_pose(w, self.barrel)
        a = -R.apply(X)
        tip = p + R.apply((-P.LUER["nozzle_len"], 0., 0.))
        return tip, a, w.point_velocity(self.barrel, tip), w.dt       # the nozzle's own velocity, not the barrel's centre's

    def tip_world(self, w):
        p, R = T.world_pose(w, self.barrel)
        return p + R.apply((-P.LUER["nozzle_len"], 0., 0.)), R

    def _nozzle_collide(self, w, on):
        """The nozzle's tip and collar are colliders while the nozzle is bare; inside a fitting the cone and the thread
        are the fit's joint, not contacts."""
        mjm = w.solver.mj_model
        for g, (ct, ca) in zip(self.g_nozzle, self._nozzle_masks):
            mjm.geom_contype[g], mjm.geom_conaffinity[g] = (ct, ca) if on else (0, 0)

    def _weld_to(self, w, f, twist, soft=0):
        """Tie the carrier to the fitting with their axes together, at the roll they have."""
        mjm = w.solver.mj_model
        e = self.eq_weld[f.name]
        q = (Rotation.from_euler("x", -twist)*f.R.inv()).as_quat()
        w.eq_row(e)[:11] = [*f.p, 0., 0., 0., q[3], q[0], q[1], q[2], 1.0]
        w.eq_enable(e, True, soft_ticks=soft)

    def mouth_clearance(self, w, f):
        """Actual tip-in-mouth geometry. No lead-in attraction across an air gap.

        6% diameter taper consumes 0.03 mm radial clearance per mm of depth;
        tilt consumes depth*sin(angle). This is an unchamfered circular mouth.
        """
        tip, R = self.tip_world(w); pf, Rf = f.mouth_world(w)
        axis = Rf.apply(X); rel = tip-pf; depth = -float(rel@axis)
        lat = float(np.linalg.norm(rel+depth*axis))
        cos = float(np.clip(R.apply(X)@axis, -1., 1.))
        clearance = .5*(P.LUER['mouth_d']-P.LUER['tip_d'])-.03*max(depth,0.)
        clear = lat+max(depth,0.)*math.sqrt(max(0.,1-cos*cos)) <= clearance
        return depth, lat, cos, clear

    # ---- the Luer joint
    def _luer(self, w):
        lu, cap, fr = P.LUER, P.LUER_CAPTURE, P.LUER_FRICTION
        mjm, d = w.solver.mj_model, w.solver.mj_data
        q_d, q_t = float(d.qpos[self.a_ls]), float(d.qpos[self.a_lt])
        if self.state == "free":
            tip, R = self.tip_world(w)
            xs = R.apply(X)
            for name in self.eq_weld:
                f = w.fluid.females[name]
                if f.mate is not None:
                    continue
                pf, Rf = f.mouth_world(w)
                af = Rf.apply(X)
                rel = tip - pf
                depth = -float(rel @ af)
                lat = float(np.linalg.norm(rel + depth*af))
                if -cap["lead_in"] <= depth <= cap["depth"] and self.mouth_clearance(w, f)[3] and \
                        float(xs @ af) >= math.cos(math.radians(cap["angle_deg"])):
                    d.qpos[self.a_ls], d.qvel[self.d_ls] = max(depth, -0.0009), 0.0
                    d.qpos[self.a_lt], d.qvel[self.d_lt] = 0.0, 0.0
                    twist = _twist_about_x(Rf.inv()*R)
                    self._weld_to(w, f, twist, soft=60)
                    self._nozzle_collide(w, False)
                    self._turn_ref = 0.0
                    self.state, self.mate, f.mate = "captured", f, self
                    w.log_event(dict(t=w.sim_time, event="luer_captured", syringe=self.name, fitting=f.name))
                    break
            mjm.dof_frictionloss[self.d_ls] = fr["slide_n"]
            mjm.dof_frictionloss[self.d_lt] = fr["tau0"]
            return
        f = self.mate
        c1 = lu["lead"]/(2*math.pi)
        stop = 0.0
        if self.state == "captured":
            if q_d < -0.0008:
                w.eq_enable(self.eq_weld[f.name], False)
                d.qpos[self.a_ls] = d.qpos[self.a_lt] = 0.0
                d.qvel[self.d_ls] = d.qvel[self.d_lt] = 0.0
                self._nozzle_collide(w, True)
                self.state, self.mate, f.mate = "free", None, None
                w.log_event(dict(t=w.sim_time, event="luer_released", syringe=self.name, fitting=f.name))
                return
            # the lugs meet the thread's start at d_thread: from there the nozzle goes deeper only as the fitting is
            # turned (a stop that moves with the turn; it does not hold the fitting on). After THREAD_CATCH_DEG of turn
            # against it the flanks are engaged both ways.
            turned = max(q_t - self._turn_ref, 0.0)
            d_stop = lu["d_thread"] + c1*turned
            stop = max(q_d - d_stop, 0.0)*fr["k_seat"]
            if turned >= math.radians(P.THREAD_CATCH_DEG) and q_d >= d_stop - 0.0001:
                w.eq_row(self.eq_thread)[0] = self._thread_c0(q_d, q_t)
                w.eq_enable(self.eq_thread, True)
                self.state = "threaded"
                stop = 0.0
                w.log_event(dict(t=w.sim_time, event="luer_threaded", syringe=self.name, fitting=f.name))
        elif self.state == "threaded" and q_d < lu["d_thread"] + c1*math.radians(0.5*P.THREAD_CATCH_DEG):
            w.eq_enable(self.eq_thread, False)
            self.state = "captured"
            self._turn_ref = q_t - (q_d - lu["d_thread"])/c1
            w.log_event(dict(t=w.sim_time, event="luer_unthreaded", syringe=self.name, fitting=f.name))
        # forces on the depth: the fitting's own spring, the seat, the pressure that parts the joint
        F = -stop
        if f.spring is not None:
            F -= f.spring(max(q_d, 0.0))
        seat = max(q_d - lu["d_seat"], 0.0)*fr["k_seat"]
        F -= seat
        F -= max(self.chamber.p - P.P_ATM, 0.0)*math.pi*lu["mouth_d"]**2/4 if self.sealed() else 0.0
        w.jf[self.j["depth"].dof] += F
        mjm.dof_frictionloss[self.d_ls] = fr["slide_n"]
        # the turn rubs on the cone's seat and on the thread's flanks: the flanks carry whatever axial load the thread
        # holds (last tick's), so the joint is self-locking under a pull as a 5 mm lead on 7.8 mm lugs is
        thread = 0.0
        if self.state == "threaded":
            ft = w.eq_force(self.eq_thread)
            thread = abs(float(ft[0])) if ft is not None else 0.0
        mjm.dof_frictionloss[self.d_lt] = fr["tau0"] + fr["mu"]*(fr["r_seat"]*seat + fr["r_thread"]*thread)
        self.last["seat_n"], self.last["thread_n"] = seat, thread
        if seat > P.LUER_OVERRIDE_N and not self.flags["overridden"]:
            self.flags["overridden"] = True
            w.log_event(dict(t=w.sim_time, event="luer_overridden", syringe=self.name, fitting=f.name))

    def depth(self, w):
        return float(w.solver.mj_data.qpos[self.a_ls])

    def sealed(self):
        return self.state == "threaded" and self._depth_now >= P.LUER["d_seat"] - 0.0002

    # ---- per tick
    def pre_step(self, w):
        mjm, d = w.solver.mj_model, w.solver.mj_data
        s, ch = self.s, self.chamber
        self._depth_now = float(d.qpos[self.a_ls])
        self._luer(w)
        self._depth_now = float(d.qpos[self.a_ls])
        x = float(d.qpos[self.a_pl])
        xd = float(d.qvel[self.d_pl])
        # the barrel's attitude and what it feels as gravity
        p, R = T.world_pose(w, self.barrel)
        v = w.bqd[self.barrel, :3].copy()
        acc = np.zeros(3) if self._v_prev is None else (v - self._v_prev)/w.dt
        self._v_prev = v
        g_eff = np.array([0., 0., P.GRAVITY]) + acc                 # 'up' as the liquid feels it, times its strength
        self._g_lp = g_eff if self._g_lp is None else self._g_lp + min(1.0, w.dt/0.05)*(g_eff - self._g_lp)
        gmag = float(np.linalg.norm(self._g_lp))
        up = self._g_lp/max(gmag, 1e-9)
        axis_up = float(R.apply(X) @ up)                              # +1: thumb rest up
        ch.move_gas(axis_up, gmag, w.dt, max(x, 0.0))
        gas_first = ch.gas_at_nozzle(axis_up)
        # the path from the nozzle
        self.path = path = self._path(w, up)
        c_h, F_bias = ch.coefficients(path, w.dt, gas_first)
        # the seal
        fb = self.first_break if self.first_break > 0 else s["seal_break"]
        rest = min(self.rest_s/P.SEAL_RECOVERY_S, 30.0)
        f_b = s["seal_glide"] + (fb - s["seal_glide"])*(1.0 - math.exp(-rest)) if self.first_break <= 0 else fb
        limit = s["seal_glide"] + (f_b - s["seal_glide"])*math.exp(-abs(xd)/P.SEAL_STRIBECK_M_S)
        mjm.dof_frictionloss[self.d_pl] = limit
        if self.flags["extracted"]:
            c_h = F_bias = 0.0
        # Chamber.coefficients eliminated the new pressure and flow in favour of the
        # new plunger velocity. Put that coefficient in a bilateral force row so
        # the implicit constraint solve includes it alongside contacts. The former dt*c
        # armature also multiplied the OLD velocity; it was not an exact implicit
        # force update and gave impacts an artificial, timestep-dependent inertia.
        c = c_h + P.SEAL_VISCOUS
        self.implicit_fluid = (c, F_bias)
        self.last.update(c_h=c_h, F_bias=F_bias, seal=limit, gas_first=gas_first, axis_up=axis_up)

    def _path(self, w, up):
        ch = self.chamber
        mate = self.mate if self.state != "free" else None
        if mate is not None and self.sealed():
            path = mate.owner.path(w, self, mate)
        else:
            path = H.Path(self.atmos, ("open", self.name))
        if path.terminal.kind != "closed":
            mu = H.viscosity(ch.stack.front_mix())
            path.add_tube(P.LUER["nozzle_len"], P.LUER["bore_d"], mu, kinetic=0.5)
        return path

    def post_step(self, w):
        d = w.solver.mj_data
        s, ch = self.s, self.chamber
        x, xd = float(d.qpos[self.a_pl]), float(d.qvel[self.d_pl])
        Q = ch.advance(self.path, -xd, x, w.dt, w.sim_time, w.fluid.ledger)
        self.last["Q"] = Q
        if abs(xd) > 2e-5:
            self.rest_s = 0.0
            if self.first_break > 0 and x < self.x0 - 5e-5:
                self.first_break = 0.0                    # the stored stopper has been freed
        else:
            self.rest_s += w.dt
        if x >= s["stroke_max"] - 1e-5 and not self.flags["extracted"] and \
                w.limit_force(self.j["plunger"].mj) > P.PLUNGER_RETENTION_N:
            # the stopper has been pulled past the barrel's retaining ring: the barrel is open at the back and what
            # it held runs out there (the plunger itself stays on its joint: a stated simplification)
            self.flags["extracted"] = True
            p, R = T.world_pose(w, self.barrel)
            back = p + R.apply((s["barrel_len"] + 0.004, 0., 0.))
            mix = ch.stack.pop_front(ch.liquid)
            n = max(int(math.ceil(float(mix[:H.AIR].sum())/(0.05*H.ML))), 1)
            for _ in range(n if float(mix[:H.AIR].sum()) > 0 else 0):
                w.fluid.spray._launch(back, np.asarray(w.bqd[self.barrel, :3], float), mix/n, self.mj_barrel)
            ch.gas0 = 0.0
            w.log_event(dict(t=w.sim_time, event="plunger_extracted", syringe=self.name,
                             spilled_ml=float(mix[:H.AIR].sum())/H.ML))
        if w.tick % 20 == 0:
            self.lump(w)

    def lump(self, w):
        """The liquid's mass rides in the barrel body (its centre on the axis, mid-column)."""
        mjm = w.solver.mj_model
        ch = self.chamber
        m_l = float(ch.stack.total() @ H._RHO)
        xc = 0.5*ch.liquid/ch.A
        Mt = self.mass0 + m_l
        ipos = (self.mass0*self.ipos0 + m_l*np.array([xc, 0., 0.]))/Mt
        set_body_mass(mjm, self.mj_barrel, Mt)
        mjm.body_ipos[self.mj_barrel] = ipos
        Lc = ch.liquid/ch.A
        extra = m_l*(Lc*Lc/12 + (xc - ipos[0])**2) + self.mass0*(self.ipos0[0] - ipos[0])**2
        mjm.body_inertia[self.mj_barrel] = self.inertia0 + np.array([0.5*m_l*(0.5*ch.bore)**2, extra, extra])

    # ---- readings
    def reading(self, w):
        d = w.solver.mj_data
        ch = self.chamber
        x = float(d.qpos[self.a_pl])
        mix = ch.stack.total()
        return dict(plunger_mm=1e3*x, scale_ml=(ch.A*x)/H.ML, liquid_ml=ch.liquid/H.ML,
                    dose_ml=max(ch.liquid - ch.dead, 0.0)/H.ML, gas_ml=ch.gas0/H.ML, gas_pos=ch.gas_pos,
                    gauge_kpa=(ch.p - P.P_ATM)/1e3, peak_gauge_kpa=ch.peak_gauge/1e3, flow_ml_s=self.last["Q"]/H.ML,
                    contents_ml={sp: float(mix[i])/H.ML for i, sp in enumerate(P.SPECIES) if mix[i] > 0},
                    luer=self.state, mate=self.mate.name if self.mate else None, luer_depth_mm=1e3*float(d.qpos[self.a_ls]),
                    luer_turn_deg=math.degrees(float(d.qpos[self.a_lt])), seat_n=self.last.get("seat_n", 0.0),
                    seal_limit_n=self.last["seal"], flags=dict(self.flags), touched=sorted(self.touch))

    def snapshot(self):
        return dict(chamber=self.chamber.snapshot(), state=self.state, mate=self.mate.name if self.mate else None, turn_ref=self._turn_ref,
                    flags=dict(self.flags), touch=dict(self.touch), rest=self.rest_s, first=self.first_break, arm=self._arm,
                    g=None if self._g_lp is None else self._g_lp.copy(), v=None if self._v_prev is None else self._v_prev.copy())

    def restore(self, w, s):
        self.chamber.restore(s["chamber"])
        self.state = s["state"]
        self.mate = w.fluid.females[s["mate"]] if s["mate"] else None
        if self.mate:
            self.mate.mate = self
        self.flags, self.touch = dict(s["flags"]), dict(s["touch"])
        self.rest_s, self.first_break = s["rest"], s["first"]
        self._turn_ref = s["turn_ref"]
        self._g_lp = None if s["g"] is None else s["g"].copy()
        self._v_prev = None if s["v"] is None else s["v"].copy()
        self._armature_set(w, s["arm"])


def T_set_q(b, joint, value):
    b.joint_q[b.joint_q_start[joint]] = float(value)


# ====================================================================== needle
class Needle:
    """Hub, tube, a bead that rides the tube inside a track, and a sheath."""

    def __init__(self, sc, name, kind, pose, sheathed=True, tight=False):
        self.sc, self.name, self.kind = sc, name, kind
        self.n = n = P.NEEDLES[kind]
        self.bore = n["bore"]*(P.NEEDLE_BORE_VARIANT if tight else 1.0)
        hub, sh = P.NEEDLE_HUB, P.NEEDLE_SHEATH
        b, wp = sc.b, sc.wp
        I = wp.transform_identity()
        Lh, Ln = hub["length"], n["length"]
        self.x_tip = -(Lh + Ln)
        self.pose0 = pose
        self.hub = Hb = T.link(sc, f"{name}_hub", pose)
        rho_h = hub["mass"]/(math.pi*(0.5*hub["od"])**2*Lh)
        # Rounded molded hub contact envelope. MuJoCo's cylinder/box CCD yields
        # one migrating contact, whereas a capsule preserves the axial contact
        # line under a small wrist turn. Same outer diameter, length and mass;
        # the separate visible cylinder preserves the hub's flat-ended appearance.
        rh = .5*hub["od"]
        T.capsule(sc, Hb, f"{name}_hub_body", (-rh, 0., 0.), (-Lh+rh, 0., 0.), rh, "collision_hidden",
                  density=hub["mass"]/(math.pi*rh*rh*(Lh-2*rh)+4/3*math.pi*rh**3), mu=P.MU["pp"])
        T.cylinder(sc, Hb, f"{name}_hub_look", (-.5*Lh, 0., 0.), rh, .5*Lh, "x", "hub_"+kind, density=0., collide=False)
        sc.box(f"{name}_hub_lugs", (-0.0008, 0., 0.), (0.0008, 0.5*P.LUER["lugs_d"], 0.0012), "hub_" + kind, body=Hb, density=rho_h,
               mu=P.MU["pp"])
        tp = T.compose(pose, (-Lh, 0., 0.))
        m_tube = 7900.*math.pi*((0.5*n["od"])**2 - (0.5*n["bore"])**2)*Ln
        self.tube = Tb = T.link(sc, f"{name}_tube", tp, mass=max(m_tube, 2e-5),
                                inertia=(1e-10, max(m_tube, 2e-5)*Ln*Ln/3, max(m_tube, 2e-5)*Ln*Ln/3), com=(-0.5*Ln, 0., 0.))
        self.shape_tube = T.capsule(sc, Tb, f"{name}_needle", (-0.5*n["od"], 0., 0.), (-Ln + 0.5*n["od"], 0., 0.), 0.5*n["od"], "steel_needle",
                                    density=0., mu=P.MU["steel"])
        bp = T.compose(pose, (self.x_tip, 0., 0.))
        self.bead = Bd = T.link(sc, f"{name}_bead", bp, mass=P.CARRIER_MASS, inertia=(1e-9, 1e-9, 1e-9))
        j_free = b.add_joint_free(Hb, label=f"{name}_free")
        # Two bending coordinates, no torsional or axial freedom. A reduced cantilever:
        # k_theta = 3 EI / free_length gives delta = F L^3/(3 EI) at the track entry.
        # The steel stays straight between hub and tip (not a beam mesh), but its
        # collision, bead and outlet now follow the elastic coordinates.
        # Above Newton's 1e-10 absolute inertia floor (below it finalize inflates
        # the tensor to 1e-6). This numerical universal-joint carrier is 0.01 mg;
        # its 2e-10 kg m2 is disclosed in the reduced-joint inertia bound.
        Gb = T.link(sc, f"{name}_elastic_carrier", tp, mass=1e-8, inertia=(2e-10, 2e-10, 2e-10))
        jy = b.add_joint_revolute(Hb, Gb, axis=(0., 1., 0.), limit_lower=-0.5, limit_upper=0.5,
                                  parent_xform=wp.transform((-Lh, 0., 0.), wp.quat_identity()), child_xform=I,
                                  label=f"{name}_elastic_y")
        jz = b.add_joint_revolute(Gb, Tb, axis=(0., 0., 1.), limit_lower=-0.5, limit_upper=0.5,
                                  parent_xform=I, child_xform=I, label=f"{name}_elastic_z")
        self.j_elastic = [T.JointRef(jy, "elastic_y"), T.JointRef(jz, "elastic_z")]
        j_bead = b.add_joint_prismatic(Tb, Bd, parent_xform=wp.transform((-Ln, 0., 0.), wp.quat_identity()), child_xform=I,
                                       axis=(1., 0., 0.), limit_lower=-0.0005, limit_upper=Ln, friction=0.0,
                                       armature=P.CARRIER_ARMATURE[0], label=f"{name}_track")
        b.add_articulation([j_free, jy, jz, j_bead], label=f"{name}_assembly")
        self.j_bead = T.JointRef(j_bead, "bead")
        sc.mutual_filter([Hb, Tb, Bd])
        self.female = Female(self, name, Hb, (0., 0., 0.))
        # the sheath: a tube over the needle, pressed onto the hub's front
        self.sheath = None
        self.sheathed = bool(sheathed)
        from newton._src.solvers.mujoco.equality import _add_equality_constraint
        self._eqc = _add_equality_constraint
        stiff = {"mujoco:eq_solref": P.FIT_SOLREF, "mujoco:eq_solimp": P.FIT_SOLIMP}
        self._stiff = stiff
        # Constant polynomial: each coordinate has an independent bilateral row.
        # An explicit second joint avoids Newton 1.5's -1/name-lookup bug; its
        # coefficient is zero and it contributes no force or kinematic relation.
        self.eq_elastic = [_add_equality_constraint(b, constraint_type=sc.newton.solvers.SolverMuJoCo.EqType.JOINT,
                            joint1=j1, joint2=j2, polycoef=[0., 0., 0., 0., 0.], enabled=True,
                            label=f"{name}_elastic_force_{i}", custom_attributes=stiff) for i, (j1, j2) in enumerate(((jy, jz), (jz, jy)))]
        if sheathed is not None:
            Ls = Ln + sh["overlap"] + 0.004
            self.sheath_len = Ls
            self.sheath_x = -Lh + sh["overlap"]              # the sheath's mouth plane in the hub frame when seated
            sp = T.compose(pose, (self.sheath_x, 0., 0.)) if sheathed else sheathed_pose_free(sc, name, pose)
            self.sheath = Sh = T.link(sc, f"{name}_sheath", sp)
            rs = 0.5*sh["od"]
            rho_s = sh["mass"]/(math.pi*(rs*rs - (rs - sh["wall"])**2)*Ls)
            a8 = rs*math.cos(math.pi/8)
            for k in range(2):
                sc.box(f"{name}_sheath{k}", (-0.5*Ls, 0., 0.), (0.5*Ls, a8, a8*math.tan(math.pi/8)), "sheath_" + kind, body=Sh,
                       rot=T.wq(Rotation.from_euler("x", 45.*k, degrees=True)),
                       density=0.5*rho_s*(rs*rs - (rs - sh["wall"])**2)*math.pi/(4*a8*a8*math.tan(math.pi/8)), mu=P.MU["pp"])
            b.add_articulation([b.add_joint_free(Sh, label=f"{name}_sheath_free")], label=f"{name}_sheath_assembly")
            self.eq_sheath = _add_equality_constraint(b, constraint_type=sc.newton.solvers.SolverMuJoCo.EqType.WELD, body1=Hb, body2=Sh,
                                                      anchor=(self.sheath_x, 0., 0.), enabled=bool(sheathed),
                                                      label=f"{name}_sheath_fit", custom_attributes=stiff)
            sc.mutual_filter([Hb, Tb, Bd, Sh])
            sc.places[Sh] = name + "_sheath"
            sc.bodies[name + "_sheath"] = Sh
        self.eq_track = {}                                # penetrable name -> connect equality
        sc.bodies[name] = Hb
        sc.places[Hb] = name
        sc.places[Tb] = name + "_tube"
        self.track = None
        self.flags = dict(bent=False, blunted=False)
        self.touch = {}
        self.moment_peak = 0.0
        self.tip_force = 0.0
        self.sheath_pull = 0.0
        self._sheath_clear = True
        self._dimple = None
        self.resheathed = []
        self.records = []                                 # punctures: where, angle, depth reached
        self.depots = {}
        self.feeding = None
        self.atmos = None
        self.terminal_closed = H.Closed()

    def declare_tracks(self, penetrables):
        for pn in penetrables:
            self.eq_track[pn.name] = self._eqc(self.sc.b, constraint_type=self.sc.newton.solvers.SolverMuJoCo.EqType.WELD,
                                               body1=self.bead, body2=pn.body, anchor=(0., 0., 0.), enabled=False,
                                               label=f"{self.name}_in_{pn.name}", custom_attributes=self._stiff)

    def bind(self, w):
        mjm = w.solver.mj_model
        self.j_bead.bind(w)
        self.d_b = int(mjm.jnt_dofadr[self.j_bead.mj])
        self.a_b = int(mjm.jnt_qposadr[self.j_bead.mj])
        mjm.dof_solref[self.d_b] = P.FIT_SOLREF
        mjm.dof_solimp[self.d_b] = P.FIT_SOLIMP
        w.keep_invweight(self.d_b, 1.0/(P.CARRIER_MASS + P.CARRIER_ARMATURE[0]))
        self.mj_tube = w._newton_to_mj[self.tube]
        self.mj_hub = w._newton_to_mj[self.hub]
        self.g_tube = g = w.geom_of_shape(self.shape_tube)
        mjm.geom_conaffinity[g] |= 4  # vial inner walls are needle-only contacts
        self._tube_mask = (int(mjm.geom_contype[g]), int(mjm.geom_conaffinity[g]))
        self._tube_on = True
        if self.sheath is not None:
            # seated: the sheath's frame is the hub's, moved along the axis to the sheath's mouth plane
            w.eq_row(self.eq_sheath)[:11] = [0., 0., 0., self.sheath_x, 0., 0., 1., 0., 0., 0., 1.0]
        self.atmos = H.Atmosphere(w.fluid.spray, lambda: self._tip_origin(w), self.bore, body=self.mj_tube)
        self.atmos.outer = self.n["od"]
        I = (math.pi/64)*(self.n["od"]**4 - self.n["bore"]**4)
        self.yield_moment = P.NEEDLE_YIELD*I/(0.5*self.n["od"])
        self.buckle_free = math.pi**2*P.E_STEEL*I/(2*self.n["length"])**2
        self.buckle_held = math.pi**2*P.E_STEEL*I/(0.7*self.n["length"])**2
        self.lateral_k = 3*P.E_STEEL*I/self.n["length"]**3
        self.EI = P.E_STEEL*I
        for joint in self.j_elastic:
            joint.bind(w)
        self.elastic_joints = np.array([joint.mj for joint in self.j_elastic])
        self.elastic_dofs = mjm.jnt_dofadr[self.elastic_joints].copy()
        self.elastic_q = mjm.jnt_qposadr[self.elastic_joints].copy()
        self._elastic_step(w)

    def _elastic_step(self, w):
        m, d = w.solver.mj_model, w.solver.mj_data
        inserted = max(float(d.qpos[self.a_b]), 0.0) if self.track else 0.0
        # At full insertion the last 1 mm is a bounded root region; the model is
        # qualified for at least 3 mm exposed length, not root crushing or plastic folding.
        free = max(self.n["length"] - inserted, 0.001)
        k = 3*self.EI/free
        m.jnt_stiffness[self.elastic_joints] = 0.0
        m.dof_damping[self.elastic_dofs] = 0.0
        # Bilateral force rows solve this spring simultaneously with the soft Luer
        # fitting and tissue constraints. Passive stiffness followed by an implicit
        # velocity correction was unstable with those compliant constraints.
        inertia = max(float(m.body_inertia[self.mj_tube, 1]), 1e-12)
        self.elastic_coeff = (k, 0.1*math.sqrt(k*inertia) + w.dt*k)
        moment = k*float(np.linalg.norm(d.qpos[self.elastic_q]))
        self.moment_peak = max(self.moment_peak, moment)
        if moment > self.yield_moment and not self.flags["bent"]:
            self.flags["bent"] = True
            w.log_event(dict(t=w.sim_time, event="needle_bent", needle=self.name, moment_nm=moment,
                            elastic_deg=math.degrees(float(np.linalg.norm(d.qpos[self.elastic_q])))))

    def tip_world(self, w):
        p, R = T.world_pose(w, self.tube)
        return p + R.apply((-self.n["length"], 0., 0.)), R

    def _tip_origin(self, w):
        tip, R = self.tip_world(w)
        # what leaves the point leaves with the point's velocity: the tube's centre's was 0.76 m/s where the point moved
        # at 1.35 m/s (a syringe landing)
        return tip, -R.apply(X), w.point_velocity(self.tube, tip), w.dt

    def _tube_collide(self, w, on):
        """The tube is a collider except where its point is at or in a face it can enter: there the face's own law
        (surface, puncture, track) carries it."""
        if on != self._tube_on:
            mjm = w.solver.mj_model
            mjm.geom_contype[self.g_tube], mjm.geom_conaffinity[self.g_tube] = self._tube_mask if on else (0, 0)
            self._tube_on = on

    def _in_sheath(self, w, tip, R):
        """Whether the point is inside its sheath (on or off the hub): the sheath's wall is then between the tube and
        whatever is outside, so the tube is no collider (the tube and its sheath do not collide with each other, and
        without this a finger gripping the sheath touched the needle through it as the sheath came off)."""
        if self.sheath is None:
            return False
        if self.sheathed:
            return True
        ps, Rs = T.world_pose(w, self.sheath)
        loc = Rs.inv().apply(tip - ps)
        return bool(-self.sheath_len <= loc[0] <= 0.002 and math.hypot(loc[1], loc[2]) < 0.5*P.NEEDLE_SHEATH["od"])

    def _touch(self, w, place):
        if place not in self.touch:
            self.touch[place] = w.sim_time
            w.log_event(dict(t=w.sim_time, event="touch", part=self.name, place=place))

    # ---- flow
    def path(self, w, syringe, female):
        n = self.n
        mu = H.viscosity(syringe.chamber.stack.front_mix())
        tr = self.track
        if self.sheathed:
            term, key = w.fluid.sink(self.name + "_sheath"), ("sheath", self.name)
        elif tr is None:
            term, key = self.atmos, ("air", self.name)
        else:
            term, key = tr["pen"].terminal_for(w, self, tr)
        path = H.Path(term, key)
        if term.kind != "closed":
            path.add_tube(n["length"] + P.NEEDLE_HUB_FLOW["embedded"], self.bore, mu, kinetic=P.K_ENTRY)
            path.add_tube(P.NEEDLE_HUB["length"] - P.NEEDLE_HUB_FLOW["embedded"], P.NEEDLE_HUB_FLOW["bore"], mu)
        return path

    # ---- the sheath
    def _sheath_step(self, w, R):
        if self.sheath is None:
            return
        sh = P.NEEDLE_SHEATH
        if self.sheathed:
            f = w.eq_force(self.eq_sheath)
            # only a pull draws the sheath off its hub (the fit's force on the hub along the hub's x is negative then);
            # a push seats the hub deeper, against the sheath's floor: counted as a pull, the 6 N with which a syringe
            # is pressed on while it is turned onto a sheathed needle let the needle out and it turned in its sheath
            pull = max(-float(f[:3] @ R.apply(X)), 0.0) if f is not None else 0.0
            self.sheath_pull = pull
            if pull > sh["pull_n"]:
                w.eq_enable(self.eq_sheath, False)
                self.sheathed = False
                self._sheath_clear = False               # it cannot seat again before it has come 5 mm off
                w.log_event(dict(t=w.sim_time, event="unsheathed", needle=self.name, pull_n=pull))
            return
        # a bare needle that has gone back into its sheath to the seat is held again (the press fit's push-on force
        # is not modelled: it latches at the seat)
        ph, Rh = T.world_pose(w, self.hub)
        ps, Rs = T.world_pose(w, self.sheath)
        e = Rh.inv().apply(ps - ph) - np.array([self.sheath_x, 0., 0.])
        if not self._sheath_clear:
            self._sheath_clear = bool(float(np.linalg.norm(e)) > 0.005)
            return
        if abs(e[0]) > 0.0006 or math.hypot(e[1], e[2]) > sh["cap_tol"] or float(Rs.apply(X) @ Rh.apply(X)) < math.cos(math.radians(6.0)):
            return
        if float(np.linalg.norm(w.bqd[self.sheath, :3] - w.bqd[self.hub, :3])) > 0.05:
            return
        held = bool(w.pair_contact_forces([self.sheath], w.pad_bodies))
        w.eq_enable(self.eq_sheath, True, soft_ticks=40)
        self.sheathed = True
        self.resheathed.append(dict(t=w.sim_time, sheath_in_gripper=held))
        w.log_event(dict(t=w.sim_time, event="resheathed", needle=self.name, sheath_in_gripper=held))

    # ---- the point and the track
    def pre_step(self, w):
        mjm, d = w.solver.mj_model, w.solver.mj_data
        self._elastic_step(w)
        tip, R = self.tip_world(w)
        ax_in = -R.apply(X)                                # the way the point goes
        scale = self.n["od"]/0.0005*(P.BLUNT_FILL_FACTOR if self.n["blunt"] else 1.0)
        self._sheath_step(w, T.world_pose(w, self.hub)[1])
        if self.track is not None:
            self._in_track(w, tip, ax_in, scale)
            return
        mjm.dof_frictionloss[self.d_b] = 0.05
        zone = None
        if not self.sheathed:
            for pn in w.fluid.penetrables:
                hit = pn.surface(w, tip)
                if hit is not None:
                    zone = (pn, hit)
                    break
        self._tube_collide(w, (zone is None or isinstance(zone[0], VialSite)) and not self._in_sheath(w, tip, R))
        if zone is None:
            self._dimple = None
            return
        pn, (depth, normal, soft, where) = zone
        if depth <= 0.0:
            self._dimple = None
            return
        if not soft:
            F = normal*min(pn.k_hard*depth, 80.0)
            w.add_force(self.tube, F, tip)
            pn.react(w, -F, tip)
            self._touch(w, f"{pn.name}:{pn.hard_name}")
            if float(np.linalg.norm(F)) > P.NEEDLE_BLUNT_N and not self.flags["blunted"]:
                self.flags["blunted"] = True
                w.log_event(dict(t=w.sim_time, event="needle_blunted", needle=self.name, place=f"{pn.name}:{pn.hard_name}",
                                 force_n=float(np.linalg.norm(F))))
            return
        self._touch(w, pn.name if where == "target" else f"{pn.name}:prepared"
                    if pn.prepared_radius is not None and pn.offset(pn.to_local(w, tip)) <= pn.prepared_radius else f"{pn.name}:off_target")
        top = P.LAYERS[pn.layers[0]]
        force = top["k"]*depth
        cosang = float(-ax_in @ normal)
        need = top["puncture"]*scale
        self.tip_force = min(force, need)
        if force >= need and cosang > math.cos(math.radians(70.0)):
            q0 = min(depth/max(cosang, 0.2), 0.5*self.n["length"])
            d.qpos[self.a_b], d.qvel[self.d_b] = q0, 0.0
            e = self.eq_track[pn.name]
            anchor2 = pn.to_local(w, tip - ax_in*q0)
            w.eq_row(e)[:6] = [0., 0., 0., *anchor2]
            w.eq_enable(e, True, soft_ticks=20)
            self.track = dict(pen=pn, eq=e, entry=anchor2, angle=math.degrees(math.acos(min(1.0, cosang))), cos=max(cosang, 0.2),
                              axis_local=pn.dir_local(w, ax_in), deepest=q0, t0=w.sim_time, where=where)
            self.records.append(dict(t=w.sim_time, target=pn.name, where=where, angle_deg=self.track["angle"],
                                     entry=[float(v) for v in anchor2], offset_mm=1e3*pn.offset(anchor2), deepest_mm=1e3*q0,
                                     lever_deg=0.0, exit_angle_deg=None, t_out=None))
            w.log_event(dict(t=w.sim_time, event="punctured", needle=self.name, target=pn.name, angle_deg=self.track["angle"],
                             force_n=force))
        else:
            Fv = normal*min(force, need)
            # the point sits in the dimple it makes: a sideways spring to where it first bore, as strong as the face is
            # deep-wise, up to the force it presses with (a friction of 1). Without it the face was ice: a syringe
            # pressed on its point from above slid off sideways at 7 N (phantom-hand rehearsal).
            if self._dimple is None or self._dimple[0] is not pn:
                self._dimple = (pn, pn.to_local(w, tip))
            p0 = pn._pose(w)
            off = tip - (p0[0] + p0[1].apply(self._dimple[1]))
            off = off - float(off @ normal)*normal
            Fl = -top["k"]*off
            cap = P.TIP_GRIP*float(np.linalg.norm(Fv))
            n_l = float(np.linalg.norm(Fl))
            if n_l > cap > 0.0:                               # it slips: the dimple goes with it
                Fl = Fl*(cap/n_l)
                self._dimple = (pn, pn.to_local(w, tip - off*(cap/top["k"]/float(np.linalg.norm(off)))))
            w.add_force(self.tube, Fv + Fl, tip)
            pn.react(w, -(Fv + Fl), tip)

    def _in_track(self, w, tip, ax_in, scale):
        mjm, d = w.solver.mj_model, w.solver.mj_data
        self._tube_collide(w, isinstance(self.track["pen"], VialSite))
        tr = self.track
        pn = tr["pen"]
        rec = self.records[-1]
        q = float(d.qpos[self.a_b])
        qd = float(d.qvel[self.d_b])
        if q <= 0.0002 and qd <= 0.0:
            w.eq_enable(tr["eq"], False)
            ax_now = pn.dir_local(w, ax_in)
            rec["exit_angle_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, float(ax_now @ tr["axis_local"])))))
            rec["t_out"] = w.sim_time
            w.log_event(dict(t=w.sim_time, event="withdrawn", needle=self.name, target=pn.name, deepest_mm=rec["deepest_mm"]))
            pn.left(w, self, tr)
            self.track = None
            d.qpos[self.a_b] = 0.0
            return
        tr["deepest"] = max(tr["deepest"], q)
        rec["deepest_mm"] = 1e3*tr["deepest"]
        fs, fk, cut, bump = pn.friction(q*tr["cos"], tr["deepest"]*tr["cos"], scale, 1.0/tr["cos"])
        mjm.dof_frictionloss[self.d_b] = fk + (fs - fk)*math.exp(-abs(qd)/P.SEAL_STRIBECK_M_S) + bump
        if qd > 1e-4 and q >= tr["deepest"] - 1e-5 and cut > 0:
            w.jf[self.j_bead.dof] -= cut                   # cutting resists only an advancing point
        # Track springs are implicit constitutive rows in World._track_constraints.
        # The same Jacobian applies equal and opposite loads to needle and trainer.
        # Explicit damping here was unstable on the thin tube's real inertia.
        ax0 = pn.dir_world(w, tr["axis_local"])
        ang = math.acos(float(np.clip(ax0 @ ax_in, -1., 1.)))
        rec["lever_deg"] = max(rec["lever_deg"], math.degrees(ang))
        entry_w = pn._pose(w)[0] + pn._pose(w)[1].apply(tr["entry"])
        off = w.bq[self.bead, :3] - entry_w
        s_lat = off - float(off @ ax0)*ax0
        rec["sideways_mm"] = max(rec.get("sideways_mm", 0.0), 1e3*float(np.linalg.norm(s_lat)))
        # The track loads the compliant tube. Yield is measured from that joint's
        # elastic moment in _elastic_step, not from a rigid-hand mismatch.
        stop = pn.floor_force(q*tr["cos"])
        if stop > 0:
            w.jf[self.j_bead.dof] -= stop
            if stop > self.buckle_held and not self.flags["bent"]:
                self.flags["bent"] = True
                w.log_event(dict(t=w.sim_time, event="needle_buckled", needle=self.name, force_n=stop))
            if stop > P.NEEDLE_BLUNT_N and not self.flags["blunted"]:
                self.flags["blunted"] = True
                w.log_event(dict(t=w.sim_time, event="needle_blunted", needle=self.name, place=pn.name + ":board", force_n=stop))

    def post_step(self, w):
        pass

    def reading(self, w):
        d = w.solver.mj_data
        tr = self.track
        return dict(sheathed=self.sheathed, in_track=tr["pen"].name if tr else None, inserted_mm=1e3*float(d.qpos[self.a_b]) if tr else 0.0,
                    layer=tr["pen"].layer_at(float(d.qpos[self.a_b])*tr["cos"]) if tr else None,
                    track_friction_n=float(w.solver.mj_model.dof_frictionloss[self.d_b]), resheathed=[dict(r) for r in self.resheathed],
                    flags=dict(self.flags), touched=sorted(self.touch), moment_peak_nm=self.moment_peak,
                    yield_moment_nm=self.yield_moment, punctures=[dict(r) for r in self.records],
                    elastic_deg=np.degrees(d.qpos[self.elastic_q]).tolist(),
                    on=self.female.mate.name if self.female.mate else None)

    def snapshot(self):
        tr = self.track
        return dict(sheathed=self.sheathed, flags=dict(self.flags), touch=dict(self.touch), moment=self.moment_peak,
                    records=[dict(r) for r in self.records],
                    track=None if tr is None else dict(tr, pen=tr["pen"].name, entry=np.array(tr["entry"]),
                                                       axis_local=tr["axis_local"].copy()),
                    resheathed=[dict(r) for r in self.resheathed], tube_on=self._tube_on, sheath_clear=self._sheath_clear,
                    depots={k: v.p for k, v in self.depots.items()})

    def restore(self, w, s):
        self.sheathed, self.flags, self.touch = s["sheathed"], dict(s["flags"]), dict(s["touch"])
        self.moment_peak, self.records = s["moment"], [dict(r) for r in s["records"]]
        t = s["track"]
        self.track = None if t is None else dict(t, pen=w.fluid.pen_by_name[t["pen"]], entry=t["entry"].copy(),
                                                 axis_local=t["axis_local"].copy())
        self.resheathed = [dict(r) for r in s["resheathed"]]
        self._sheath_clear = s["sheath_clear"]
        self._tube_collide(w, s["tube_on"])
        for k, p in s["depots"].items():
            if k in self.depots:
                self.depots[k].p = p


def sheathed_pose_free(sc, name, pose):
    """A sheath that starts off its needle lies beside it on the tray (set by the scene through `sc.loose[name]`)."""
    return sc.loose.get(name + "_sheath", T.compose(pose, (0., 0.03, 0.)))


class TipCap:
    """The screw cap on a prefilled syringe's nozzle: a closed female fitting."""

    def __init__(self, sc, name, pose):
        self.name = name
        self.body = Bc = T.link(sc, name, pose)
        T.cylinder(sc, Bc, f"{name}_body", (-0.006, 0., 0.), 0.0045, 0.006, "x", "tip_cap", density=1100., mu=P.MU["pp"])
        sc.box(f"{name}_wings", (-0.004, 0., 0.), (0.004, 0.0065, 0.0015), "tip_cap", body=Bc, density=1100., mu=P.MU["pp"])
        sc.b.add_articulation([sc.b.add_joint_free(Bc, label=f"{name}_free")], label=f"{name}_assembly")
        self.female = Female(self, name, Bc, (0., 0., 0.))
        self.closed = H.Closed()
        sc.bodies[name] = Bc
        sc.places[Bc] = name

    def path(self, w, syringe, female):
        return H.Path(self.closed, ("cap", self.name))


# ====================================================================== what a needle enters
class Penetrable:
    """A face a needle can go through, on a body, with the layers behind it. The face is a disc or a rectangle on the
    body; inside `soft_radius` (or everywhere, if none) it gives and is punctured, outside it is hard."""
    k_hard = 2.0e4
    hard_name = "hard"
    prepared_radius = None
    tilt_limit = None
    lateral = P.TRACK_LATERAL["pad"]

    def __init__(self, name, body, origin, normal, layers, face_radius=None, face_half=None, soft_radius=None, target_radius=None):
        self.name, self.body = name, int(body)
        self.o, self.n = np.asarray(origin, float), np.asarray(normal, float)/np.linalg.norm(normal)
        self.layers = list(layers)
        self.face_radius, self.face_half, self.soft_radius, self.target_radius = face_radius, face_half, soft_radius, target_radius
        self.bounds = np.cumsum([P.LAYERS[k]["thick"] for k in self.layers])
        u = np.cross(self.n, [1., 0., 0.])
        if np.linalg.norm(u) < 0.1:
            u = np.cross(self.n, [0., 1., 0.])
        self.u = u/np.linalg.norm(u)
        self.v = np.cross(self.n, self.u)

    def _pose(self, w):
        return (np.zeros(3), I3) if self.body < 0 else T.world_pose(w, self.body)

    def to_local(self, w, p_world):
        p, R = self._pose(w)
        return R.inv().apply(np.asarray(p_world) - p)

    def dir_local(self, w, v_world):
        return self._pose(w)[1].inv().apply(v_world)

    def dir_world(self, w, v_local):
        return self._pose(w)[1].apply(v_local)

    def normal_world(self, w):
        return self._pose(w)[1].apply(self.n)

    def offset(self, p_local):
        """Distance of a point of the body from the face's centre, in the face's plane."""
        loc = np.asarray(p_local, float) - self.o
        return float(np.linalg.norm(loc - float(loc @ self.n)*self.n))

    def surface(self, w, tip_world):
        """-> (depth below the face, outward normal, soft?, 'target' or 'off_target') if the point is over the face
        and within 2 mm above or 8 mm below it, else None."""
        p, R = self._pose(w)
        loc = R.inv().apply(tip_world - p) - self.o
        h = float(loc @ self.n)
        if h > 0.002 or h < -0.008:
            return None
        a, c = float(loc @ self.u), float(loc @ self.v)
        r = math.hypot(a, c)
        if self.face_radius is not None and r > self.face_radius:
            return None
        if self.face_half is not None and (abs(a) > self.face_half[0] or abs(c) > self.face_half[1]):
            return None
        soft = self.soft_radius is None or r <= self.soft_radius
        where = "target" if (self.target_radius is None or r <= self.target_radius) else "off_target"
        return -h, R.apply(self.n), soft, where

    def react(self, w, F, p_world):
        if self.body >= 0:
            w.add_force(self.body, F, p_world)

    def react_torque(self, w, tau):
        if self.body >= 0:
            w.add_torque(self.body, tau)

    def layer_at(self, depth):
        for name, bnd in zip(self.layers, self.bounds):
            if depth < bnd:
                return name
        return "beyond"

    def friction(self, depth, deepest, scale, per_depth=1.0):
        """Static and sliding friction limit (N) of a track whose point is `depth` below the face (`per_depth`: track
        length per unit depth), the cutting force of the layer at the point, and the bump of a further layer's skin
        at its boundary."""
        fs = fk = 0.0
        lo = 0.0
        cut = bump = 0.0
        for name, bnd in zip(self.layers, self.bounds):
            L = P.LAYERS[name]
            seg = max(min(depth, bnd) - lo, 0.0)*per_depth
            fs += L["fs"]*seg*scale
            fk += L["fk"]*seg*scale
            if lo <= depth < bnd:
                cut = L["cut"]*scale
            if lo > 0 and abs(depth - lo) < 0.0006 and depth >= deepest - 1e-5 and L["puncture"] > 0:
                bump = L["puncture"]*scale
            lo = bnd
        return fs, fk, cut, bump

    def pivot_k(self, q):
        return P.TRACK_PIVOT_K*q**3/3.0

    def floor_force(self, depth):
        over = depth - float(self.bounds[-1])
        return 2.0e4*over if over > 0 else 0.0

    def terminal_for(self, w, needle, tr):
        raise NotImplementedError

    def left(self, w, needle, tr):
        pass


class PadSite(Penetrable):
    """The injection pad's skin, fat and muscle over a board."""
    prepared_radius = 0.025

    def terminal_for(self, w, needle, tr):
        d = w.solver.mj_data
        depth = float(d.qpos[needle.a_b])*tr["cos"]
        layer = self.layer_at(depth)
        if layer == "beyond":
            return needle.terminal_closed, ("board", needle.name)
        key = (self.name, layer)
        if key not in needle.depots:
            needle.depots[key] = H.Depot(w.fluid.ledger, f"{self.name}:{layer}", layer, needle.bore)
        needle.feeding = needle.depots[key]
        return needle.depots[key], ("depot", needle.name, self.name, layer)

    def left(self, w, needle, tr):
        """What is still stored under pressure when the needle leaves comes back up the track in part."""
        for (pad, layer), dep in needle.depots.items():
            if pad != self.name:
                continue
            stored = dep.stored()
            if stored > 1e-12:
                back = P.TISSUE["track_leak"]*stored
                led = w.fluid.ledger
                have = led.places.get(dep.place)
                if have is not None and float(have[:H.AIR].sum()) > 0:
                    f = min(back/float(have[:H.AIR].sum()), 1.0)
                    mix = have*f
                    led.places[dep.place] = have - mix
                    led.add(f"{self.name}:surface", mix, w.sim_time)
                dep.p = P.P_ATM


class Vial:
    """A 10R vial with its closure; a rigid closed vessel behind a rubber target."""

    def __init__(self, sc, name, pose, liquid="medication", liquid_ml=10.0, gauge_kpa=0.0, adapter=False, mount="free"):
        self.sc, self.name = sc, name
        v = P.VIAL
        self.body = Bv = T.link(sc, name, pose)
        r = 0.5*v["od"]
        H_ = v["height"]
        body_h = H_ - v["neck_h"] - v["cap_h"]
        glass_vol = math.pi*(r*r - (r - v["wall"])**2)*body_h + math.pi*r*r*0.0012
        rho = 0.0102/glass_vol
        # the body stands on three low pads inside its footprint (a flat disc on a plane is degenerate in MuJoCo:
        # the repository's sweeping repair)
        self.glass_shape = T.cylinder(sc, Bv, f"{name}_glass", (0., 0., 0.0006 + 0.5*body_h), r, 0.5*body_h - 0.0006, "z", "vial_glass",
                   density=rho*(1 - ((r - v["wall"])/r)**2), mu=P.MU["glass"])
        for k in range(3):
            a = 2*math.pi*k/3
            sc.box(f"{name}_foot{k}", (0.7*r*math.cos(a), 0.7*r*math.sin(a), 0.0006), (0.003, 0.003, 0.0006), "vial_glass", body=Bv,
                   density=2500., mu=P.MU["glass"])
        self.neck_shape = T.cylinder(sc, Bv, f"{name}_neck", (0., 0., body_h + 0.5*v["neck_h"]), 0.5*v["flange_d"] - 0.002, 0.5*v["neck_h"], "z",
                   "vial_glass", density=600., mu=P.MU["glass"])
        self.cap_shape = T.cylinder(sc, Bv, f"{name}_cap", (0., 0., H_ - 0.5*v["cap_h"]), 0.5*v["flange_d"] + 0.0003, 0.5*v["cap_h"], "z",
                                    "vial_cap", density=1300., mu=P.MU["pp"])
        if mount == "fixed":
            jv = sc.b.add_joint_fixed(-1, Bv, parent_xform=sc.wp.transform(tuple(map(float, pose[0])), T.wq(pose[1])),
                                      child_xform=sc.wp.transform_identity(), label=f"{name}_clamp")
        else:
            jv = sc.b.add_joint_free(Bv, label=f"{name}_free")
        sc.b.add_articulation([jv], label=f"{name}_assembly")
        self.r_in = r - v["wall"]
        interior = (v["overflow_ml"] - v["plug_ml"])*H.ML
        self.stopper_in = H_-P.LAYERS["stopper"]["thick"]
        self.body_h = body_h
        self.z0 = body_h-(interior-math.pi*(v["bore"]/2)**2*(self.stopper_in-body_h))/(math.pi*self.r_in**2)
        self.h_in = self.stopper_in-self.z0
        gas0 = (interior - liquid_ml*H.ML)*(1.0 + gauge_kpa*1e3/P.P_ATM)
        self.vessel = H.Vessel(interior, H.parcel(liquid_ml*H.ML, liquid), gas_at_atm_m3=gas0)
        self.stopper_in = self.z0 + self.h_in                # same inner face as puncture law
        top = H_
        self.site = VialSite(name + "_stopper", Bv, (0., 0., top), (0., 0., 1.), ["stopper"], face_radius=0.5*v["flange_d"],
                             soft_radius=0.5*v["target_d"], target_radius=0.5*v["target_d"])
        self.site.vial = self
        # sample points of the interior, for the liquid's plane surface at any attitude
        g = np.linspace(-1, 1, 17)
        xx, yy = np.meshgrid(g, g)
        m = xx*xx + yy*yy <= 1.0
        zz = np.linspace(0, 1, 25)[:, None]
        self._pts = np.stack([np.tile(xx[m]*self.r_in, len(zz)), np.tile(yy[m]*self.r_in, len(zz)),
                              (self.z0 + self.h_in*(zz + 0*xx[m][None, :])).ravel()], 1)
        # A zero-mass geometry supplies the vial body reference for analytic
        # unilateral wall/neck/bottom contact rows. Native convex exterior
        # proxies remain for all other objects; World supplies the hollow cavity.
        self.inner_shapes = [sc.box(f'{name}_inner_reference',(0,0,self.z0/2),
            (.001,.001,self.z0/2),'collision_hidden',body=Bv,density=0.,mu=P.MU['glass'])]
        self._pts = self._pts[np.linalg.norm(self._pts[:,:2],axis=1) <= np.where(self._pts[:,2] < body_h,self.r_in,v["bore"]/2)]
        self.female = None
        self.adapter = bool(adapter)
        if adapter:
            a = P.ADAPTER
            T.cylinder(sc, Bv, f"{name}_adapter", (0., 0., H_ + 0.5*a["height"] - 0.006), 0.5*a["od"] + 0.001, 0.5*a["height"] - 0.006, "z",
                       "adapter", density=a["mass"]/(math.pi*(0.5*a["od"])**2*a["height"]), mu=P.MU["pp"])
            T.cylinder(sc, Bv, f"{name}_adapter_port", (0., 0., H_ + a["height"] - 0.006), 0.0036, 0.006, "z", "adapter", density=900.,
                       mu=P.MU["pp"])
            self.female = Female(self, name + "_port", Bv, (0., 0., H_ + a["height"]), Rotation.from_euler("y", -90., degrees=True),
                                 spring=lambda dpt: P.CONNECTOR["spring0"] + P.CONNECTOR["spring_k"]*dpt)
            self.vessel.vented = True
        sc.bodies[name] = Bv
        sc.places[Bv] = name
        self.mass_glass = None
        self.touch_cap = []

    def bind(self, w):
        mjm = w.solver.mj_model
        self.mj = w._newton_to_mj[self.body]
        self.outer_geoms = {w.geom_of_shape(sh) for sh in (self.glass_shape,self.neck_shape,self.cap_shape)}
        for sh in self.inner_shapes:
            g=w.geom_of_shape(sh);mjm.geom_contype[g]=0;mjm.geom_conaffinity[g]=0
        self.mass0 = float(mjm.body_mass[self.mj])
        self.ipos0 = mjm.body_ipos[self.mj].copy()
        self.lump(w)

    def plane(self, w):
        """Height of the liquid's plane surface along 'up', for the vial's present attitude: (up in the vial's frame, h)."""
        p, R = T.world_pose(w, self.body)
        up = R.inv().apply([0., 0., 1.])
        hgt = self._pts @ up
        frac = min(max(self.vessel.liquid_volume/self.vessel.interior, 0.0), 1.0)
        k = int(round(frac*(len(hgt) - 1)))
        return up, float(np.partition(hgt, k)[k]) if frac > 0 else float(hgt.min()) - 1.0

    def contains(self, port_local):
        p=np.asarray(port_local);radius=self.r_in if p[2] < self.body_h else P.VIAL["bore"]/2
        return bool(self.z0 < p[2] < self.stopper_in and np.linalg.norm(p[:2]) < radius)

    def port_state(self, w, port_local):
        up, h = self.plane(w)
        z = float(np.asarray(port_local) @ up)
        self.vessel.wet = self.contains(port_local) and z < h
        self.vessel.head_pa = max(h - z, 0.0)*H.density(self.vessel.liquid)*P.GRAVITY if self.vessel.wet else 0.0
        return self.vessel.wet

    def path(self, w, syringe, female):
        """Through the adapter's spike: the port is at the inner face of the stopper."""
        depth = syringe.depth(w)
        opened = min(max((depth - P.CONNECTOR["open_from"])/(P.CONNECTOR["open_full"] - P.CONNECTOR["open_from"]), 0.0), 1.0)
        if opened <= 0.0:
            return H.Path(H.Closed(), ("adapter_closed", self.name))
        self.port_state(w, (0., 0., self.stopper_in - 0.002))
        path = H.Path(self.vessel, ("adapter", self.name, self.vessel.wet))
        path.add_tube(0.03, 0.0015, H.viscosity(syringe.chamber.stack.front_mix()) if not self.vessel.wet else H.viscosity(self.vessel.liquid),
                      kinetic=1.0)
        path.r_liquid += P.CONNECTOR["r_open"]/opened
        return path

    def lump(self, w):
        mjm = w.solver.mj_model
        m_l = float(self.vessel.liquid @ H._RHO)
        Mt = self.mass0 + m_l
        hz = 0.5*self.vessel.liquid_volume/(math.pi*self.r_in**2)
        set_body_mass(mjm, self.mj, Mt)
        mjm.body_ipos[self.mj] = (self.mass0*self.ipos0 + m_l*np.array([0., 0., self.z0 + hz]))/Mt

    def reading(self, w):
        p, R = T.world_pose(w, self.body)
        mix = self.vessel.liquid
        return dict(liquid_ml=self.vessel.liquid_volume/H.ML, gauge_kpa=(self.vessel.gas_pressure() - P.P_ATM)/1e3,
                    upright_cos=float(R.apply([0., 0., 1.])[2]), position=[float(v) for v in p],
                    contents_ml={sp: float(mix[i])/H.ML for i, sp in enumerate(P.SPECIES) if mix[i] > 0})


class VialSite(Penetrable):
    vial = None
    hard_name = "cap"
    lateral = P.TRACK_LATERAL["stopper"]
    tilt_limit = math.radians(25.0)        # the neck: a tube through the stopper can lean this far (authored from the 12.6 mm bore)

    def pivot_k(self, q):
        return 0.02

    def floor_force(self, depth):
        return 0.0

    def terminal_for(self, w, needle, tr):
        d = w.solver.mj_data
        depth = float(d.qpos[needle.a_b])*tr["cos"]
        if depth < P.LAYERS["stopper"]["thick"]:
            return needle.terminal_closed, ("rubber", needle.name)
        tip, _ = needle.tip_world(w)
        local=self.to_local(w, tip)
        if not self.vial.contains(local):
            self.vial.vessel.wet=False
            return needle.terminal_closed, ("vial_wall_or_outside", needle.name, self.name)
        wet = self.vial.port_state(w, local)
        return self.vial.vessel, ("vial", needle.name, self.name, wet)


# ====================================================================== the intravenous lock
class IVLock:
    """Needle-free connector on a short extension with a slide clamp, a cannula, and the trainer's vein."""

    def __init__(self, sc, name, connector_pose, clamp_pose, cannula="g20", clamp_closed=True, blocked=False):
        self.sc, self.name = sc, name
        c, ex = P.CONNECTOR, P.EXTENSION
        b, wp, nt = sc.b, sc.wp, sc.newton
        self.pose0 = connector_pose
        # the connector: on the skin under a strip of tape (a six-way spring)
        self.body = Bc = T.link(sc, f"{name}_connector", connector_pose)
        # its collider is a capsule of the connector's length, drawn as the flat-faced cylinder it is: a cylinder held
        # between two flat pads gets one contact point as soon as it is a fraction of a degree off the pads' faces,
        # and the pinched connector turns in the grip until a pad's edge stands 5 mm into it at 80 N (as with the
        # barrel). A capsule against a box returns both ends of the line.
        rc = 0.5*c["od"]
        T.capsule(sc, Bc, f"{name}_connector_body", (-c["length"] + rc, 0., 0.), (-rc, 0., 0.), rc, "collision_hidden",
                  density=c["mass"]/(math.pi*rc*rc*(c["length"] - 2*rc) + 4.0/3.0*math.pi*rc**3), mu=P.MU["pp"])
        T.cylinder(sc, Bc, f"{name}_connector_look", (-0.5*c["length"], 0., 0.), rc, 0.5*c["length"], "x", "connector", collide=False)
        Jc = nt.ModelBuilder.JointDofConfig
        lin = [Jc(axis=a, target_ke=c["mount_k"], target_kd=c["mount_c"], armature=0.002, limit_lower=-0.05, limit_upper=0.05,
                  actuator_mode=nt.JointTargetMode.POSITION) for a in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))]
        ang = [Jc(axis=a, target_ke=c["mount_kr"], target_kd=c["mount_cr"], armature=2e-7, limit_lower=-1.0, limit_upper=1.0,
                  actuator_mode=nt.JointTargetMode.POSITION) for a in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))]
        p, R = connector_pose
        j = b.add_joint_d6(-1, Bc, linear_axes=lin, angular_axes=ang, parent_xform=wp.transform(tuple(map(float, p)), T.wq(R)),
                           child_xform=wp.transform_identity(), label=f"{name}_tape")
        b.add_articulation([j], label=f"{name}_connector_assembly")
        self.female = Female(self, name, Bc, (0., 0., 0.), spring=lambda d: c["spring0"] + c["spring_k"]*d)
        # the slide clamp on the extension
        self.clamp = Bk = T.link(sc, f"{name}_clamp", T.compose(clamp_pose, (ex["clamp_travel"] if clamp_closed else 0., 0., 0.)))
        sc.box(f"{name}_clamp_body", (0., 0., 0.), (0.007, 0.0045, 0.005), "clamp", body=Bk, density=1100., mu=P.MU["pp"])
        pc, Rc = clamp_pose
        jc = b.add_joint_prismatic(-1, Bk, parent_xform=wp.transform(tuple(map(float, pc)), T.wq(Rc)), child_xform=wp.transform_identity(),
                                   axis=(1., 0., 0.), limit_lower=0.0, limit_upper=ex["clamp_travel"], friction=ex["clamp_n"], armature=0.002,
                                   label=f"{name}_clamp_slide")
        b.add_articulation([jc], label=f"{name}_clamp_assembly")
        T_set_q(b, jc, ex["clamp_travel"] if clamp_closed else 0.0)
        self.j_clamp = T.JointRef(jc, "clamp")
        self.cannula = dict(P.CANNULAS[cannula])
        self.blocked = bool(blocked)
        vol = math.pi*ex["bore"]**2/4*ex["length"] + math.pi*self.cannula["bore"]**2/4*self.cannula["length"] + c["prime_ml"]*H.ML
        self.line = H.Stack([H.parcel(vol, "saline")])
        self.line_volume = vol
        self.vein = None
        self.node = H.Node(ex["compliance"])
        self.closed = H.Closed()
        self.peak_gauge = 0.0
        self.max_shift = 0.0
        sc.bodies[name] = Bc
        sc.places[Bc] = name
        sc.places[Bk] = name + "_clamp"

    def bind(self, w):
        mjm = w.solver.mj_model
        self.j_clamp.bind(w)
        self.d_k = int(mjm.jnt_dofadr[self.j_clamp.mj])
        self.a_k = int(mjm.jnt_qposadr[self.j_clamp.mj])
        mjm.dof_solref[self.d_k] = P.FIT_SOLREF
        mjm.dof_solimp[self.d_k] = P.FIT_SOLIMP
        w.keep_invweight(self.d_k, 1.0/(float(mjm.body_mass[w._newton_to_mj[self.clamp]]) + 0.002))
        self.vein = H.Reservoir(w.fluid.ledger, "vein", P.VEIN_GAUGE_PA)

    def clamp_open(self, w):
        return float(w.solver.mj_data.qpos[self.a_k]) < 0.5*P.EXTENSION["clamp_travel"]

    def path(self, w, syringe, female):
        c, ex = P.CONNECTOR, P.EXTENSION
        depth = syringe.depth(w)
        opened = min(max((depth - c["open_from"])/(c["open_full"] - c["open_from"]), 0.0), 1.0)
        if opened <= 0.0:
            return H.Path(self.closed, ("valve_closed", self.name))
        mu = self.line.mean_viscosity()
        if not self.clamp_open(w) or self.blocked:
            path = H.Path(self.node, ("line_closed", self.name))
            path.r_liquid += c["r_open"]/opened
            path.add_tube(0.4*ex["length"], ex["bore"], mu)
            return path
        path = H.Path(self.vein, ("line", self.name))
        path.r_liquid += c["r_open"]/opened
        path.add_tube(ex["length"], ex["bore"], mu)
        path.add_tube(self.cannula["length"], self.cannula["bore"], mu, kinetic=1.0)
        path.queue = self.line
        path.tag = self.name
        return path

    def post_step(self, w):
        p, _ = T.world_pose(w, self.body)
        self.max_shift = max(self.max_shift, float(np.linalg.norm(p - self.pose0[0])))
        m = self.female.mate
        if m is not None and m.sealed() and m.path is not None and m.path.terminal.kind != "closed":
            self.peak_gauge = max(self.peak_gauge, m.chamber.p - P.P_ATM)

    def reading(self, w):
        tot = self.line.total()
        return dict(clamp_open=self.clamp_open(w), clamp_mm=1e3*float(w.solver.mj_data.qpos[self.a_k]),
                    line_ml={sp: float(tot[i])/H.ML for i, sp in enumerate(P.SPECIES + ("air",)) if tot[i] > 1e-12},
                    line_segments=[[float(x)/H.ML for x in p] for p in self.line.items], peak_gauge_kpa=self.peak_gauge/1e3,
                    connector_shift_mm=1e3*self.max_shift, mate=self.female.mate.name if self.female.mate else None,
                    blocked=self.blocked)

    def snapshot(self):
        return dict(line=[p.copy() for p in self.line.items], node=(self.node.p, self.node.held.copy()), peak=self.peak_gauge,
                    shift=self.max_shift)

    def restore(self, w, s):
        self.line.items = [p.copy() for p in s["line"]]
        self.node.p, self.node.held = s["node"][0], s["node"][1].copy()
        self.peak_gauge, self.max_shift = s["peak"], s["shift"]


class Sink(H.Terminal):
    """Liquid that leaves into a named place at ambient pressure (the inside of a sheath)."""
    kind = "fixed"

    def __init__(self, ledger, place):
        self.ledger, self.place = ledger, place

    def give(self, mix, gas_at_atm, t):
        self.ledger.add(self.place, mix, t)


# ====================================================================== the manager
class Fluidics:
    def __init__(self, scene):
        self.scene = scene
        self.ledger = H.Ledger()
        self.spray = H.Spray(self.ledger)
        self.syringes, self.needles, self.vials = {}, {}, {}
        self.females, self.penetrables, self.pen_by_name = {}, [], {}
        self.locks, self.caps = {}, {}
        self._sinks = {}
        self.initial = None

    def sink(self, place):
        if place not in self._sinks:
            self._sinks[place] = Sink(self.ledger, place)
        return self._sinks[place]

    def add_female(self, f):
        self.females[f.name] = f

    def add_penetrable(self, pn):
        self.penetrables.append(pn)
        self.pen_by_name[pn.name] = pn

    def finish(self):
        """Declare every nozzle-to-fitting weld and every needle-to-target track (before the model is finalised)."""
        fem = list(self.females.values())
        for s in self.syringes.values():
            s.declare_mates(fem)
        for n in self.needles.values():
            n.declare_tracks(self.penetrables)

    def bind(self, w):
        import mujoco
        self._mujoco = mujoco
        for dev in (*self.needles.values(), *self.vials.values(), *self.locks.values(), *self.syringes.values()):
            dev.bind(w)
        self.spray.locate = lambda p0, p1, src: self._locate(w, p0, p1, src)
        self._tube_geoms = {}
        mjm = w.solver.mj_model
        for n in self.needles.values():
            self._tube_geoms[n.g_tube] = n
        self._tip_geoms = {s.g_tip: s for s in self.syringes.values()}
        self.initial = self.totals()

    def _locate(self, w, p0, p1, src):
        m, d = w.solver.mj_model, w.solver.mj_data
        seg = np.asarray(p1, float) - np.asarray(p0, float)
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            return None
        gid = np.zeros(1, np.int32)
        dist = self._mujoco.mj_ray(m, d, np.asarray(p0, float), seg/L, None, 1, int(src), gid)
        if dist < 0 or dist > L:
            return None
        g = int(gid[0])
        body = int(m.geom_bodyid[g])
        point = np.asarray(p0, float) + seg/L*dist
        nb = w._mj_to_newton.get(body, -1)
        place = self.scene.place_of(nb, int(w._geom_shape[g]) if g < len(w._geom_shape) else -1)
        Rm = d.xmat[body].reshape(3, 3)
        return point, place, nb, Rm.T @ (point - d.xpos[body])

    def pre_step(self, w):
        self._touches(w)
        for n in self.needles.values():
            n.pre_step(w)
        for s in self.syringes.values():
            s.pre_step(w)

    def post_step(self, w):
        for s in self.syringes.values():
            s.post_step(w)
        for lk in self.locks.values():
            lk.post_step(w)
        for n in self.needles.values():
            for dep in n.depots.values():
                if dep is not n.feeding:
                    dep.relax(w.dt)
            n.feeding = None
        self.spray.step(w.dt)
        if w.tick % 20 == 0:
            for v in self.vials.values():
                v.lump(w)

    def _touches(self, w):
        """What every bare needle tube and bare nozzle tip is touching (the last step's contacts)."""
        m, d = w.solver.mj_model, w.solver.mj_data
        if not d.ncon:
            return
        f6 = np.zeros(6)
        for i in range(d.ncon):
            g0, g1 = int(d.contact.geom[i][0]), int(d.contact.geom[i][1])
            for g, other in ((g0, g1), (g1, g0)):
                dev = self._tube_geoms.get(g) or self._tip_geoms.get(g)
                if dev is None:
                    continue
                nb = w._mj_to_newton.get(int(m.geom_bodyid[other]), -1)
                place = self.scene.place_of(nb, int(w._geom_shape[other]) if other < len(w._geom_shape) else -1)
                if other in w._glass_geoms:
                    place += ":inside"
                # inside the container's walls what a needle meets is the container: a point that lands at 2 m/s sinks
                # through its 3 mm floor in the soft contact and meets the counter top under it for a tick or two
                box = getattr(self.scene, "sharps_box", None)
                if box is not None and place == "counter" and all(box["lo"][k] <= d.contact.pos[i][k] <= box["hi"][k] for k in range(3)):
                    place = "sharps_inside"
                if place not in dev.touch:
                    dev.touch[place] = w.sim_time
                    w.log_event(dict(t=w.sim_time, event="touch", part=dev.name, place=place))
                if g in self._tube_geoms:
                    self._mujoco.mj_contactForce(m, d, i, f6)
                    at_point = float(np.linalg.norm(np.asarray(d.contact.pos[i]) - dev.tip_world(w)[0])) < 0.002
                    if at_point and abs(float(f6[0])) > P.NEEDLE_BLUNT_N and not dev.flags["blunted"]:
                        dev.flags["blunted"] = True
                        w.log_event(dict(t=w.sim_time, event="needle_blunted", needle=dev.name, place=place, force_n=abs(float(f6[0]))))

    # ---- accounting
    def totals(self):
        tot = np.zeros(H.N_SPECIES)
        gas = 0.0
        for s in self.syringes.values():
            tot += s.chamber.stack.total()
            gas += s.chamber.gas0
        for v in self.vials.values():
            tot += v.vessel.liquid
            gas += v.vessel.gas0
        for lk in self.locks.values():
            tot += lk.line.total() + lk.node.held
        for v in self.ledger.places.values():
            tot += v
        if len(self.spray.mix):
            tot += self.spray.mix.sum(0)
        for a, _ in self.spray.pending.values():
            tot += a
        return dict(liquid=tot[:H.AIR].copy(), air_in_lines=float(tot[H.AIR]), gas_in_chambers=gas)

    def conservation_error_ml(self):
        now, was = self.totals(), self.initial
        vein_blood = 0.0
        return float(np.abs(now["liquid"][:3] - was["liquid"][:3]).max())/H.ML

    def snapshot(self):
        return dict(ledger=self.ledger.snapshot(), spray=self.spray.snapshot(),
                    syringes={k: v.snapshot() for k, v in self.syringes.items()},
                    needles={k: v.snapshot() for k, v in self.needles.items()},
                    vials={k: v.vessel.snapshot() for k, v in self.vials.items()},
                    locks={k: v.snapshot() for k, v in self.locks.items()},
                    mates={k: (f.mate.name if f.mate else None) for k, f in self.females.items()})

    def restore(self, w, s):
        self.ledger.restore(s["ledger"])
        self.spray.restore(s["spray"])
        for f in self.females.values():
            f.mate = None
        for k, v in s["syringes"].items():
            self.syringes[k].restore(w, v)
        for k, v in s["needles"].items():
            self.needles[k].restore(w, v)
        for k, v in s["vials"].items():
            self.vials[k].vessel.restore(v)
        for k, v in s["locks"].items():
            self.locks[k].restore(w, v)
