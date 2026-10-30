"""Hold-down strap clamps over the pre-clamped workpieces of the bench.

A plain strap clamp as a machinist sets one: TE-CO plain clamp 30402 on a 3/8-16 stud, a spherical flange nut 41803 on
a spherical washer 42742, a steel heel block; forces from Carr Lane's strap-clamp table (params.HOLD_DOWN_*). Every
part that is drawn is a collider on its own body (only the taper of the strap's nose is drawn without one: the
strap's colliders keep the full height to the tip), and the workpiece is held by nothing but contact:

  stud        a 3/8 in rod standing in the bench top (a link fixed to the world), through the strap's slot
  heel block  a steel block of the stack's height on the bench (the same fixed link), under the strap's heel end
  strap       a FREE body: it lies on the work with its toe and on the heel block with its heel, the stud in its slot.
              Its two bearing lines are rounded (two contact spheres each, params.HOLD_DOWN_BEARING_M)
  nut         on the stud above the strap: a slide along the stud. The stud's and the strap's elasticity (E A/L in
              series with the strap's bending between toe and heel) is the spring that holds the nut to its place on the
              stud (a MuJoCo position actuator, the centre punch's pattern)
  washer      under the nut, on the strap's two rails either side of the slot: it rocks in the nut's spherical seat (a
              hinge across the strap on the stud's axis), so the stud's pull meets the strap on the stud's axis however
              the strap lies. The hinge stands in the strap's bearing plane, the level its heel turns about, so
              that the washer goes with the strap's top face while the strap settles (the friction under the washer
              then ends well inside its limit: 0.3-0.6 kN of 1.16 kN measured). The seat turns freely

The nut has no turn: a wrench-tightened flange nut cannot be undone by fingers and no task asks for it. Its tightening
is the start state: before hand-over the nut runs down the stud until the stud tension reads params.HOLD_DOWN_PRELOAD_N
(the registered nut torque through T = K F d), then its place on the stud is fixed for good. From then on the clamp is
passive: the stud tension presses the strap down, the strap shares it between the work and the heel block by its lever
arms, and the work is held by the friction of its faces under that load. Push or twist the work harder than that
friction and it slips; lift it against the strap and the stud tension rises with the spring.

Not modelled: the stud's bending. The rod is rigid on the bench, so the nut does not give sideways. A real 3/8 in
stud bends (3 E I/L^3: 2e6 N/m at the 39 mm of the panel stack, 8e4 N/m at the 114 mm of the timber stack), and across
the straps the real clamp on the tall stack holds less than this one: there the strap can swing with its stud. With
that bending built in (two slides with the cantilever spring), the timber stack leaned and drifted sideways (straps
0.5 mm in 8 s, the top board's lean changing sign): this engine's contacts let a stack of light parts rock far more
easily than wood does, and under 5.5 kN the stack needs the clamps' sideways hold to stand. Also not modelled: the
heel block's own footing (fixed on the bench); the nut's thread (no turn, no loosening by vibration); the spherical
seat's true centre (a seat radius above the washer), its second axis (it rocks about one axis, across the strap) and
its dry friction (mu F R_seat, 37 N m at the preload, which holds a real seat still: here the seat is free).
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import tools as T

WOODS = ("douglas_fir", "plywood")


class StrapClamp(T.Tool):
    """One strap clamp. Clamp frame: origin on the bench top at the stud's axis, x along the strap towards the work,
    z up. `work_top_z`: the height of the face the toe bears on (world); `work_material`: that part's material."""

    def __init__(self, sc, name, stud_xy, toward_xy, work_top_z, work_material):
        self.name = name
        b = sc.b
        pr = T.Parts(sc)
        L, W, H = P.HOLD_DOWN_STRAP_M
        slot_l, slot_w = P.HOLD_DOWN_SLOT_LEN_M, P.HOLD_DOWN_SLOT_W_M
        d = P.HOLD_DOWN_STUD_D_M
        nut_od, nut_h, fl_t, nut_af = P.HOLD_DOWN_NUT_M
        w_od, w_t, _ = P.HOLD_DOWN_WASHER_M
        heel_l, heel_w, heel_lap = P.HOLD_DOWN_HEEL_M
        rho = P.DENSITY["steel"]
        mu_ss = P.MU["steel_steel"]
        h = float(work_top_z) - P.BENCH_TOP_Z                       # the stack's height: the strap lies level on it
        yaw = math.atan2(float(toward_xy[1]), float(toward_xy[0]))
        base_pose = (np.array([float(stud_xy[0]), float(stud_xy[1]), P.BENCH_TOP_Z]), Rotation.from_euler("z", yaw))
        x_toe = P.HOLD_DOWN_STUD_OFFSET_M + P.HOLD_DOWN_TOE_OVERLAP_M      # the toe's tip, from the stud
        x_edge = x_toe - L + heel_lap                               # the heel block's edge under the strap
        # ---- stud and heel block: fixed on the bench
        Bs = T.link(sc, f"{name}_base", base_pose)
        stud_len = h + H + w_t + nut_h + 0.004                      # two threads proud of the nut
        pr.cyl(Bs, f"{name}_stud", (0., 0., 0.5*stud_len), 0.5*d, 0.5*stud_len, "z", "black_oxide", density=rho, mu=mu_ss)
        heel = sc.box(f"{name}_heel", (x_edge - 0.5*heel_l, 0., 0.5*h), (0.5*heel_l, 0.5*heel_w, 0.5*h), "heel_blue",
                      body=Bs, density=rho, mu=mu_ss)
        # ---- nut (slides on the stud) and washer (rocks in the nut's spherical seat, bears on the strap's rails). The
        # round washer is a square of the same area: a round rim on a box face is one contact point in MuJoCo
        Wp = T.compose(base_pose, (0., 0., h + H))
        Wa = T.link(sc, f"{name}_washer", Wp)
        s = 0.25*w_od*math.sqrt(math.pi)
        washer = sc.box(f"{name}_washer_pad", (0., 0., 0.5*w_t), (s, s, 0.5*w_t), "black_oxide", body=Wa, density=rho,
                        mu=mu_ss)
        Np = T.compose(base_pose, (0., 0., h + H + w_t))
        N = T.link(sc, f"{name}_nut", Np)
        pr.cyl(N, f"{name}_nut_flange", (0., 0., 0.5*fl_t), 0.5*nut_od, 0.5*fl_t, "z", "black_oxide", density=rho, mu=mu_ss)
        pr.cyl(N, f"{name}_nut_hex", (0., 0., fl_t + 0.5*(nut_h - fl_t)), 0.5*nut_af, 0.5*(nut_h - fl_t), "z",
               "black_oxide", density=rho, mu=mu_ss)
        # ---- strap: toe block, two rails beside the slot, tail block (strap frame: x towards the toe, origin at centre).
        # It bears on the work and on the heel block through rounded lines: two contact spheres at the toe's tip and
        # two under its heel, a little proud of the flat underside (params.HOLD_DOWN_BEARING_M); the boxes stand back
        # by that much and carry the strap's whole mass
        Sp = T.compose(base_pose, (x_toe - 0.5*L, 0., h + 0.5*H))
        S = T.link(sc, f"{name}_strap", Sp)
        e = 0.5*(L - slot_l)                                        # solid length at each end
        rw = 0.5*(W - slot_w)                                       # rail width
        r_b, proud = P.HOLD_DOWN_BEARING_M
        hz, zc, rho_s = 0.5*(H - proud), 0.5*proud, rho*H/(H - proud)
        toe_mu = P.MU["steel_wood"] if work_material in WOODS else mu_ss
        toe = sc.box(f"{name}_toe", (0.5*(L - e), 0., zc), (0.5*e, 0.5*W, hz), "black_oxide", body=S, density=rho_s,
                     mu=toe_mu)
        tail = sc.box(f"{name}_tail", (-0.5*(L - e), 0., zc), (0.5*e, 0.5*W, hz), "black_oxide", body=S, density=rho_s,
                      mu=mu_ss)
        rails = [sc.box(f"{name}_rail{k}", (0., sgn*0.5*(slot_w + rw), zc), (0.5*slot_l, 0.5*rw, hz), "black_oxide",
                        body=S, density=rho_s, mu=mu_ss) for k, sgn in enumerate((-1, 1))]
        x_heel = (x_edge - 0.5*heel_lap) - (x_toe - 0.5*L)          # the heel's bearing line, in the strap frame
        feet = [pr.sphere(S, f"{name}_{part}_bearing{k}", (x, sgn*(0.5*W - r_b), -0.5*H + r_b), r_b, "collision_hidden",
                          density=0., mu=mu)
                for part, x, mu in (("toe", 0.5*L - r_b, toe_mu), ("heel", x_heel, mu_ss)) for k, sgn in enumerate((-1, 1))]
        # hardened clamping faces (case-hardened strap, washer, steel heel block): the stiff steel contact of the vise
        # jaws (world.py), so that 5.8 kN does not sink a 0.27 kg strap a millimetre into its seats
        sc.stiff_shapes.update([heel, washer, toe, tail, *rails, *feet])
        sc.clamp_shapes.update([heel, washer, toe, tail, *rails, *feet])
        sc.bearing_shapes.update([heel, toe, tail, *feet])           # the faces the strap bears with (world.py)
        # ---- the spring: the stud in tension over its grip, in series with the strap bending between its bearing lines
        E = P.HOLD_DOWN_E_PA
        k_stud = E*P.HOLD_DOWN_STUD_AT_M2/(h + H + w_t)
        span = (x_toe - r_b) - (x_edge - 0.5*heel_lap)
        k_strap = 48.*E*((W - slot_w)*H**3/12.)/span**3
        self.k = 1./(1./k_stud + 1./k_strap)
        m_a = P.HOLD_DOWN_NUT_ARMATURE_KG
        self.c = 2.*math.sqrt(self.k*m_a)                           # critical on the slide's inertia
        I = sc.wp.transform_identity()
        lim = P.HOLD_DOWN_SEAT_LIMIT_RAD
        jb = b.add_joint_fixed(-1, Bs, parent_xform=sc.wp.transform(tuple(map(float, base_pose[0])), T.wq(base_pose[1])),
                               child_xform=I, label=f"{name}_in_bench")
        jn = b.add_joint_prismatic(Bs, N, parent_xform=T.rel_xform(sc, base_pose, Np), child_xform=I, axis=(0., 0., 1.),
                                   limit_lower=-(H + 0.005), limit_upper=0.004, target_pos=0., target_ke=self.k,
                                   target_kd=self.c, armature=m_a, effort_limit=10.*P.HOLD_DOWN_REC_TENSION_N,
                                   actuator_mode=sc.newton.JointTargetMode.POSITION, label=f"{name}_nut_slide")
        # the spherical seat, as a hinge across the strap on the stud's axis, in the strap's bearing plane (the strap's
        # thickness under the washer). The strap settles by turning about its heel's bearing line, which lies in that
        # plane, so a washer hinged there goes with the strap's top face, and the stud's pull passes through the axis
        # (measured: centre of pressure under the washer 0.2-1.7 mm from the axis; friction under it 0.3-0.6 kN of the
        # 1.16 kN it could carry; both stacks at rest below 0.002 mm/s). Two other places were tried.
        # At the seat's centre, a seat radius above the washer, the washer had to slide 0.3 mm on
        # the strap as the strap tilted; friction held it and the nut bore 7 mm off the axis. In the washer's own
        # bearing face, with flat bearing faces at toe and heel, the strap's top slid under the washer (thickness x
        # tilt), the friction there ended on its limit (1161 N of 0.2 x 5805 N) and the clamp relaxed in jolts (three
        # in 6 s at rest, each 0.5 mm/s for 3 ms, the stud tension 3 N lower after each)
        jw = b.add_joint_revolute(N, Wa, parent_xform=sc.wp.transform((0., 0., -(w_t + H)), sc.wp.quat_identity()),
                                  child_xform=sc.wp.transform((0., 0., -H), sc.wp.quat_identity()), axis=(0., 1., 0.),
                                  limit_lower=-lim, limit_upper=lim, label=f"{name}_seat")
        b.add_articulation([jb, jn, jw], label=f"{name}_assembly")
        js = b.add_joint_free(S, label=f"{name}_strap_free")
        b.add_articulation([js], label=f"{name}_strap_assembly")
        sc.mutual_filter([Bs, N, Wa])                               # nut and washer ride on the stud they sit on
        self.bodies = [Bs, N, Wa, S]
        self.base, self.nut, self.washer, self.body = Bs, N, Wa, S  # `body`: the strap (scene.bodies[name])
        self.shapes = dict(heel=heel, washer=washer, toe=toe, tail=tail, rail0=rails[0], rail1=rails[1])
        self.j = dict(nut=T.JointRef(jn, "nut"), seat=T.JointRef(jw, "seat"))
        x_tb, x_hb = x_toe - r_b, x_edge - 0.5*heel_lap             # the two bearing lines, from the stud
        self.geometry = dict(stack_height_m=h, toe_tip_from_stud_m=x_toe, heel_edge_from_stud_m=x_edge,
                             toe_bearing_from_stud_m=x_tb, heel_bearing_from_stud_m=x_hb,
                             lever_share=-x_hb/(x_tb - x_hb), k_stud_n_m=k_stud, k_strap_n_m=k_strap, k_n_m=self.k,
                             toe_mu=toe_mu, work_material=work_material)
        self.s_target = 0.               # the nut's place on the stud (slide coordinate, m; negative = run down)
        self.s_set = False               # tightened: the nut's place is fixed
        self.s_ticks, self.s_quiet = 0, 0
        self.s_tension_n = 0.            # stud tension from last tick's state (N)
        self.s_set_n = None              # the tension the tightening ended with
        sc.tool_bodies.update(self.bodies)

    def joints(self):
        return list(self.j.values())

    def tension(self, w):
        """Stud tension (N): the spring's pull on the nut towards the bench at the nut's present place."""
        return self.k*(self.j["nut"].pos(w) - self.s_target)

    def pre_step(self, w):
        F = self.tension(w)
        self.s_tension_n = F
        if not self.s_set:
            # start-state preparation: the nut is run down until the stud tension reads the preload, then left
            g = P.HOLD_DOWN_TIGHTEN
            self.s_ticks += 1
            err = P.HOLD_DOWN_PRELOAD_N - F
            v = float(np.clip(g["gain_per_s"]*err/self.k, -g["max_speed_m_s"], g["max_speed_m_s"]))
            self.s_target -= v*w.dt
            quiet = abs(err) < g["tol"]*P.HOLD_DOWN_PRELOAD_N
            self.s_quiet = self.s_quiet + 1 if quiet else 0
            if self.s_quiet >= g["hold_ticks"] or self.s_ticks*w.dt >= g["limit_s"]:
                self.s_set = True
                self.s_set_n = float(F)
                w.log_event(dict(type="clamp_tightened", clamp=self.name, stud_tension_n=round(float(F), 1),
                                 after_s=round(self.s_ticks*w.dt, 3), converged=bool(self.s_quiet >= g["hold_ticks"])))
        w.tq[self.j["nut"].q] = self.s_target

    def contacts(self, w):
        """What presses on the strap now (owner diagnostics, E21): per contact partner the normal force (N), the
        friction force (N, in the bench plane), the centre of pressure along the strap from the stud (m, towards the
        toe), the friction coefficient the engine uses there and the deepest penetration. Partners: 'nut' (its washer),
        'heel', 'stud', and the name of any other body (the work)."""
        mujoco = w._mujoco
        m, d = w.solver.mj_model, w.solver.mj_data
        lab = w.model.body_label
        q = w.bq[self.base]
        ax = Rotation.from_quat(q[3:]).apply([1., 0., 0.])
        heel_shape = self.shapes["heel"]
        out = {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b0, b1 = (w._mj_to_newton.get(int(m.geom_bodyid[g]), -1) for g in c.geom)
            if self.body not in (b0, b1):
                continue
            other = b1 if b0 == self.body else b0
            if other in (self.nut, self.washer):
                key = "nut"
            elif other == self.base:
                key = "heel" if int(w._geom_shape[int(c.geom[1] if b0 == self.body else c.geom[0])]) == heel_shape else "stud"
            else:
                key = str(lab[other]) if other >= 0 else "world"
            mujoco.mj_contactForce(m, d, i, f6)
            F_w = np.asarray(c.frame).reshape(3, 3).T @ f6[:3]  # world frame, on the contact's second geom
            e = out.setdefault(key, dict(normal_n=0., force_w=np.zeros(3), moment=0., mu=float(c.friction[0]),
                                         points=0, max_depth_mm=0.))
            e["normal_n"] += abs(float(f6[0]))
            e["force_w"] += F_w if b1 == self.body else -F_w    # the force on the strap
            e["moment"] += abs(float(f6[0]))*float((np.asarray(c.pos) - q[:3]) @ ax)
            e["points"] += 1
            e["max_depth_mm"] = max(e["max_depth_mm"], -float(c.dist)*1e3)
        for e in out.values():
            fw = e.pop("force_w")
            mo = e.pop("moment")
            e["friction_n"] = float(np.linalg.norm(fw[:2]))
            e["centre_from_stud_m"] = mo/e["normal_n"] if e["normal_n"] > 0. else None
        return out

    def state(self):
        w = self.w
        return dict(stud_tension_n=round(float(self.tension(w)), 1), tightened=bool(self.s_set),
                    preload_n=round(P.HOLD_DOWN_PRELOAD_N, 1), nut_m=round(self.j["nut"].pos(w), 6))


def hold_down(sc, spec):
    """The two strap clamps over a pre-clamped part (layout: `clamped=True`), one at each end along the part's y. The
    part and whatever lies under it are free bodies. Returns the clamps."""
    p, hh = np.asarray(spec["xyz"], float), np.asarray(spec["half"], float)
    out = []
    for k, sgn in enumerate((-1, 1)):
        y_stud = p[1] + sgn*(hh[1] + P.HOLD_DOWN_STUD_OFFSET_M)
        out.append(sc.add_tool(StrapClamp(sc, f"{spec['name']}_clamp{k}", (p[0], y_stud), (0., -sgn), p[2] + hh[2],
                                          spec["material"])))
    return out


def stiffen_stack(sc, spec):
    """The wooden parts of a clamped stack (the clamped part `spec` and what lies under it) get the stiff contact of
    clamping faces (scene.stiff_shapes: world.py), when params.HOLD_DOWN_STACK_STIFF is set. Call when every part of
    the stack is built. The steel sheet keeps the metal workpieces' own contact (scene.stiff_solids). The reason and
    the measurements are in the parameter's basis. Returns the names of the parts stiffened."""
    if not P.HOLD_DOWN_STACK_STIFF:
        return []
    p, hh = np.asarray(spec["xyz"], float), np.asarray(spec["half"], float)
    labels = list(sc.b.shape_label)
    parts = {q["name"]: q for q in sc.layout.get("parts", []) if "half" in q}
    out = []
    for name, d in sc.drillables.items():
        part = parts.get(name)
        if part is None or d.material not in WOODS:
            continue
        c, h = np.asarray(part["xyz"], float), np.asarray(part["half"], float)
        under = c[2] <= p[2] + 1e-9 and abs(c[0] - p[0]) < h[0] + hh[0] and abs(c[1] - p[1]) < h[1] + hh[1]
        if under and f"{name}_solid" in labels:
            sc.stiff_shapes.add(labels.index(f"{name}_solid"))
            sc.stack_shapes.add(labels.index(f"{name}_solid"))
            out.append(name)
    return out
