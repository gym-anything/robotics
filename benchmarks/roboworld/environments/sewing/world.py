"""Coupled world of the sewing cell: two arms with Franka Hands (Newton 1.5 -> MuJoCo on the CPU) and the cloth, the
lockstitch machine and its presser foot (sewing/cloth.py and sewing/machine.py on the GPU), on one 0.5 ms clock.

Per tick:
  1. the arms' motion generator advances the joint targets (rate- and acceleration-limited, as the controllers do);
  2. MuJoCo steps the arms with the cloth's reactions of the previous tick on the finger and hand bodies;
  3. the Franka Hands' finger boxes and housing are posed as kinematic shapes in the cloth solver (their pose at the
     tick's start and end, so a moving finger carries cloth by friction);
  4. the machine advances its shaft (control box), the cloth solver takes its implicit step (contacts, friction, the
     foot body, needle holes and threads inside it) and the machine reads punctures, loops and stitches;
  5. the cloth's forces on each finger/hand shape are summed into a wrench on its body, low-pass filtered with
     REACTION_TAU (1 ms: a numerical damper of this explicit coupling) and applied on the next MuJoCo step.
Robot-machine contact (fingers on the head, the needle bar, the foot shank) is MuJoCo's: those parts are static
colliders in scene.py. The machine is commanded only through `machine_io` (the control box's signals: pedal speed,
needle stop, presser lift, touch-back reverse, trimmer), which the GAR backend exposes as the robot cell's tool I/O.
"""

import copy
import math
import time
from dataclasses import dataclass, asdict

import numpy as np

from . import layout as LY
from . import params as P
from .bench import place_rect_ply
from .cloth import ClothSolver, SHAPE_BOX, KIND_KINEMATIC
from .machine import Machine, MachineSettings
from .mesh import rect_panel, rounded_panel, mesh_panel
from .scene import Scene, HAND_HOUSING

REACTION_TAU = P.REACTION_TAU  # s, filter on cloth -> rigid reactions (numerical damper)
REFLEX_TORQUE_NM = (10., 20.)  # libfranka default collision thresholds (nominal, accelerating) [read]
REFLEX_RECOVERY_S = 1.
FRANKA_HAND_FINGER_SPEED = .05  # m/s per finger [read: Franka Hand datasheet]
CONTACT_TIMECONST = .004       # s, MuJoCo contact time constant of the cell's rigid contacts (as welding)


@dataclass
class Config:
    preset: str = "sample-room"
    task: str = "plain-seam"
    seed: int = 0
    robot: str = None
    device: str = "cuda:0"
    dt: float = P.DT                       # one clock for arms, cloth and machine (0.5 ms)
    solver_iterations: int = 50
    contact_capacity: int = 2000
    fabric: str = "muslin_125"
    plies: int = 2
    ply_width: float = 0.120               # across the seam (m); the seam edge is the ply's right edge
    ply_length: float = 0.300              # along the seam (m)
    seam_allowance: float = LY.SEAM_ALLOWANCE
    corner_radius: float = 0.0             # > 0: the plies' near-right corner rounded (task curve, A6)
    start: str = "under_foot"              # under_foot | in_front | tray
    ply_offset: tuple = (0., 0., 0.)       # (dx m, dy m, yaw deg) of the stack (variants.py seeds)
    misalign: float = 0.                   # top ply shifted across the seam (m) (variants.py seeds)
    stitch_length: float = 2.5e-3
    presser_force: float = P.PRESSER_F
    needle_nm: int = 80
    needle_tension: float = P.NEEDLE_TENSION_REF
    bobbin_tension: float = P.BOBBIN_TENSION
    cloth: bool = True                     # False: rigid cell only (layout and reach probes)


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0; found {newton.__version__}")
    newton.use_coord_layout_targets = True
    return newton, wp


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz])


def _R(q):
    x, y, z, w = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def drape_over_front_edge(pos, y_edge, radius=0.004):
    """Lay the part of a flat ply that lies beyond y_edge (where the table's rounded front edge begins) over the edge,
    as cloth laid on a table falls: a bend of `radius` (the edge radius plus the ply's height) around the edge, then
    hanging straight down (arc length preserved). Without it the overhang starts in mid air and is still falling at
    tick 0 (E14: 229 mm in the first 0.25 s)."""
    out = pos.copy()
    s = y_edge - pos[:, 1]
    m = s > 0
    z0 = pos[m, 2]
    a = np.pi/2*radius
    bend = m.copy()
    bend[m] = s[m] <= a
    hang = m & ~bend
    phi = s[bend]/radius
    out[bend, 1] = y_edge - radius*np.sin(phi)
    out[bend, 2] = pos[bend, 2] - radius + radius*np.cos(phi)
    out[hang, 1] = y_edge - radius
    out[hang, 2] = pos[hang, 2] - radius - (s[hang] - a)
    return out


class World:
    def __init__(self, config=None):
        self.config = c = config or Config()
        newton, wp = self.newton, self.wp = require_engine()
        self.scene = Scene(c, newton, wp)
        b = self.scene.b
        with wp.ScopedDevice("cpu"):
            self.model = b.finalize(device="cpu")
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            ref = self.model.mujoco.solref.numpy()
            ref[:] = (CONTACT_TIMECONST, 1.)
            mass = self.model.body_mass.numpy()
            for i, body in enumerate(self.model.shape_body.numpy()):
                if body >= 0 and "finger" in self.model.body_label[body]:
                    ref[i] = (.005*math.sqrt(float(mass[body])/.0291), 1.)
            self.model.mujoco.solref.assign(ref)
        self.rig = self.scene.rig_pending.bind(self.model)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=c.contact_capacity, broad_phase="sap",
                                                 reduce_contacts=True)
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                                                  enable_multiccd=True, integrator="implicitfast", cone="elliptic",
                                                  solver="newton", iterations=c.solver_iterations, ls_iterations=20,
                                                  impratio=10., nconmax=c.contact_capacity,
                                                  njmax=c.contact_capacity*3)
        self.state, self.next_state = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.joints = dict(self.rig.joints)
        self.arms = tuple(self.rig.instances)
        self.g1 = bool(getattr(self.scene, "g1", False))
        # the hands the cloth couples to: the Franka Hand's arm, or the G1's two Dex3 sides
        self.hand_arms = ("left", "right") if self.g1 else \
            tuple(a for a in self.arms if self.scene.arm_hw[a].get("hand"))
        mj, m = self.solver._mujoco, self.solver.mj_model
        g2s = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
        sb = self.model.shape_body.numpy()
        self._g2s = g2s
        self._g2b = np.where(g2s >= 0, sb[np.clip(g2s, 0, None)], -1)
        # MuJoCo's midphase culls geom pairs with per-body bounding trees built when the model is compiled; the robot
        # adapter places the robots' collision geoms afterwards, up to 22 mm outside their trees here (finger boxes
        # and links, checked per body; welding E19 found 4-596 mm and a hand closing through a part), so arm-table and
        # arm-head contacts could be dropped without a trace. The midphase is a culling optimisation only: off, every
        # body pair goes to the exact narrowphase.
        m.opt.disableflags = int(m.opt.disableflags) | int(mj.mjtDisableBit.mjDSBL_MIDPHASE)
        names = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j): int(m.jnt_dofadr[j]) for j in range(m.njnt)}
        self._arm_mj_dofs = {}
        self._ext_cache = (-1, None)
        for name in ([] if self.g1 else self.joints):      # the G1 has no libfranka collision reflex (g1.py)
            if "finger" in name:
                continue
            arm, jn = name.split("/", 1)
            hit = [dof for mn, dof in names.items() if mn and mn.startswith(arm + "_") and mn.endswith("_" + jn)]
            if len(hit) != 1:
                raise RuntimeError(f"no unique MuJoCo joint for {name}: {hit}")
            self._arm_mj_dofs.setdefault(arm, []).append((name, hit[0]))
        self._robot_bodies = {i for i, lab in enumerate(self.model.body_label)
                              if any(lab.startswith(a + "/") for a in self.arms)}
        if self.g1:                    # the ready posture over the profile's standing pose (g1.READY)
            from . import g1 as G1
            q, t = self.model.joint_q.numpy(), self.control.joint_target_q.numpy()
            for key, v in G1.READY.items():
                q[self.joints[key]["q"]] = t[self.joints[key]["q"]] = v
            self.model.joint_q.assign(q)
            self.control.joint_target_q.assign(t)
        self.state.joint_q.assign(self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.targets = self.control.joint_target_q.numpy().copy()
        self._cmd_v = np.zeros(len(self.targets))
        self._finger_speed = {j: FRANKA_HAND_FINGER_SPEED for a in self.hand_arms for j in self.gripper_joints(a)}
        self._accel = self._joint_accel_limits()
        self.grip_force = {a: 0. for a in self.hand_arms}
        ke = self.model.joint_target_ke.numpy()
        self._finger_kp = {j: float(ke[self.joints[j]["dof"]]) for a in self.hand_arms for j in self.gripper_joints(a)}
        self.finger_squeeze = {}
        self._cmd_close = {}             # force mode: the finger's commanded position while closing past its target
        self.tick, self.fault = 0, None
        self.events, self.reflexes = [], []
        self._reflex_until = {}
        self.refused_targets = 0
        self.timing = dict(rigid_s=0., cloth_s=0., coupling_s=0.)
        self.assist = None               # ASSISTED fixtures (demos and probes): callable(world) per tick
        self.assist_log = []
        if self.g1:
            from .g1 import G1HandIK
            self.kin = G1HandIK(self)
            self._g1_setup()
        else:
            from .kinematics import Kinematics
            self.kin = Kinematics(self)
        self._mj_bad = self._mj_warnings()
        self.body_wrench = np.zeros((self.model.body_count, 6), np.float32)
        self._wrench_filtered = np.zeros((self.model.body_count, 6))
        self.cloth = self.machine = None
        if c.cloth:
            self._build_cloth()

    # ------------------------------------------------------------------ cloth, machine, pads
    def _build_cloth(self):
        c = self.config
        f = dict(P.FABRICS[c.fabric])
        f["grain_deg"] = 0.0
        self.fabric = f
        self.cloth = cl = P.solver_attrs(ClothSolver(device=c.device, dt=c.dt, **P.solver_kwargs()))
        self._unresolved_seen = 0.0           # the solver's count of unconverged ticks already checked (_check_solve)
        self.machine = M = Machine(MachineSettings(stitch_length=c.stitch_length, presser_force=c.presser_force,
                                                   needle_nm=c.needle_nm, needle_tension=c.needle_tension,
                                                   bobbin_tension=c.bobbin_tension))
        # the fine band follows every seam the task sews: the pivot's second leg runs along the near edge (v = 10 mm),
        # which the rectangle's band (the right edge) alone leaves at the 7.5 mm bulk size, 2.5 mm stitches in 7.5 mm
        # triangles (leg 2 drifts and stalls)
        near = [np.array([[0.0, 0.0], [c.ply_width, 0.0]]), np.array([[0.0, c.seam_allowance],
                                                                       [c.ply_width, c.seam_allowance]])]
        spec = (rounded_panel("ply", c.ply_width, c.ply_length, c.corner_radius, seam_offset=c.seam_allowance,
                              band=P.FINE_BAND)
                if c.corner_radius > 0 else rect_panel("ply", c.ply_width, c.ply_length, seam_offset=c.seam_allowance,
                                                       extra_curves=near if c.task == "pivot" else (), band=P.FINE_BAND))
        self.mesh = mesh_panel(spec)
        th_free = max(f["thickness"], f.get("thickness_free", f["thickness"]))
        dx, dy, yaw = c.ply_offset
        if c.start == "under_foot":
            # laid and the foot lowered on them (the operator holds the plies until the foot is down: with the foot
            # lifted for the settle, the part hanging over the table's front edge pulled a 420 mm stack off the table)
            right, near = c.seam_allowance, LY.LEAD - c.ply_length
            M.presser = "down"
        elif c.start == "in_front":
            right, near = c.seam_allowance, -0.020 - c.ply_length
            M.presser = "lift2"
        elif c.start == "tray":
            right, near = -0.200, -0.170
            M.presser = "down"
        else:
            raise ValueError(f"unknown start {c.start!r}")
        z0 = max(0.0, M.dog(M.theta)[0]) + 0.05e-3
        Ry = np.array([[math.cos(math.radians(yaw)), -math.sin(math.radians(yaw)), 0.],
                       [math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0.], [0., 0., 1.]])
        self.ply_ids = []
        for k in range(c.plies):
            origin, R = place_rect_ply(c.ply_width, c.ply_length, right + dx + (c.misalign if k == c.plies - 1 else 0.),
                                       near + dy, z0 + 0.5*th_free + k*th_free + 1e-5)
            pivot = np.array([right + dx, near + dy + c.ply_length/2, 0.0])
            origin = pivot + Ry @ (origin - pivot)
            X = self.mesh["X"]
            pos = origin[None] + np.column_stack([X, np.zeros(len(X))]) @ (Ry @ R).T
            # concentric with the table's rounded front edge (centre at the edge radius behind and below the edge)
            pos = drape_over_front_edge(pos, LY.TABLE["y"][0] + P.TABLE_EDGE_RADIUS,
                                        radius=P.TABLE_EDGE_RADIUS + z0 + (k + 0.5)*th_free + 1e-5)
            self.ply_ids.append(cl.add_ply(f"ply{k}", X, self.mesh["tri"], origin, Ry @ R, f, mu=1.0, positions=pos))
        M.add_to_cloth(cl, f)
        self._add_pads()
        M.obstacles = [(i, cl.shapes[i]["name"]) for i, _, _, _ in self.pads]
        cl.build()
        for arm, (cb, hb) in self.hand_body.items():
            # the arm carries the hand's weight (MuJoCo has its mass); the spring carries only what touches the pads
            cl.set_particle_force(cb, (0., 0., P.HAND_MASS*9.81))
        for fb in self.finger_body.values():
            cl.set_particle_force(fb["cb"], (0., 0., P.FINGER_PAD_MASS*9.81))
        M.after_build()
        self._place_pads(first=True)

    def _add_pads(self):
        """The hands as the cloth sees them: the URDF finger boxes and the housing, with the pad friction (catalog
        fingertip mu 0.8 [repo]). With HAND_COMPLIANT the shapes ride on bodies of the cloth solve:
        the Franka Hand's housing (and all of a Dex3 hand) on one body per hand that moves along the vertical, held to
        the hand's pose by the spring HAND_STIFFNESS; each Franka finger's boxes on a body of their own that moves
        along the vertical and along the finger's closing axis, held to the finger's pose by the arm's vertical
        compliance (half of HAND_STIFFNESS each) and the fingertip's own (FINGER_STIFFNESS). Otherwise they are
        kinematic, posed from the arm."""
        cl = self.cloth
        mu = self.fabric["mu_pad"]
        self.pads = []                     # (cloth shape, body, local p, local q)
        self.hand_body = {}                # arm -> cloth body particle (compliant hands)
        self.finger_body = {}              # MuJoCo finger body -> dict(cb, arm, axis (finger frame), ...)
        q0 = self.state.body_q.numpy().astype(float)
        for arm in self.hand_arms:
            hb = self.scene.hand_bodies.get(arm)
            cb = -1
            fingers = self.scene.finger_boxes.get(arm, [])
            ref = hb if hb is not None else (fingers[0][0] if fingers else None)     # an arm-fixed vertical reference
            if P.HAND_COMPLIANT and ref is not None:
                z0 = float(q0[ref, 2])
                cb = cl.add_body((0., 0., z0), P.HAND_MASS, free=(0, 0, 1))
                cl.set_spring(cb, P.HAND_STIFFNESS, (0., 0., z0))
                self.hand_body[arm] = (cb, ref)
            links = sorted({body for body, _, _, _ in fingers})
            own = P.HAND_COMPLIANT and not self.g1 and len(links) == 2
            if own:
                # the closing axis in each finger's frame: the line between the two finger bodies with the hand open
                d = q0[links[0], :3] - q0[links[1], :3]
                assert np.linalg.norm(d) > 1e-3, "the Franka Hand's fingers start apart"
                d = d/np.linalg.norm(d)
                for body in links:
                    fb = cl.add_body((0., 0., 0.), P.FINGER_PAD_MASS, free=(1, 1, 1))
                    self.finger_body[body] = dict(cb=fb, arm=arm, axis=_R(q0[body, 3:]).T @ d, hand=ref,
                                                  K=np.zeros((3, 3)), P=np.eye(3), a=d.copy())
            for body, p, q, half in fingers:
                i = cl.add_shape(f"{arm}_finger", SHAPE_BOX, half, (0., 0., 1.), kind=KIND_KINEMATIC, mu=mu,
                                 air=False, body=self.finger_body[body]["cb"] if own else cb)
                self.pads.append((i, body, np.asarray(p, float), np.asarray(q, float)))
            if hb is not None:
                hh = self.scene.hand_housing.get(arm, (HAND_HOUSING["center"], HAND_HOUSING["half"]))
                centre, half = hh[0], hh[1]
                qh = np.asarray(hh[2], float) if len(hh) > 2 else np.array([0., 0., 0., 1.])
                i = cl.add_shape(f"{arm}_hand", SHAPE_BOX, half, (0., 0., 1.), kind=KIND_KINEMATIC,
                                 mu=P.HAND_HOUSING_MU, air=False, body=cb)
                self.pads.append((i, hb, np.asarray(centre, float), qh))

    def _finger_springs(self, q):
        """Each finger body's free subspace and spring for this tick. The body's position is the pad's deflection from
        where the arm holds the finger: zero unloaded. It moves in the plane of the vertical and the closing axis a.
        Its compliance is the arm's along the vertical in series with the fingertip's along a (C = z z^T / k_arm +
        a a^T / k_finger; the spring is C's inverse on that plane), with the structural damping of the hand's (a
        damper over one tick is a spring c/dt toward the body's start-of-tick deflection, as for the hand body)."""
        cl = self.cloth
        k_arm, k_fin = 0.5*float(P.HAND_STIFFNESS), float(P.FINGER_STIFFNESS)
        ez = np.array([0., 0., 1.])
        for body, fb in self.finger_body.items():
            a = _R(q[body, 3:]) @ fb["axis"]
            C = np.outer(ez, ez)/k_arm + np.outer(a, a)/k_fin
            w_, V = np.linalg.eigh(C)
            keep = w_ > 1e-3*w_.max()            # one direction when the closing axis is itself vertical
            Pm = V[:, keep] @ V[:, keep].T
            d0 = cl.x[fb["cb"]:fb["cb"] + 1].numpy()[0].astype(float)
            K = np.zeros((3, 3))
            t = np.zeros(3)
            for lam, u in zip(w_[keep], V[:, keep].T):
                k = 1.0/lam
                kd = 2.0*float(P.HAND_DAMPING_RATIO)*math.sqrt(k*float(P.FINGER_PAD_MASS))/self.config.dt
                K += (k + kd)*np.outer(u, u)
                t += u*(kd*float(u @ d0)/(k + kd))
            fb["K"], fb["P"], fb["a"] = K, Pm, a
            cl.set_body_free(fb["cb"], Pm)
            cl.set_spring_matrix(fb["cb"], K, t)

    def _place_pads(self, first=False):
        q = self.state.body_q.numpy().astype(float)
        cl = self.cloth
        zt = {}
        for arm, (cb, hb) in self.hand_body.items():
            zt[cb] = float(q[hb, 2])
            if first:
                cl.set_spring_target(cb, (0., 0., zt[cb]))
                self._hand_zt_prev = {}
            else:
                # spring k plus damper c (zeta HAND_DAMPING_RATIO): over the tick the damper is a spring c/dt toward the
                # body's start-of-tick height moved with the target; both fold into one implicit spring
                k = float(P.HAND_STIFFNESS)
                c = 2.0*float(P.HAND_DAMPING_RATIO)*math.sqrt(k*float(P.HAND_MASS))
                kd = c/self.config.dt
                zb = float(cl.x[cb:cb + 1].numpy()[0][2])
                zp = self._hand_zt_prev.get(cb, zt[cb])
                t_eff = (k*zt[cb] + kd*(zb + zt[cb] - zp))/(k + kd)
                cl.set_spring_target(cb, (0., 0., t_eff), k=k + kd)
            self._hand_zt_prev[cb] = zt[cb]
        if self.finger_body:
            self._finger_springs(q)
        for i, body, pl, ql in self.pads:
            R = _R(q[body, 3:])
            p = q[body, :3] + R @ pl
            qq = _quat_mul(q[body, 3:], ql)
            b = cl.shapes[i]["body"]
            if b >= 0 and b in zt:
                p = p - np.array([0., 0., zt[b]])         # an offset from the hand body (which carries the vertical)
            # (a finger's body carries its deflection only: the shape's offset is the arm's pose of it)
            cl.set_shape_targets(i, p, qq)
            if first:
                cl._sp_host[i] = p
                cl._sq_host[i] = qq
        if first:
            cl.s_p0.assign(cl._sp_host)
            cl.s_p1.assign(cl._sp_host)
            cl.s_q0.assign(cl._sq_host)
            cl.s_q1.assign(cl._sq_host)

    def _pad_reactions(self):
        """Cloth forces on the pads -> wrenches about each body's COM for the next MuJoCo step. Contact forces are
        filtered (REACTION_TAU). Along the directions a pad's body is solved with the cloth, the arm feels that body's
        spring instead of the contact impulses: the hand body's along the vertical (filtered with the rest), a
        finger body's in its plane, unfiltered (the spring is the fingertip's compliance; MuJoCo integrates
        the finger against it explicitly, with no tick of delay)."""
        cl = self.cloth
        F, tau, _ = cl.all_reactions()
        q = self.state.body_q.numpy().astype(float)
        com = self.model.body_com.numpy().astype(float)
        raw = np.zeros_like(self._wrench_filtered)
        direct = np.zeros_like(self._wrench_filtered)
        sp = cl._sp_host.copy()
        fcb = {fb["cb"]: (body, fb) for body, fb in self.finger_body.items()}
        if self.hand_body or fcb:
            xb = cl.positions()
            for i, body, _, _ in self.pads:
                if cl.shapes[i]["body"] >= 0:
                    sp[i] = sp[i] + xb[cl.shapes[i]["body"]]          # a body's shape: its offset plus the body
        for i, body, _, _ in self.pads:
            f = F[i]
            if not np.any(f):
                continue
            pc = q[body, :3] + _R(q[body, 3:]) @ com[body]
            b = cl.shapes[i]["body"]
            if b in fcb:
                f = f - fcb[b][1]["P"] @ f                 # the in-plane part reaches the finger through its spring
            raw[body, :3] += f
            raw[body, 3:] += tau[i] + np.cross(sp[i] - pc, F[i])
        # a compliant hand: the arm feels the spring, not the pads' contact impulses, along the vertical
        for arm, (cb, hb) in self.hand_body.items():
            fz_contact = sum(F[i][2] for i, body, _, _ in self.pads if cl.shapes[i]["body"] == cb)
            raw[hb, 2] += float(cl.spring_force(cb)[2]) - fz_contact
        self.finger_squeeze = {}
        for cb, (body, fb) in fcb.items():
            fs = cl.spring_force(cb)
            direct[body, :3] += fs
            # the squeeze: the spring's force along the closing axis, positive when the pad is pushed back away from the
            # other finger (the finger squeezing what it holds)
            other = [b for b, f in self.finger_body.items() if f["arm"] == fb["arm"] and b != body]
            u = fb["a"]
            if other and float(np.dot(q[other[0], :3] - q[body, :3], u)) < 0.0:
                u = -u                                           # u: from this finger toward the other
            self.finger_squeeze[body] = float(-fs @ u)
        a = self.config.dt/(REACTION_TAU + self.config.dt)
        self._wrench_filtered += a*(raw - self._wrench_filtered)
        self.body_wrench[:] = (self._wrench_filtered + direct).astype(np.float32)

    # ------------------------------------------------------------------ helpers
    @property
    def time(self):
        return self.tick*self.config.dt

    def _g1_setup(self):
        """G1 upper-body servos: position PD plus a gravity feed-forward torque from the model at the commanded
        configuration, as Unitree's G1 manipulation stack sends it for the arms (xr_teleoperate robot_arm_ik.py:
        pinocchio rnea(q_target, 0, 0) -> motor_cmd.tau; web-sourced in barcode scanning). The waist gets
        the same (on the real robot the balance controller holds it; barcode scanning measured a 4.2 deg sag without
        it). Legs and Dex3 fingers are PD only."""
        import mujoco
        mjm = self.solver.mj_model
        names = {mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(mjm.njnt)}
        arm = [n for n in self.joints if any(k in n for k in ("_shoulder_", "_elbow_", "_wrist_", "/waist_"))]
        mj_j = [[j for k, j in names.items() if k and k.endswith("_" + n.split("/")[-1])][0] for n in arm]
        self._g1_ff = dict(names=arm, q=[self.joints[n]["q"] for n in arm], dof=[self.joints[n]["dof"] for n in arm],
                           mj_q=[int(mjm.jnt_qposadr[j]) for j in mj_j], mj_dof=[int(mjm.jnt_dofadr[j]) for j in mj_j],
                           limit=np.array([float(self.joints[n]["effort_limit"]) for n in arm]),
                           data=mujoco.MjData(mjm), tau=np.zeros(len(arm)),
                           every=max(1, int(round(.005/self.config.dt))))

    def _g1_gravity_feedforward(self, jf, cur):
        """Gravity torque of the G1 waist and arm joints at their commanded positions (the rest of the robot as
        measured), refreshed every 5 ms (Unitree's teleoperation stack streams it at 250 Hz), clipped to the URDF
        effort limits."""
        ff = self._g1_ff
        if self.tick % ff["every"] == 0:
            mujoco = self.solver._mujoco
            m, d = self.solver.mj_model, ff["data"]
            d.qpos[:] = self.solver.mj_data.qpos
            d.qvel[:] = 0.
            d.qpos[ff["mj_q"]] = cur[ff["q"]]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            out = np.zeros(m.nv)
            mujoco.mj_rne(m, d, 0, out)
            ff["tau"] = np.clip(out[ff["mj_dof"]], -ff["limit"], ff["limit"])
        jf[ff["dof"]] += ff["tau"]

    def _joint_accel_limits(self):
        """The motion generator's acceleration per commanded joint, by robot model (params JOINT_ACCEL_*): an
        arm's joints 1-7 in the catalog's order; every G1 joint (body and Dex3) the
        environment's ramp, the G1 having no motion generator. Fingers keep the Franka Hand's speed-limited closing."""
        out = {}
        for arm in self.arms:
            model = str(self.scene.arm_hw[arm].get("model", ""))
            lims = {"fr3": P.JOINT_ACCEL_FR3, "panda": P.JOINT_ACCEL_PANDA}.get(model)
            if lims is None:
                for name in self.joints:
                    if name.startswith(arm + "/") or self.g1:
                        out[name] = float(P.JOINT_ACCEL_G1)
                continue
            for k, j in enumerate(self.scene.arm_hw[arm]["arm_joints"]):
                out[f"{arm}/{j}"] = float(lims[min(k, len(lims) - 1)])
        return out

    def gripper_joints(self, arm):
        if self.g1:                    # the Dex3 hands are commanded joint by joint (named targets)
            return []
        return [f"{arm}/{j}" for j in self.scene.arm_hw[arm]["gripper_joints"]]

    def body_pose(self, body):
        q = self.state.body_q.numpy()[body].astype(float)
        return q[:3], _R(q[3:])

    def _mj_warnings(self):
        mj = self.solver._mujoco
        d = self.solver.mj_data
        W = mj.mjtWarning
        return sum(int(d.warning[int(k)].number) for k in (W.mjWARN_BADQACC, W.mjWARN_BADQPOS, W.mjWARN_BADQVEL))

    # ------------------------------------------------------------------ commands
    def set_targets(self, targets):
        latched = {arm for arm, until in self._reflex_until.items() if self.time < until}
        if latched:
            kept = {n: v for n, v in targets.items() if n.split("/", 1)[0] not in latched or "finger" in n}
            self.refused_targets += len(targets) - len(kept)
            targets = kept
        for name, value in targets.items():
            if name not in self.joints:
                raise ValueError(f"Not a robot actuator: {name}")
            spec = self.joints[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not spec["lower"] <= value <= spec["upper"]:
                raise ValueError(f"Invalid joint target {name}={value}")
        for name, value in targets.items():
            self.targets[self.joints[name]["q"]] = value

    def set_grip_force(self, arm, force):
        if self.g1:
            raise ValueError("the G1's Dex3 hands have no gripper force mode; command their joints by name")
        if arm not in self.hand_arms:
            raise ValueError(f"No hand on arm {arm}")
        limit = float(self.joints[self.gripper_joints(arm)[0]]["effort_limit"])
        if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) \
                or not 0 <= force <= limit:
            raise ValueError(f"Grip force must be in [0, {limit}] N per finger")
        self.grip_force[arm] = float(force)

    def set_joint_impedance(self, arm, stiffness=None, damping=None):
        """Joint impedance of one arm's seven joints: the joint servos become
        tau = K (q_d - q) - D dq, the arm's gravity compensated as before (the real FR3 and Panda compensate it in
        their internal controller). K per joint in [0, 14250] N m/rad, libfranka's Robot::setJointImpedance range;
        without D, D keeps the damping ratios of libfranka's joint impedance example (params JOINT_IMPEDANCE_*).
        stiffness None restores the catalog's position servo. Targets are commanded as before. The G1's joints are
        not offered (no sourced range for its low-level gains here)."""
        if self.g1:
            raise ValueError("joint impedance is offered for the FR3 and Panda arms only (the G1's low-level gains have "
                             "no sourced range in this cell)")
        if arm not in self._arm_mj_dofs:
            raise ValueError(f"No arm {arm!r}; arms: {', '.join(self._arm_mj_dofs)}")
        names = [n for n, _ in self._arm_mj_dofs[arm]]
        dofs = [self.joints[n]["dof"] for n in names]
        if not hasattr(self, "_servo_gains"):
            ke0, kd0 = self.model.joint_target_ke.numpy(), self.model.joint_target_kd.numpy()
            self._servo_gains = {a: (ke0[[self.joints[n]["dof"] for n, _ in ds]].copy(),
                                     kd0[[self.joints[n]["dof"] for n, _ in ds]].copy())
                                 for a, ds in self._arm_mj_dofs.items()}
            self.impedance = {}

        def vec(v, what, hi):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                v = [v]*len(names)
            if not isinstance(v, (list, tuple)) or len(v) != len(names) or \
                    any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                        or not 0 <= x <= hi for x in v):
                raise ValueError(f"{what} must be a number or {len(names)} numbers in [0, {hi:g}]")
            return np.array(v, float)
        if stiffness is None:
            if damping is not None:
                raise ValueError("damping needs a stiffness; omit both to restore the position servo")
            K, D = self._servo_gains[arm]
            self.impedance.pop(arm, None)
        else:
            K = vec(stiffness, "stiffness (N m/rad)", float(P.JOINT_IMPEDANCE_MAX))
            ex = P.JOINT_IMPEDANCE_EXAMPLE
            if damping is None:
                D = np.array(ex["D"])*np.sqrt(K/np.array(ex["K"]))
            else:
                D = vec(damping, "damping (N m s/rad)", float(max(ex["D"]))*math.sqrt(P.JOINT_IMPEDANCE_MAX/min(ex["K"])))
            self.impedance[arm] = dict(stiffness=[float(x) for x in K], damping=[float(x) for x in D])
        ke, kd = self.model.joint_target_ke.numpy(), self.model.joint_target_kd.numpy()
        ke[dofs], kd[dofs] = K, D
        self.model.joint_target_ke.assign(ke)
        self.model.joint_target_kd.assign(kd)
        self.solver.notify_model_changed(self.newton.ModelFlags.JOINT_DOF_PROPERTIES)
        # Newton 1.5 writes the new gains to its MuJoCo-Warp model only; the CPU MuJoCo this world steps keeps the old
        # ones unless they are copied over (a joint load deflected the arm the same at K 50 as at the servo's 600)
        sm = self.solver
        sm.mj_model.actuator_gainprm[:] = sm.mjw_model.actuator_gainprm.numpy()[0]
        sm.mj_model.actuator_biasprm[:] = sm.mjw_model.actuator_biasprm.numpy()[0]
        return self.impedance.get(arm)

    MACHINE_IO = ("speed_spm", "needle_stop", "presser", "reverse", "trim")

    def machine_io(self, **outputs):
        """The machine's control box (what its pedal, knee lifter, touch-back switch and trimmer would signal; an
        automated cell drives the same inputs through the control box's I/O). Validated by Machine.command."""
        if self.machine is None:
            raise ValueError("this world was built without the cloth and machine")
        unknown = set(outputs) - set(self.MACHINE_IO)
        if unknown:
            raise ValueError(f"unknown machine outputs {sorted(unknown)}; outputs: {', '.join(self.MACHINE_IO)}")
        self.machine.command(**outputs)
        self.events.append(dict(tick=self.tick, machine_io=dict(outputs)))

    # ------------------------------------------------------------------ protective stops (FR3 reflex, as welding)
    def external_joint_torques(self):
        m, d = self.solver.mj_model, self.solver.mj_data
        mj = self.solver._mujoco
        tau = np.zeros(m.nv)
        if d.nefc:
            f = np.array(d.efc_force[:d.nefc])
            f[np.asarray(d.efc_type[:d.nefc]) < int(mj.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0.
            if f.any():
                mj.mj_mulJacTVec(m, d, tau, f)
        return {arm: tau[[dof for _, dof in dofs]] for arm, dofs in self._arm_mj_dofs.items()}

    def external_torque_estimate(self):
        """Each arm joint's external torque (N m) from the contacts on the robot and the cloth's filtered wrenches on its
        hands, by joint name: the quantity libfranka reports to a controller as the estimated external torque
        (tau_ext_hat) and the collision reflex judges. Computed once per tick. The G1 has no entry (its joint interface
        has no such estimate in this cell, and no reflex)."""
        if self._ext_cache[0] != self.tick:
            ext = self.external_joint_torques()
            if self.cloth is not None:
                cj = self._cloth_joint_torques()
                ext = {a: ext[a] + cj[a] for a in ext}
            self._ext_cache = (self.tick, ext)
        ext = self._ext_cache[1]
        return {name: float(ext[arm][k]) for arm, dofs in self._arm_mj_dofs.items()
                for k, (name, _) in enumerate(dofs)}

    def _cloth_joint_torques(self):
        """External joint torques from the cloth's (filtered) wrenches on the hand bodies: J^T of each wrench."""
        out = {arm: np.zeros(len(dofs)) for arm, dofs in self._arm_mj_dofs.items()}
        wb = self._wrench_filtered
        if not np.any(wb):
            return out
        mj = self.solver._mujoco
        m, d = self.solver.mj_model, self.solver.mj_data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for body in np.nonzero(np.abs(wb).sum(1) > 0)[0]:
            mjb = self._newton_to_mj_body(int(body))
            if mjb is None:
                continue
            mj.mj_jacBodyCom(m, d, jacp, jacr, mjb)
            tau = jacp.T @ wb[body, :3] + jacr.T @ wb[body, 3:]
            for arm, dofs in self._arm_mj_dofs.items():
                out[arm] += tau[[dof for _, dof in dofs]]
        return out

    def _newton_to_mj_body(self, body):
        if not hasattr(self, "_nb2mj"):
            mj = self.solver._mujoco
            m = self.solver.mj_model
            self._nb2mj = {}
            for nb, lab in enumerate(self.model.body_label):
                name = lab.replace("/", "_")
                i = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)
                if i >= 0:
                    self._nb2mj[nb] = i
        return self._nb2mj.get(body)

    def _stop_arm(self, arm, q):
        cur = self.control.joint_target_q.numpy()
        for name, spec in self.joints.items():
            if name.startswith(arm + "/") and "finger" not in name:
                self.targets[spec["q"]] = cur[spec["q"]] = float(q[spec["q"]])
                self._cmd_v[spec["q"]] = 0.
        self.control.joint_target_q.assign(cur)

    def _protective_stops(self):
        if self.g1:                    # the G1's low-level joint interface has no collision or velocity reflex
            return
        q = self.state.joint_q.numpy()
        qd = self.state.joint_qd.numpy()
        ext = None
        for arm in self.arms:
            hit = None
            for name, spec in self.joints.items():
                if name.startswith(arm + "/") and "finger" not in name and \
                        abs(float(qd[spec["dof"]])) > 1.05*spec["velocity_limit"]:
                    hit = dict(kind="velocity", joint=name, speed_rad_s=float(qd[spec["dof"]]))
                    break
            if hit is None:
                if ext is None:
                    ext = self.external_joint_torques()
                    if self.cloth is not None:
                        cj = self._cloth_joint_torques()
                        ext = {a: ext[a] + cj[a] for a in ext}
                    self._ext_cache = (self.tick, ext)
                cur = self.control.joint_target_q.numpy()
                dofs = self._arm_mj_dofs[arm]
                moving = any(abs(float(self.targets[self.joints[n]["q"]] - cur[self.joints[n]["q"]])) > 1e-6
                             for n, _ in dofs)
                lim = REFLEX_TORQUE_NM[1] if moving else REFLEX_TORQUE_NM[0]
                e = np.abs(ext[arm])
                k = int(np.argmax(e))
                if e[k] > lim:
                    hit = dict(kind="collision", joint=dofs[k][0], external_torque_nm=float(e[k]), limit_nm=lim)
            if hit:
                self._reflex_until[arm] = self.time + REFLEX_RECOVERY_S
                self._stop_arm(arm, q)
                if not self.reflexes or self.reflexes[-1]["tick"] < self.tick - 50 or self.reflexes[-1]["arm"] != arm:
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, **hit))

    # ------------------------------------------------------------------ stepping
    def _motion_generator(self):
        c = self.config
        cur = self.control.joint_target_q.numpy()
        cur0 = cur.copy()
        dt = c.dt
        for name, spec in self.joints.items():
            j = spec["q"]
            vlim = spec["velocity_limit"]
            if name in self._finger_speed:
                vlim = min(vlim, self._finger_speed[name])
                e = float(self.targets[j] - cur[j])
                cur[j] += float(np.clip(e, -vlim*dt, vlim*dt))
                continue
            amax = self._accel[name]
            e = float(self.targets[j] - cur[j])
            v0 = self._cmd_v[j]
            v_stop = math.copysign(min(vlim, math.sqrt(2.*amax*abs(e))), e) if e else 0.
            v1 = min(max(v_stop, v0 - amax*dt), v0 + amax*dt)
            step = v1*dt
            if e == 0. or (e > 0. and step >= e) or (e < 0. and step <= e):
                cur[j] = self.targets[j]
                v1 = 0.
            else:
                cur[j] += step
            self._cmd_v[j] = v1
        jf = np.zeros(self.model.joint_dof_count, np.float32)
        q_now = self.state.joint_q.numpy()
        # a grip force (N per finger): the finger's commanded position closes at the hand's speed and is never more
        # than force / kp ahead of the finger, so its servo pushes with that force at most and with exactly that
        # force once the finger is stopped by what it holds: the Franka Hand's grasp (closes at `speed` until the
        # fingers stop, then applies `force`). Handed a bare force with no speed limit once it stalls, a finger closes
        # at 0.5 m/s at 20 N
        for arm, force in self.grip_force.items():
            if not force:
                continue
            for name in self.gripper_joints(arm):
                spec = self.joints[name]
                j = spec["q"]
                lead = float(force)/self._finger_kp[name]
                qn = float(q_now[j])
                if self.targets[j] < qn:
                    # closing: on past the commanded width (the fingers close until something stops them, then
                    # squeeze), never more than `lead` ahead of the finger
                    cur[j] = max(self._cmd_close.get(j, float(cur0[j])) - FRANKA_HAND_FINGER_SPEED*dt, qn - lead)
                else:
                    cur[j] = min(max(cur[j], qn - lead), qn + lead)
                self._cmd_close[j] = cur[j]
        self.control.joint_target_q.assign(cur)
        if self.g1:
            self._g1_gravity_feedforward(jf, cur)
        self.control.joint_f.assign(jf)

    def step(self):
        if self.fault:
            raise RuntimeError(self.fault)
        c = self.config
        try:
            t0 = time.perf_counter()
            if self.assist is not None:
                self.assist(self)
            self._motion_generator()
            self.state.clear_forces()
            self.state.body_f.assign(self.body_wrench)
            self.solver.step(self.state, self.next_state, self.control, self.contacts, c.dt)
            bad = self._mj_warnings()
            if bad > self._mj_bad:
                raise FloatingPointError(f"MuJoCo reported an unstable state at tick {self.tick}")
            self.state, self.next_state = self.next_state, self.state
            for attr in ("body_q", "body_qd", "joint_q"):
                if not np.isfinite(getattr(self.state, attr).numpy()).all():
                    raise FloatingPointError(f"Non-finite rigid {attr} at tick {self.tick}")
            if self._finger_kp:
                # a gripper finger outside its travel is not a state of a real hand
                jq = self.state.joint_q.numpy()
                for name in self._finger_kp:
                    spec = self.joints[name]
                    v = float(jq[spec["q"]])
                    if not spec["lower"] - P.FINGER_RANGE_FAULT <= v <= spec["upper"] + P.FINGER_RANGE_FAULT:
                        raise FloatingPointError(f"Gripper finger {name} left its travel ({v*1e3:.1f} mm; "
                                                 f"{spec['lower']*1e3:.0f} to {spec['upper']*1e3:.0f} mm) at tick "
                                                 f"{self.tick}")
            t1 = time.perf_counter()
            self.timing["rigid_s"] += t1 - t0
            if self.cloth is not None:
                self._place_pads()
                self.machine.pre_tick(c.dt)
                self.cloth.step()
                self.machine.post_tick(self.tick + 1)
                t2 = time.perf_counter()
                self.timing["cloth_s"] += t2 - t1
                self._pad_reactions()
                self._check_solve()
                if self.tick % 20 == 0:
                    self._check_cloth()
                self.timing["coupling_s"] += time.perf_counter() - t2
            self.tick += 1
            self._protective_stops()
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def _check_solve(self):
        """A cloth step the solve did not resolve ends the episode: the tick's Newton loop ended at its cap
        with its last step above UNRESOLVED_STEP_FAULT tolerances (1 m/s of velocity error at 1e4). The motion it
        leaves is not the world's: at the machine's top speed with nobody guiding, such steps grew for ~600 ticks into
        a run-away the velocity guard caught only at 278 m/s."""
        c = self.cloth.newton_ctl.numpy()
        if c[4] < self._unresolved_seen:      # the counters were restored with a snapshot
            self._unresolved_seen = float(c[4])
        if c[4] > self._unresolved_seen:
            self._unresolved_seen = float(c[4])
            if c[0] > float(P.UNRESOLVED_STEP_FAULT):
                raise FloatingPointError(f"Cloth step not resolved at tick {self.tick}: its Newton loop ended at the "
                                         f"cap with a last step of {c[0]:.0f} tolerances ({c[0]*float(P.NEWTON_TOL_V):.2f} "
                                         f"m/s of velocity error; the fault is at {float(P.UNRESOLVED_STEP_FAULT):.0f})")

    def _check_cloth(self):
        x = self.cloth.x.numpy()
        if not np.isfinite(x).all():
            raise FloatingPointError(f"Non-finite cloth state at tick {self.tick}")
        v = self.cloth.v.numpy()
        if not np.isfinite(v).all() or np.abs(v).max() > 50.0:
            raise FloatingPointError(f"Cloth velocity out of physical range ({np.abs(v).max():.3g} m/s) at tick "
                                     f"{self.tick}")

    def settle(self, seconds):
        for _ in range(int(round(seconds/self.config.dt))):
            self.step()

    def relax(self, seconds, every=10):
        """Start-state preparation only: dynamic relaxation of the cloth (kinetic damping, the classical method for the
        static equilibrium of cable nets and cloth: the whole world steps normally, and the cloth's velocities are
        zeroed whenever its kinetic energy has passed a peak). Finds a rest state of the undamped model from a laid
        shape the discrete mesh cannot hold exactly. Returns the log."""
        cl = self.cloth
        ke_prev, resets, n = 0.0, 0, int(round(seconds/self.config.dt))
        for k in range(n):
            self.step()
            if (k + 1) % every == 0:
                ke = cl.kinetic_energy()
                if ke < ke_prev:
                    cl.zero_velocities()
                    resets += 1
                    ke = 0.0
                ke_prev = ke
        cl.zero_velocities()
        return dict(relax_s=seconds, velocity_resets=resets)

    # ------------------------------------------------------------------ snapshots
    def snapshot(self):
        snap = dict(rigid={a: getattr(self.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q",
                                                                               "joint_qd")},
                    targets=self.targets.copy(), control=self.control.joint_target_q.numpy().copy(),
                    cmd_v=self._cmd_v.copy(), tick=self.tick, grip_force=dict(self.grip_force),
                    reflex_until=dict(self._reflex_until), body_wrench=self.body_wrench.copy(),
                    cmd_close=dict(self._cmd_close),
                    reflexes=copy.deepcopy(self.reflexes), refused=self.refused_targets,
                    events=copy.deepcopy(self.events), wrench=self._wrench_filtered.copy(),
                    impedance=copy.deepcopy(getattr(self, "impedance", {})))
        if self.cloth is not None:
            snap["cloth"] = self.cloth.snapshot()
            snap["machine"] = self.machine.snapshot()
        return snap

    def restore(self, snap):
        for a, v in snap["rigid"].items():
            getattr(self.state, a).assign(v)
            getattr(self.next_state, a).assign(v)
        self.targets = snap["targets"].copy()
        self.control.joint_target_q.assign(snap["control"])
        self._cmd_v = snap["cmd_v"].copy()
        self.tick = snap["tick"]
        self.grip_force = dict(snap["grip_force"])
        self._cmd_close = dict(snap.get("cmd_close", {}))
        self._reflex_until, self.reflexes = dict(snap["reflex_until"]), copy.deepcopy(snap["reflexes"])
        self.refused_targets = snap["refused"]
        self.events = copy.deepcopy(snap["events"])
        self._wrench_filtered = snap["wrench"].copy()
        self.body_wrench[:] = snap["body_wrench"]         # the filtered contacts plus the finger springs' reactions
        imp = snap.get("impedance", {})
        if imp or getattr(self, "impedance", {}):
            for arm in self._arm_mj_dofs:
                if arm in imp:
                    self.set_joint_impedance(arm, imp[arm]["stiffness"], imp[arm]["damping"])
                elif arm in getattr(self, "impedance", {}):
                    self.set_joint_impedance(arm)
        self.solver._update_mjc_data(self.solver.mj_data, self.model, self.state)
        self.solver._mujoco.mj_forward(self.solver.mj_model, self.solver.mj_data)
        if self.cloth is not None:
            self.cloth.restore(snap["cloth"])
            self.machine.restore(snap["machine"])
        self.fault = None

    # ------------------------------------------------------------------ readouts
    def ply_positions(self, k):
        p = self.cloth.plies[self.ply_ids[k]]
        return self.cloth.positions()[p["v0"]:p["v0"] + p["nv"]]

    def tcp(self, arm):
        return self.kin.forward(arm)

    def finger_force(self, arm):
        """Net cloth force (N) on the arm's fingers and hand (the filtered value the arm feels)."""
        F = np.zeros(3)
        for i, body, _, _ in self.pads:
            if self.cloth.shapes[i]["name"].startswith(arm + "_"):
                F += self.cloth.shape_reaction(i)[0]
        return F
