"""Room 112 as one world: the rigid world of classroom_arrangement (Newton 1.5 + MuJoCo on the CPU, 0.5 ms ticks,
the Mobile FR3 Duo with its servos, reflexes and base controller) and the bristle and debris solver (solver.py, Warp
on the GPU, two 0.25 ms substeps per tick), coupled explicitly.

Order within a tick:
  1. the rigid world steps with the forces the bristles and the debris exerted during the previous tick applied to
     the bodies they touched (tools, pans, containers, furniture, the robot's base and fingertips);
  2. the solver is given the new poses and velocities of every rigid shape it can touch and of every brush block,
     takes its two substeps, and returns the mean force and moment on each of them.
The lag is one tick. Nothing in the solver is attached to anything: a brush moves debris only through bristle
contact, a pan carries debris only by its floor, walls and friction.

What the solver sees of the rigid world: boxes, capsules and cylinders (the floor plane is its own); every
installed part and every part of a free body that starts below 0.40 m, every table and counter top, every tool and
container, the supplies on the craft table, and of the robot: its fingers and the boxes of its hands as they are,
every arm link, hand body and part of the torso as a box around its collision mesh (the solver has no mesh colliders), its wheels, and
a skirt box for the chassis.
"""

import collections
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
import classroom_arrangement.world as CW
from classroom_arrangement.world import World
from . import params as P
from . import debris as D
from .scene import SweepScene
from .dust import DustFilm, EVERY as DUST_EVERY
from .solver import Sweeper, BOX, CAPSULE, CYLINDER, SUBSTEPS

LOW = 0.40                    # parts that start below this height are colliders of the debris solver
ACTIVE_GAP = 0.030            # a bank is simulated while a straight bristle tip is this close to a surface ...
ACTIVE_HOLD = 0.25            # ... and for this long after (s)
BODY_SPEED_FAULT = 12.0       # m/s: no tool, container or piece of furniture can be this fast in this room (_sweep_tick)
ROOM_MARGIN = 0.30            # m: ... or this far outside its walls, under its floor or above its ceiling
FLOOR_DEPTH_FAULT = 1.0e-3    # m: no rigid shape stands this deep in the room's floor ...
FLOOR_DEPTH_PER_SPEED = 2.0e-3   # s: ... plus this much for every m/s its body had in the last hundredths of a second (the
                              # rigid contact is a critically damped spring of 4 ms: a landing at v goes v/(e*263/s) = 1.4 mm
                              # per m/s into the floor and comes back)
CHASSIS = dict(centre=(0.0, 0.0, 0.150), half=(0.40, 0.29, 0.115))    # skirt of the TMR in base_link (clearance 35 mm)


def mesh_box(vertices):
    """A box around a convex collision mesh: the smaller of its bounding box in the shape's own axes and in the
    principal axes of its vertices. Returns centre, rotation (shape frame), half extents."""
    v = np.asarray(vertices, float)
    lo, hi = v.min(0), v.max(0)
    best = ((lo + hi)/2, Rotation.identity(), (hi - lo)/2)
    c = v.mean(0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    u = (v - c) @ vt.T
    lo, hi = u.min(0), u.max(0)
    if np.prod(hi - lo) < 0.9*np.prod(2*best[2]):
        best = (c + ((lo + hi)/2) @ vt, Rotation.from_matrix(vt.T), (hi - lo)/2)
    return best


class SweepWorld(World):
    # The consolidated classroom World builds `self.scene_cls` (machine-2 hook); the module-global swap below is the
    # machine-3 hook this environment was built against. Both select the same SweepScene.
    scene_cls = SweepScene

    def __init__(self, config, spec):
        original = CW.Scene
        CW.Scene = SweepScene          # the classroom world builds its scene by this module global
        try:
            super().__init__(config, spec)
        finally:
            CW.Scene = original
        self.ext_wrench = np.zeros_like(self.body_wrench)      # ASSISTED probes add phantom forces here
        self._sweep_wrench = np.zeros_like(self.body_wrench)
        self.sweep_recorders = []
        self.timing["sweep_s"] = 0.0
        self._body_com = self.model.body_com.numpy().astype(float)
        # the hard body under each fingertip's rubber (scene.pad_cores) governs its own contacts (priority 1, as the pads,
        # glides and wheels) and is rigid against the arm behind it (params.PAD_CORE_SOLIMP)
        self.core_geoms = np.array([self._shape_geom[s] for s in self.scene.pad_core_shapes], int)
        for g in self.core_geoms:
            self.mjm.geom_priority[g] = 1
            self.mjm.geom_solimp[g, :3] = P.PAD_CORE_SOLIMP
        self._build_sweeper()
        # fine dust: a film on the floor (reduced-order, sweeping/dust.py), only in start states that have any
        self.dust = DustFilm(self.sweeper, {n: t["spec"] for n, t in self.scene.tools.items()}, spec["dust"], seed=int(spec.get("seed", 0)),
                             pose=self.tool_pose, active=lambda: self.tool_active, clock=lambda: self.sim_time) if spec.get("dust") else None

    # ------------------------------------------------------------------ construction
    def _build_sweeper(self):
        sc, b, newton = self.scene, self.scene.b, self.newton
        G = newton.GeoType
        kinds = {int(G.BOX): BOX, int(G.CAPSULE): CAPSULE, int(G.CYLINDER): CYLINDER}
        q0 = self.state.body_q.numpy().astype(float)
        flags = np.asarray(b.shape_flags)
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        tool_names = [n for n, t in sc.tools.items() if t.get("bank") is not None]
        self.tool_names = tool_names
        self.tool_body = np.array([sc.tools[n]["body"] for n in tool_names], int)
        tool_of_body = {int(sc.tools[n]["body"]): k for k, n in enumerate(tool_names)}
        always = set()                      # shapes included whatever their height
        for n, t in sc.tools.items():
            always.update(t["shapes"]); always.update(t.get("handle_shapes", []))
        for n, it in sc.items.items():
            if it["kind"] == "supply":
                always.update(it["shapes"])
            if it["kind"] in ("table", "teacher_desk"):
                always.update(s for s in it["shapes"] if sc.part_of_shape[s][1].role == "top")
        robot_shapes = set(sc.robot_shapes)
        cols, self.coll_owner = [], []
        for i in range(b.shape_count):
            # a container's bottom disc takes no part in rigid contact and is what pieces lie on; the squares the rigid
            # world stands the container on are not colliders for pieces (scene.add_part, "THE BOTTOM OF A ROUND CONTAINER")
            if i in sc.rigid_only or int(b.shape_type[i]) not in kinds or not ((int(flags[i]) & collide) or i in sc.debris_only):
                continue
            body = int(b.shape_body[i])
            if i in robot_shapes and i not in sc.pad_shapes:
                continue
            tf = np.asarray(b.shape_transform[i], float)
            sz = np.asarray(b.shape_scale[i], float)
            k = kinds[int(b.shape_type[i])]
            label = b.shape_label[i] if hasattr(b, "shape_label") else str(i)
            if "casework_top" in label or "sand_tub" in label or label == "mat":
                always.add(i)
            if i not in always and i not in sc.pad_shapes:
                # lowest point of the shape at the start
                R = Rotation.from_quat(tf[3:])
                p = tf[:3]
                if body >= 0:
                    Rb = Rotation.from_quat(q0[body, 3:])
                    p, R = q0[body, :3] + Rb.apply(p), Rb*R
                ext = np.abs(R.as_matrix()) @ (sz if k == BOX else np.array([sz[0], sz[0], sz[1] + (sz[0] if k == CAPSULE else 0.)]))
                if p[2] - ext[2] > LOW:
                    continue
            lp, lq = tf[:3].copy(), tf[3:].copy()
            part = sc.part_of_shape.get(i, (None, None))[1]
            if "cove" in label and body < 0 and k == BOX:
                # The cove base is the wall as far as bristles and pieces are concerned: a strip 10 mm thick and
                # 0.10 m high with nothing behind it. A bristle tip pressed against it got past its middle, was
                # pushed out of its far side, into the wall, and stayed hooked there: a broom set down 25 mm from
                # the alcove's back wall was held back with 18 N and the stroke out of the corner did not happen
                # For the solver the strip is therefore 0.20 m thicker, toward the wall
                # behind it, and 0.30 m high; its face toward the room is where it was.
                thin = 0 if sz[0] < sz[1] else 1
                out = 1.0 if lp[thin] > (P.ROOM[thin] + P.ROOM[thin + 2])/2 else -1.0
                sz = sz.copy()
                sz[thin] += 0.10
                lp[thin] += out*0.10
                lp[2] += 0.15 - sz[2]
                sz[2] = 0.15
            entry = dict(type=k, size=tuple(sz), mu=float(b.shape_material_mu[i]), tool=tool_of_body.get(body, -1), body=body, lp=lp,
                         lq=lq, shape=i, pad=i in sc.pad_shapes)
            # A pan's floor slab and lip are thin plates, and the loose pieces see them as the boxes they are, from
            # both sides (solver.sd_box_from decides the side from where a sphere came).
            cols.append(entry)
            self.coll_owner.append(label)
        # the robot's arms, hands and torso: finger and hand boxes as they are, each collision mesh as a box around
        # it (the fingertip pads are already in the list). Without them a hand or a forearm went through loose
        # pieces as if it were not there
        # Which of the robot's parts: every colliding box or mesh except the fingertip pads (already in the list), the
        # wheels (their own primitives, below) and the bodies of a wheeled base, which the skirt box stands for. The
        # bodies of the base are found from the robot's own tree: whatever lies between a wheel and the root.
        parent = {int(c): int(q) for q, c in zip(b.joint_parent, b.joint_child)}
        robot_bodies = set(int(k) for k in sc.robot_bodies)
        wheel_shapes = set(sc.wheel_shapes)
        base_bodies, base = set(), None
        for i in sc.wheel_shapes:
            k = parent.get(int(b.shape_body[i]), -1)
            while k >= 0 and k in robot_bodies:
                base_bodies.add(k)
                base = k
                k = parent.get(k, -1)
        for i in sorted(robot_shapes):
            body = int(b.shape_body[i])
            if (not (int(flags[i]) & collide) or i in sc.pad_shapes or i in sc.pad_core_shapes or i in wheel_shapes
                    or body in base_bodies):
                continue
            tf = np.asarray(b.shape_transform[i], float)
            st = int(b.shape_type[i])
            if st == int(G.BOX):
                lp, lq, half = tf[:3].copy(), tf[3:].copy(), np.asarray(b.shape_scale[i], float)
            elif st == int(G.MESH):
                c, Rm, half = mesh_box(np.asarray(b.shape_source[i].vertices, float)*np.asarray(b.shape_scale[i], float))
                Rs = Rotation.from_quat(tf[3:])
                lp, lq = tf[:3] + Rs.apply(c), (Rs*Rm).as_quat()
            else:
                continue
            cols.append(dict(type=BOX, size=tuple(float(h) for h in half), mu=float(b.shape_material_mu[i]), tool=-1, body=body,
                             lp=np.asarray(lp, float), lq=np.asarray(lq, float), shape=i))
            self.coll_owner.append("robot/arm")
        # the robot's base: wheels (as their own primitives where they are primitives) and a skirt box
        # (the skirt's size is this mobile base's, measured from its model; a robot without wheels has none)
        if base is not None:
            cols.append(dict(type=BOX, size=CHASSIS["half"], mu=P.TOOL_MU, tool=-1, body=base, lp=np.array(CHASSIS["centre"]),
                             lq=np.array([0., 0., 0., 1.]), shape=-1))
            self.coll_owner.append("robot/chassis_skirt")
        for i in sc.wheel_shapes:
            if int(b.shape_type[i]) in kinds:
                tf = np.asarray(b.shape_transform[i], float)
                cols.append(dict(type=kinds[int(b.shape_type[i])], size=tuple(np.asarray(b.shape_scale[i], float)), mu=0.6, tool=-1,
                                 body=int(b.shape_body[i]), lp=tf[:3], lq=tf[3:], shape=i))
                self.coll_owner.append("robot/wheel")
        # what each shape's body weighs, for the pieces that are held under it (solver.contact_world)
        mass = self.model.body_mass.numpy()
        for c in cols:
            c["invm"] = 0.0 if c["body"] < 0 or float(mass[c["body"]]) <= 0.0 else 1.0/float(mass[c["body"]])
        # A shape of a FREE body (a tool, a container, a piece of furniture) also tells the solver where the body's
        # centre of mass is in the shape's own frame and the body's inverse inertia there: a held piece under one end
        # of a long light tool stops the tool with what that end weighs, not with the tool's whole mass (solver,
        # contact_world; a broom let go lying 30 mm above debris was thrown out of the room).
        # The robot's links are held by their joints and answer with their mass alone.
        inertia = self.model.body_inertia.numpy()
        for c, owner in zip(cols, self.coll_owner):
            bdy = c["body"]
            if bdy < 0 or c["invm"] <= 0.0 or owner.startswith("robot") or bdy in robot_bodies:
                continue
            Rc = Rotation.from_quat(c["lq"]).as_matrix()
            I = np.asarray(inertia[bdy], float).reshape(3, 3)
            if not np.isfinite(I).all() or abs(np.linalg.det(I)) < 1.0e-18:
                continue
            c["lcom"] = Rc.T @ (self._body_com[bdy] - np.asarray(c["lp"], float))
            c["iinv"] = Rc.T @ np.linalg.inv(I) @ Rc
        # a wheel does not give way with its own 0.2 to 2.5 kg: to lift it over a grain the robot has to be lifted.
        # Each wheel counts as its share of the whole robot.
        wheels = [k for k, o in enumerate(self.coll_owner) if o == "robot/wheel"]
        robot_kg = float(sum(mass[i] for i in set(sc.robot_body.values())))
        for k in wheels:
            cols[k]["invm"] = len(wheels)/robot_kg
            cols[k]["roll"] = 1
        for k, o in enumerate(self.coll_owner):
            if o == "robot/chassis_skirt":
                cols[k]["roll"] = 2                  # pushes along the floor only (solver.contact_world)
        self.coll_shape = [c["shape"] for c in cols]
        self._c_body = np.array([c["body"] for c in cols], int)
        self._c_dyn = np.flatnonzero(self._c_body >= 0)
        # Which body takes what the loose pieces put on a shape: the shape's own, except for the wheels, whose
        # forces go to the base at the wheel's place. A caster is a 0.2 kg link on a free swivel: with the push of
        # a grain under it (which answers for a quarter of the robot, see `invm` above) put on the link itself,
        # the caster whirled about its swivel at 270 rad/s and its rim hit the next pieces at 10 m/s (traced in the
        # wheel's own frame).
        self._c_force_body = self._c_body.copy()
        if base is not None:
            self._c_force_body[wheels] = base
        self._c_lp = np.array([c["lp"] for c in cols], float)
        self._c_tool = np.array([int(c["tool"]) for c in cols], int)             # which tool's block a collider belongs to (-1: none)
        self._c_ext = np.array([float(max(c["size"])) for c in cols], float)       # largest half size of each collider
        self._c_hist = collections.deque(maxlen=int(round(0.1/P.DT)) + 1)          # poses of the last tenth of a second
        self._c_net = np.zeros(len(cols))
        self._c_lq = np.array([c["lq"] for c in cols], float)
        self._c_base = np.array([o.startswith("robot/") for o in self.coll_owner], bool)
        spec = self.spec
        banks = [sc.tools[n]["bank"] for n in tool_names]
        self.debris_classes = D.classes()
        # the collider cells reach 0.3 m beyond the floor so that the walls themselves are found from inside them
        # (with cells ending at the wall faces, a grain flicked at a wall went through it)
        grid = (P.ROOM[0] - 0.3, P.ROOM[1] - 0.3, P.ROOM[2] + 0.3, P.ROOM[3] + 0.3)
        self.sweeper = sw = Sweeper(banks, self.debris_classes, spec["debris"], dt_tick=P.DT, room=grid,
                                    device=getattr(self.config, "device", "cuda:0"), kin=P.KINETIC_RATIO,
                                    rest_disp=P.REST_DISP, rest_time=P.REST_TIME, slip_disp=P.SLIP_DISP, cell=P.CELL,
                                    kv_zeta=P.BRISTLE_ZETA_TOP)
        sw.v_dep = P.DEPENETRATION_SPEED
        sw.set_colliders(cols)
        self.colliders = cols
        # straight tips of a sample of each bank's clusters (to decide when a bank must be simulated)
        self._tip_local = []
        for bk in banks:
            idx = np.unique(np.concatenate([np.arange(0, len(bk.roots), max(1, len(bk.roots)//40)), [len(bk.roots) - 1]]))
            self._tip_local.append(bk.roots[idx] + bk.dirs[idx]*bk.length[idx, None])
        # horizontal surfaces a bank can work on besides the floor: (top z, x0, y0, x1, y1), refreshed from poses
        self._surface_ids = [k for k, c in enumerate(cols) if c["type"] == BOX and c["shape"] >= 0 and (
            "top" in self.coll_owner[k] or self.coll_owner[k] == "mat")]
        self._active_until = np.full(len(tool_names), -1.0)
        self.tool_active = np.zeros(len(tool_names), bool)
        self._cpos = None
        self.awake = len(spec["debris"])
        self._asleep_all = False
        self._asleep_ticks = 0
        self._quiet = 0
        # every rigid body that is not the robot's (tools, their handles and wheels, containers, furniture, supplies)
        self._free_bodies = np.array(sorted(set(range(self.model.body_count)) - robot_bodies), int)
        # for _floor_fault: the floor's geom, each geom's body (-1: fixed), and each body's recent speed
        labels = self.scene.b.shape_label
        gs = np.asarray(self._geom_shape)
        fl = [g for g in range(len(gs)) if gs[g] >= 0 and labels[int(gs[g])] == "floor"]
        self._floor_geom = int(fl[0]) if fl else None
        sbody = np.asarray(self.scene.b.shape_body, int)
        self._geom_body = np.array([int(sbody[int(s)]) if s >= 0 else -1 for s in gs], int)
        # how much rubber stands in front of each geom's rigid part: the fingertip pads' (scene.pad_cores), none elsewhere
        self._rubber = np.zeros(len(gs))
        self._rubber[self.pad_geoms] = P.PAD_RUBBER
        self._speed_mem = self._turn_mem = None
        self._push_poses(first=True)
        sw.place_rods()

    # ------------------------------------------------------------------ poses of what the solver sees
    def _poses(self):
        q = self.state.body_q.numpy().astype(float)
        qd = self.state.body_qd.numpy().astype(float)
        K = len(self._c_body)
        pos = self._c_lp.copy()
        quat = self._c_lq.copy()
        vel = np.zeros((K, 3)); om = np.zeros((K, 3))
        d = self._c_dyn
        bd = self._c_body[d]
        Rb = Rotation.from_quat(q[bd, 3:])
        pos[d] = q[bd, :3] + Rb.apply(self._c_lp[d])
        quat[d] = (Rb*Rotation.from_quat(self._c_lq[d])).as_quat()
        com = q[bd, :3] + Rb.apply(self._body_com[bd])
        om[d] = qd[bd, 3:]
        vel[d] = qd[bd, :3] + np.cross(qd[bd, 3:], pos[d] - com)
        return q, qd, pos, quat, vel, om

    def _push_poses(self, first=False):
        sw = self.sweeper
        q, qd, pos, quat, vel, om = self._poses()
        self._cpos = pos
        self._qd_now = qd
        self._q_now = q
        # the same speed at which a moving shape wakes a sleeping piece it touches (solver: v_wake)
        moving = float(np.abs(vel).max(initial=0.0)) > 1.0e-3 or float(np.abs(om).max(initial=0.0)) > 5.0e-3
        tb = self.tool_body
        tq = tv = tw = np.zeros((0, 3)); tp = np.zeros((0, 3)); tquat = np.zeros((0, 4))
        active = np.zeros(0, bool)
        # each collider's net speed over the last tenth of a second: how far its centre went and how far it turned (at
        # its own largest half size), over that time. The debris solver uses it to tell a support that moves from one
        # that trembles in place (solver.contact_world, NET SPEED).
        hist = self._c_hist
        hist.append((pos.copy(), quat.copy()))
        p0, q0 = hist[0]
        span = max(1, len(hist) - 1)*P.DT
        dq = np.abs(np.einsum("ij,ij->i", quat, q0)).clip(0.0, 1.0)
        net = (np.linalg.norm(pos - p0, axis=1) + self._c_ext*2.0*np.arccos(dq))/span
        if len(hist) < 3:
            net = np.linalg.norm(vel, axis=1) + self._c_ext*np.linalg.norm(om, axis=1)
        self._c_net = net
        if len(tb):
            Rt = Rotation.from_quat(q[tb, 3:])
            com = q[tb, :3] + Rt.apply(self._body_com[tb])
            tv = qd[tb, :3] + np.cross(qd[tb, 3:], q[tb, :3] - com)
            # a bank is simulated while its straight tips are near the floor or a top
            t = self.sim_time
            surf = [(pos[k][2] + self.colliders[k]["size"][2], pos[k][0] - self.colliders[k]["size"][0] - 0.05,
                     pos[k][1] - self.colliders[k]["size"][1] - 0.05, pos[k][0] + self.colliders[k]["size"][0] + 0.05,
                     pos[k][1] + self.colliders[k]["size"][1] + 0.05) for k in self._surface_ids]
            for k in range(len(tb)):
                tips = q[tb[k], :3] + Rt[k].apply(self._tip_local[k])
                near = bool((tips[:, 2] < ACTIVE_GAP).any())
                if not near:
                    for z, x0, y0, x1, y1 in surf:
                        m = (tips[:, 0] > x0) & (tips[:, 0] < x1) & (tips[:, 1] > y0) & (tips[:, 1] < y1)
                        if m.any() and bool((np.abs(tips[m, 2] - z) < ACTIVE_GAP).any() or
                                            ((tips[m, 2] < z) & (q[tb[k], 2] > z)).any()):
                            near = True
                            break
                if near:
                    self._active_until[k] = t + ACTIVE_HOLD
            active = self._active_until >= t
            tp, tquat, tw = q[tb, :3], q[tb, 3:], qd[tb, 3:]
        # one upload, and only when something the solver sees moves (or a bank changes state)
        # (the net speeds go on changing for a tenth of a second after the last motion, and must reach the solver as zero)
        net_moving = float(net.max(initial=0.0)) > 1.0e-6
        changed = (first or moving or getattr(self, "_was_moving", True) or net_moving or getattr(self, "_was_net", True)
                   or bool((active != self.tool_active).any()))
        if changed:
            # a bank whose tool has had no net motion for a tenth of a second is told so (2 instead of 1)
            code = active.astype(int)
            for k in range(len(tb)):
                m = self._c_tool == k
                if active[k] and m.any() and float(net[m].max()) < self.sweeper.rest_disp/0.1:
                    code[k] = 2
            sw.push(pos, quat, vel, om, tp, tquat, tv, tw, code, net=self._c_net)
        self._was_moving = moving
        self._was_net = net_moving
        self.tool_active = active
        return q, pos, moving

    # ------------------------------------------------------------------ stepping
    def step(self):
        self.body_wrench[:] = self.ext_wrench
        self.body_wrench += self._sweep_wrench
        super().step()
        try:
            t0 = time.perf_counter()
            self._sweep_tick()
            if self.dust is not None and self.tick % DUST_EVERY == 0:
                self.dust.update(DUST_EVERY*P.DT)
            self.timing["sweep_s"] += time.perf_counter() - t0
            for rec in self.sweep_recorders:
                rec(self)
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def _sweep_tick(self):
        sw = self.sweeper
        q, pos, moving = self._push_poses()
        # Nothing in this room can make a tool, a container or a piece of furniture faster than BODY_SPEED_FAULT (a
        # fall from the ceiling gives 8 m/s; the arms move a held tool at 2 to 3 m/s, the base at 1 m/s) or take it
        # through a wall or the ceiling. A body that does either is a fault of the world, said so at once.
        if self._free_bodies is not None and len(self._free_bodies):
            fb = self._free_bodies
            sp = np.linalg.norm(self._qd_now[fb, :3], axis=1)
            x = q[fb, :3]
            out = ((x[:, 0] < P.ROOM[0] - ROOM_MARGIN) | (x[:, 0] > P.ROOM[2] + ROOM_MARGIN) | (x[:, 1] < P.ROOM[1] - ROOM_MARGIN)
                   | (x[:, 1] > P.ROOM[3] + ROOM_MARGIN) | (x[:, 2] > P.ROOM_HEIGHT + ROOM_MARGIN) | (x[:, 2] < -ROOM_MARGIN))
            bad = np.flatnonzero((sp > BODY_SPEED_FAULT) | out | ~np.isfinite(sp))
            if len(bad):
                k = int(bad[0])
                name = self.model.body_label[int(fb[k])].split("/")[-1]
                raise FloatingPointError(f"A rigid body ({name}) " + ("left the room" if out[k] else f"moves at {sp[k]:.1f} m/s, faster than "
                                         f"{BODY_SPEED_FAULT:.0f} m/s") + f" at tick {self.tick}")
        # Nothing stands in the floor. A shape whose contact with the floor is deeper than a landing at its body's
        # recent speed can make it is a fault of the world.
        self._floor_fault()
        # nothing the solver sees is moving and every piece sleeps: the solver is not stepped, and the forces of the
        # last stepped tick stay on (a broom standing on its bristles keeps their support, a pan the weight in it).
        # That tick must be one in which every piece slept from its first substep on: only then is the force the
        # weight of what lies there and not the last contact of a piece just falling asleep.
        self._quiet = 0 if moving else self._quiet + 1
        if self._asleep_ticks >= 2 and self._quiet > int(ACTIVE_HOLD/P.DT):
            sw.skipped_ticks += 1
            return
        sw.begin_tick()
        for _ in range(SUBSTEPS):
            sw.substep()
        tf, ttq, cf, ctq, faults, awake = sw.pull()
        if faults:
            raise FloatingPointError(f"Debris faster than {sw.v_limit:.1f} m/s or non-finite at tick {self.tick} "
                                     f"({faults} piece-substeps)")
        self._asleep_all = awake == 0
        self._asleep_ticks = self._asleep_ticks + 1 if awake == 0 else 0
        self.awake = awake
        w = np.zeros_like(self._sweep_wrench)
        tb = self.tool_body
        if len(tb):
            com = q[tb, :3] + Rotation.from_quat(q[tb, 3:]).apply(self._body_com[tb])
            w[tb, :3] += tf[:len(tb)]
            w[tb, 3:] += ttq[:len(tb)] + np.cross(q[tb, :3] - com, tf[:len(tb)])
        d = self._c_dyn
        if len(d):
            bd = self._c_force_body[d]
            com = q[bd, :3] + Rotation.from_quat(q[bd, 3:]).apply(self._body_com[bd])
            np.add.at(w[:, :3], bd, cf[d])
            np.add.at(w[:, 3:], bd, ctq[d] + np.cross(pos[d] - com, cf[d]))
        if not np.isfinite(w).all():
            raise FloatingPointError(f"Non-finite force from the bristle and debris solver at tick {self.tick}")
        self._sweep_wrench = w.astype(np.float32)
        self.last_tool_wrench = (tf[:len(tb)].copy(), ttq[:len(tb)].copy())

    # ------------------------------------------------------------------ queries
    def debris(self):
        """Positions (B, 3), quaternions (B, 4 xyzw), velocities, asleep flags, masses, class index and tags."""
        sw = self.sweeper
        x, qq, v, sl = sw.bodies()
        return dict(pos=x.astype(float), quat=qq.astype(float), vel=v.astype(float), asleep=sl.astype(bool),
                    mass=sw.body_mass.copy(), cls=sw.body_class.copy(), tag=list(sw.body_tag))

    def bristles(self):
        """Node positions (R, N, 3) of every cluster and the slice of each tool."""
        sw = self.sweeper
        return sw.rod_x.numpy().copy(), {n: s for n, s in zip(self.tool_names, sw.rod_slices)}

    def tool_pose(self, name):
        return self.body_pose(self.scene.tools[name]["body"])

    def _floor_fault(self):
        d = self.mjd
        n = int(d.ncon)
        # each body's largest linear and angular speed of the last hundredths of a second (a decaying maximum)
        sp = np.linalg.norm(self._qd_now[:, :3], axis=1)
        om = np.linalg.norm(self._qd_now[:, 3:], axis=1)
        if self._speed_mem is None:
            self._speed_mem, self._turn_mem = sp, om
        else:
            self._speed_mem, self._turn_mem = np.maximum(sp, 0.98*self._speed_mem), np.maximum(om, 0.98*self._turn_mem)
        if not n or self._floor_geom is None:
            return
        g = d.contact.geom[:n]
        dist = d.contact.dist[:n]
        on = (g[:, 0] == self._floor_geom) | (g[:, 1] == self._floor_geom)
        if not on.any():
            return
        # How deep the rigid part of each shape stands. A fingertip pad's box is its rubber's outer face; its rigid part
        # is the hard body PAD_RUBBER behind every face (scene.pad_cores), whose deepest corner stands PAD_RUBBER times
        # the summed vertical components of the box's axes less deep than the box's. Other shapes: their contact depth.
        depth = -dist
        on_at = np.flatnonzero(on)
        touching = np.where(g[on_at, 0] == self._floor_geom, g[on_at, 1], g[on_at, 0])
        rubber = self._rubber[touching]
        if rubber.any():
            axes_z = np.abs(np.asarray(d.geom_xmat)[touching].reshape(-1, 3, 3)[:, 2, :]).sum(axis=1)
            depth[on_at] -= rubber*axes_z
        deep = np.flatnonzero(on & (depth > FLOOR_DEPTH_FAULT))
        if not len(deep):
            return
        other = np.where(g[deep, 0] == self._floor_geom, g[deep, 1], g[deep, 0])
        body = self._geom_body[other]
        free = body >= 0
        b = np.maximum(body, 0)
        # how fast the touching point can have been: the body's speed and its turning at the point's distance from it
        arm = np.linalg.norm(np.asarray(d.contact.pos[:n])[deep] - self._q_now[b, :3], axis=1)
        fast = np.where(free, self._speed_mem[b] + self._turn_mem[b]*arm, 0.0)
        allowed = FLOOR_DEPTH_FAULT + FLOOR_DEPTH_PER_SPEED*fast
        bad = np.flatnonzero(depth[deep] > allowed)
        if len(bad):
            k = int(bad[np.argmax(depth[deep][bad])])
            label = self.scene.b.shape_label[int(self._geom_shape[int(other[k])])]
            if self._rubber[int(other[k])]:
                label += f", the hard body under its {1e3*self._rubber[int(other[k])]:g} mm of rubber"
            raise FloatingPointError(f"A rigid shape ({label}) stands {1e3*float(depth[deep][k]):.2f} mm in the floor, deeper than a "
                                     f"landing at its recent {float(fast[k]):.2f} m/s allows ({1e3*float(allowed[k]):.2f} mm), at tick {self.tick}")

    def settle_debris(self, max_s=3.0):
        """Step until every piece sleeps (start of an episode); returns the time taken and whether all slept."""
        n = int(max_s/P.DT)
        for k in range(n):
            self.step()
            if self._asleep_ticks >= 2:
                return (k + 1)*P.DT, True
        return n*P.DT, self._asleep_all

    def snapshot(self):
        return dict(sweeper=self.sweeper.state(), wrench=self._sweep_wrench.copy(), active=self._active_until.copy(),
                    dust=None if self.dust is None else self.dust.state())

    def restore_snapshot(self, snap):
        self.sweeper.restore(snap["sweeper"])
        self._sweep_wrench = snap["wrench"].copy()
        self._active_until = snap["active"].copy()
        if self.dust is not None:
            self.dust.restore(snap["dust"])
        self._asleep_all = False
        self._asleep_ticks = 0
        self._c_hist.clear()
        self._speed_mem = self._turn_mem = None
        self._push_poses(first=True)
