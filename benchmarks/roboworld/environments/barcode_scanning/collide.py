"""Whole-robot collision test for planned configurations (ASSISTED scripted operator only, never a policy action).

MuJoCo's own collision detection, on a scratch MjData that copies the live state and replaces one arm's joint angles
with the candidate's: every geom of the model at its real shape (convex hulls, boxes), MuJoCo's pair filtering (so
the arm's adjacent links are exempt exactly as in the simulation), the other arm and every free body where they are
now. An object held in the hand can be carried along (its pose relative to the hand kept): its contacts with
anything but that arm are checked too. A configuration collides if any such contact is closer than `clearance`
(MuJoCo reports pairs within their detection margin, 2 mm for hulls, so a clearance of 0 means "touching").

A key-press planner that checked only the hand, the wrist camera and link 7 against the scene drove the left arm
into itself on its path to the "1" key (links 2 and 6, seven collision reflexes), and an unchecked book presentation
put the wrist camera's bracket on the arm's own link (nine reflexes).
"""
import numpy as np


CLEARANCE_M = .003     # arm (or held object) to anything else: the servos lag the planned targets by a few mm


class Checker:
    def __init__(self, world, clearance=CLEARANCE_M):
        import copy
        import mujoco
        self.w = world
        self.mujoco = mujoco
        # a private copy of the model whose detection margins are widened by `clearance`, so that pairs closer than
        # that are reported (with their true distance); the live model is untouched. Plans at zero clearance touched
        # in the simulation: the arm tracks its targets a few mm behind (link 7 on the scanner head)
        self.m = copy.deepcopy(world.solver.mj_model)
        self.m.geom_margin[:] = self.m.geom_margin + clearance
        self.clearance = clearance
        self.d = mujoco.MjData(self.m)
        # an unaltered copy for "touching" tests: MuJoCo's box-box routine reported a finger pad 12-22 mm inside a
        # neighbouring keycap when both boxes were only inside the widened margin
        self.m0 = copy.deepcopy(world.solver.mj_model)
        self.d0 = mujoco.MjData(self.m0)
        m = self.m
        self.qadr = {}
        for arm, dofs in world._arm_mj_dofs.items():
            self.qadr[arm] = {name: int(m.jnt_qposadr[int(m.dof_jntid[dof])]) for name, dof in dofs}
        labels = world.model.body_label
        shape_body = world.scene.b.shape_body
        gs = world._geom_shape
        self.geom_arm = {}
        self.hand_bodies = {i for i, l in enumerate(labels) if "/" in l and ("finger" in l or "hand" in l)}
        self.geom_body = np.full(len(gs), -1)
        for g, s in enumerate(gs):
            if s < 0:
                continue
            b = int(shape_body[int(s)])
            self.geom_body[g] = b
            if b >= 0:
                arm = labels[b].split("/", 1)[0]
                if arm in self.qadr:
                    self.geom_arm[g] = arm
        # Newton body -> MuJoCo body, and a free body's qpos address
        nb = np.asarray(world.solver.mjc_body_to_newton.numpy()).reshape(-1)
        self.mj_body = {int(b): i for i, b in enumerate(nb) if b >= 0}
        self.free_qadr = {}
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_qadr[int(m.jnt_bodyid[j])] = int(m.jnt_qposadr[j])
        # each Franka Hand's finger joints (qpos addresses), for testing a path with the opening the hand will have
        mjn = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
        self.fadr = {}
        for arm in self.qadr:
            names = world.gripper_joints(arm) if arm in getattr(world.scene, "arm_hw", {}) else []
            self.fadr[arm] = [int(m.jnt_qposadr[j]) for n in names for k, j in mjn.items()
                              if k and k.startswith(arm + "_") and k.endswith("_" + n.split("/", 1)[1])]

    def _hand_body(self, arm):
        labels = self.w.model.body_label
        cands = [i for i, l in enumerate(labels) if l.startswith(f"{arm}/") and l.endswith("link7")]
        return cands[0]

    def contacts(self, arm, joints, ignore_bodies=(), clearance=None, carried=(), ignore_labels=(), fingers=None):
        """Contacts (geom labels, distance m) of `arm` at `joints` ({name: rad}) -- and of the `carried` bodies, moved
        with its link 7 -- with anything but `ignore_bodies` (and, for carried bodies, the arm itself): every pair that
        touches (the unaltered model), and, with a clearance, every pair closer than it (the widened model), keycaps
        excepted (2.5 mm apart: touching only). A widened-model penetration the unaltered model does not confirm is a
        box-box artefact of the widened margin and is reported as merely near (a finger pad "12-22 mm inside" a keycap
        that was clear). `fingers`: each finger's opening (m) to test with instead of the current one."""
        clearance = self.clearance if clearance is None else clearance
        touching = self._pairs(arm, joints, ignore_bodies, carried, ignore_labels, exact=True, fingers=fingers)
        out = {k: v for k, v in touching.items() if v < 0.}
        if clearance > 0.:
            near = self._pairs(arm, joints, ignore_bodies, carried, tuple(ignore_labels) + ("keycap_",), exact=False,
                               own_links_touching_only=True, fingers=fingers)
            for k, v in near.items():
                if k in out or v >= clearance:
                    continue
                out[k] = v if v >= 0. else 1e-4
        return [(a, b, d) for (a, b), d in out.items()]

    def _pairs(self, arm, joints, ignore_bodies, carried, ignore_labels, exact, own_links_touching_only=False, fingers=None):
        """{(geom label, geom label): closest distance} of the pairs MuJoCo reports for the configuration."""
        mj = self.mujoco
        m, d = (self.m0, self.d0) if exact else (self.m, self.d)
        live = self.w.solver.mj_data
        d.qpos[:] = live.qpos
        for name, q in joints.items():
            if name in self.qadr[arm]:
                d.qpos[self.qadr[arm][name]] = q
        if fingers is not None:
            d.qpos[self.fadr.get(arm, [])] = fingers
        carried = [int(b) for b in carried]
        if carried:
            h = self.mj_body[self._hand_body(arm)]
            p0, R0 = live.xpos[h].copy(), live.xmat[h].reshape(3, 3).copy()
            mj.mj_kinematics(m, d)
            p1, R1 = d.xpos[h].copy(), d.xmat[h].reshape(3, 3).copy()
            for b in carried:
                mb = self.mj_body.get(b)
                if mb is None or mb not in self.free_qadr:
                    continue
                a = self.free_qadr[mb]
                pb, Rb = live.xpos[mb].copy(), live.xmat[mb].reshape(3, 3).copy()
                p_new = p1 + R1 @ (R0.T @ (pb - p0))
                R_new = R1 @ R0.T @ Rb
                qw = np.empty(4)
                mj.mju_mat2Quat(qw, R_new.reshape(-1))
                # a free joint's qpos is the body frame origin; xpos is the body frame too
                d.qpos[a:a + 3] = p_new
                d.qpos[a + 3:a + 7] = qw
        mj.mj_kinematics(m, d)
        mj.mj_collision(m, d)
        out = {}
        ignore = set(int(b) for b in ignore_bodies if b is not None)
        carried_set = set(carried)
        labels = self.w.model.shape_label
        gs = self.w._geom_shape
        name = lambda g: labels[int(gs[g])] if gs[g] >= 0 else str(g)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            b1, b2 = int(self.geom_body[g1]), int(self.geom_body[g2])
            a1, a2 = self.geom_arm.get(g1) == arm, self.geom_arm.get(g2) == arm
            # the arm's own non-adjacent links may pass within a few mm of each other in ordinary poses: touching only;
            # but not the wrist camera's bracket and case, which the servos' lag then drove into link 6 (a stop, and a
            # set-down that left the hand no clear move away)
            if a1 and a2 and own_links_touching_only and not ("camera" in name(g1) or "camera" in name(g2)):
                continue
            c1, c2 = b1 in carried_set, b2 in carried_set
            if (a1 and c2) or (a2 and c1) or (c1 and c2):
                continue                                   # the held object in its own hand
            if a1 and a2 and b1 in self.hand_bodies and b2 in self.hand_bodies:
                continue                                   # the closed fingers' pads on each other
            if not (a1 or a2 or c1 or c2):
                continue
            other = b2 if (a1 or c1) else b1
            if other in ignore:
                continue
            og = g2 if (a1 or c1) else g1
            if ignore_labels and name(og).startswith(tuple(ignore_labels)):
                continue
            key = (name(g1), name(g2))
            out[key] = min(float(c.dist), out.get(key, float(c.dist)))
        return out

    def path_clear(self, arm, names, path, ignore_bodies=(), clearance=None, every=2, carried=(), ignore_labels=(),
                   fingers=None):
        """First colliding waypoint of a joint path (list of arrays in `names` order) and its contacts, or None.
        A pair already inside the clearance at the arm's current configuration may stay there if it gets no closer
        (0.5 mm tolerance): a path must be able to leave a close pose (an arm left 0.1 mm from the cradled scanner by
        a failed press could not even be tucked away)."""
        live = {k: float(v) for k, v in zip(names, self.w.solver.mj_data.qpos[[self.qadr[arm][n] for n in names]])}
        start = {}
        for a, b, dist in self.contacts(arm, live, ignore_bodies, clearance, carried, ignore_labels, fingers):
            start[(a, b)] = min(dist, start.get((a, b), dist))
        idx = list(range(0, len(path), max(1, every)))
        if idx[-1] != len(path) - 1:
            idx.append(len(path) - 1)
        for i in idx:
            hits = [h for h in self.contacts(arm, dict(zip(names, map(float, path[i]))), ignore_bodies, clearance, carried,
                                             ignore_labels, fingers)
                    if not ((h[0], h[1]) in start and h[2] >= start[(h[0], h[1])] - .0005)]
            if hits:
                return i, hits
        return None

    def free_body_contacts(self, body, position, R, clearance=None):
        """Contacts (geom labels, distance m) of the free `body` placed at `position`/`R` (its frame, world) -- with the
        links jointed to it (a book's cover) carried along -- and everything else where it is now: every pair closer
        than `clearance` (default the checker's) with anything but the body and its links. For the phantom customer's
        or patron's carry of an item (skills.take_item)."""
        mj = self.mujoco
        clearance = self.clearance if clearance is None else clearance
        m, d = self.m, self.d
        live = self.w.solver.mj_data
        d.qpos[:] = live.qpos
        mb = self.mj_body[int(body)]
        a = self.free_qadr[mb]
        qw = np.empty(4)
        mj.mju_mat2Quat(qw, np.asarray(R, float).reshape(-1))
        d.qpos[a:a + 3] = position
        d.qpos[a + 3:a + 7] = qw
        mj.mj_kinematics(m, d)
        mj.mj_collision(m, d)
        own = {b for b in range(m.nbody) if b == mb or any(p == mb for p in self._ancestors(b))}
        labels = self.w.model.shape_label
        gs = self.w._geom_shape
        name = lambda g: labels[int(gs[g])] if gs[g] >= 0 else str(g)
        out = {}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            o1, o2 = int(m.geom_bodyid[g1]) in own, int(m.geom_bodyid[g2]) in own
            if o1 == o2 or float(c.dist) >= clearance:
                continue
            key = (name(g1), name(g2)) if o1 else (name(g2), name(g1))
            out[key] = min(float(c.dist), out.get(key, float(c.dist)))
        return [(x, y, dist) for (x, y), dist in out.items()]

    def _ancestors(self, b):
        while b > 0:
            b = int(self.m.body_parentid[b])
            yield b
