"""Per-arm FR3 kinematics on the moving base (IK computes joint targets only; the servos move the arm).

Each arm's chain is its own subtree of the robot URDF (link0 -> hand_tcp). Its base is the arm's link0 in the world,
i.e. the Spine carriage body's current pose times the fixed Duo-mount transform, so a Cartesian target is reached
wherever the base has driven and at whatever spine height.
"""

import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from robots.controllers.kinematics import SerialKinematics, transform


def _origin(joint):
    o = joint.find("origin")
    xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
    rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
    return transform(xyz, rpy)


def fixed_chain(xml, parent, child):
    by_child = {j.find("child").get("link"): j for j in xml.findall("joint")}
    T, node = np.eye(4), child
    while node != parent:
        j = by_child[node]
        if j.get("type") != "fixed":
            raise ValueError(f"non-fixed joint {j.get('name')} between {parent} and {child}")
        T = _origin(j) @ T
        node = j.find("parent").get("link")
    return T


class ArmKinematics:
    def __init__(self, world):
        self.world = world
        urdf = world.scene.handle.metadata["urdf"]
        urdf = urdf.decode() if isinstance(urdf, bytes) else urdf
        xml = ET.fromstring(urdf)
        inst = world.scene.robot_instance
        self.carriage = world.scene.robot_body["franka_spine_mounting_point"]
        self.arms = {}
        for arm in world.arms:
            pre = f"{arm}_fr3_"
            sub = ET.Element("robot", name=arm)
            for l in xml.findall("link"):
                if l.get("name").startswith(pre):
                    sub.append(l)
            for j in xml.findall("joint"):
                if j.get("name").startswith(pre) and j.find("parent").get("link").startswith(pre):
                    sub.append(j)
            payload = ET.tostring(sub)
            chain = SerialKinematics(payload, f"{pre}hand_tcp", base=np.eye(4))
            mount = fixed_chain(xml, "franka_spine_mounting_point", f"{pre}link0")
            names = [f"{inst}/{n}" for n in chain.names]
            self.arms[arm] = (chain, names, mount)
        import mujoco
        self._mj = mujoco
        m = world.mjm
        names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
        self._qadr = {}
        for arm, (chain, jn, _) in self.arms.items():
            self._qadr[arm] = [int(m.jnt_qposadr[[j for n, j in names.items() if n and n.endswith("_" + x.split("/")[-1])][0]])
                               for x in jn]
        robot_mj = {world._mj_body[i] for i in world.scene.robot_bodies}
        self._robot_body = np.zeros(m.nbody, bool); self._robot_body[list(robot_mj)] = True
        self._arm_body = {}
        for arm in self.arms:
            ids = [world._mj_body[i] for i in world.scene.robot_bodies if world.model.body_label[i].split("/")[-1].startswith(f"{arm}_fr3_")]
            mask = np.zeros(m.nbody, bool); mask[ids] = True
            self._arm_body[arm] = mask
        fingers = [world._mj_body[i] for i in world.scene.robot_bodies if "finger" in world.model.body_label[i]]
        self._finger = np.zeros(m.nbody, bool); self._finger[fingers] = True
        # self-contact checks run on a copy of the model in which only robot geoms collide: the room's ~1000
        # furniture geoms would make each check cost ~0.2 s (a 5 cm Cartesian delta would take 4-9 s)
        import copy as _copy
        self._mself = _copy.deepcopy(m)
        other = ~self._robot_body[self._mself.geom_bodyid]
        self._mself.geom_contype[other] = 0
        self._mself.geom_conaffinity[other] = 0
        self._scratch = mujoco.MjData(self._mself)
        from robots.catalog import model
        setup = world.scene.robot_setup.to_dict()["instances"][inst]
        stowed = model(setup["model"]).to_dict()["poses"]["stowed"]
        self.stowed = {arm: np.array([stowed[n.split("/")[-1]] for n in self.arms[arm][1]]) for arm in self.arms}
        self.check()
        # reach envelope of each arm's TCP from its link0 frame, sampled once (refuses far targets without a search)
        rng = np.random.default_rng(0)
        self.reach = {}
        for arm, (chain, _, _) in self.arms.items():
            b0 = chain.base
            chain.base = np.eye(4)
            pts = np.array([chain.fk(rng.uniform(chain.lower, chain.upper))[:3, 3] for _ in range(3000)])
            chain.base = b0
            self.reach[arm] = float(np.linalg.norm(pts, axis=1).max())

    def base(self, arm):
        p, R = self.world.body_pose(self.carriage)
        T = np.eye(4); T[:3, :3], T[:3, 3] = R, p
        return T @ self.arms[arm][2]

    def joints_now(self, arm):
        chain, names, _ = self.arms[arm]
        q = self.world.state.joint_q.numpy()
        return np.array([q[self.world.joints[n]["q"]] for n in names])

    def commanded(self, arm):
        """The arm's commanded joint targets (Franka's O_T_EE_c analogue): relative Cartesian commands start from the
        pose last commanded, not from the measured one, so a small tracking error or a contact cannot ratchet the hand
        (planning deltas from the measured pose sink a held hand 16 mm per command)."""
        chain, names, _ = self.arms[arm]
        return np.array([float(self.world.targets[self.world.joints[n]["q"]]) for n in names])

    def forward(self, arm, q=None):
        chain, _, _ = self.arms[arm]
        chain.base = self.base(arm)
        return chain.fk(self.joints_now(arm) if q is None else np.asarray(q, float))

    def check(self):
        """The IK chain's tip must coincide with the simulated hand TCP frame (1 mm)."""
        for arm in self.arms:
            fk = self.forward(arm)[:3, 3]
            ref = np.asarray(self.world.rig.frame_pose(f"{self.world.scene.robot_instance}/{arm}_tcp", self.world.state)["position_m"])
            if np.linalg.norm(fk - ref) > 1e-3:
                raise RuntimeError(f"{arm}: IK tip {fk} differs from the rig TCP {ref}")

    # FR3 "ready" (franka_description group_state), mirrored for the right arm: a restart seed in front of the robot
    READY = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785])
    MIRROR = np.array([-1., 1., -1., 1., -1., 1., -1.])

    def self_contact(self, arm, q, clearance=0.0):
        """True if the arm at joints q would touch another part of the robot (scratch MuJoCo data, collision only)."""
        m, d, mj = self._mself, self._scratch, self._mj
        d.qpos[:] = self.world.mjd.qpos
        d.qpos[self._qadr[arm]] = q
        mj.mj_kinematics(m, d)
        mj.mj_collision(m, d)
        gb = m.geom_bodyid
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = gb[c.geom[0]], gb[c.geom[1]]
            if self._finger[b1] and self._finger[b2]:
                continue                                     # the two fingers of a closed hand touch each other
            if c.dist < clearance and self._robot_body[b1] and self._robot_body[b2] and (self._arm_body[arm][b1] or self._arm_body[arm][b2]):
                return True
        return False

    def solutions(self, arm, position, orientation_xyzw, seeds=24, tol_m=2e-3, tol_rad=0.0175):
        """Distinct converged IK solutions (from the current joints, ready, stowed and random seeds), nearest first."""
        chain, names, _ = self.arms[arm]
        chain.base = self.base(arm)
        now = self.joints_now(arm)
        ready = self.READY*(self.MIRROR if arm == "right" else 1.)
        rng = np.random.default_rng(int(1e3*abs(float(np.sum(position)))) % 9973)
        out = []
        for s0 in [now, ready, self.stowed[arm], 0.5*(ready + self.stowed[arm])] + [rng.uniform(chain.lower, chain.upper) for _ in range(seeds)]:
            c = chain.solve(position, orientation_xyzw, np.clip(s0, chain.lower + 1e-6, chain.upper - 1e-6), max_evaluations=400)
            if c["position_error_m"] <= tol_m and c["orientation_error_rad"] <= tol_rad:
                q = np.array(c["joints"])
                if all(np.abs(q - o).max() > 0.05 for o in out):
                    out.append(q)
        return sorted(out, key=lambda q: float(np.abs(q - now).max()))

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=2e-3, tol_rad=0.0175, restarts=8, check=None,
              budget_s=None):
        """Joint targets reaching the TCP pose.

        1. local solve from the current joints; if it does not converge, a continuation along the straight TCP path
           (4 sub-goals, each solved from the previous solution);
        2. if that fails too and restarts are allowed: restarts from the ready, stowed and mid poses, then random
           seeds, keeping the acceptable solution nearest the current joints. With a time budget (the policy path:
           1 s) the search stops at the first acceptable solution within 1 rad of the current joints or once the
           budget is spent; without one (scripted operator) every seed is tried.
        check: also require no self-contact of the solution (default: whenever restarts are allowed)."""
        import time
        check = restarts > 0 if check is None else check
        if arm not in self.arms:
            raise ValueError(f"No arm {arm!r} in this robot ({', '.join(self.arms)})")
        chain, names, _ = self.arms[arm]
        chain.base = self.base(arm)
        now = self.joints_now(arm)
        seed = now if seed is None else np.asarray(seed, float)
        seed = np.clip(seed, chain.lower + 1e-6, chain.upper - 1e-6)
        T0 = chain.fk(seed)
        if orientation_xyzw is None:
            orientation_xyzw = Rotation.from_matrix(T0[:3, :3]).as_quat()
        position = np.asarray(position, float)
        if float(np.linalg.norm(position - chain.base[:3, 3])) > 1.02*self.reach[arm]:
            raise ValueError(f"IK unreachable for {arm}: target {np.linalg.norm(position - chain.base[:3, 3]):.2f} m from "
                             f"the arm base, reach {self.reach[arm]:.2f} m; no action executed")
        conv = lambda c: c["position_error_m"] <= tol_m and c["orientation_error_rad"] <= tol_rad
        ok = lambda c: conv(c) and (not check or not self.self_contact(arm, np.array(c["joints"])))
        t_start = time.time()
        r = chain.solve(position, orientation_xyzw, seed, max_evaluations=400)
        if not conv(r) and (budget_s is None or time.time() - t_start < budget_s):
            from scipy.spatial.transform import Slerp
            sl = Slerp([0., 1.], Rotation.from_matrix([T0[:3, :3], Rotation.from_quat(orientation_xyzw).as_matrix()]))
            s = seed
            for f in (0.25, 0.5, 0.75, 1.0):
                c = chain.solve(T0[:3, 3] + f*(position - T0[:3, 3]), sl([f])[0].as_quat(), s, max_evaluations=400)
                s = np.clip(np.array(c["joints"]), chain.lower + 1e-6, chain.upper - 1e-6)
            c = chain.solve(position, orientation_xyzw, s, max_evaluations=400)
            if conv(c):
                r = c
        if not ok(r) and restarts:
            rng = np.random.default_rng(len(self.arms[arm][1]) + int(1e3*abs(float(np.sum(position)))) % 9973)
            ready = self.READY*(self.MIRROR if arm == "right" else 1.)
            best = None
            for s0 in [ready, self.stowed[arm], 0.5*(ready + self.stowed[arm])] + [rng.uniform(chain.lower, chain.upper) for _ in range(3*restarts)]:
                c = chain.solve(position, orientation_xyzw, np.clip(s0, chain.lower + 1e-6, chain.upper - 1e-6), max_evaluations=400)
                if ok(c):
                    dq = float(np.abs(np.array(c["joints"]) - now).max())
                    if best is None or dq < best[0]:
                        best = (dq, c)
                if budget_s is not None and (best is not None and best[0] <= 1.0 or time.time() - t_start > budget_s):
                    break
            if best is not None:
                r = best[1]
        if not ok(r):
            if conv(r):
                raise ValueError(f"IK: every solution found for {arm} touches the robot itself; no action executed")
            raise ValueError(f"IK unreachable for {arm} ({r['position_error_m']*1e3:.1f} mm, "
                             f"{math.degrees(r['orientation_error_rad']):.1f} deg); no action executed")
        return dict(zip(names, map(float, r["joints"])))
