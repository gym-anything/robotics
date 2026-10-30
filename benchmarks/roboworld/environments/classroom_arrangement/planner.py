"""Collision-aware joint-space motion planning for the ASSISTED scripted operator (never part of the policy API).

With straight Cartesian or joint interpolation the Mobile FR3 Duo's arms hit their own links, the Spine and the
furniture on the way, and the FR3 reflex freezes them. Here every
configuration is checked against the live scene in a scratch MuJoCo data (the world's current state with one arm's
joints replaced; kinematics + collision only, ~2 ms), and paths come from: a direct joint segment if it is free, else
RRT-Connect, then shortcut smoothing. One arm moves at a time; the other arm, the base and the furniture are obstacles
where they are now. `allow` names bodies the hand may touch (the object about to be grasped or pushed) so a final
approach can end in contact.
"""

import numpy as np


class ArmPlanner:
    def __init__(self, world, ik, margin=0.008, step=0.04, seed=0):
        self.world, self.ik, self.margin, self.step = world, ik, margin, step
        self.m, self.mj = world.mjm, ik._mj
        self.d = self.mj.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.gb = self.m.geom_bodyid
        self.checks = 0
        # MuJoCo body id -> Newton body label (for `allow` and diagnostics)
        self.label = {world._mj_body[i]: world.model.body_label[i].split("/")[-1] for i in range(len(world._mj_body))
                      if world._mj_body[i] >= 0}

    def _bodies(self, names):
        out = set()
        for n in names or ():
            b = self.world.scene.bodies.get(n) if hasattr(self.world.scene, "bodies") else None
            if b is not None:
                out.add(self.world._mj_body[b])
        return out

    def contacts(self, arm, q, allow=(), margin=None):
        """Contacts (pairs of body labels, distance) of the arm at joints q with anything but `allow` bodies."""
        m, d = self.m, self.d
        margin = self.margin if margin is None else margin
        d.qpos[:] = self.world.mjd.qpos
        d.qpos[self.ik._qadr[arm]] = q
        self.mj.mj_kinematics(m, d)
        self.mj.mj_collision(m, d)
        self.checks += 1
        armb, fing = self.ik._arm_body[arm], self.ik._finger
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= margin:
                continue
            b1, b2 = self.gb[c.geom[0]], self.gb[c.geom[1]]
            if not (armb[b1] or armb[b2]):
                continue
            if fing[b1] and fing[b2]:
                continue
            other = b2 if armb[b1] else b1
            if other in allow or (armb[b1] and armb[b2] and abs(int(b1) - int(b2)) == 1):
                continue
            out.append((self.label.get(int(b1), str(b1)), self.label.get(int(b2), str(b2)), float(c.dist)))
        return out

    def free(self, arm, q, allow=()):
        return not self.contacts(arm, q, allow)

    def segment_free(self, arm, a, b, allow=()):
        n = max(1, int(np.ceil(np.abs(b - a).max()/self.step)))
        return all(self.free(arm, a + (b - a)*k/n, allow) for k in range(1, n + 1))

    def plan(self, arm, q0, q1, allow=(), max_nodes=3000):
        """Joint path q0 -> q1 avoiding collisions (q0 itself may be in light contact: the first step leaves it)."""
        chain = self.ik.arms[arm][0]
        lo, hi = chain.lower + 1e-4, chain.upper - 1e-4
        q0, q1 = np.clip(np.asarray(q0, float), lo, hi), np.clip(np.asarray(q1, float), lo, hi)
        if not self.free(arm, q1, allow):
            raise ValueError(f"{arm}: goal configuration in collision: {self.contacts(arm, q1, allow)[:3]}")
        if self.segment_free(arm, q0, q1, allow):
            return [q0, q1]
        trees = [[q0], [q1]]
        parents = [[-1], [-1]]
        ext = 0.25

        def extend(t, target):
            nodes = np.array(trees[t])
            i = int(np.argmin(np.abs(nodes - target).max(axis=1)))
            near = nodes[i]
            d = target - near
            dist = np.abs(d).max()
            new = target if dist <= ext else near + d*(ext/dist)
            if self.segment_free(arm, near, new, allow):
                trees[t].append(new); parents[t].append(i)
                return len(trees[t]) - 1, np.allclose(new, target)
            return None, False

        for it in range(max_nodes):
            target = q1 if it % 10 == 0 else self.rng.uniform(lo, hi)
            a, _ = extend(it % 2, target)
            if a is None:
                continue
            new = trees[it % 2][a]
            # connect the other tree toward the new node greedily
            o = 1 - it % 2
            while True:
                b, reached = extend(o, new)
                if b is None:
                    break
                if reached:
                    return self._smooth(arm, self._join(trees, parents, it % 2, a, o, b), allow)
        raise ValueError(f"{arm}: no collision-free path found ({self.checks} checks)")

    @staticmethod
    def _join(trees, parents, ta, a, tb, b):
        def chainup(t, i):
            out = []
            while i >= 0:
                out.append(trees[t][i]); i = parents[t][i]
            return out
        pa, pb = chainup(ta, a), chainup(tb, b)
        path = pa[::-1] + pb[1:]
        return path if ta == 0 else path[::-1]

    def _smooth(self, arm, path, allow, tries=60):
        path = list(path)
        for _ in range(tries):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.segment_free(arm, path[i], path[j], allow):
                path = path[:i + 1] + path[j:]
        return path


def time_path(path, vmax, dt):
    """Waypoints every dt along a joint path at no more than vmax (per joint, rad/s), smoothstep in each segment."""
    out = []
    for a, b in zip(path[:-1], path[1:]):
        T = max(np.abs(b - a).max()/vmax, dt)
        n = max(1, int(np.ceil(T/dt)))
        for k in range(1, n + 1):
            u = k/n
            s = u*u*(3 - 2*u)
            out.append(a + (b - a)*s)
    return out
