"""State-only grading and connectivity helpers. No physics imports or mutations."""
import math

import numpy as np


def clip(x):
    return float(np.clip(x, 0., 1.)) if np.isfinite(x) else 0.


def ramp(x, good, zero):
    return 0. if x is None else clip((zero - x)/(zero - good))


def gained(value, start):
    return clip((value - start)/max(1. - start, 1e-12))


def angle(a, b):
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b)/den, -1., 1.)))) if den else 180.


def min_connection(n, edges, source, target, allowed=None):
    """Unit-capacity physical connections; terminal edges cannot be the minimum cut."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_flow
    source, target = np.asarray(source, int), np.asarray(target, int)
    edges = np.asarray(edges, int).reshape(-1, 2)
    if allowed is not None:
        edges = edges[allowed[edges].all(1)]
        source, target = source[allowed[source]], target[allowed[target]]
    if not len(source) or not len(target):
        return 0
    if np.intersect1d(source, target).size:
        raise ValueError("Connectivity terminals overlap")
    u, v = edges.T
    r = np.concatenate((u, v, np.full(len(source), n), target))
    c = np.concatenate((v, u, source, np.full(len(target), n + 1)))
    cap = np.concatenate((np.ones(2*len(edges), int), np.full(len(source) + len(target), len(edges) + 1)))
    graph = coo_matrix((cap, (r, c)), shape=(n + 2, n + 2)).tocsr()
    graph.sum_duplicates()
    graph.data = graph.data.astype(np.int64)
    return int(maximum_flow(graph, n, n + 1).flow_value)


class Connections:
    """Tet faces, cohesive bonds, flesh/bone ties and bone joints in one graph.

    Endpoints are tet/body identities, never a chosen robot path. Fully torn bonds
    count just as cut bonds. The actual settled starting graph is the denominator.
    """
    def __init__(self, world):
        self.world = world
        self.nt = len(world.cut.orig)
        self.n = self.nt + len(world.tissue.body_states()["p"])
        self.centres = world.tissue.hX[world.cut.orig].mean(1)
        t = np.repeat(np.arange(self.nt), 4)
        u = world.cut._nbr[:self.nt].ravel()
        self.faces = np.stack((t[u > t], u[u > t]), 1)
        self.initial_faces = set(world.cut.cut_faces)
        self.start = self.edges()

    def edges(self):
        w, nt = self.world, self.nt
        ts = w.tissue
        active = np.asarray(ts.hactive[:nt], bool)
        faces = self.faces
        cut = {tuple(sorted(f)) for f in w.cut.cut_faces}
        keep = active[faces].all(1) & np.array([tuple(f) not in cut for f in faces], bool)
        out = [faces[keep]]
        nodes = np.full(ts.n, -1, int)
        ai = np.flatnonzero(active)
        nodes[ts.htets[ai].ravel()] = np.repeat(ai, 4)
        if getattr(ts, "n_bonds", 0):
            a, b = nodes[ts.bonds["a"].numpy()], nodes[ts.bonds["b"].numpy()]
            ok = (a >= 0) & (b >= 0) & (a != b) & (ts.bonds["damage"].numpy() < 1.)
            out.append(np.stack((a[ok], b[ok]), 1))
        if getattr(ts, "n_bties", 0):
            t = nodes[ts.bties["node"].numpy()]
            b = ts.bties["body"].numpy()
            ok = (t >= 0) & (ts.bties["damage"].numpy() < 1.)
            out.append(np.stack((t[ok], nt + b[ok]), 1))
        if getattr(ts, "n_joints", 0):
            on = ts.joints["on"].numpy() > 0
            out.append(nt + np.stack((ts.joints["a"].numpy()[on], ts.joints["b"].numpy()[on]), 1))
        return np.concatenate(out).astype(int)

    def remaining(self, source, target, edges=None, allowed=None, start=False):
        return min_connection(self.n, self.start if start else (self.edges() if edges is None else edges),
                              source, target, allowed)

    def partial_slice(self, lab, frags, edges, mesh_h):
        """Best unfinished cross-cut in a current fragment, wherever the agent cut it.

        Candidate planes come from new, unseparating crack faces, grouped by x at
        two mesh lengths. Already separated pieces cannot pay again here. Both
        terminal sets are in that same current fragment; no fixed cut recipe.
        """
        xs = {}
        for a, b in self.world.cut.cut_faces - self.initial_faces:
            if lab[a] >= 0 and lab[a] == lab[b]:
                xs.setdefault(int(lab[a]), []).append(float((self.centres[a, 0] + self.centres[b, 0])/2))
        best = 0.
        for fid, values in xs.items():
            if fid not in frags or frags[fid]["mass"] < .03:
                continue
            values = np.sort(values)
            groups = np.split(values, np.flatnonzero(np.diff(values) > 2*mesh_h) + 1)
            ids = frags[fid]["tets"]
            allowed = np.zeros(self.n, bool)
            allowed[ids] = True
            for group in groups:
                plane = float(np.median(group))
                source = ids[self.centres[ids, 0] < plane - 2*mesh_h]
                target = ids[self.centres[ids, 0] > plane + 2*mesh_h]
                base = self.remaining(source, target, allowed=allowed, start=True)
                now = self.remaining(source, target, edges, allowed)
                if base and now:  # zero = completed separation, credited as a product instead
                    best = max(best, clip(1. - now/base))
        return best


def knife_return_geometry(heel, axis, segment_centres, target_heel, target_R, blade_length, blade_height,
                          half_gap=.0018):
    """Finite slot volume, not an infinite thin slab. All coordinates in metres."""
    local = (np.asarray(segment_centres) - target_heel) @ target_R
    # The segment COM is above its edge; the slot floor/ceiling in scene.py
    # are -0.3 mm and blade_height + 0.5 mm in this same frame.
    excess = np.maximum.reduce((np.abs(local[:, 1]) - (half_gap + .001),
                               -.0025 - local[:, 0], local[:, 0] - (blade_length + .005),
                               -.0013 - local[:, 2], local[:, 2] - (blade_height + .0015)))
    insertion = float(np.mean(np.clip(1. - np.maximum(excess, 0.)/.03, 0., 1.))) if len(local) else 0.
    return dict(heel_error=float(np.linalg.norm(heel - target_heel)), axis_error=angle(axis, target_R[:, 0]),
                inside=bool(len(local) and (excess <= 0.).all()), insertion=insertion)
