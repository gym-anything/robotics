"""Tearing: flesh separates where it is pulled apart, not only where a blade passes.

Without tearing, flesh pulled by a knife pressed into the held head of A1's control stretches 5.6 times, past where
its stable neo-Hookean law stays convex, and runs away.

Every `split_every` ticks the elastic Cauchy stress of every tet is computed on the GPU (k_tet_cauchy); tets whose
largest principal stress reaches the lowest strength of their material are screened, and each face of a screened tet
toward an uncut neighbour is tested with the two tets' mean stress resolved on the face's current normal n
(normal traction t_n, shear t_s):
  * fish muscle, faces within SHEET_ALIGN_DEG of the local myoseptal sheet plane: the sheet law
    sqrt(<t_n>^2 + (beta t_s)^2) >= SHEET_SIGMA, fracture energy SHEET_G per area;
  * every other face: <t_n> >= the material's tensile strength, fracture energy its toughness R (the knife-edge law's
    R: Atkins' work of fracture), anisotropic against the sheets as the knife-edge law has it.
An opened face becomes a crack face through the knife-cut topology code (topology._cut_faces), the extrinsic cohesive
approach (Camacho & Ortiz 1996): where node splitting separates the two sides of torn faces, the two copies of the
node are joined by a bilinear cohesive bond (bodies.k_bonds) carrying the faces' strength times the node's share of
their area, with fracture energy G times that area. A tear is resisted and paid for exactly as a pre-inserted
interface is: the ledger books G A as the bond fails ('tie'); a blade passing a torn face releases its bonds.

Skin: a face with an edge on the skin opens only where the skin's own membrane tension across that edge reaches its
strength times its thickness (SKIN_STRENGTH; SILVER_STRENGTH on the tenderloin): over a flesh tear the skin holds
the surface nodes together. (A cohesive bond cannot carry the skin: its stiffness is capped by the nodes' explicit
stability at ~2.5 kN/m, a tenth of what 9 kN/m of skin needs within its 0.3 mm failure opening.) When the skin
fails, its elastic energy is released and damped; its own fracture energy is not booked separately.
"""
import numpy as np
import warp as wp

from . import params as PR


@wp.kernel
def k_tet_cauchy(n_tets: int, tets: wp.array(dtype=wp.vec4i), active: wp.array(dtype=wp.int32),
                 x: wp.array(dtype=wp.vec3), Dm_inv: wp.array(dtype=wp.mat33), tmat: wp.array(dtype=wp.int32),
                 mat_mu: wp.array(dtype=float), mat_lam: wp.array(dtype=float),
                 sig: wp.array(dtype=wp.mat33), smax: wp.array(dtype=float)):
    """Elastic Cauchy stress sigma = P F^T / J = (mu F F^T + lam' (J - alpha) J I) / J of the stable neo-Hookean law
    (tissue.k_tet_forces) and its largest principal value. Inverted or degenerate tets report no tension."""
    t = wp.tid()
    if t >= n_tets:
        return
    if active[t] == 0:
        smax[t] = 0.0
        return
    idx = tets[t]
    x0 = x[idx[0]]
    F = wp.matrix_from_cols(x[idx[1]] - x0, x[idx[2]] - x0, x[idx[3]] - x0) * Dm_inv[t]
    J = wp.determinant(F)
    if J < 1.0e-3:
        smax[t] = 0.0
        sig[t] = wp.mat33(0.0)
        return
    m = tmat[t]
    mu = mat_mu[m]
    lamnh = mat_lam[m] + mu
    alpha = 1.0 + mu / lamnh
    S = (mu * (F * wp.transpose(F)) + wp.identity(n=3, dtype=float) * (lamnh * (J - alpha) * J)) / J
    sig[t] = S
    Q = wp.mat33(0.0)
    d = wp.vec3(0.0)
    wp.eig3(S, Q, d)
    smax[t] = wp.max(wp.max(d[0], d[1]), d[2])


def _skin_edges(ts, orig):
    """Edges of the skin (or silver skin) membrane, as sorted pairs of ORIGINAL node ids, mapped to their triangles."""
    if not getattr(ts, "n_tris", 0):
        return {}
    faces = np.array([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)])
    tri = orig[ts.mem_tet][np.arange(len(ts.mem_tet))[:, None], faces[ts.mem_face]]
    out = {}
    for t, (a, b, c) in enumerate(tri):
        for u, v in ((a, b), (b, c), (c, a)):
            out.setdefault((min(u, v), max(u, v)), []).append(t)
    return out


class Tearing:
    """Tear insertion for one world (see the module docstring). `step()` once per split cadence after the knife's
    split; `bond_new_twins(n_before)` after any split, so a node separated across torn faces by a knife batch is
    bonded too."""

    def __init__(self, world):
        self.world = world
        ts, cut = world.tissue, world.cut
        cut._adjacency()
        mats = ts.mats
        n = len(mats.names)
        self.sigma_t = np.asarray(getattr(mats, "sigma_t", [np.inf]*n), float)
        self.sheets = np.asarray(getattr(mats, "sheets", [False]*n), bool)
        self.Rpar = np.asarray(mats.Rpar, float)
        self.Rperp = np.asarray(mats.Rperp, float)
        self.sheet_sigma = PR.v(PR.SHEET_SIGMA)
        self.sheet_G = PR.v(PR.SHEET_G)
        self.beta = PR.v(PR.SHEET_BETA)
        # screening threshold per material: the lowest strength any of its faces can have
        self.screen = np.where(self.sheets, np.minimum(self.sigma_t, self.sheet_sigma), self.sigma_t)
        self.sig = wp.zeros(ts.Tc, dtype=wp.mat33, device=ts.device)
        self.smax = wp.zeros(ts.Tc, dtype=float, device=ts.device)
        self.skin = _skin_edges(ts, cut.orig)
        mt = getattr(ts, "mem_tear", None)
        self.skin_line_strength = float(mt["strength"]*mt["thickness"]) if mt else np.inf
        # torn faces: (tet a, tet b) -> (strength Pa, fracture energy J/m2, rest area m2, mode)
        # tets at a point attachment (a flesh-bone tie's node, a pre-inserted interface bond's node, a rod's tie tet)
        # neither start nor grow a tear: the attachment lumps a distributed traction onto a node, so the stress in the
        # element beside it depends on the mesh, not the material (at rest the trout's flesh reaches 39.8 kPa there,
        # enough to open 60 tears before any cut), and the attachment's own cohesive law governs its failure
        tied = set()
        if getattr(ts, "n_bties", 0):
            tied |= set(int(n_) for n_ in ts.bties["node"].numpy()[:ts.n_bties])
        if getattr(ts, "n_bonds", 0):
            tied |= set(int(n_) for n_ in ts.bonds["a"].numpy()[:ts.n_bonds])
            tied |= set(int(n_) for n_ in ts.bonds["b"].numpy()[:ts.n_bonds])
        nt0 = ts.nt
        mask = np.zeros(ts.Tc, bool)
        if tied:
            mask[:nt0] = np.isin(ts.htets[:nt0], np.fromiter(tied, np.int64)).any(1)
        if getattr(ts, "n_rods", 0):
            rt = ts.rods["tie_tet"].numpy()
            mask[rt[rt >= 0]] = True
        self.at_attachment = mask
        self.rejected = {}                  # face -> tick: faces the crumb rule refused (not retried for 200 ticks)
        if not hasattr(cut, "torn"):
            cut.torn = {}
        self.events = []

    # ------------------------------------------------------------------ membrane tension across a skin edge
    def _skin_tension(self, tris, x, idx):
        """Membrane tension (N/m) of skin triangles `tris` as 3x3 in-plane Cauchy-like tensors (P F^T / Ja); idx: the
        membrane's current nodes (Tissue._membrane_nodes)."""
        ts = self.world.tissue
        if not hasattr(self, "_tri_cache"):
            self._tri_cache = dict(Dmi=ts.tris["Dm_inv"].numpy().astype(float), mu=ts.tris["mu"].numpy().astype(float),
                                   lam=ts.tris["lam"].numpy().astype(float))
        tc = self._tri_cache
        out = []
        for t in tris:
            i = idx[t]
            e1, e2 = x[i[1]] - x[i[0]], x[i[2]] - x[i[0]]
            D = tc["Dmi"][t]
            Fa = e1*D[0, 0] + e2*D[1, 0]
            Fb = e1*D[0, 1] + e2*D[1, 1]
            nrm = np.cross(Fa, Fb)
            Ja = max(np.linalg.norm(nrm), 1e-6)
            nn = nrm/Ja
            mu, lam = tc["mu"][t], tc["lam"][t]
            lamnh = lam + mu
            s = lamnh*(Ja - (1 + mu/lamnh))
            Pa = mu*Fa + s*np.cross(Fb, nn)
            Pb = mu*Fb + s*np.cross(nn, Fa)
            out.append((np.outer(Pa, Fa) + np.outer(Pb, Fb))/Ja)
        return out

    def _skin_holds(self, fo, x, idx):
        """True if the skin bridges this face: one of its edges lies on the skin and the membrane tension across that
        edge (mean of the edge's triangles, in their plane, normal to the edge) is below the skin's line strength."""
        ts, cut = self.world.tissue, self.world.cut
        for u, v in ((fo[0], fo[1]), (fo[1], fo[2]), (fo[2], fo[0])):
            tris = self.skin.get((min(u, v), max(u, v)))
            if not tris:
                continue
            # the edge's current nodes, through the first triangle's current node ids
            T = self._skin_tension(tris, x, idx)
            faces = np.array([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)])
            tens = []
            for t, S in zip(tris, T):
                P = x[idx[t]]
                # edge direction: the two triangle corners whose original ids are u and v
                oi = cut.orig[ts.mem_tet[t]][faces[ts.mem_face[t]]]
                pu, pv = P[list(oi).index(u)], P[list(oi).index(v)]
                ed = pv - pu
                ed /= max(np.linalg.norm(ed), 1e-12)
                nt_ = np.cross(P[1] - P[0], P[2] - P[0])
                nt_ /= max(np.linalg.norm(nt_), 1e-12)
                nu_ = np.cross(nt_, ed)
                tens.append(float(nu_ @ S @ nu_))
            if np.mean(tens) < self.skin_line_strength:
                return True
        return False

    # ------------------------------------------------------------------ the knife's process zone
    EDGE_ZONE_H = 1.5           # tets with a current centroid this many mean edge lengths from the knife edge

    def _outside_edge_zone(self, cand):
        """Screened tets outside the knife edge's process zone. The edge law ties each anchored edge sample to one
        material point and pulls it with up to R*l (0.6 N per 2 mm sample): a point load, like the attachments
        excluded above, so the stress in the tets around it depends on the mesh, not the material. There the edge law
        is the fracture law (the crack follows the edge); the bulk and sheet laws act outside it. (If they
        acted under the edge too, a press cut through the 80 mm coupon would tear 834 faces beside the knife's ~530,
        and the cohesive bonds of the torn faces that lie off the blade's slab would survive its passage and hold the
        cut halves together as one piece; with the zone, 2 pieces and no bond left.)"""
        ts, cut = self.world.tissue, self.world.cut
        e = getattr(ts, "edge", None)
        if e is None or not len(cand):
            return cand
        pos = e["pos"].numpy().astype(float)
        if not len(pos):
            return cand
        from scipy.spatial import cKDTree
        x = ts.x.numpy()[:ts.n].astype(float)
        cen = x[ts.htets[cand]].mean(1)
        d, _ = cKDTree(pos).query(cen, k=1)
        return cand[d > self.EDGE_ZONE_H*cut.h0()]

    # ------------------------------------------------------------------ the tear step
    def _laws(self, cand, sig, x):
        """Per screened tet: the failure law that is met, if any, as (rest crack-plane normal, strength, fracture
        energy, mode, overstress ratio). The sheet law acts on the material's own sheet plane, whose current normal is
        F^-T m; the bulk law on the plane normal to the largest principal stress, carried back to the rest
        configuration by F^T. (Testing each face against its own normal does not work: on the lattice mesh no face is
        normal to the pull, every face turns toward the pull as the tissue stretches, and a bar stretches 4 times
        under 335 kPa without a face opening.)"""
        ts = self.world.tissue
        cand = np.asarray(cand)
        Xc = x[ts.htets[cand]]
        Ds = np.stack([Xc[:, 1] - Xc[:, 0], Xc[:, 2] - Xc[:, 0], Xc[:, 3] - Xc[:, 0]], 2)
        F = Ds @ ts.hDminv[cand]
        S = sig[cand].astype(float)
        mat = ts.htmat[cand]
        m = ts.hsheet[cand].astype(float)
        lm = np.linalg.norm(m, axis=1)
        has_m = lm > 0
        m = m/np.maximum(lm, 1e-30)[:, None]
        out = {}
        # sheet law
        sheet_ok = np.zeros(len(cand), bool)
        sh = np.nonzero(self.sheets[mat] & has_m)[0]
        if len(sh):
            mc = np.linalg.solve(np.transpose(F[sh], (0, 2, 1)), m[sh][:, :, None])[:, :, 0]
            mc /= np.maximum(np.linalg.norm(mc, axis=1), 1e-12)[:, None]
            t = np.einsum("nij,nj->ni", S[sh], mc)
            tn = np.einsum("ni,ni->n", mc, t)
            tsh = np.linalg.norm(t - tn[:, None]*mc, axis=1)
            r = np.hypot(np.maximum(tn, 0.), self.beta*tsh)/self.sheet_sigma
            for i, rr in zip(sh[r >= 1.], r[r >= 1.]):
                out[int(cand[i])] = (m[i], self.sheet_sigma, self.sheet_G, "sheet", float(rr))
                sheet_ok[i] = True
        # bulk law (where the sheet law is not met)
        rest = np.nonzero(~sheet_ok)[0]
        if len(rest):
            ev, evec = np.linalg.eigh(S[rest])
            r = ev[:, -1]/self.sigma_t[mat[rest]]
            hit = np.nonzero(r >= 1.)[0]
            if len(hit):
                i = rest[hit]
                N = np.einsum("nji,nj->ni", F[i], evec[hit][:, :, -1])
                N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
                c = np.where(has_m[i], np.abs(np.einsum("ni,ni->n", N, m[i])), 0.)
                G = self.Rpar[mat[i]]*c*c + self.Rperp[mat[i]]*(1 - c*c)
                for k, ii in enumerate(i):
                    out[int(cand[ii])] = (N[k], float(self.sigma_t[mat[ii]]), float(G[k]), "bulk", float(r[hit][k]))
        return out

    def step(self):
        """Test screened tets and open crack faces where a failure law is met. Returns the number of faces opened.

        Each tear is a crack with a fixed plane (rest normal N through a point p, its strength, fracture energy and
        law), as a knife's cut follows its swept sheet. It STARTS at the most overstressed tet of a connected
        overstressed region that touches no crack, on the plane the law gives (the sheet plane, or the plane normal
        to the largest principal stress), through the tet's face best aligned with it. It GROWS into screened tets
        that touch its torn vertices, that its plane passes through (the crack tip, never its flanks), and where the
        traction across its plane reaches its strength. In each such tet the faces whose two tets' centroids lie on
        either side of the plane are cut (the knife's rule, topology._fan_split). A region that holds a crack grows
        it and does not start another. (On a uniformly pulled bar, opening every overstressed face shatters it,
        starting a tear in every region at every check breaks it into many pieces, and deriving a growing tear's
        plane from its torn faces nearby lets it drift one element per check along the bar.)"""
        w = self.world
        ts, cut = w.tissue, w.cut
        nt = ts.nt
        wp.launch(k_tet_cauchy, dim=nt, inputs=[nt, ts.tets, ts.tactive, ts.x, ts.Dm_inv, ts.tmat, ts.mat_mu,
                                                ts.mat_lam, self.sig, self.smax], device=ts.device)
        smax = self.smax.numpy()[:nt]
        act = ts.hactive[:nt] > 0
        cand = np.nonzero(act & (smax >= self.screen[ts.htmat[:nt]]) & ~self.at_attachment[:nt])[0]
        if not len(cand):
            return 0
        cand = self._outside_edge_zone(cand)
        if not len(cand):
            return 0
        sig = self.sig.numpy()
        x = ts.x.numpy().astype(float)
        nbr, orig = cut._nbr, cut.orig
        X = ts.hX
        if not hasattr(cut, "cracks"):
            cut.cracks, cut.vertex_cracks = [], {}
        cracks, vcr = cut.cracks, cut.vertex_cracks
        # ---- growth
        grow = {}
        touching = set()
        if vcr:
            Xc = x[ts.htets[cand]]
            Fc = np.stack([Xc[:, 1] - Xc[:, 0], Xc[:, 2] - Xc[:, 0], Xc[:, 3] - Xc[:, 0]], 2) @ ts.hDminv[cand]
            for k, a in enumerate(cand):
                a = int(a)
                ids = set()
                for v_ in orig[a].tolist():
                    ids |= vcr.get(v_, set())
                if not ids:
                    continue
                touching.add(a)
                Xa = X[orig[a]]
                for cid in sorted(ids):
                    c = cracks[cid]
                    d = (Xa - c["p"]) @ c["N"]
                    if d.min() >= 0. or d.max() <= 0.:
                        continue
                    nc = np.linalg.solve(Fc[k].T, c["N"])
                    nc /= max(np.linalg.norm(nc), 1e-12)
                    t = sig[a].astype(float) @ nc
                    tn = float(nc @ t)
                    tsh = float(np.linalg.norm(t - tn*nc))
                    teff = np.hypot(max(tn, 0.), self.beta*tsh) if c["mode"] == "sheet" else max(tn, 0.)
                    if teff >= c["sc"]:
                        grow[a] = cid
                        break
        # ---- initiation
        laws = self._laws(cand, sig, x)
        start = {}
        if laws:
            keys = list(laws)
            pos = {a: i for i, a in enumerate(keys)}
            par = list(range(len(keys)))

            def find(i):
                while par[i] != i:
                    par[i] = par[par[i]]
                    i = par[i]
                return i
            for a in keys:
                for j in range(4):
                    b = int(nbr[a, j])
                    if b in pos and (min(a, b), max(a, b)) not in cut.cut_faces:
                        ra, rb = find(pos[a]), find(pos[b])
                        if ra != rb:
                            par[ra] = rb
            holds = {find(pos[a]) for a in keys if a in touching}
            best = {}
            for a in keys:
                r_ = find(pos[a])
                if r_ in holds:
                    continue
                if r_ not in best or laws[a][4] > laws[best[r_]][4]:
                    best[r_] = a
            for a in best.values():
                N, sc, G, mode, _ = laws[a]
                fbest, abest = None, -1.
                for j in range(4):
                    fo = orig[a][[k_ for k_ in range(4) if k_ != j]]
                    nf = np.cross(X[fo[1]] - X[fo[0]], X[fo[2]] - X[fo[0]])
                    al = abs(float(nf @ N))/max(np.linalg.norm(nf), 1e-15)
                    if al > abest:
                        fbest, abest = X[fo].mean(0), al
                start[a] = dict(N=np.asarray(N, float), p=fbest, sc=float(sc), G=float(G), mode=mode, tick=int(w.tick))
        if not grow and not start:
            return 0
        cen = X[ts.htets[:nt]].mean(1)
        pre = w.assembly.get("interface_faces", set()) if getattr(w, "assembly", None) is not None else set()
        mem_idx = ts._membrane_nodes() if self.skin else None
        work = [(a, cracks[cid], cid) for a, cid in grow.items()] + [(a, c, None) for a, c in start.items()]
        new, meta = {}, {}
        for a, c, cid in work:
            N, p = c["N"], c["p"]
            for j in range(4):
                b = int(nbr[a, j])
                if b < 0 or not ts.hactive[b]:
                    continue
                key = (a, b) if a < b else (b, a)
                if key in cut.cut_faces or key in pre or key in new or w.tick - self.rejected.get(key, -10**9) < 200:
                    continue
                if float((cen[a] - p) @ N)*float((cen[b] - p) @ N) >= 0:
                    continue
                fo = orig[a][[k_ for k_ in range(4) if k_ != j]]
                if self.skin and self._skin_holds(fo, x, mem_idx):
                    continue
                Pr = X[fo]
                nf = np.cross(Pr[1] - Pr[0], Pr[2] - Pr[0])
                Af = .5*np.linalg.norm(nf)
                new[key] = (N, p)
                meta[key] = (c["sc"], c["G"], float(Af*abs(float(nf @ N))/max(2*Af, 1e-15)), c["mode"], a, cid)
        if not new:
            return 0
        n_before = ts.n
        e0 = ts.energies()
        proposed = list(new)
        cut._cut_faces(new, "tear")
        for key in proposed:
            if key not in cut.cut_faces:
                self.rejected[key] = w.tick
        if ts.n == n_before and not any(key in cut.cut_faces for key in proposed):
            # every proposed face refused by the crumb rule: nothing changed, nothing to bond or book
            self.refused_batches = getattr(self, "refused_batches", 0) + 1
            return 0
        made = {}
        for key, m in meta.items():         # crumb control may have dropped some
            if key not in cut.cut_faces:
                continue
            cid = m[5]
            if cid is None:
                a_ = m[4]
                if a_ not in made:
                    made[a_] = len(cracks)
                    cracks.append(start[a_])
                cid = made[a_]
            cut.torn[key] = (m[0], m[1], m[2], m[3], int(w.tick), cid)
            a, b = key
            for v_ in set(orig[a].tolist()) & set(orig[b].tolist()):
                vcr.setdefault(v_, set()).add(cid)
        bonded = self.bond_new_twins(n_before)
        e1 = ts.energies()
        ts.book("topo", sum(e0[k] - e1[k] for k in ("ke", "pe", "psi")))
        ev = dict(tick=w.tick, event="tear", faces=len(new), sheet=sum(1 for k in new if meta[k][3] == "sheet"),
                  grown=len(grow), started=len(made), cracks=len(cracks), nodes_split=ts.n - n_before, bonds=bonded)
        w.events.append(ev)
        self.events.append(ev)
        return len(new)

    def bond_new_twins(self, n_before):
        """Cohesive bonds for nodes split since `n_before` across torn faces: each new copy and the node it was split
        from are joined with the torn faces' strength and fracture energy times the node's third of their area.
        Faces cut by a blade carry nothing."""
        w = self.world
        ts, cut = w.tissue, w.cut
        if ts.n <= n_before or not cut.torn:
            return 0
        new_nodes = [t for t in range(n_before, ts.n) if t in cut.twin and cut.twin[t] < t]
        if not new_nodes:
            return 0
        act = np.nonzero(ts.hactive[:ts.nt])[0]
        rows = ts.htets[act]
        rel = np.array(sorted(set(new_nodes) | {cut.twin[t] for t in new_nodes}))
        hit = np.isin(rows, rel)
        tets_of = {}
        for r, k in zip(*np.nonzero(hit)):
            tets_of.setdefault(int(rows[r, k]), set()).add(int(act[r]))
        nbr = cut._nbr
        pa, pb, fm, df = [], [], [], []
        for tgt in new_nodes:
            node = cut.twin[tgt]
            other = tets_of.get(node, set())
            fmax = ga = 0.
            for t1 in tets_of.get(tgt, ()):
                for j in range(4):
                    t2 = int(nbr[t1, j])
                    if t2 < 0 or t2 not in other:
                        continue
                    m = cut.torn.get((t1, t2) if t1 < t2 else (t2, t1))
                    if m is None:
                        continue
                    sc, G, Ar = m[:3]
                    fmax += sc*Ar/3
                    ga += G*Ar/3
            if fmax > 0:
                pa.append(node); pb.append(tgt); fm.append(fmax); df.append(2*ga/fmax)
        if not pa:
            return 0
        from .assembly import _append_bonds
        pa, pb = np.asarray(pa), np.asarray(pb)
        fm, df = np.asarray(fm), np.asarray(df)
        meff = 1/(1/ts.hmass[pa] + 1/ts.hmass[pb])
        # extrinsic: the face carried its strength until it opened, so the bond starts (nearly) at its peak and only
        # softens; its rise to the peak is 2 % of the failure opening (the pre-inserted interfaces' 20 % let the load
        # keep rising after a tear began, and a uniformly pulled bar started 87 tears). The energy, fmax dfull / 2, does
        # not depend on it.
        k = np.minimum(fm/(.02*df), .1*meff/ts.dt**2)
        # where the stability cap binds, the peak opening fmax/k would pass dfull and the bilinear law would be
        # undefined: such a bond keeps its strength and opens further (more energy than G A; counted)
        capped = fm/k > .5*df
        self.stiffness_capped = getattr(self, "stiffness_capped", 0) + int(capped.sum())
        df = np.where(capped, 2*fm/k, df)
        _append_bonds(ts, pa, pb, k, fm, df, .2*np.sqrt(k*meff))
        ts.recompute_mass()
        return len(pa)
