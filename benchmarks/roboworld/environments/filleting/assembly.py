"""Physical assembly of an anatomical workpiece (filleting/anatomy) into the tissue solver.

Takes the generator's geometry and fields and builds, in world coordinates on the board:
  * the tet mesh of the tissue (isosurface stuffing), region -> material from params.py, myoseptal sheet normals;
  * the rigid bones (head, backbone segments) as bodies of the tissue solver: capsule shapes (tissue contact),
    sphere samples on the capsules (knife, board, finger contact), intervertebral joints, flesh-bone ties;
  * the thin bones (ribs, pin bones, supraneurals, fin supports) as rods bonded along their shaft to the flesh and
    at their root to their vertebral segment;
  * pre-inserted, bonded anatomical interfaces (median septum, rib-cage lining): the faces crossing the interface
    surface are split at build time and the node copies bonded by the interface's cohesive law;
  * the skin membrane on the skin-bearing boundary faces;
  * per-tet reference-product labels for the verifier (never used by the physics).
Every constant is read from params.py (provenance there); numerical measures are named where they are applied.
"""

import math

import numpy as np

from . import layout as LY
from . import mesher
from . import params as PR
from . import tissue as T
from .bodies import capsule_frame, quat_inv, quat_mul, quat_rot

# numerical/authored assembly constants (disclosed)
TIE_TOL_FRAC = .45          # a node within 0.45 h of a bone surface is bonded to it
SAMPLE_SPACING = 1e-3       # bone contact samples along a capsule (m)
BACKBONE_JOINT_K_ANG = P_BACKBONE = .02    # N m/rad per intervertebral joint [estimate; open: E6 sweep]
BACKBONE_JOINT_K_LIN = 5e4                 # N/m (a 10 N pull opens a joint 0.2 mm) [numerical, stable at 2e-5 s]
DISABLE = set()                            # development switches: "ties", "rods", "skin", "interfaces", "bodies"


def _material_table(name, region_names):
    """Region -> material parameters (E, nu, rho, tau_v, R_par, R_perp, tensile strength, sheets) from the parameter
    ledger."""
    v = PR.v
    fish = (v(PR.FISH_E), v(PR.FISH_NU), v(PR.FISH_DENSITY), v(PR.FISH_TAU_V), v(PR.FISH_R_PAR), v(PR.FISH_R_PERP),
            v(PR.FISH_SIGMA_T), True)
    pork = (v(PR.PORK_E), v(PR.FISH_NU), v(PR.PORK_DENSITY), v(PR.FISH_TAU_V), v(PR.PORK_R), v(PR.PORK_R),
            v(PR.PORK_SIGMA_T), False)
    fat = (v(PR.FAT_E), v(PR.FISH_NU), 920., v(PR.FISH_TAU_V), v(PR.FAT_R), v(PR.FAT_R), v(PR.FAT_SIGMA_T), False)
    sb = v(PR.SALMON_BELLY)
    belly = (v(PR.FISH_E)*sb["E_factor"], v(PR.FISH_NU), v(PR.FISH_DENSITY), v(PR.FISH_TAU_V),
             v(PR.FISH_R_PAR)*sb["R_factor"], v(PR.FISH_R_PERP)*sb["R_factor"], v(PR.FISH_SIGMA_T), True)
    out = []
    for r in region_names:
        # the workpiece is named "pork_tenderloin"
        if "tenderloin" in name:
            out.append(fat if "fat" in r else pork)            # pork flake fat: chilled pork back fat
        elif r == "belly_fat":
            out.append(belly)          # the salmon's fatty belly strip is fish muscle (params.SALMON_BELLY), not pork fat
        else:
            out.append(fish)
    return out


def _rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def boundary_faces_with_tets(tets):
    """Boundary faces as (tet, local face index 0..3 = face opposite that vertex)."""
    faces = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]])
    tid = np.tile(np.arange(len(tets)), 4)
    fj = np.repeat(np.arange(4), len(tets))
    key = np.sort(faces, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    b = cnt[inv] == 1
    return tid[b], fj[b], faces[b]


def build(world, wpc, preset, knife_bodies, knife_joints):
    """Returns dict with the tissue, recorder inputs and bookkeeping; `knife_bodies` / `knife_joints` come first in
    the body list (the driven handle is body 0)."""
    c = world.config
    h = c.mesh_h
    pose = LY.WORKPIECE_POSE[preset]
    R = _rz(math.radians(pose["yaw_deg"]))
    off = np.array([pose["center"][0], pose["center"][1], LY.BOARD_TOP_Z + 1e-4])
    to_w = lambda P: np.asarray(P) @ R.T + off
    vec_w = lambda V: np.asarray(V) @ R.T
    lo, hi = wpc.bounds
    Xl, tets = mesher.mesh_sdf(wpc.sdf, lo, hi, h)
    X = to_w(Xl)
    cen_l = Xl[tets].mean(1)
    reg = np.asarray(wpc.region(cen_l), np.int32)
    mats = T.Materials()
    for name, (E, nu, rho, tau, rpar, rperp, sig_t, sheets) in zip(wpc.region_names,
                                                                    _material_table(wpc.name, wpc.region_names)):
        mats.add(name, E, nu, rho, tau, rpar, rperp, sigma_t=sig_t, sheets=sheets)
    sheet = vec_w(wpc.sheet_normal(cen_l))
    ts = T.Tissue(X, tets, reg, mats, sheet_normal=sheet, dt=c.tissue_dt, device=c.device)
    nk = len(knife_bodies)
    # ---------------- rigid bones
    bodies = list(knife_bodies)
    joints = list(knife_joints)
    names = {}
    capsules_w = []
    for i, rb in enumerate(wpc.rigid_bodies):
        bi = nk + i
        names[rb["name"]] = bi
        com = to_w(np.asarray(rb["com"])[None])[0]
        I = R @ np.asarray(rb["inertia"]) @ R.T
        caps = [(to_w(np.asarray(p0)[None])[0], to_w(np.asarray(p1)[None])[0], float(r)) for p0, p1, r in rb["capsules"]]
        capsules_w.append(caps)
        pts, rad, sp = [], [], []
        for p0, p1, r in caps:
            L = np.linalg.norm(p1 - p0)
            n = max(2, int(np.ceil(L/min(SAMPLE_SPACING, max(r, 2e-4)))) + 1)
            for t in np.linspace(0, 1, n):
                pts.append(p0 + t*(p1 - p0)); rad.append(r); sp.append(L/(n - 1) if n > 1 else r)
        bodies.append(dict(name=rb["name"], mass=float(rb["mass"]), com=com, inertia=I, samples=np.array(pts),
                           sample_r=np.array(rad), sample_spacing=np.array(sp), group=2))
    for i, rb in enumerate(wpc.rigid_bodies):
        j = rb.get("joint")
        if not j or rb.get("parent") is None:
            continue
        par = rb["parent"]
        a = names[par] if isinstance(par, str) else nk + int(par)
        b = nk + i
        ma, mb = bodies[a]["mass"], bodies[b]["mass"]
        meff = ma*mb/(ma + mb)
        k_lin = min(BACKBONE_JOINT_K_LIN, .05*meff/c.tissue_dt**2)
        Ia = np.linalg.eigvalsh(bodies[a]["inertia"]).min()
        Ib = np.linalg.eigvalsh(bodies[b]["inertia"]).min()
        Ieff = min(Ia, Ib)
        # per joint, the backbone's bending stiffness over the length the joint stands for: BACKBONE_JOINT_K_ANG is
        # per joint of 62/13 = 4.8 vertebrae; a joint between shorter segments is stiffer in proportion (the anterior
        # one-vertebra segments, anatomy/trout.py N_SINGLE)
        nv_ = [b_.get("vertebrae") for b_ in (rb, next((r_ for r_ in wpc.rigid_bodies if r_["name"] == par), {}))
               if isinstance(b_, dict)]
        span = np.mean([v_[1] - v_[0] + 1 for v_ in nv_ if v_]) if any(nv_) else 62/13
        k_ang = min(BACKBONE_JOINT_K_ANG*(62/13)/span, .05*Ieff/c.tissue_dt**2)
        joints.append(dict(a=a, b=b, anchor=to_w(np.asarray(j["anchor"])[None])[0], k=k_lin,
                           c=2*.3*math.sqrt(k_lin*meff), k_ang=k_ang, c_ang=2*.3*math.sqrt(k_ang*Ieff),
                           name=f"{par}->{rb['name']}", gap=j.get("gap"), radius=j.get("radius")))
    ts.add_bodies(bodies, joints)
    # ---------------- flesh-bone ties: nodes within TIE_TOL_FRAC h of a bonded bone surface
    xn = ts.hX[:ts.n]
    sigma, G = PR.v(PR.FLESH_BONE_SIGMA), PR.v(PR.FLESH_BONE_G)
    tie_nodes, tie_body, tie_anchor = [], [], []
    tol = TIE_TOL_FRAC*h
    for i, rb in enumerate(wpc.rigid_bodies):
        if not rb.get("bond", False):
            continue
        dmin = np.full(ts.n, np.inf)
        foot = np.zeros((ts.n, 3))
        for p0, p1, r in capsules_w[i]:
            ax = p1 - p0
            L2 = max(float(ax @ ax), 1e-18)
            t = np.clip(((xn - p0) @ ax)/L2, 0, 1)
            q = p0 + t[:, None]*ax
            dv = xn - q
            dl = np.linalg.norm(dv, axis=1)
            d = np.abs(dl - r)
            better = d < dmin
            dmin[better] = d[better]
            foot[better] = (q + dv/np.maximum(dl, 1e-12)[:, None]*r)[better]
        if "sdf" in rb and rb["sdf"] is not None:          # the head solid: bonded across its whole face
            to_l = lambda P: (np.asarray(P) - off) @ R
            ds = np.abs(rb["sdf"](to_l(xn)))
            better = ds < dmin
            dmin[better] = ds[better]
            foot[better] = xn[better]
        sel = np.nonzero(dmin < tol)[0]
        tie_nodes.extend(sel.tolist()); tie_body.extend([nk + i]*len(sel)); tie_anchor.extend(foot[sel].tolist())
    # flesh overlapping a bone capsule at rest (small generator overlaps at capsule ends and spines) is embedded in
    # that bone: exempt from contact with it (it is bonded instead). Without this the overlap kept light backbone
    # segments in a ~50 rad/s contact/bond limit cycle at rest (development probe).
    exempt = np.full(ts.Nc, -1, np.int32)
    for i, caps in enumerate(capsules_w):
        inside = np.zeros(ts.n, bool)
        for p0, p1, r in caps:
            ax = p1 - p0
            t = np.clip(((xn - p0) @ ax)/max(float(ax @ ax), 1e-18), 0, 1)
            inside |= np.linalg.norm(xn - (p0 + t[:, None]*ax), axis=1) < r
        exempt[:ts.n][inside] = nk + i
    ts.exempt.assign(exempt)
    ts.n_exempt = int((exempt >= 0).sum())
    if tie_nodes and "ties" not in DISABLE:
        tie_nodes = np.array(tie_nodes)
        A_node = .5*h*h
        fmax = np.full(len(tie_nodes), sigma*A_node)
        dfull = np.full(len(tie_nodes), 2*G/sigma)
        m = ts.hmass[tie_nodes]
        k = np.minimum(fmax/(.2*dfull), .1*m/ts.dt**2)
        ts.add_body_ties(tie_nodes, np.array(tie_body), np.array(tie_anchor), k, fmax, dfull, .2*np.sqrt(k*m))
    # ---------------- rods
    rod_info = []
    if wpc.rods and "rods" not in DISABLE:
        rods_w = [(to_w(np.asarray(rd["points"])), np.asarray(rd["radius"], float)) for rd in wpc.rods]
        ids = ts.add_rods(rods_w, PR.v(PR.BONE_E), PR.v(PR.BONE_DENSITY), uts=PR.v(PR.BONE_UTS))
        # shaft bonds: every rod node to the tet that contains it (rest), cohesive (pin: PIN_SHAFT_TAU; others:
        # flesh-bone); root node to its vertebral segment at bone strength
        from scipy.spatial import cKDTree
        tt = ts.htets[:ts.nt]
        P = ts.hX[tt]
        cen = P.mean(1)
        tree = cKDTree(cen)
        Tm = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1)
        Tinv = np.linalg.inv(Tm)
        t_nodes, t_tets, t_bary, t_k, t_f, t_d = [], [], [], [], [], []
        r_nodes, r_rod, r_body, r_anchor, r_k, r_f = [], [], [], [], [], []
        for ri, (rd, nodes) in enumerate(zip(wpc.rods, ids)):
            tau = PR.v(PR.PIN_SHAFT_TAU) if rd.get("kind") == "pin" else PR.v(PR.FLESH_BONE_SIGMA)
            Gi = PR.v(PR.FLESH_BONE_G)
            pts = ts.hX[nodes]
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            lnode = np.zeros(len(nodes)); lnode[:-1] += seg/2; lnode[1:] += seg/2
            for j, (nd, p) in enumerate(zip(nodes, pts)):
                _, cand = tree.query(p, k=12)
                found = -1
                for cc in np.atleast_1d(cand):
                    l = Tinv[cc] @ (p - P[cc, 0])
                    b0 = 1 - l.sum()
                    if b0 >= -1e-6 and (l >= -1e-6).all():
                        found = int(cc); bary = np.array([b0, *l]); break
                if found < 0:
                    continue                      # outside the tissue (a fin ray base): free
                A = math.pi*2*ts.hradius[nd]*lnode[j]
                fm = tau*A
                df = 2*Gi/tau
                mn = ts.point_mass[nd]
                dts = ts.dt/ts.rod_nsub
                t_nodes.append(nd); t_tets.append(found); t_bary.append(bary)
                t_k.append(min(fm/(.2*df), .2*mn/dts**2)); t_f.append(fm); t_d.append(df)
            root = rd.get("root")
            if root is not None:
                bi = names[root] if isinstance(root, str) else nk + 1 + int(root) if "head" in names else nk + int(root)
                nd = nodes[0]
                r0 = float(np.asarray(rd["radius"])[0])
                fm = PR.v(PR.BONE_UTS)*math.pi*r0*r0*.5          # root breaks at half the shaft's tensile capacity
                mn = ts.point_mass[nd]
                dts = ts.dt/ts.rod_nsub
                r_nodes.append(nd); r_rod.append(ri); r_body.append(bi); r_anchor.append(ts.hX[nd])
                r_k.append(.2*mn/dts**2); r_f.append(fm)
            rod_info.append(dict(name=rd.get("name"), kind=rd.get("kind"), nodes=np.asarray(nodes), root=root))
        if t_nodes:
            ts.set_rod_ties(t_nodes, t_tets, t_bary, t_k, t_f, t_d)
        if r_nodes:
            # the root bond's reaction reaches the vertebral segment explicitly at the tissue substep: the summed root
            # stiffness on a segment must stay within its budget (0.05 m/dt^2); ~10 roots per segment at the rod
            # sub-step stiffness made the segments ring and snap their ribs (development probe)
            counts = np.bincount(np.asarray(r_body), minlength=ts.nb)
            mb = ts.rb["m"].numpy()
            r_k = [min(k_, .05*mb[b_]/(max(counts[b_], 1)*ts.dt**2)) for k_, b_ in zip(r_k, r_body)]
            ts.set_rod_roots(r_rod, r_body, r_anchor, r_k, r_f)
    # ---------------- skin membrane on skin-bearing boundary faces
    btet, bface, bf = boundary_faces_with_tets(tets)
    fcen_l = Xl[bf].mean(1)
    sk = wpc.skin
    mask = np.asarray(sk["skin_mask"](fcen_l), bool)
    if mask.any() and "skin" not in DISABLE:
        # the membrane's own modulus: fish skin for the trout and the salmon, the silver skin (epimysium) for the
        # tenderloin
        E_m = PR.v(getattr(PR, sk.get("modulus_param", "SKIN_E")))
        # its cutting toughness (SKIN_R; SILVER_R for the tenderloin), paid by the edge law in the membrane's thickness
        R_m = PR.v(getattr(PR, sk.get("toughness_param", "SKIN_R")))
        # its mass: only the skin's EXCESS over the flesh it lies on, whose tets already fill the volume up to the
        # surface
        rho_under = float(np.median(np.asarray(ts.mat_rho)[np.asarray(reg)[btet[mask]]]))
        rho_m = max(PR.v(PR.SKIN_DENSITY) - rho_under, 0.)
        ts.add_membrane(btet[mask], bface[mask], sk["thickness"], E_m, .45, rho_m, toughness=R_m)
        # what a tear must overcome where it reaches the membrane (tearing.py)
        ts.mem_tear = dict(thickness=float(sk["thickness"]), strength=PR.v(getattr(PR, sk.get("strength_param", "SKIN_STRENGTH"))))
    return dict(tissue=ts, mesh_local=(Xl, tets), to_world=to_w, to_local=lambda P: (np.asarray(P) - off) @ R,
                rot=R, offset=off, body_names=names, n_knife_bodies=nk, rods=rod_info,
                ref_tet=np.asarray(wpc.reference(cen_l), np.int32), region_tet=reg, skin_faces=(btet[mask], bface[mask]),
                capsules=capsules_w, ties=len(tie_nodes))


def _append_body_ties(ts, nodes, body, anchor_w, k, fmax, dfull):
    import warp as wp
    if not ts.n_bties:
        ts.add_body_ties(nodes, body, anchor_w, k, fmax, dfull, .2*np.sqrt(np.asarray(k)*ts.point_mass[nodes]))
        return
    bt = ts.bties
    p = ts.rb["p"].numpy(); q = ts.rb["q"].numpy()
    loc = np.array([quat_rot(quat_inv(q[b]), np.asarray(a) - p[b]) for b, a in zip(body, anchor_w)], np.float32)
    kd = .2*np.sqrt(np.asarray(k)*ts.point_mass[nodes])
    cat = lambda key, extra, dt=np.float32: np.concatenate([bt[key].numpy(), np.asarray(extra, dt)])
    d = ts.device
    new = dict(node=cat("node", nodes, np.int32), body=cat("body", body, np.int32), loc=cat("loc", loc),
               k=cat("k", k), fmax=cat("fmax", fmax), dfull=cat("dfull", dfull), damage=cat("damage", np.zeros(len(nodes))),
               kd=cat("kd", kd))
    for key, arr in new.items():
        dtype = wp.int32 if arr.dtype == np.int32 else (wp.vec3 if arr.ndim == 2 else float)
        bt[key] = wp.array(arr, dtype=dtype, device=d)
    ts.n_bties = len(new["node"])
    ts._graphs = {}
    np.add.at(ts.h_btie_k, np.asarray(nodes), np.asarray(k, float))


def insert_interfaces(world, wpc, asm, cut):
    """Split the mesh along each interface surface (faces whose centroid segment changes the interface sdf's sign,
    midpoint inside the interface's extent) and bond the node copies with the interface's cohesive law."""
    ts = asm["tissue"]
    cut._adjacency()
    nt = len(cut.orig)
    to_l = asm["to_local"]
    cen_l = to_l(ts.hX[ts.htets[:nt]].mean(1))
    report = {}
    for name, it in (wpc.interfaces.items() if "interfaces" not in DISABLE else ()):
        if name == "head_joint":
            continue            # flesh bonded to the rigid head: flesh-bone ties (build), not a tissue split
        if it.get("insert") is False:
            continue            # a reference line for planning and evaluation, continuous tissue in the world
        s = np.asarray(it["sdf"](cen_l))
        a = np.repeat(np.arange(nt), 4)
        b = cut._nbr.ravel()
        m = (b > a) & (s[a]*s[np.maximum(b, 0)] < 0)
        a, b = a[m], b[m]
        mid = .5*(cen_l[a] + cen_l[b])
        ok = np.asarray(it["extent"](mid), bool)
        a, b = a[ok], b[ok]
        twins_before = set(cut.twin)
        # interface normal at each face from the sdf gradient (local -> world), oriented like the sdf
        eps = 2e-4
        grad = np.stack([(np.asarray(it["sdf"](mid[ok] + e)) - np.asarray(it["sdf"](mid[ok] - e)))/(2*eps)
                         for e in np.eye(3)*eps], 1)
        grad = grad/np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
        grad_w = grad @ asm["rot"].T
        new = {}
        for u, v_, nrm in zip(a, b, grad_w):
            new[(int(u), int(v_))] = (nrm, .5*(ts.hX[ts.htets[u]].mean(0) + ts.hX[ts.htets[v_]].mean(0)))
        cut._cut_faces(new, len(new))
        asm.setdefault("interface_faces", set()).update((min(u, v_), max(u, v_)) for u, v_ in new)
        pairs = [(o, n_) for n_, o in cut.twin.items() if n_ not in twins_before and n_ > o]
        report[name] = dict(faces=len(new), node_pairs=len(pairs))
        if not pairs:
            continue
        gk, sk = it["params"]
        G, sigma = PR.v(getattr(PR, gk)), PR.v(getattr(PR, sk))
        h = world.config.mesh_h
        A = .5*h*h
        pa = np.array([p[0] for p in pairs]); pb = np.array([p[1] for p in pairs])
        fmax = np.full(len(pairs), sigma*A)
        dfull = np.full(len(pairs), 2*G/sigma)
        meff = 1/(1/ts.hmass[pa] + 1/ts.hmass[pb])
        k = np.minimum(fmax/(.2*dfull), .1*meff/ts.dt**2)
        _append_bonds(ts, pa, pb, k, fmax, dfull, .2*np.sqrt(k*meff))
    ts.recompute_mass()
    return report


def _append_bonds(ts, a, b, k, fmax, dfull, kd):
    if not getattr(ts, "n_bonds", 0):
        ts.add_bonds(a, b, k, fmax, dfull, kd)
        return
    import warp as wp
    bd = ts.bonds
    cat = lambda key, extra, dt=np.float32: np.concatenate([bd[key].numpy(), np.asarray(extra, dt)])
    arrs = dict(a=cat("a", a, np.int32), b=cat("b", b, np.int32), k=cat("k", k), fmax=cat("fmax", fmax),
                dfull=cat("dfull", dfull), kd=cat("kd", kd), damage=cat("damage", np.zeros(len(a))),
                cut=cat("cut", np.zeros(len(a), np.int32), np.int32))
    for key, arr in arrs.items():
        bd[key] = wp.array(arr, dtype=wp.int32 if arr.dtype == np.int32 else float, device=ts.device)
    ts.n_bonds = len(arrs["a"])
    ts._graphs = {}
    np.add.at(ts.h_bond_k, np.asarray(a), np.asarray(k, float))
    np.add.at(ts.h_bond_k, np.asarray(b), np.asarray(k, float))
