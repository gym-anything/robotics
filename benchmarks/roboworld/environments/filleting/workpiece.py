"""Workpiece assembly and the rigid <-> tissue coupling tick (see world.py for the coupling contract).

`build_tissue(world, kind)` meshes the workpiece, creates the tissue solver with every rigid collider that can touch
it as a solver shape, and the crack recorder. `tissue_tick(world, q_before)` advances the tissue over the rigid tick
just taken and feeds the reactions back to the rigid bodies.

Workpieces: "coupon" (a fish-muscle block on the board, the component-test workpiece) and the anatomical presets
from filleting/anatomy (trout, salmon side, tenderloin) once their generators are wired in (build order section 12).
"""

import math
import time

import numpy as np

from . import knife as KN
from . import layout as LY
from . import mesher
from . import params as PR
from . import tissue as T
from . import topology as TP

REACTION_TAU = 5e-3


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


def coupon_geometry(size=(.12, .04, .025)):
    """A fish-muscle block (loin-sized) lying on the board centre, rest frame = world."""
    L, W, H = size
    cx, cy = LY.BOARD["center"]
    z0 = LY.BOARD_TOP_Z + 1e-4
    c = np.array([cx, cy, z0 + H/2])

    def sdf(p):
        q = np.abs(p - c) - np.array([L/2, W/2, H/2])
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    return sdf, c - np.array([L/2, W/2, H/2]), c + np.array([L/2, W/2, H/2])


def build_tissue(world, kind="coupon"):
    c = world.config
    sc = world.scene
    h = c.mesh_h
    knife = _knife_bodies(world)
    asm = None
    if kind == "coupon":
        sdf, lo, hi = coupon_geometry()
        X, tets = mesher.mesh_sdf(sdf, lo, hi, h)
        mats = T.Materials()
        mats.add("white_muscle", PR.v(PR.FISH_E), PR.v(PR.FISH_NU), PR.v(PR.FISH_DENSITY), PR.v(PR.FISH_TAU_V),
                 PR.v(PR.FISH_R_PAR), PR.v(PR.FISH_R_PERP), sigma_t=PR.v(PR.FISH_SIGMA_T), sheets=True)
        tmat = np.zeros(len(tets), np.int32)
        sheet = np.tile([1., 0, 0], (len(tets), 1))       # sheets across the block's length (cut across them)
        ts = T.Tissue(X, tets, tmat, mats, sheet_normal=sheet, dt=c.tissue_dt, device=c.device)
        ts.add_bodies(knife["bodies"], knife["joints"])
        world.workpiece = None
    else:
        from . import anatomy, assembly
        wpc = anatomy.load(kind, c.seed) if hasattr(anatomy, "load") else _load_anatomy(kind, c.seed)
        asm = assembly.build(world, wpc, c.preset, knife["bodies"], knife["joints"])
        ts = asm["tissue"]
        world.workpiece = wpc
    world.assembly = asm
    # ---- solver shapes: static colliders, then kinematic ones carried by rigid bodies, then bodies of this solver
    shapes, owner = [], []          # owner: (newton body or -1 static or -2 tissue-solver body, local p, local q)
    mu_hdpe = (PR.v(PR.MU_FLESH_HDPE), PR.v(PR.MU_SKIN_HDPE), PR.v(PR.MU_BLADE_BONE))
    mu_steel = (PR.v(PR.MU_BLADE_TISSUE), PR.v(PR.MU_BLADE_TISSUE), PR.v(PR.MU_BLADE_BONE))
    board_ids = []
    for name, center, half, q in sc.static_boxes:
        if name.startswith("table_leg") or name == "pedestal":
            continue
        if name == "board":
            board_ids.append(len(shapes))
        shapes.append(dict(type=T.BOX, half=tuple(half), mu=mu_hdpe if name == "board" else mu_steel,
                           com_local=(0, 0, 0), name=name))
        owner.append((-1, np.asarray(center, float), np.asarray(q, float)))
    # the room's floor (MuJoCo's ground plane) for the tissue solver too: a piece that falls off the bench lands on
    # it. Without it a small fragment thrown off the board in a fillet demonstration fell for 40 s, reached z = -5 km
    # and its coordinates overflowed
    shapes.append(dict(type=T.BOX, half=(5., 5., .05), mu=mu_steel, com_local=(0, 0, 0), name="floor"))
    owner.append((-1, np.array([0., 0., -.05]), np.array([0., 0., 0., 1.])))
    world.tissue_static = len(shapes)
    mu_pad = (PR.v(PR.MU_PAD_FLESH), PR.v(PR.MU_PAD_SKIN), PR.v(PR.MU_TWEEZER_BONE))
    world.finger_shape0 = len(shapes)
    for arm, body, name, p, q, half, role in list(sc.finger_boxes) + list(sc.hand_boxes):
        shapes.append(dict(type=T.BOX, half=tuple(half), mu=mu_pad, com_local=(0, 0, 0), name=name, arm=arm,
                           role=role))
        owner.append((body, np.asarray(p, float), np.asarray(q, float)))
    world.knife_body0 = 0                      # tissue-solver body index of the driven handle
    ts.n_fast_bodies = 1 + world.scene.blade.segments   # the knife's bodies: wide contact-candidate margin
    from .scene import HANDLE_L, HANDLE_W, HANDLE_H
    shapes.append(dict(type=T.BOX, half=(HANDLE_L/2, HANDLE_W/2, HANDLE_H/2), mu=mu_pad, com_local=(0, 0, 0),
                       name="knife_handle", body=0, loc_p=(0, 0, 0), loc_q=(0, 0, 0, 1)))
    owner.append((-2, None, None))
    blade = sc.blade
    world.blade_shape0 = len(shapes)
    zc = knife["seg_com_z"]
    for s, sh in enumerate(blade.segment_shapes(*mu_steel)):
        sh = dict(sh)
        sh.update(body=1 + s, loc_p=(0, 0, -zc), loc_q=(0, 0, 0, 1), com_local=(0, 0, zc))
        shapes.append(sh)
        owner.append((-2, None, None))
    world.bone_shape0 = len(shapes)
    if asm is not None:
        mu_bone = (.3, .3, PR.v(PR.MU_BLADE_BONE))           # wet flesh on bone ~0.3 [estimate]; steel on bone
        st = ts.body_states()
        for i, caps in enumerate(asm["capsules"]):
            bi = asm["n_knife_bodies"] + i
            com = st["p"][bi]
            for (p0, p1, r) in caps:
                cc, qc, hl = _capsule(p0, p1)
                loc_p = cc - com
                com_local = _quat_rot(_quat_inv(qc), -loc_p)
                shapes.append(dict(type=T.CAPSULE, half=(hl, r, 0.), mu=mu_bone, body=bi, loc_p=tuple(loc_p),
                                   loc_q=tuple(qc), com_local=tuple(com_local), name=f"bone_{bi}"))
                owner.append((-2, None, None))
    ts.set_shapes(shapes)
    ts.set_blade_shapes(world.blade_shape0, blade.segments)
    ts.set_edge_samples(blade.samples_per_segment()*blade.segments)
    lay = blade.edge_layout(world.blade_shape0)
    ts.set_edge_layout(lay["shape"], lay["loc"], lay["ell"], lay["thick"])
    ts.bsi = blade.bsi
    world.knife_desc = knife
    world.tissue = ts
    world.tissue_shapes = shapes
    world.tissue_owner = owner
    world.cut = TP.CrackRecorder(ts, record_every=2)
    world.cut.supports = board_ids
    world.tissue_filtered = np.zeros((world.model.body_count, 6))
    world.edge_log = []
    q_now = world.state.body_q.numpy().astype(float)
    _pose_shapes(world, q_now, q_now)
    _drive_handle(world, q_now, q_now)
    if asm is not None:
        from . import assembly
        world.interface_report = assembly.insert_interfaces(world, world.workpiece, asm, world.cut)
    ts.recompute_mass()
    ts.rebuild_bvh()
    world.tearing = None
    if c.tearing:
        from .tearing import Tearing
        world.tearing = Tearing(world)


def _capsule(p0, p1):
    from .bodies import capsule_frame
    return capsule_frame(p0, p1)


def _quat_rot(q, v):
    from .bodies import quat_rot
    return quat_rot(q, v)


def _quat_inv(q):
    from .bodies import quat_inv
    return quat_inv(q)


def _load_anatomy(kind, seed):
    import importlib
    mod = importlib.import_module(f"filleting.anatomy.{kind}")
    return mod.build(seed=seed)


def _pose_shapes(world, q0, q1):
    """Shape poses at the tick start (q0) and twists that carry them to q1 over the tick."""
    dt = world.config.rigid_dt
    owner = world.tissue_owner
    n = len(owner)
    poses = np.zeros((n, 7))
    tw = np.zeros((n, 6))
    for i, (b, pl, ql) in enumerate(owner):
        if b == -2:
            continue                      # a body of the tissue solver: posed there
        if b < 0:
            poses[i, :3], poses[i, 3:] = pl, ql
            continue
        Rb0, Rb1 = _R(q0[b, 3:]), _R(q1[b, 3:])
        p0 = q0[b, :3] + Rb0 @ pl
        p1 = q1[b, :3] + Rb1 @ pl
        qs = _quat_mul(q0[b, 3:], ql)
        poses[i, :3], poses[i, 3:] = p0, qs
        tw[i, :3] = (p1 - p0)/dt
        # angular velocity from the relative rotation R1 R0^T (axis-angle / dt)
        from scipy.spatial.transform import Rotation
        rv = Rotation.from_matrix(Rb1 @ Rb0.T).as_rotvec()
        tw[i, 3:] = rv/dt
    world.tissue.set_shape_states(poses, tw)
    return poses, tw


def _knife_bodies(world):
    """Tissue-solver bodies of the knife: [driven handle, blade segment 0..4] and their joints.

    Segment s: rigid body in the segment's edge-centre frame (knife.py), COM 0.6 H above the edge (flat-ground
    profile, spine-heavy), mass from its steel volume (scene.py). Joints (handle-seg0 and seg-seg) at the segment
    boundaries, mid-height, with per-axis rotational stiffness in the parent's frame:
      flat-wise (about the blade height, z): EI_flat/l with EI tapering with the spine (the fillet knife's flex);
      edge-wise (about the thickness, y): EI_edge/l = E t H^3/12/l (true value, ~9e3 N m/rad);
      torsion (about the blade, x): GJ/l, GJ = G H t^3/3.
    The edge-wise spring is stable at the 2e-5 s substep only with I_yy >= 2 k dt^2/0.2 (~3.6e-5 kg m^2 against the
    true ~5e-7): that one rotational inertia is raised (disclosed numerical measure; it slows edge-wise angular
    dynamics of a blade that barely bends edge-wise, and leaves mass, gravity, flat-wise flex and torsion true).
    Linear joint stiffness 2e5 N/m (0.15 mm under 30 N)."""
    from scipy.spatial.transform import Rotation
    sc = world.scene
    blade = sc.blade
    kh = sc.bodies["knife_handle"]
    q = world.state.body_q.numpy().astype(float)
    Rh = _R(q[kh, 3:])
    ph = q[kh, :3] + Rh @ sc.handle_ref_local
    qh = q[kh, 3:]
    m_h = sc.knife_mass["handle"]
    from .scene import HANDLE_L, HANDLE_W, HANDLE_H
    Ih = np.diag([m_h*(HANDLE_W**2 + HANDLE_H**2)/12, m_h*(HANDLE_L**2 + HANDLE_H**2)/12, m_h*(HANDLE_L**2 + HANDLE_W**2)/12])
    # contact samples on the grip box (a shell of 1.5 mm spheres, ~6 mm apart), active only while the knife is free
    # (not gripped: World._update_knife_mode); while gripped MuJoCo resolves the handle's contacts
    r_s = 1.5e-3
    gx, gy, gz = (np.linspace(-h_ + r_s, h_ - r_s, n_) for h_, n_ in
                  ((HANDLE_L/2, 24), (HANDLE_W/2, 3), (HANDLE_H/2, 6)))
    G_ = np.array(np.meshgrid(gx, gy, gz, indexing="ij")).reshape(3, -1).T
    shell = (np.isclose(np.abs(G_[:, 0]), HANDLE_L/2 - r_s) | np.isclose(np.abs(G_[:, 1]), HANDLE_W/2 - r_s)
             | np.isclose(np.abs(G_[:, 2]), HANDLE_H/2 - r_s))
    hs = G_[shell] + sc.handle_ref_local
    bodies = [dict(name="knife_handle", mass=m_h, com=ph, quat=qh, inertia=Rh @ Ih @ Rh.T, driven=True, group=1,
                   samples=ph + (hs - sc.handle_ref_local) @ Rh.T, sample_r=np.full(len(hs), r_s),
                   sample_spacing=np.full(len(hs), 6e-3))]
    dt = world.config.tissue_dt
    E, G = PR.v(PR.STEEL_E), PR.v(PR.STEEL_G)
    zc = sc.blade_seg_com_z
    L, H = blade.seg_len, blade.H
    joints = []
    for s in range(blade.segments):
        m = sc.blade_seg_mass[s]
        t = blade.spine_at((s + .5)/blade.segments)
        k_edge = E*t*H**3/12/L
        # stability inertia (inertia, never weight) for the stiff edge-wise spring: on the edge-wise axis, and on the
        # twist axis too, onto which a flat-wise bend of theta turns sin^2(theta) of it. With the twist inertia real,
        # a blade pressed flat on the board and bent 60 deg at two hinges passed the explicit limit there and went
        # non-finite (A2). Twist then responds at ~1000 rad/s instead of ~10000
        Iyy = max(m*(L*L + H*H)/12, 2*2*k_edge*dt*dt/.2)
        I = np.diag([max(m*(H*H + t*t)/12, Iyy), Iyy, m*(L*L + t*t)/12])
        origin = q[kh, :3] + Rh @ sc.blade_seg_local[s]          # handle body frame origin + local offset
        com = origin + Rh @ np.array([0, 0, zc])
        # samples: the edge (every 2 mm, r 0.15 mm) and the spine (every 2 mm, r t/2), segment frame -> world
        xs = np.arange(-L/2 + 1e-3, L/2, 2e-3)
        pts = np.vstack([np.column_stack([xs, 0*xs, 0*xs + .15e-3]), np.column_stack([xs, 0*xs, 0*xs + H - t/2])])
        rad = np.concatenate([np.full(len(xs), .15e-3), np.full(len(xs), t/2)])
        bodies.append(dict(name=f"knife_blade_{s}", mass=m, com=com, quat=qh, inertia=Rh @ I @ Rh.T,
                           samples=origin + pts @ Rh.T, sample_r=rad, sample_spacing=np.full(len(pts), 2e-3), group=1))
        EI_flat = blade.EI_heel*(t/blade.t)**3
        GJ = G*H*t**3/3
        k_ang = np.array([GJ/L, k_edge, EI_flat/L])
        a = 0 if s == 0 else s                                   # parent: handle (0) or previous segment
        anchor = origin + Rh @ np.array([-L/2, 0, H/2])
        m_par = m if s == 0 else sc.blade_seg_mass[s - 1]
        m_eff = m*m_par/(m + m_par) if s else m
        # linear joint spring at the segment ends: acting at a half-segment lever it also stiffens the flat-wise
        # rotation (k (l/2)^2 on I_zz ~ m l^2/12, omega = sqrt(6 k/m)); 2e5 N/m keeps omega*dt ~0.45 per body (the
        # chain's top mode ~2x) and deflects 0.15 mm under 30 N. 5e5 N/m with half-critical dampers at the same lever
        # was outside the explicit stability region (blade chain diverged in ~2 ms: development probe).
        k_lin = 2e5
        # damping: linear 10 % of critical; rotational 2 % structural on the flat-wise axis, 20 % on the stiff axes. The
        # stiff edge-wise damper is sized on the smallest inertia it can act on (the twist axis's, which a flat-wise
        # bend turns it onto: sized on a larger one it passed the explicit limit there at a ~33 deg bend, A2); with the
        # twist axis carrying the same stability inertia, that is Iyy
        I_eff = np.array([I[0, 0], min(Iyy, I[0, 0]), I[2, 2]])
        zeta = np.array([.2, .2, .02])
        joints.append(dict(a=a, b=1 + s, anchor=anchor, k=k_lin, c=2*.1*np.sqrt(k_lin*m_eff),
                           k_ang=k_ang, c_ang=2*zeta*np.sqrt(k_ang*I_eff)))
    return dict(bodies=bodies, joints=joints, seg_com_z=zc)


def _drive_handle(world, q0, q1):
    """The tissue solver's handle body follows the MuJoCo handle over the tick (pose at start, twist)."""
    from scipy.spatial.transform import Rotation
    kh = world.scene.bodies["knife_handle"]
    com_l = world.scene.handle_ref_local
    R0, R1 = _R(q0[kh, 3:]), _R(q1[kh, 3:])
    p0 = q0[kh, :3] + R0 @ com_l
    p1 = q1[kh, :3] + R1 @ com_l
    dt = world.config.rigid_dt
    w = Rotation.from_matrix(R1 @ R0.T).as_rotvec()/dt
    world.tissue.set_driven(world.knife_body0, p0, q0[kh, 3:], (p1 - p0)/dt, w)


BLADE_AIR_TICKS = 20      # ticks with no tissue load on a MuJoCo-owned knife's blade before the blade follows it rigidly


def _blade_in_air(world, q0, q1):
    """While MuJoCo owns the knife (held, in the block, or lying on rigid geometry) and its blade has carried no tissue
    load for BLADE_AIR_TICKS, the tissue solver's flexible blade segments follow MuJoCo's rigid knife at their rest
    shape (driven bodies); the first tissue load makes them dynamic again. In that state the segments meet only tissue
    (MuJoCo's rigid blade meets the board and the robot), so nothing else could load or damp them. (Left dynamic, the
    tissue-side blade of a knife dropped on the board swings flat-wise +-25 mm through the board at 2 Hz, 2-4.5 m/s, for
    as long as it lies there, while the knife lies still in MuJoCo: E15 drop.) The elastic and kinetic energy the snap
    to the rest shape removes is booked with the knife joints' damping."""
    ts = world.tissue
    k = world.knife_body0
    segs = list(range(k + 1, k + 1 + world.scene.blade.segments))
    air = world.phantom_knife is None and getattr(world, "_blade_air", 0) >= BLADE_AIR_TICKS
    rigid = bool(ts.h_rb_fixed[segs[0]] == 2)
    if air != rigid:
        e0 = ts.energies()
        fixed = ts.rb["fixed"].numpy()
        fixed[segs] = 2 if air else 0
        ts.rb["fixed"].assign(fixed)
        ts.h_rb_fixed = fixed
        ts.invalidate_graphs()
        world.events.append(dict(tick=world.tick, event="blade_rigid_in_air" if air else "blade_flexible"))
        if not air:
            # the segments become bodies of this solver again: their weight and motion, carried by the driver while they
            # followed MuJoCo, enter the world's energy as the driver's work (otherwise every first touch would create
            # the blade's 0.16 J of potential energy)
            e1 = ts.energies()
            ts.book("work_in", sum(e1[x] - e0[x] for x in ("ke", "pe")))
            ts.book("damp_joints", e0["psi"] - e1["psi"])
    world._blade_rigid = air
    if not air:
        return
    from scipy.spatial.transform import Rotation
    sc = world.scene
    kh = sc.bodies["knife_handle"]
    zc = world.knife_desc["seg_com_z"]
    R0, R1 = _R(q0[kh, 3:]), _R(q1[kh, 3:])
    dt = world.config.rigid_dt
    w = Rotation.from_matrix(R1 @ R0.T).as_rotvec()/dt
    P0, V = [], []
    for s in range(sc.blade.segments):
        loc = sc.blade_seg_local[s] + np.array([0., 0., zc])
        c0, c1 = q0[kh, :3] + R0 @ loc, q1[kh, :3] + R1 @ loc
        P0.append(c0); V.append((c1 - c0)/dt)
    ts.set_driven_many(segs, np.array(P0), np.tile(q0[kh, 3:], (len(segs), 1)), np.array(V), np.tile(w, (len(segs), 1)))
    if air != rigid:
        # the driver takes the segments over: their potential energy leaves the world's energy as work done on the driver
        # (not as the joints' damping, 0.16 J per lift); what the snap to the rest shape removes
        # from their motion relative to the handle and their elastic energy is dissipation
        e1 = ts.energies()
        ts.book("work_in", e1["pe"] - e0["pe"])
        ts.book("damp_joints", (e0["ke"] + e0["psi"]) - (e1["ke"] + e1["psi"]))


def _chain_momentum(world):
    """Linear momentum of the blade chain and its angular momentum about the tissue handle's reference point, with
    that point, its velocity, the chain's mass-weighted position sum (for the weight's moment) and its mass."""
    ts = world.tissue
    st = ts.body_states()
    k = world.knife_body0
    idx = list(range(k + 1, k + 1 + world.scene.blade.segments))
    if not hasattr(ts, "_h_Ib"):
        ts._h_Ib = ts.rb["Ib"].numpy().astype(float)
        ts._h_m = ts.rb["m"].numpy().astype(float)
    c, vc = st["p"][k], st["v"][k]
    P, L, S, M = np.zeros(3), np.zeros(3), np.zeros(3), 0.
    for i in idx:
        m = ts._h_m[i]
        R = _R(st["q"][i])
        P += m*st["v"][i]
        L += m*np.cross(st["p"][i] - c, st["v"][i]) + R @ ts._h_Ib[i] @ R.T @ st["w"][i]
        S += m*(st["p"][i] - c)
        M += m
    return dict(P=P, L=L, c=c, vc=vc, S=S, M=M)


def _knife_external_load(world, ch0, f_handle, t_handle, q_after):
    """The loads the world applies to the knife through the tissue solver (blade and handle contacts), for MuJoCo's
    rigid knife: the mean load on the driven handle over the tick plus the blade chain's rate of change of momentum
    (Newton-Euler on the chain, about the moving handle point: dL_c/dt + v_c x P), minus the chain's weight and its
    moment (MuJoCo's knife carries the blade's mass and applies its gravity itself). The chain's own inertial
    reaction, which reached the handle through the stiff root joint, no longer crosses the 1 ms boundary."""
    dt = world.config.rigid_dt
    ch1 = _chain_momentum(world)
    g = np.array([0., 0., -9.81])
    F = f_handle + (ch1["P"] - ch0["P"])/dt - ch1["M"]*g
    Pm = .5*(ch0["P"] + ch1["P"])
    vc = .5*(ch0["vc"] + ch1["vc"])
    S = .5*(ch0["S"] + ch1["S"])
    tau_c = t_handle + (ch1["L"] - ch0["L"])/dt + np.cross(vc, Pm) - np.cross(S, g)
    kh = world.scene.bodies["knife_handle"]
    com_w = q_after[kh, :3] + _R(q_after[kh, 3:]) @ world.model.body_com.numpy()[kh].astype(float)
    c = .5*(ch0["c"] + ch1["c"])
    return F, tau_c + np.cross(c - com_w, F)


def _rig_tick(fn, t, dt):
    """Pose of an ASSISTED rig at the tick start and the SECANT twist to its pose at the tick end, so the driven body
    lands exactly on the path every tick. (Handing the solver the path's instantaneous velocity overshot every
    waypoint that fell inside a tick and snapped the body back at the next one: a 0.1 mm snap on the knife's 2e5 N/m
    root joint injected ~1 mJ that no work term saw -- iface_drag_check.)"""
    from scipy.spatial.transform import Rotation
    p0, q0 = fn(t)[:2]
    p1, q1 = fn(t + dt)[:2]
    p0, q0, p1, q1 = (np.asarray(a, float) for a in (p0, q0, p1, q1))
    w = (Rotation.from_quat(q1)*Rotation.from_quat(q0).inv()).as_rotvec()/dt
    return p0, q0, (p1 - p0)/dt, w


def blade_segment_poses(world):
    """World poses (R, p of the edge-centre frame) of the blade segments, from the tissue solver's bodies."""
    st = world.tissue.body_states()
    zc = world.knife_desc["seg_com_z"]
    out = []
    for s in range(world.scene.blade.segments):
        R = _R(st["q"][1 + s])
        out.append((R, st["p"][1 + s] - R @ np.array([0, 0, zc])))
    return out


def _joint_cut_rule(world, poses, blades):
    """An intervertebral (or head) joint is cut when the knife's blade has passed through the joint: the joint's
    anchor point lies in a blade segment's slab on the spine side of the edge. The centra are rigid capsules, so the
    blade can only get there through the joint gap ('finding the joint', A1). The joint tissue's
    fracture work (JOINT_R x the joint disc area) is owed by the blade segment that cut it and paid from the knife's
    motion by the interface drag (bodies.k_body_integrate, booked to the iface ledger)."""
    ts = world.tissue
    if not ts.n_joints:
        return
    on = ts.joints["on"].numpy()
    st = ts.body_states()
    nk = world.assembly["n_knife_bodies"]
    pa = ts.joints["pa"].numpy()
    ja = ts.joints["a"].numpy()
    shapes = world.tissue_shapes
    changed = False
    for j, jd in enumerate(ts.joint_desc):
        if not on[j] or ja[j] < nk:
            continue                         # knife joints are never cut
        a = ja[j]
        anchor = st["p"][a] + _R(st["q"][a]) @ pa[j]
        for s in blades:
            X = ts.s_p0.numpy()[s]; Q = ts.s_q0.numpy()[s]
            Rs = _R(Q)
            pl = Rs.T @ (anchor - X)
            hx, hy, hz = shapes[s]["half"]
            if abs(pl[0]) <= hx and -1e-4 <= pl[2] <= 2*hz and abs(pl[1]) <= hy + 3e-4:
                on[j] = 0
                changed = True
                r = jd.get("radius") or 1.5e-3
                debt = ts.rb["debt"].numpy()
                debt[int(ts.h_s_body[s])] += PR.v(PR.JOINT_R)*math.pi*r*r
                ts.rb["debt"].assign(debt)
                # the joint's springs vanish with it: their stored energy leaves the elastic energy here and is booked
                # with the interface releases (the kernels' counterpart, bodies.k_bond_release)
                ts.book("iface", _joint_energy(ts, j, st))
                world.events.append(dict(tick=world.tick, event="joint_cut", joint=jd.get("name")))
                break
    if changed:
        ts.joints["on"].assign(on)


def _joint_energy(ts, j, st):
    """Stored energy of joint j's springs now (the potential of bodies.k_body_joints: k d^2/2 + sum ka rv^2/2)."""
    from scipy.spatial.transform import Rotation
    jn = ts.joints
    a, b = int(jn["a"].numpy()[j]), int(jn["b"].numpy()[j])
    Ra, Rb = Rotation.from_quat(st["q"][a]), Rotation.from_quat(st["q"][b])
    d = (st["p"][b] + Rb.apply(jn["pb"].numpy()[j])) - (st["p"][a] + Ra.apply(jn["pa"].numpy()[j]))
    qe = (Rb*(Ra*Rotation.from_quat(jn["q0"].numpy()[j])).inv()).as_quat()
    s_ = -1. if qe[3] < 0 else 1.
    rv = Ra.inv().apply(qe[:3]*2*s_)
    return float(.5*float(jn["k"].numpy()[j])*float(d @ d) + .5*float((jn["ka"].numpy()[j]*rv*rv).sum()))


def tissue_tick(world, q_before):
    c = world.config
    ts, cut = world.tissue, world.cut
    t0 = time.perf_counter()
    q_after = world.state.body_q.numpy().astype(float)
    poses, tw = _pose_shapes(world, q_before, q_after)
    t_tick = world.tick*c.rigid_dt - c.rigid_dt
    if world.phantom_knife is not None:
        ts.set_driven(world.knife_body0, *_rig_tick(world.phantom_knife, t_tick, c.rigid_dt))
    elif world.knife_mode == "held":
        _drive_handle(world, q_before, q_after)
        _blade_in_air(world, q_before, q_after)
    for i, fn in world.driven_bones.items():
        ts.set_driven(i, *_rig_tick(fn, t_tick, c.rigid_dt))
    blade = world.scene.blade
    k0 = world.blade_shape0
    held = world.phantom_knife is None and world.knife_mode == "held"
    chain0 = _chain_momentum(world) if held else None
    t1 = time.perf_counter()
    capped0 = ts.substep_capped
    w = ts.step_tick(c.rigid_dt)
    if ts.substep_capped > capped0:
        # the tissue's stability bound asked for more substeps than the cap: this tick ran outside it (recorded)
        last = next((e for e in reversed(world.events) if e.get("event") == "substep_cap"), None)
        if last is None or world.tick - last["tick"] > 50:
            world.events.append(dict(tick=world.tick, event="substep_cap", needed=int(ts.last_n_need),
                                     cap=int(ts.max_substeps)))
    t2 = time.perf_counter()
    cut.record()
    ta = time.perf_counter()
    tb = tc = ta
    if world.tick % c.split_every == 0:
        n0 = ts.n
        cut.split_booked()
        tb = time.perf_counter()
        if getattr(world, "tearing", None) is not None:
            if ts.n > n0 and world.cut.torn:
                # a knife batch that separates torn faces' nodes: their bonds re-lump the stability mass, booked as
                # the split's own changes are (CrackRecorder.split_booked)
                e0_ = ts.energies()
                if world.tearing.bond_new_twins(n0):
                    e1_ = ts.energies()
                    ts.book("topo", sum(e0_[k] - e1_[k] for k in ("ke", "pe", "psi")))
            world.tearing.step()
        tc = time.perf_counter()
    # the topology bucket's parts
    tm = world.timing
    tm["topo_record_s"] = tm.get("topo_record_s", 0.) + (ta - t2)
    tm["topo_split_s"] = tm.get("topo_split_s", 0.) + (tb - ta)
    tm["topo_tear_s"] = tm.get("topo_tear_s", 0.) + (tc - tb)
    blades = list(range(k0, k0 + blade.segments))
    if ts.n_bties:
        ts.release_body_ties(blades)
    if getattr(ts, "n_bonds", 0):
        ts.release_bonds(blades)
    if getattr(ts, "n_rods", 0) and world.tick % 5 == 0:
        ts.break_rods()
    if world.assembly is not None and world.tick % 5 == 0:
        _joint_cut_rule(world, poses, blades)
    t3 = time.perf_counter()
    # reactions -> rigid bodies (about each body's COM), filtered
    body_com = world.model.body_com.numpy()
    wb = np.zeros((world.model.body_count, 6))
    fh, th = ts.driven_load()
    kh = world.scene.bodies["knife_handle"]
    world.knife_load = (fh[world.knife_body0].copy(), th[world.knife_body0].copy())
    if held and getattr(world, "_blade_rigid", False):
        # the blade follows MuJoCo's rigid knife (in air, _blade_in_air): no chain dynamics to hand back, only the
        # handle's own contact load (MuJoCo carries the knife's mass and weight)
        F, tau = fh[world.knife_body0].copy(), th[world.knife_body0].copy()
        world.knife_ext_load = (F, tau)
        wb[kh, :3] += F
        wb[kh, 3:] += tau
    elif held:
        F, tau = _knife_external_load(world, chain0, fh[world.knife_body0], th[world.knife_body0], q_after)
        world.knife_ext_load = (F, tau)
        wb[kh, :3] += F
        wb[kh, 3:] += tau                           # about MuJoCo's knife COM (handle + blade)
    for i, (b, pl, ql) in enumerate(world.tissue_owner):
        if b < 0 or not np.any(w[i]):
            continue
        F, tau = w[i, :3], w[i, 3:]
        Xs = poses[i]
        cs = Xs[:3] + _R(Xs[3:]) @ np.asarray(world.tissue_shapes[i].get("com_local", (0, 0, 0)))
        cb = q_after[b, :3] + _R(q_after[b, 3:]) @ body_com[b]
        wb[b, :3] += F
        wb[b, 3:] += tau + np.cross(cs - cb, F)
    a = c.rigid_dt/(REACTION_TAU + c.rigid_dt)
    world.tissue_filtered = (1 - a)*world.tissue_filtered + a*wb
    world.body_wrench = world.tissue_filtered.astype(np.float32)
    world.last_tissue_wrench = wb
    world.timing["tissue_s"] += t2 - t1
    world.timing["couple_s"] += (t1 - t0) + (time.perf_counter() - t3)
    world.timing["topo_s"] += t3 - t2
