"""Soil: Newton 1.5 implicit MPM on the GPU, two-way coupled to the robot world's rigid bodies.

One Newton model holds the soil particles, the static colliders (each a zero-mass kinematic body so it can carry its own
interface friction and adhesion) and one dynamic collider body per coupled rigid body of the MuJoCo world (tools,
stones, root ball, post, robot hands and base). Each soil step (params.SOIL_DT, every SOIL_TICKS robot ticks):

  1. the caller writes each coupled body's pose and its velocity *minus the previous soil wrench's effect over the
     step* ("rewind", Newton's example_mpm_twoway_coupling): the MPM then solves soil + contact with the body's own
     mass and inertia inside the contact solve and returns the *total* soil impulse, not an increment;
  2. one CUDA graph replays the MPM step and a kernel that sums the collider impulses per body into a wrench at the
     body's centre of mass (world frame);
  3. the caller applies the wrench to the MuJoCo body over the next SOIL_TICKS ticks.

With the rewind a held tool's response is under-relaxed by m_tool / m_effective (the arm adds inertia the MPM does not
see): stable, converging over a few soil steps.

Particles carry their material id, origin tag (in-situ block / placed pile) and initial position (ledgers). The grid is
a rebuildable sparse grid with reserved capacity (graph-capturable); overflow, a non-finite particle or a particle faster
than MAX_SOIL_SPEED faults the soil (as in filleting).
"""

import math
import time
import numpy as np
import warp as wp

from . import params as P
from . import materials as M

ORIGIN_BLOCK, ORIGIN_PILE = 0, 1


# ------------------------------------------------------------------------------ kernels
@wp.kernel
def k_body_forces(scale: float, collider_ids: wp.array(dtype=int), impulses: wp.array(dtype=wp.vec3),
                  impulse_pos: wp.array(dtype=wp.vec3), collider_body: wp.array(dtype=int),
                  body_q: wp.array(dtype=wp.transform), body_com: wp.array(dtype=wp.vec3),
                  body_f: wp.array(dtype=wp.spatial_vector)):
    # scale = -cell_volume / dt: the impulse field's dofs are per unit volume and act on the soil (sign flip), as in
    # SolverImplicitMPM.collect_collider_impulses (which allocates a scaled copy, so it cannot sit in a CUDA graph)
    i = wp.tid()
    cid = collider_ids[i]
    if cid >= 0 and cid < collider_body.shape[0]:
        b = collider_body[cid]
        if b < 0:
            return
        f = impulses[i]*scale
        r = impulse_pos[i] - wp.transform_point(body_q[b], body_com[b])
        wp.atomic_add(body_f, b, wp.spatial_vector(f, wp.cross(r, f)))


@wp.kernel
def k_cut(q: wp.array(dtype=wp.vec3), jp: wp.array(dtype=float), jcut: wp.array(dtype=float),
          body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector), body_com: wp.array(dtype=wp.vec3),
          cut_body: wp.array(dtype=int),
          cut_tf: wp.array(dtype=wp.transform), cut_half: wp.array(dtype=wp.vec3), cut_kind: wp.array(dtype=int),
          cut_edge: wp.array(dtype=wp.vec3), n_cut: int, band: float, vmin: float):
    """Blade cut closure: a particle a cutting region passes through edge-first is remolded to its loose state,
    J_p >= J_p loose (cohesion and tension lost, friction kept). A 2-3 mm blade parts the soil along a surface the 2 cm
    grid cannot hold open; without this the particles pushed aside close behind the blade and stay bonded (B5: a spit
    cut on four sides still did not lift). The cut is the surface the edge sweeps: the band (half a voxel) widens a plate
    across its thickness only (a pick or spike radially only) -- never ahead of the edge or past the plate's sides -- and
    only while the region advances along its edge direction (`cut_edge`, region frame) faster than vmin: a blade
    withdrawn, levered face-first or scraped along a floor does not cut, so faces are not widened by the band and
    floors a blade touches are not re-loosened."""
    i = wp.tid()
    x = q[i]
    for c in range(n_cut):
        X = body_q[cut_body[c]]
        xb = wp.transform_point(wp.transform_inverse(X), x)
        d = wp.transform_point(wp.transform_inverse(cut_tf[c]), xb)
        hh = cut_half[c]
        inside = False
        if cut_kind[c] == 0:
            inside = wp.abs(d[0]) <= hh[0] + band and wp.abs(d[1]) <= hh[1] and wp.abs(d[2]) <= hh[2]
        else:
            inside = wp.sqrt(d[0]*d[0] + d[1]*d[1]) <= hh[0] + band and wp.abs(d[2]) <= hh[2]
        if inside:
            v = body_qd[cut_body[c]]
            vp = wp.spatial_top(v) + wp.cross(wp.spatial_bottom(v), x - wp.transform_point(X, body_com[cut_body[c]]))
            e = wp.transform_vector(X, wp.transform_vector(cut_tf[c], cut_edge[c]))
            if wp.dot(vp, e) > vmin:
                jp[i] = wp.max(jp[i], jcut[i])


@wp.kernel
def k_guard(q: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    x = q[i]
    v = qd[i]
    s = wp.length(v)
    if not (wp.isfinite(x[0]) and wp.isfinite(x[1]) and wp.isfinite(x[2]) and wp.isfinite(s)):
        wp.atomic_add(out, 0, 1.0)
    else:
        wp.atomic_max(out, 1, s)


# ------------------------------------------------------------------------------ geometry helpers
def _rot(quat):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(quat).as_matrix()


def sdf(points, shape):
    """Signed distance (m, negative inside) of points (N, 3) to a shape dict(kind, size, pos, quat) given in the frame of
    the points. Kinds: box (half extents), cylinder / capsule (radius, half height along local z), ellipsoid (radii,
    approximate), sphere (radius,)."""
    R = _rot(shape.get("quat", (0., 0., 0., 1.)))
    x = (np.asarray(points, float) - np.asarray(shape["pos"], float)) @ R
    k, s = shape["kind"], shape["size"]
    if k == "box":
        d = np.abs(x) - np.asarray(s, float)
        return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0)
    if k == "cylinder":
        r, hh = s
        dr = np.hypot(x[:, 0], x[:, 1]) - r
        dz = np.abs(x[:, 2]) - hh
        return np.hypot(np.maximum(dr, 0), np.maximum(dz, 0)) + np.minimum(np.maximum(dr, dz), 0)
    if k == "capsule":
        r, hh = s
        z = np.clip(x[:, 2], -hh, hh)
        return np.linalg.norm(x - np.stack([np.zeros_like(z), np.zeros_like(z), z], 1), axis=1) - r
    if k == "sphere":
        return np.linalg.norm(x, axis=1) - s[0]
    if k == "ellipsoid":
        r = np.asarray(s, float)
        k0 = np.linalg.norm(x/r, axis=1)
        k1 = np.linalg.norm(x/(r*r), axis=1)
        return k0*(k0 - 1.0)/np.maximum(k1, 1e-12)
    if k == "mesh":
        v, f = shape["mesh"]
        import trimesh
        tm = trimesh.Trimesh(np.asarray(v)*np.asarray(shape.get("scale", (1, 1, 1))), np.asarray(f), process=False)
        return -trimesh.proximity.signed_distance(tm, x)
    raise ValueError(k)


def unit_sphere(n_lat=12, n_lon=24):
    """UV sphere of radius 1: (vertices (N, 3) float32, triangles (T, 3) int32), outward winding."""
    v = [(0., 0., 1.)]
    for i in range(1, n_lat):
        th = math.pi*i/n_lat
        for j in range(n_lon):
            ph = 2*math.pi*j/n_lon
            v.append((math.sin(th)*math.cos(ph), math.sin(th)*math.sin(ph), math.cos(th)))
    v.append((0., 0., -1.))
    f = []
    for j in range(n_lon):
        f.append((0, 1 + j, 1 + (j + 1) % n_lon))
    for i in range(n_lat - 2):
        a, b = 1 + i*n_lon, 1 + (i + 1)*n_lon
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            f += [(a + j, b + j, b + j2), (a + j, b + j2, a + j2)]
    last = len(v) - 1
    a = 1 + (n_lat - 2)*n_lon
    for j in range(n_lon):
        f.append((a + j, last, a + (j + 1) % n_lon))
    return np.asarray(v, np.float32), np.asarray(f, np.int32)


def smooth_noise(xy, scale, amp, seed):
    """Smooth 2-D noise (sum of a few random plane waves), amplitude amp (m) at wavelength ~scale (m)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(xy))
    for _ in range(6):
        th = rng.uniform(0, 2*np.pi)
        lam = scale*rng.uniform(0.6, 1.6)
        ph = rng.uniform(0, 2*np.pi)
        out += np.cos((np.cos(th)*xy[:, 0] + np.sin(th)*xy[:, 1])*2*np.pi/lam + ph)
    return amp*out/np.sqrt(3.0)


def block_particles(lo, hi, layers, h, seed=0, undulation=0.01, jitter=0.0):
    """Lattice particles (spacing h) filling the box [lo, hi]; `layers` = [(thickness, material), ...] from the top
    (the last layer fills to the bottom). Layer interfaces undulate smoothly by +-undulation (m). Returns
    (pos (N, 3) float32, material index (N,) int32)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.maximum(np.round((hi - lo)/h).astype(int), 1)
    axes = [lo[i] + (np.arange(n[i]) + 0.5)*(hi[i] - lo[i])/n[i] for i in range(3)]
    pos = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    if jitter:
        pos += np.random.default_rng(seed + 7).uniform(-jitter, jitter, pos.shape)*h
    depth = hi[2] - pos[:, 2]
    mat = np.full(len(pos), M.NAMES.index(layers[-1][1]), np.int32)
    top = 0.0
    for k, (th, name) in enumerate(layers[:-1]):
        und = smooth_noise(pos[:, :2], 0.35, undulation, seed + 101*k) if undulation else 0.
        bot = top + th + und
        sel = (depth >= top) & (depth < bot) if k == 0 else (depth >= prev) & (depth < bot)
        mat[sel] = M.NAMES.index(name)
        prev = bot
        top = top + th
    return pos.astype(np.float32), mat


def pile_particles(center, radius, mass, material, h, seed=0, slope_deg=30.0, base_z=0.0, fraction=None):
    """A loose spoil mound (cone frustum with the given base radius, side slope <= slope_deg) holding `mass` kg of
    `material` at loose packing (spacing h / fraction^(1/3), fraction = CRITICAL_FRACTION by default)."""
    rho = P.MATERIALS[material]["rho"]
    f = fraction or P.CRITICAL_FRACTION
    hl = h/f**(1.0/3.0)
    vol = mass/rho/f           # loose volume
    t = math.tan(math.radians(slope_deg))
    # frustum of base radius R and slope t holding vol: solve for height H (cone if needed)
    R = radius
    Hc = R*t
    vc = math.pi*R*R*Hc/3
    if vol >= vc:
        R = (3*vol/(math.pi*t))**(1/3)
        H = R*t
        rtop = 0.0
    else:
        lo_, hi_ = 0.0, Hc
        for _ in range(60):
            H = 0.5*(lo_ + hi_)
            rt = R - H/t
            v = math.pi*H/3*(R*R + R*rt + rt*rt)
            lo_, hi_ = (H, hi_) if v < vol else (lo_, H)
        rtop = R - H/t
    n = int(math.ceil(2*R/hl)) + 1
    g = (np.arange(n) - (n - 1)/2)*hl
    zz = (np.arange(int(math.ceil(H/hl)) + 1) + 0.5)*hl
    X, Y, Z = np.meshgrid(g, g, zz, indexing="ij")
    rr = np.hypot(X, Y)
    inside = rr <= np.maximum(R - Z/t, rtop if rtop > 0 else -1)
    pts = np.stack([X[inside], Y[inside], Z[inside]], 1)
    rng = np.random.default_rng(seed)
    pts += rng.uniform(-0.15, 0.15, pts.shape)*hl
    # trim to the exact mass
    m_p = rho*h**3
    want = int(round(mass/m_p))
    if len(pts) > want:
        pts = pts[np.argsort(pts[:, 2])[:want]]
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    pts[:, 2] += base_z
    return pts.astype(np.float32), np.full(len(pts), M.NAMES.index(material), np.int32)


# ------------------------------------------------------------------------------ the soil system
class SoilFault(RuntimeError):
    pass


class SoilSystem:
    """spec: dict(voxel, moisture, particles=dict(pos, mat, loose, origin), statics=[dict(name, iface, shapes)],
    bounds=(lo, hi)) ; couplers: [dict(name, shapes, mass, com, inertia, iface, pose)] -- pose is (p, quat xyzw) of the
    body origin; com/inertia in the body frame (MuJoCo's values)."""

    def __init__(self, spec, couplers=(), device="cuda:0", capture=True, iterations=None, tolerance=None,
                 velocity_mode="forward", critical_fraction=None, extra_config=None, dt=None):
        import newton
        from newton.solvers import SolverImplicitMPM
        self.newton, self.wp = newton, wp
        self.device = wp.get_device(device)
        self.spec = spec
        self.voxel = h_v = float(spec["voxel"])
        self.dt = float(dt or P.SOIL_DT)                 # dt: convergence benches only (the world steps at SOIL_DT)
        self.moisture = spec.get("moisture", "moist")
        t0 = time.perf_counter()
        pp = spec["particles"]
        pos = np.asarray(pp["pos"], np.float32)
        mat = np.asarray(pp["mat"], np.int32)
        loose = np.asarray(pp.get("loose", np.zeros(len(pos), bool)), bool)
        self.origin = np.asarray(pp.get("origin", np.where(loose, ORIGIN_PILE, ORIGIN_BLOCK)), np.int8)
        self.pos0 = pos.copy()
        self.mat = mat
        self.n = len(pos)
        h = h_v/P.PPC_AXIS
        self.h = h
        params = [M.solver_params(name, self.moisture, override=(spec.get("overrides") or {}).get(name))
                  for name in M.NAMES]
        self.material_params = params
        rho = np.array([p["rho"] for p in params])[mat]
        frac = np.array([p["fraction"] for p in params])[mat]
        self.mass = (rho*h**3).astype(np.float32)
        radius = (0.5*h*frac**(1/3)).astype(np.float32)
        jp = np.array([p["jp0"] for p in params])[mat].astype(np.float32)
        jp[loose] = np.array([p["jp_loose"] for p in params])[mat][loose]
        b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -P.GRAVITY))
        SolverImplicitMPM.register_custom_attributes(b)
        self.static_names, self.coupler_names = [], []
        adhesion, masses, coms, invI = [], [], [], []
        for st in spec.get("statics", []):
            body = b.add_body(xform=wp.transform_identity(), mass=0.0, is_kinematic=True, label=f"static/{st['name']}")
            mu, ca = st["iface"]
            for s in st["shapes"]:
                self._add_shape(b, body, s, s.get("mu", mu))          # per-shape friction within one collider
            self.static_names.append(st["name"])
            adhesion.append(M.contact_normal_offset(mu, float(ca)*(P.MOISTURE[self.moisture]["ca"] if st.get("moisture_scaled", True) else 1.0)))
            masses.append(0.0); coms.append((0., 0., 0.)); invI.append(np.zeros((3, 3)))
        self.n_static = len(self.static_names)
        for c in couplers:
            p, q = c["pose"]
            body = b.add_body(xform=wp.transform(wp.vec3(*map(float, p)), wp.quat(*map(float, q))), mass=float(c["mass"]),
                              label=f"coupled/{c['name']}")
            mu, ca = c["iface"]
            mo = P.MOISTURE[self.moisture]
            for s in c["shapes"]:
                self._add_shape(b, body, s, mu*mo["mut"])
            self.coupler_names.append(c["name"])
            adhesion.append(M.contact_normal_offset(mu*mo["mut"], float(ca)*mo["ca"]))
            masses.append(float(c["mass"])); coms.append(tuple(map(float, c["com"])))
            I = np.asarray(c["inertia"], float).reshape(3, 3)
            invI.append(np.linalg.inv(I) if c["mass"] > 0 else np.zeros((3, 3)))       # mass 0: kinematic collider
        self.n_coupled = len(self.coupler_names)
        # cutting regions (blade cut closure, k_cut): body index, shape transform, half extents, kind (0 box, 1 cylinder)
        cuts = []
        for ci, c in enumerate(couplers):
            for cs in c.get("cuts", []):
                z = cs["size"]
                half = (z[0], z[1], z[2]) if cs["kind"] == "box" else (z[0], z[0], z[1])
                cuts.append((self.n_static + ci, tuple(cs["pos"]) + tuple(cs.get("quat", (0., 0., 0., 1.))), half,
                             0 if cs["kind"] == "box" else 1, tuple(cs.get("edge", (0., 0., -1.)))))
        self.n_cut = len(cuts)
        self.cut_band = float(P.CUT_CLOSURE["band_voxels"]*h_v)
        self.cut_vmin = float(P.CUT_CLOSURE["v_min"])
        cuts = cuts or [(0, (0., 0., 0., 0., 0., 0., 1.), (0., 0., 0.), 0, (0., 0., -1.))]
        self.cut_body = wp.array([c[0] for c in cuts], dtype=int, device=self.device)
        self.cut_tf = wp.array([wp.transform(wp.vec3(*c[1][:3]), wp.quat(*c[1][3:])) for c in cuts], dtype=wp.transform,
                               device=self.device)
        self.cut_half = wp.array([c[2] for c in cuts], dtype=wp.vec3, device=self.device)
        self.cut_kind = wp.array([c[3] for c in cuts], dtype=int, device=self.device)
        self.cut_edge = wp.array([c[4] for c in cuts], dtype=wp.vec3, device=self.device)
        attrs = {f"mpm:{k}": [params[m]["attrs"][k] for m in range(len(params))] for k in params[0]["attrs"]}
        per = {k: np.asarray(v, np.float32)[mat].tolist() for k, v in attrs.items()}
        b.add_particles(pos=pos.tolist(), vel=np.zeros_like(pos).tolist(), mass=self.mass.tolist(), radius=radius.tolist(),
                        custom_attributes=per)
        self.model = model = b.finalize(device=self.device)
        cfg = SolverImplicitMPM.Config()
        cfg.voxel_size = h_v
        cfg.max_iterations = int(iterations or P.MPM_ITERATIONS)
        cfg.tolerance = float(tolerance or P.MPM_TOLERANCE)
        cfg.collider_velocity_mode = velocity_mode
        cfg.critical_fraction = float(P.SOLVER_CRITICAL_FRACTION if critical_fraction is None else critical_fraction)
        cfg.grid_type = "sparse"
        cfg.collider_basis = P.COLLIDER_BASIS
        # capacity from the material's own cells (8 particles per cell at bank packing), not the bounding box: the
        # implicit solver keeps several 6x6 blocks per active cell, and a box-sized reserve of a 2.6 x 2.6 x 1.1 m yard
        # domain exhausts the shared GPU's free memory
        cfg.max_active_cell_count = int(max(4096, P.GRID_CAPACITY*self.n/P.PPC_AXIS**3))
        for k, v in (extra_config or {}).items():
            setattr(cfg, k, v)
        self.config = cfg
        self.solver = SolverImplicitMPM(model, config=cfg)
        self.body_mass = wp.array(np.asarray(masses, np.float32), dtype=float, device=self.device)
        self.body_com = wp.array(np.asarray(coms, np.float32), dtype=wp.vec3, device=self.device)
        self.body_inv_inertia = wp.array(np.asarray(invI, np.float32), dtype=wp.mat33, device=self.device)
        if len(masses):
            self.solver.setup_collider(body_mass=self.body_mass, body_com=self.body_com, body_inv_inertia=self.body_inv_inertia,
                                       collider_adhesion=adhesion)
        self.state = model.state()
        self.state.mpm.particle_Jp.assign(jp)
        self.jp_cut = wp.array(np.array([p["jp_loose"] for p in params])[mat].astype(np.float32), dtype=float, device=self.device)
        self.collider_body = self.solver.collider_body_index
        nb = model.body_count
        self.body_f = wp.zeros(nb, dtype=wp.spatial_vector, device=self.device)
        self.guard = wp.zeros(2, dtype=float, device=self.device)
        self._host_q = wp.zeros(max(self.n_coupled, 1), dtype=wp.transform, device="cpu", pinned=self.device.is_cuda)
        self._host_qd = wp.zeros(max(self.n_coupled, 1), dtype=wp.spatial_vector, device="cpu", pinned=self.device.is_cuda)
        self._host_f = wp.zeros(nb, dtype=wp.spatial_vector, device="cpu", pinned=self.device.is_cuda)
        self._host_guard = wp.zeros(2, dtype=float, device="cpu", pinned=self.device.is_cuda)
        self.graph = None
        self.steps = 0
        self.timing = dict(build_s=time.perf_counter() - t0, step_s=0.0, steps=0)
        self.fault = None
        self.capture_enabled = bool(capture) and self.device.is_cuda
        self._imp = None

    # -------------------------------------------------------------- construction helpers
    def _add_shape(self, b, body, s, mu):
        newton, wp_ = self.newton, self.wp
        cfg = newton.ModelBuilder.ShapeConfig(mu=float(mu), density=0.0, margin=float(s.get("margin", 0.0)),
                                              has_shape_collision=False, has_particle_collision=True)
        tf = wp_.transform(wp_.vec3(*map(float, s["pos"])), wp_.quat(*map(float, s.get("quat", (0., 0., 0., 1.)))))
        k, z = s["kind"], s["size"]
        if k == "box":
            return b.add_shape_box(body, xform=tf, hx=float(z[0]), hy=float(z[1]), hz=float(z[2]), cfg=cfg)
        if k == "cylinder":
            return b.add_shape_cylinder(body, xform=tf, radius=float(z[0]), half_height=float(z[1]), cfg=cfg)
        if k == "capsule":
            return b.add_shape_capsule(body, xform=tf, radius=float(z[0]), half_height=float(z[1]), cfg=cfg)
        if k == "sphere":
            return b.add_shape_sphere(body, xform=tf, radius=float(z[0]), cfg=cfg)
        if k == "ellipsoid":
            # the MPM collider rasterises meshes (Newton 1.5 has no ellipsoid SDF there): a scaled unit sphere
            v, f = unit_sphere()
            mesh = newton.Mesh(v*np.asarray(z, np.float32), f.reshape(-1))
            return b.add_shape_mesh(body, xform=tf, mesh=mesh, cfg=cfg)
        if k == "mesh":
            v, f = s["mesh"]
            mesh = newton.Mesh(np.asarray(v, np.float32), np.asarray(f, np.int32).reshape(-1))
            return b.add_shape_mesh(body, xform=tf, mesh=mesh, scale=tuple(s.get("scale", (1., 1., 1.))), cfg=cfg)
        raise ValueError(k)

    # -------------------------------------------------------------- stepping
    def _device_step(self):
        if self.n_cut and self.cut_band > 0.0:
            wp.launch(k_cut, dim=self.n, inputs=[self.state.particle_q, self.state.mpm.particle_Jp, self.jp_cut,
                                                 self.state.body_q, self.state.body_qd, self.body_com, self.cut_body, self.cut_tf,
                                                 self.cut_half, self.cut_kind, self.cut_edge, self.n_cut, self.cut_band,
                                                 self.cut_vmin],
                      device=self.device)
        self.solver.step(self.state, self.state, None, None, self.dt)
        self.body_f.zero_()
        s = self.state
        imp, ipos, ids = s.impulse_field.dof_values, s.collider_position_field.dof_values, s.collider_ids
        n = min(imp.shape[0], ids.shape[0], ipos.shape[0])
        if n:
            wp.launch(k_body_forces, dim=n, inputs=[-self.voxel**3/self.dt, ids, imp, ipos, self.collider_body,
                                                    self.state.body_q, self.body_com, self.body_f], device=self.device)
        self.guard.zero_()
        wp.launch(k_guard, dim=self.n, inputs=[self.state.particle_q, self.state.particle_qd, self.guard], device=self.device)
        return (imp.ptr, ipos.ptr, ids.ptr, n)

    def set_masses(self, masses):
        """Per-coupled-body MPM masses for the next step (0 = kinematic: the soil cannot move the body, its reaction is
        the force needed to move the soil out of the body's way)."""
        if not self.n_coupled:
            return
        if not hasattr(self, "_host_m"):
            self._host_m = wp.zeros(self.n_static + self.n_coupled, dtype=float, device="cpu", pinned=self.device.is_cuda)
        m = np.asarray(masses, np.float32)
        if getattr(self, "_last_m", None) is not None and np.array_equal(m, self._last_m):
            return
        self._host_m.numpy()[self.n_static:] = m
        wp.copy(self.body_mass, self._host_m, dest_offset=self.n_static, src_offset=self.n_static, count=self.n_coupled)
        self._last_m = m.copy()

    def set_bodies(self, q, qd):
        """q: (n_coupled, 7) [p, quat xyzw] body-origin transforms; qd: (n_coupled, 6) [v_com, w] world frame."""
        if not self.n_coupled:
            return
        self._host_q.numpy()[:self.n_coupled] = np.asarray(q, np.float32)
        self._host_qd.numpy()[:self.n_coupled] = np.asarray(qd, np.float32)
        wp.copy(self.state.body_q, self._host_q, dest_offset=self.n_static, src_offset=0, count=self.n_coupled)
        wp.copy(self.state.body_qd, self._host_qd, dest_offset=self.n_static, src_offset=0, count=self.n_coupled)

    def step(self, q=None, qd=None):
        """One soil step. Returns the soil wrench on each coupled body (n_coupled, 6): force and torque about its COM,
        world frame, to be applied over the next soil step."""
        if self.fault:
            raise SoilFault(self.fault)
        t0 = time.perf_counter()
        if q is not None:
            self.set_bodies(q, qd)
        try:
            with wp.ScopedDevice(self.device):
                if self.graph is not None:
                    wp.capture_launch(self.graph)
                elif self.capture_enabled and self.steps >= 2:
                    # the step's scratch buffers are borrowed per step outside a graph; inside one they are the graph's
                    # own allocations, so the whole step (solver + wrench kernel + guard) is captured once and replayed
                    self._device_step()
                    with wp.ScopedCapture(device=self.device) as cap:
                        self._device_step()
                    self.graph = cap.graph
                    self.capture_note = f"captured at soil step {self.steps}"
                else:
                    self._imp = self._device_step()
                wp.copy(self._host_f, self.body_f)
                wp.copy(self._host_guard, self.guard)
                wp.synchronize_device(self.device)
                self.solver.check_sparse_grid_rebuild_status()
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise SoilFault(self.fault) from exc
        g = self._host_guard.numpy()
        if g[0] > 0:
            self.fault = f"non-finite soil particles ({int(g[0])}) at soil step {self.steps}"
            raise SoilFault(self.fault)
        if g[1] > P.MAX_SOIL_SPEED:
            self.fault = f"soil particle at {g[1]:.1f} m/s at soil step {self.steps} (> {P.MAX_SOIL_SPEED} m/s)"
            raise SoilFault(self.fault)
        self.max_speed = float(g[1])
        self.steps += 1
        dt = time.perf_counter() - t0
        self.timing["step_s"] += dt
        self.timing["steps"] += 1
        f = self._host_f.numpy()[self.n_static:self.n_static + self.n_coupled].astype(float)
        if not np.isfinite(f).all():
            self.fault = f"non-finite soil wrench at soil step {self.steps}"
            raise SoilFault(self.fault)
        return f

    def settle(self, seconds=None, q=None, qd=None):
        """Soil-only gravity settle with the coupled bodies held where they are (zero velocity)."""
        n = int(round((P.SOIL_SETTLE_S if seconds is None else seconds)/self.dt))
        if q is not None:
            self.set_bodies(q, np.zeros((self.n_coupled, 6)))
        for _ in range(n):
            self.step()
        return n

    # -------------------------------------------------------------- state access
    def particles(self):
        s = self.state
        return dict(pos=s.particle_q.numpy(), vel=s.particle_qd.numpy(), jp=s.mpm.particle_Jp.numpy(), mat=self.mat,
                    mass=self.mass, origin=self.origin, pos0=self.pos0)

    STATE_FIELDS = ("particle_q", "particle_qd")
    MPM_FIELDS = ("particle_qd_grad", "particle_elastic_strain", "particle_Jp", "particle_stress", "particle_transform")

    def snapshot(self):
        s = self.state
        snap = {k: getattr(s, k).numpy().copy() for k in self.STATE_FIELDS}
        snap.update({k: getattr(s.mpm, k).numpy().copy() for k in self.MPM_FIELDS})
        snap["body_q"] = s.body_q.numpy().copy()
        snap["body_qd"] = s.body_qd.numpy().copy()
        snap["steps"] = self.steps
        return snap

    def restore(self, snap):
        s = self.state
        for k in self.STATE_FIELDS:
            getattr(s, k).assign(snap[k])
        for k in self.MPM_FIELDS:
            getattr(s.mpm, k).assign(snap[k])
        s.body_q.assign(snap["body_q"])
        s.body_qd.assign(snap["body_qd"])
        self.steps = snap["steps"]
        self.fault = None
        try:
            self.solver._clear_sparse_grid_rebuild_status()
        except Exception:
            pass

    def info(self):
        return dict(particles=int(self.n), voxel_m=self.voxel, steps=int(self.steps), dt_s=self.dt,
                    iterations=int(self.config.max_iterations), tolerance=float(self.config.tolerance),
                    critical_fraction=float(self.config.critical_fraction), grid_capacity=int(self.config.max_active_cell_count),
                    graph=self.graph is not None, coupled=list(self.coupler_names), statics=list(self.static_names),
                    moisture=self.moisture, build_s=round(self.timing["build_s"], 2),
                    ms_per_step=round(1e3*self.timing["step_s"]/max(1, self.timing["steps"]), 2))
