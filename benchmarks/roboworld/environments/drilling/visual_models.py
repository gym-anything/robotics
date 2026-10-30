"""Procedural appearance models (visual only: no collision, no mass, no physics parameter comes from here).

Built with manifold3d (booleans, hulls, lathes, twisted extrusions) and direct swept-tube meshes, each fitted to the
collider it draws within a few millimetres:

  * rotary_hammer(): the SDS-plus hammer's housing (DeWalt DCH273 class) on the housing body, in the housing frame
    (x along the bit, z up, origin at the front of the chuck sleeve), following tools.RotaryHammer's colliders;
  * shape_visual(): a render-side registry keyed by tool class or part kind and shape label. It replaces a collision
    primitive's drawing (render.export) by a detailed model in that shape's own frame: the robot cart, the kit trolley,
    the steel workbench and the press cabinet stand (static colliders), the hammer's chuck, trigger, mode dial and
    depth rod, fluted bits, hex-head and carriage bolts, wedge anchors with nut and washer, the vacuum wand, the hole
    brush, the ball-peen hammer, the centre punch, the press vise and the hold-down clamps;
  * the HEPA wet/dry vacuum drawn on the vacuum unit's colliders (drum_vacuum(), RIDGID RV2400HF class; a boxy
    DWV012-class extractor, shop_vacuum(), when the unit has no drum colliders) and its rocker switch on the switch's
    own body (rocker_switch()), when the layout has a vacuum (render.export; its hose: render.Hose);
  * the welded test stand under a fixed test block of the component benches (test_stand(), dressing.py);
  * room furniture shared with dressing.py (workbench(), press_cabinet(), pegboard(), conduit runs, junction boxes).

A part is (name, vertices, triangles, material, opts); opts may carry {"smooth": True} (vertex normals) or
{"crease": degrees} (normals split at sharper edges; the default) and {"uv": per_metre} for textured materials.
"""

import math
import numpy as np


def _m3():
    import manifold3d as m3
    return m3


# ====================================================================================== geometry helpers
def _rot_z_to(axis):
    """3x3 rotation taking +z onto `axis`."""
    a = np.asarray(axis, float)
    a = a/np.linalg.norm(a)
    z = np.array([0., 0., 1.])
    c = float(z @ a)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1., -1., -1.])
    v = np.cross(z, a)
    s = np.linalg.norm(v)
    k = v/s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang)*K + (1 - math.cos(ang))*K @ K


def _rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _tf(man, R=None, t=(0., 0., 0.)):
    T = np.zeros((3, 4))
    T[:, :3] = np.eye(3) if R is None else R
    T[:, 3] = t
    return man.transform(T.tolist())


def box(c, size):
    m3 = _m3()
    return m3.Manifold.cube([float(s) for s in size], True).translate([float(x) for x in c])


def box2(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return box(0.5*(lo + hi), hi - lo)


def rbox(c, size, r, n=12):
    """Rounded box: hull of corner spheres (r clipped to the smallest half extent)."""
    m3 = _m3()
    c = np.asarray(c, float)
    size = np.asarray(size, float)
    r = float(min(r, 0.499*size.min()))
    h = size/2 - r
    pts = [c + h*np.array(s) for s in ((sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))]
    return m3.Manifold.batch_hull([m3.Manifold.sphere(r, n).translate([float(x) for x in p]) for p in pts])


def rbox2(lo, hi, r, n=12):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return rbox(0.5*(lo + hi), hi - lo, r, n)


def cyl(a, b, r0, r1=None, n=32):
    """Cylinder (or cone) from point a to point b."""
    m3 = _m3()
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    c = m3.Manifold.cylinder(L, float(r0), float(r0 if r1 is None else r1), n)
    return _tf(c, _rot_z_to(d/L), a)


def ring(a, b, r_out, r_in, n=32):
    m3 = _m3()
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    c = m3.Manifold.cylinder(L, float(r_out), float(r_out), n) - m3.Manifold.cylinder(L + 2e-4, float(r_in), float(r_in),
                                                                                       n).translate([0., 0., -1e-4])
    return _tf(c, _rot_z_to(d/L), a)


def sphere(c, r, n=24):
    return _m3().Manifold.sphere(float(r), n).translate([float(x) for x in c])


def lathe(profile, a=(0., 0., 0.), axis=(0., 0., 1.), n=48):
    """Solid of revolution of a closed (r, s) profile (r >= 0) about `axis` through a (s measured along the axis)."""
    m3 = _m3()
    pts = np.asarray(profile, float)
    area = 0.5*float(np.sum(pts[:, 0]*np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1)*pts[:, 1]))
    if area < 0:
        pts = pts[::-1]
    cs = m3.CrossSection([pts])
    man = m3.Manifold.revolve(cs, n, 360.)
    return _tf(man, _rot_z_to(axis), np.asarray(a, float))


def hexagon(a, b, af, R=None):
    """Hexagonal prism of across-flats `af` from a to b. R (optional): the prism frame (its z along b - a); the flats
    are normal to that frame's x axis."""
    m3 = _m3()
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    rc = af/math.sqrt(3.)
    pts = np.array([[rc*math.cos(math.pi/6 + k*math.pi/3), rc*math.sin(math.pi/6 + k*math.pi/3)] for k in range(6)])
    man = m3.Manifold.extrude(m3.CrossSection([pts]), L)
    return _tf(man, _rot_z_to(d/L) if R is None else R, a)


def sweep(points, r, bend=0.03, n=16, corrugate=None, cap=True, step=0.004):
    """Tube mesh along a polyline with filleted corners (bend radius `bend`). corrugate=(amplitude, pitch) modulates
    the radius. Returns (vertices, triangles)."""
    P = [np.asarray(p, float) for p in points]
    path = [P[0]]
    for k in range(1, len(P) - 1):
        a, b, c = P[k - 1], P[k], P[k + 1]
        u, w = a - b, c - b
        lu, lw = np.linalg.norm(u), np.linalg.norm(w)
        u, w = u/lu, w/lw
        cosang = float(np.clip(u @ w, -1., 1.))
        ang = math.acos(cosang)                       # interior angle
        if ang > math.pi - 1e-3 or ang < 1e-3:        # straight on, or folded straight back: no fillet to cut
            path.append(b)
            continue
        t = min(bend/math.tan(ang/2), 0.45*lu, 0.45*lw)
        if t < 1e-7:
            path.append(b)
            continue
        rr = t*math.tan(ang/2)
        p0, p1 = b + u*t, b + w*t
        bis = (u + w)/np.linalg.norm(u + w)
        cen = b + bis*rr/math.sin(ang/2)
        v0, v1 = p0 - cen, p1 - cen
        th = math.acos(float(np.clip(v0 @ v1/(np.linalg.norm(v0)*np.linalg.norm(v1)), -1., 1.)))
        m = max(3, int(th*rr/step) + 1)
        for j in range(m + 1):
            s = j/m
            q = (math.sin((1 - s)*th)*v0 + math.sin(s*th)*v1)/math.sin(th)
            path.append(cen + q)
    path.append(P[-1])
    # resample straight runs
    pts = [path[0]]
    for k in range(1, len(path)):
        a, b = pts[-1], path[k]
        L = np.linalg.norm(b - a)
        if L < 1e-9:
            continue
        m = max(1, int(L/step)) if corrugate else 1
        for j in range(1, m + 1):
            pts.append(a + (b - a)*j/m)
    pts = np.asarray(pts)
    # parallel-transport frames
    T = np.gradient(pts, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.array([0., 0., 1.]) if abs(T[0][2]) < 0.9 else np.array([1., 0., 0.])
    N = np.cross(T[0], ref)
    N /= np.linalg.norm(N)
    Ns = [N]
    for k in range(1, len(pts)):
        N = N - (N @ T[k])*T[k]
        N /= np.linalg.norm(N)
        Ns.append(N)
    Ns = np.asarray(Ns)
    Bs = np.cross(T, Ns)
    s = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    rad = np.full(len(pts), float(r))
    if corrugate:
        amp, pitch = corrugate
        rad = rad + amp*np.cos(2*math.pi*s/pitch)
    ang = 2*math.pi*np.arange(n)/n
    ring_ = (np.cos(ang)[None, :, None]*Ns[:, None, :] + np.sin(ang)[None, :, None]*Bs[:, None, :])
    V = (pts[:, None, :] + rad[:, None, None]*ring_).reshape(-1, 3)
    F = []
    M = len(pts)
    for k in range(M - 1):
        for j in range(n):
            a, b_ = k*n + j, k*n + (j + 1) % n
            c, d = a + n, b_ + n
            F += [[a, b_, d], [a, d, c]]
    if cap:
        c0 = len(V)
        V = np.vstack([V, pts[0], pts[-1]])
        for j in range(n):
            F.append([c0, (j + 1) % n, j])
            F.append([c0 + 1, (M - 1)*n + j, (M - 1)*n + (j + 1) % n])
    return np.asarray(V, np.float32), np.asarray(F, np.int32)


def threaded(r_major, pitch, z0, z1, n_theta=28, depth=None, chamfer=True):
    """Threaded rod mesh along +z from z0 to z1 (single start, 60-degree V-thread approximated by a triangle wave),
    with 45-degree chamfered ends. Returns (vertices, triangles)."""
    depth = 0.6*pitch if depth is None else depth
    L = z1 - z0
    nz = max(8, int(L/(pitch/8.)))
    th = 2*math.pi*np.arange(n_theta)/n_theta
    zs = np.linspace(z0, z1, nz + 1)
    TH, Z = np.meshgrid(th, zs)
    phase = ((Z - z0)/pitch - TH/(2*math.pi)) % 1.
    tri = 1. - 2.*np.abs(phase - 0.5)                         # 0 at the root, 1 at the crest
    R = r_major - depth*(1. - tri)
    if chamfer:
        cut = np.minimum(Z - z0, z1 - Z)
        R = np.minimum(R, r_major - depth + cut)
        R = np.maximum(R, 0.5*(r_major - depth))
    V = np.stack([R*np.cos(TH), R*np.sin(TH), Z], -1).reshape(-1, 3)
    F = []
    for k in range(nz):
        for j in range(n_theta):
            a, b = k*n_theta + j, k*n_theta + (j + 1) % n_theta
            F += [[a, b, b + n_theta], [a, b + n_theta, a + n_theta]]
    c0 = len(V)
    V = np.vstack([V, [0., 0., z0], [0., 0., z1]])
    for j in range(n_theta):
        F.append([c0, (j + 1) % n_theta, j])
        F.append([c0 + 1, nz*n_theta + j, nz*n_theta + (j + 1) % n_theta])
    return np.asarray(V, np.float32), np.asarray(F, np.int32)


def param_cyl(rfun, z0, z1, n_theta=48, nz=60):
    """Closed surface of revolution r = rfun(theta, z) (arrays) about +z from z0 to z1: knurls, ribs."""
    th = 2*math.pi*np.arange(n_theta)/n_theta
    zs = np.linspace(z0, z1, nz + 1)
    TH, Z = np.meshgrid(th, zs)
    R = rfun(TH, Z)
    V = np.stack([R*np.cos(TH), R*np.sin(TH), Z], -1).reshape(-1, 3)
    F = []
    for k in range(nz):
        for j in range(n_theta):
            a, b = k*n_theta + j, k*n_theta + (j + 1) % n_theta
            F += [[a, b, b + n_theta], [a, b + n_theta, a + n_theta]]
    c0 = len(V)
    V = np.vstack([V, [0., 0., z0], [0., 0., z1]])
    for j in range(n_theta):
        F.append([c0, (j + 1) % n_theta, j])
        F.append([c0 + 1, nz*n_theta + j, nz*n_theta + (j + 1) % n_theta])
    return np.asarray(V, np.float32), np.asarray(F, np.int32)


def _xf_mesh(vf, R=None, t=(0., 0., 0.)):
    v, f = vf
    v = np.asarray(v, float)
    if R is not None:
        v = v @ np.asarray(R).T
    return (v + np.asarray(t, float)).astype(np.float32), f


def _mesh(man):
    m = man.to_mesh()
    return np.asarray(m.vert_properties, np.float32)[:, :3], np.asarray(m.tri_verts, np.int32)


class Model:
    """Accumulates parts; merges same-material, same-option parts into one mesh each."""

    def __init__(self):
        self.items = []

    def add(self, geom, mat, name=None, **opts):
        if geom is None:
            return
        if isinstance(geom, tuple):
            v, f = geom
        else:
            if geom.is_empty():
                return
            v, f = _mesh(geom)
        if not len(f):
            return
        self.items.append((name, np.asarray(v, np.float32), np.asarray(f, np.int32), mat, dict(opts)))

    def extend(self, parts):
        for p in parts:
            name, v, f, mat = p[:4]
            self.items.append((name, v, f, mat, dict(p[4]) if len(p) > 4 else {}))

    def parts(self, R=None, t=(0., 0., 0.)):
        groups = {}
        for name, v, f, mat, opts in self.items:
            key = (name or mat, mat, tuple(sorted(opts.items())))
            groups.setdefault(key, []).append((v, f))
        out = []
        for (name, mat, opts), lst in groups.items():
            vs, fs, off = [], [], 0
            for v, f in lst:
                vs.append(v)
                fs.append(f + off)
                off += len(v)
            v = np.concatenate(vs).astype(float)
            if R is not None:
                v = v @ np.asarray(R).T
            v = (v + np.asarray(t, float)).astype(np.float32)
            out.append((str(name).replace("/", "_"), v, np.concatenate(fs).astype(np.int32), mat, dict(opts)))
        return out


def transform_parts(parts, R=None, t=(0., 0., 0.)):
    M = Model()
    M.extend(parts)
    return M.parts(R, t)


# ====================================================================================== rotary hammer (housing body)
def rotary_hammer():
    """DeWalt DCH273-class SDS-plus cordless rotary hammer, housing body only (the chuck, trigger, mode dial, depth
    rod and bit are drawn on their own moving bodies by shape_visual). Housing frame: x along the bit, z up, y left,
    origin at the front of the chuck sleeve. Envelope = tools.RotaryHammer's housing colliders:
      gearbox/percussion tube x -0.23..-0.03 r 36 mm; motor x -0.165..-0.075, z -0.115..-0.035; D-handle top bar
      x -0.315..-0.225, z +-0.014; grip (12 degrees) centred (-0.30, -0.075); battery x -0.315..-0.225, z -0.20..-0.13;
      side-handle collar x -0.034..-0.010 r 34 mm, stem to z -0.095, grip z -0.20..-0.09 r 16 mm."""
    m3 = _m3()
    M = Model()
    # ---- gearbox / percussion housing: yellow upper clamshell, a slight oval (wider than tall at the rear)
    gear = (cyl((-0.212, 0, 0.002), (-0.075, 0, 0.002), 0.0355, 0.0355, 48)
            + sphere((-0.212, 0., 0.002), 0.0355, 40).scale([1., 1., 1.]).translate([0., 0., 0.])
            ).scale([1., 1.02, 0.98])
    gear = m3.Manifold.batch_hull([gear, sphere((-0.226, 0., -0.004), 0.024, 24)])
    nose = cyl((-0.080, 0, 0.001), (-0.034, 0, 0.0), 0.0345, 0.0300, 48)
    motor = rbox2((-0.163, -0.0395, -0.113), (-0.077, 0.0395, -0.010), 0.013)
    # lower link from the motor to the battery mount (the black foot of the D-handle loop)
    foot = rbox2((-0.236, -0.0335, -0.136), (-0.085, 0.0335, -0.104), 0.010)
    upper = gear + motor
    cut_lo = box2((-0.30, -0.1, -0.30), (0.05, 0.1, -0.058))      # clamshell split: yellow above, black below
    yellow = upper - cut_lo
    black_motor = (motor ^ cut_lo) + foot
    # vent slots on both motor sides (black lower half) and on the gearbox rear top
    slots = None
    for k in range(6):
        x = -0.150 + 0.011*k
        for sy in (-1, 1):
            s_ = rbox2((x - 0.0022, sy*0.0395 - 0.004, -0.104), (x + 0.0022, sy*0.0395 + 0.004, -0.066), 0.0015, 6)
            slots = s_ if slots is None else slots + s_
    black_motor = black_motor - slots
    for k in range(5):
        x = -0.205 + 0.010*k
        sl = rbox2((x - 0.0025, -0.016, 0.026), (x + 0.0025, 0.016, 0.050), 0.0015, 6)
        yellow = yellow - sl
    M.add(yellow, "hammer_yellow", "housing_yellow", crease=50)
    M.add(black_motor, "hammer_black", "housing_black", crease=50)
    M.add(nose, "hammer_grey", "nose", smooth=True)
    # badge panels on both gearbox sides (black, the maker's logo field)
    for sy in (-1, 1):
        if sy > 0:
            badge = rbox2((-0.212, 0.0325, -0.010), (-0.160, 0.0372, 0.016), 0.004, 8)
        else:
            badge = rbox2((-0.205, -0.0372, -0.012), (-0.090, -0.0325, 0.016), 0.004, 8)
        M.add(badge ^ (gear.scale([1.0, 1.05, 1.05])), "hammer_black", "badges", crease=50)
    # mode selector boss on the left (+y) side under the dial, with the three mode marks (rotary, hammer drill, hammer)
    M.add(cyl((-0.13, 0.026, 0.010), (-0.13, 0.0402, 0.010), 0.0245, 0.0245, 48), "hammer_black", "dial_boss", crease=50)
    for k, th in enumerate((0., math.radians(60.), math.radians(120.))):
        cx, cz = -0.13 + 0.0212*math.cos(th), 0.010 - 0.0212*math.sin(th)
        M.add(cyl((cx, 0.0400, cz), (cx, 0.0406, cz), 0.0022, 0.0022, 16), "decal_white", "dial_marks")
    # ---- D-handle: yellow core top bar with a black rubber overmould on top, black rubber grip down to the battery
    top_bar = rbox2((-0.318, -0.0165, -0.0155), (-0.200, 0.0165, 0.0155), 0.012)
    M.add(top_bar - box2((-0.33, -0.1, 0.004), (-0.24, 0.1, 0.1)), "hammer_yellow", "handle_bar")
    M.add(top_bar ^ box2((-0.33, -0.1, 0.004), (-0.24, 0.1, 0.1)), "hammer_grip", "handle_pad", crease=50)
    Rg = _ry(math.radians(-12.))
    grip = _tf(rbox((0., 0., 0.), (0.0345, 0.0385, 0.135), 0.0145, 16), Rg, (-0.300, 0., -0.0745))
    corner = sphere((-0.309, 0., -0.008), 0.0165, 24)
    grip = grip + corner
    # two-tone like the real handle: yellow structural front (the trigger side), black rubber overmould behind it
    front = _tf(box((0.0135, 0., 0.), (0.015, 0.06, 0.20)), Rg, (-0.300, 0., -0.0745))
    M.add(grip - front, "hammer_grip", "grip", crease=60)
    M.add(grip ^ front, "hammer_yellow", "grip_front", crease=60)
    # battery mount under the grip (black) and the 20 V slide-on pack
    M.add(rbox2((-0.3185, -0.0395, -0.1385), (-0.222, 0.0395, -0.1235), 0.005), "hammer_black", "battery_mount")
    pack = rbox2((-0.3165, -0.0385, -0.2015), (-0.2235, 0.0385, -0.1385), 0.008)
    for k in range(5):
        x = -0.300 + 0.0145*k
        for sy in (-1, 1):
            pack = pack - rbox2((x - 0.0025, sy*0.0385 - 0.0015, -0.192), (x + 0.0025, sy*0.0385 + 0.0015, -0.150),
                                0.001, 6)
    M.add(pack, "drill_battery", "battery", crease=50)
    M.add(rbox2((-0.3185, -0.0392, -0.1440), (-0.2215, 0.0392, -0.1385), 0.002, 6), "hammer_yellow", "battery_band")
    M.add(rbox2((-0.2245, -0.020, -0.1505), (-0.2205, 0.020, -0.1420), 0.0015, 6), "hammer_grey", "battery_release")
    M.add(rbox2((-0.3190, -0.022, -0.180), (-0.3155, 0.022, -0.160), 0.0012, 6), "hammer_yellow", "battery_logo")
    for k in range(3):
        y = -0.008 + 0.008*k
        M.add(cyl((-0.3165, y, -0.192), (-0.3175, y, -0.192), 0.0018, 0.0018, 12), "led_green", "battery_gauge")
    M.add(rbox2((-0.312, -0.030, -0.2045), (-0.228, 0.030, -0.2005), 0.0015, 6), "rubber_grey", "battery_pad")
    # ---- side handle on its clamp band around the nose, with the depth-rod clamp on the left
    M.add(ring((-0.034, 0, 0), (-0.010, 0, 0), 0.0345, 0.0295, 64), "hammer_black", "collar_band", crease=50)
    for k in range(3):
        x = -0.031 + 0.009*k
        M.add(ring((x, 0, 0), (x + 0.003, 0, 0), 0.0352, 0.0300, 64), "hammer_black", "collar_band_ribs", smooth=True)
    M.add(rbox2((-0.0335, -0.0115, -0.046), (-0.0105, 0.0115, -0.026), 0.004), "hammer_black", "side_boss")
    M.add(cyl((-0.022, 0, -0.050), (-0.022, 0, -0.088), 0.0075, 0.0075, 32), "steel_dark", "side_stem", smooth=True)
    prof = [(0., -0.112), (0.0125, -0.112), (0.0140, -0.108)]
    for k in range(9):
        z = -0.104 + 0.010*k
        prof += [(0.0152, z - 0.003), (0.0163, z), (0.0152, z + 0.003)]
    prof += [(0.0148, -0.005), (0.0168, -0.002), (0.0175, 0.003), (0.0160, 0.008), (0., 0.008)]
    prof = [(r, -z) for r, z in prof][::-1]              # flip: the knob end at the bottom
    grip_s = lathe(prof, (-0.022, 0., -0.200 + 0.008), (0., 0., 1.), 48)
    M.add(grip_s, "hammer_grip", "side_grip", smooth=True)
    M.add(cyl((-0.022, 0., -0.0915), (-0.022, 0., -0.0885), 0.0140, 0.0140, 32), "hammer_grey", "side_ferrule", smooth=True)
    clamp = (rbox2((-0.031, 0.0300, -0.0285), (-0.013, 0.0530, -0.0115), 0.003)
             - cyl((-0.040, 0.045, -0.020), (0.0, 0.045, -0.020), 0.0036, 0.0036, 24))
    M.add(clamp, "hammer_black", "depth_clamp")
    M.add(cyl((-0.022, 0.045, -0.0285), (-0.022, 0.045, -0.040), 0.0055, 0.0055, 20)
          + cyl((-0.022, 0.045, -0.040), (-0.022, 0.045, -0.044), 0.0085, 0.0085, 6), "hammer_grey", "depth_screw")
    return [(n, v, f, m, o) for n, v, f, m, o in M.parts()]


# ====================================================================================== hammer moving parts
# ====================================================================================== impact wrench (housing body)
def impact_wrench():
    """Hilti SIW 6AT-A22 class 1/2 in cordless impact wrench with the SI-AT-A22 Adaptive Torque module, housing body
    only (the socket and the trigger are drawn on their own bodies). Housing frame (tools.ImpactWrench): x along the
    anvil towards the socket's mouth, z up, origin at the socket's mouth. Envelope = its colliders: the head (r 30 mm,
    x -0.188..-0.068), the grip (x -0.167..-0.133, z -0.140..-0.030) and the battery box (x -0.200..-0.080, y +-0.040,
    z -0.210..-0.140), which holds the tool's foot, the module ('slots in between the impact wrench and battery',
    Hilti product page https://www.hilti.com/r3414399) and the B22 battery. Colours: Hilti red motor housing and foot,
    a black hammer case at the front, a black rubber grip, a black battery (the tool as photographed in use,
    fastfixtechnology.com, 2019)."""
    M = Model()
    m3 = _m3()
    # motor housing (red): a shell of revolution about x, the rear cap black with vent slots
    prof = [(0., -0.1885), (0.020, -0.1885), (0.026, -0.1860), (0.0292, -0.1810), (0.0300, -0.1740), (0.0300, -0.1180),
            (0., -0.1180)]
    M.add(lathe(prof, (0, 0, 0), (1, 0, 0), 64), "hilti_red", "motor_housing", smooth=True)
    cap = lathe([(0., -0.1890), (0.0205, -0.1890), (0.0215, -0.1860), (0., -0.1860)], (0, 0, 0), (1, 0, 0), 48)
    M.add(cap, "hammer_black", "rear_cap", smooth=True)
    for k in range(8):
        a = 2*math.pi*k/8
        c = np.array([-0.1745, 0.0298*math.cos(a), 0.0298*math.sin(a)])
        M.add(_tf(box((0., 0., 0.), (0.020, 0.0035, 0.0012)), _rx(a), c), "vent_dark", "vents")
    # white label band on both sides of the motor housing (the HILTI mark)
    for sy in (-1, 1):
        M.add(box((-0.150, sy*0.0299, 0.006), (0.038, 0.0008, 0.009)), "decal_white", "label")
    # hammer case (black) tapering to the anvil's bearing collar
    nose = [(0., -0.1185), (0.0292, -0.1185), (0.0288, -0.1040), (0.0262, -0.0880), (0.0228, -0.0760), (0.0205, -0.0700),
            (0.0160, -0.0685), (0., -0.0685)]
    M.add(lathe(nose, (0, 0, 0), (1, 0, 0), 64), "hammer_black", "hammer_case", smooth=True)
    M.add(ring((-0.1190, 0., 0.), (-0.1170, 0., 0.), 0.0303, 0.0285, 64), "steel_dark", "case_seam", smooth=True)
    # grip: black rubber overmold, the red housing coming down to it in front, a trigger guard ahead of the trigger
    grip = m3.Manifold.batch_hull([rbox((-0.150, 0., -0.034), (0.034, 0.032, 0.008), 0.010, 12),
                                   rbox((-0.151, 0., -0.090), (0.032, 0.030, 0.010), 0.011, 12),
                                   rbox((-0.150, 0., -0.135), (0.034, 0.032, 0.010), 0.010, 12)])
    M.add(grip, "hammer_grip", "grip", crease=40)
    M.add(rbox((-0.130, 0., -0.0285), (0.028, 0.026, 0.010), 0.006, 10), "hilti_red", "trigger_housing", crease=40)
    for sy in (-1, 1):                                   # forward / reverse button through the housing above the trigger
        M.add(cyl((-0.139, sy*0.0130, -0.025), (-0.139, sy*0.0175, -0.025), 0.0045, 0.0045, 20), "hammer_black",
              "reverse_button", smooth=True)
    # the tool's foot (red, with the work light), the SI-AT-A22 module (black, a status window), the B22 battery
    M.add(rbox2((-0.198, -0.034, -0.152), (-0.086, 0.034, -0.140), 0.006, 10), "hilti_red", "foot", crease=40)
    M.add(box((-0.0862, 0., -0.146), (0.0016, 0.020, 0.005)), "decal_white", "work_light")
    M.add(rbox2((-0.200, -0.036, -0.168), (-0.082, 0.036, -0.152), 0.004, 10), "controller_black", "at_module",
          crease=40)
    # the module's status: Hilti's SI-AT-A22 shows its result by LEDs "on both sides of the module" after the wrench has
    # switched itself off (green: tightened as set; red: not). The window on its front is drawn dark; on each side a
    # green and a red LED that the renderer shows by the tool's simulated light (render.wrench_light_prims)
    M.add(box((-0.0818, 0., -0.160), (0.0012, 0.022, 0.008)), "vent_dark", "at_status")
    for sy in (-1, 1):
        M.add(box((-0.104, sy*0.0366, -0.160), (0.010, 0.0010, 0.0045)), "vent_dark", "at_led_lens")
        M.add(box((-0.104, sy*0.0372, -0.160), (0.0085, 0.0010, 0.0035)), "led_green", "at_led_green")
        M.add(box((-0.104, sy*0.0372, -0.160), (0.0085, 0.0010, 0.0035)), "led_red", "at_led_red")
    M.add(rbox2((-0.200, -0.040, -0.210), (-0.080, 0.040, -0.168), 0.008, 12), "drill_battery", "battery", crease=40)
    for k in range(5):                                   # grip ribs along the battery's sides
        for sy in (-1, 1):
            M.add(box((-0.175 + 0.018*k, sy*0.0402, -0.189), (0.006, 0.0012, 0.026)), "rubber_grey", "battery_ribs")
    M.add(rbox2((-0.0815, -0.016, -0.186), (-0.0790, 0.016, -0.176), 0.002, 6), "hilti_red", "battery_release",
          crease=40)
    for k in range(4):
        M.add(box((-0.2006, -0.012 + 0.008*k, -0.190), (0.0012, 0.004, 0.0025)), "led_green" if k < 3 else "vent_dark",
              "charge_leds")
    return M.parts()


def impact_socket(r, h):
    """3/4 in deep 1/2 in-drive impact socket on the anvil, in its cylinder frame (z from -h at the drive end to +h at
    the mouth): black phosphate, a chamfered mouth with the 6-point opening (3/4 in across flats, 10 mm deep), the
    deep bore the stud passes into, a retaining-ring groove at the drive end; the anvil's collar behind it."""
    M = Model()
    prof = [(0., -h), (r*0.96, -h), (r, -h + 0.002), (r, -h + 0.006), (r*0.92, -h + 0.0065), (r*0.92, -h + 0.0095),
            (r, -h + 0.010), (r, h - 0.0025), (r*0.90, h), (0., h)]
    body = lathe(prof, (0, 0, 0), (0, 0, 1), 64)
    af = 0.0195
    hexa = hexagon((0, 0, h - 0.010), (0, 0, h + 0.001), af)
    bore = cyl((0, 0, -h + 0.008), (0, 0, h), 0.0086, 0.0086, 40)
    drive = box((0., 0., -h + 0.004), (0.0128, 0.0128, 0.0102))
    M.add(body - hexa - bore - drive, "black_oxide", "socket", crease=35)
    M.add(cyl((0, 0, -h - 0.0055), (0, 0, -h), 0.0105, 0.0105, 40), "steel_dark", "anvil_collar", smooth=True)
    return M.parts()


def impact_trigger(half):
    """Trigger of the impact wrench on its blade collider (box half extents)."""
    hx, hy, hz = half
    M = Model()
    M.add(_m3().Manifold.batch_hull([rbox((0.001, 0., hz - 0.003), (2*hx - 0.002, 2*hy, 0.006), 0.0025, 8),
                                     rbox((0.003, 0., -hz + 0.003), (2*hx - 0.003, 2*hy - 0.001, 0.006), 0.0025, 8)]),
          "drill_trigger", "trigger", crease=40)
    return M.parts()


def _hammer_chuck(r, h):
    """SDS-plus chuck sleeve with grip ribs and a rubber dust cap; shape frame: z along the bit, z +h at the front."""
    M = Model()
    m3 = _m3()
    sleeve = cyl((0, 0, -h), (0, 0, h - 0.0065), r - 0.0008, r - 0.0008, 48)
    for k in range(12):
        a = 2*math.pi*k/12
        rib = _tf(box((0, 0, 0), (0.0026, 0.0034, 0.020)), _rz(a), (math.cos(a)*(r - 0.0016), math.sin(a)*(r - 0.0016),
                                                                      h - 0.0065 - 0.010))
        sleeve = sleeve + rib
    M.add(sleeve, "chuck_black", "sleeve", crease=40)
    cap = lathe([(0.0074, h - 0.0068), (r - 0.0040, h - 0.0068), (r - 0.0036, h - 0.0020), (r - 0.0048, h),
                 (0.0082, h), (0.0074, h - 0.0012)], (0, 0, 0), (0, 0, 1), 48)
    M.add(cap, "rubber_dust", "dust_cap", smooth=True)
    return M.parts()


def _hammer_trigger(h):
    """Curved trigger blade on the trigger body: finger face to +x."""
    m3 = _m3()
    blade = m3.Manifold.batch_hull([rbox((-0.002, 0, 0), (0.008, 0.0090, 2*h[2]), 0.002, 8),
                                    sphere((0.0045, 0, -0.002), 0.0045, 16).scale([1., 0.95, 1.6])])
    M = Model()
    M.add(blade, "hammer_black", "trigger", smooth=True)
    return M.parts()


def _hammer_dial(r, h):
    """Mode selector knob (dial frame: z out of the housing side), a raised paddle across its face."""
    M = Model()
    knob = lathe([(0., -h), (r, -h), (r, h - 0.002), (r - 0.002, h), (0., h)], (0, 0, 0), (0, 0, 1), 48)
    for k in range(24):
        a = 2*math.pi*k/24
        knob = knob - _tf(box((0, 0, 0), (0.0016, 0.0016, 2*h + 0.002)), _rz(a), (math.cos(a)*r, math.sin(a)*r, 0.))
    paddle = rbox((0.0, 0.0, h + 0.0035), (0.030, 0.0075, 0.007), 0.0025, 8)
    M.add(knob + paddle, "hammer_black", "dial", crease=45)
    M.add(box((0.0115, 0., h + 0.0071), (0.008, 0.0028, 0.0006)), "hammer_yellow", "dial_arrow")
    return M.parts()


def _depth_rod(r, h):
    M = Model()
    M.add(cyl((0, 0, -h), (0, 0, h), r, r, 16), "steel_bright", "rod", smooth=True)
    for k in range(10):
        z = -h + 0.02 + 0.018*k
        M.add(ring((0, 0, z), (0, 0, z + 0.0008), r + 0.0002, r - 0.0002, 16), "steel_dark", "graduation", smooth=True)
    M.add(lathe([(0., h - 0.001), (r + 0.0012, h - 0.001), (r + 0.0012, h + 0.002), (0., h + 0.002)], (0, 0, 0),
                (0, 0, 1), 16), "hammer_black", "rod_tip", smooth=True)
    return M.parts()


# ====================================================================================== bits
def fluted_bit(r, h, kind="hss", n=40):
    """Fluted bit body in a cylinder's frame (z from -h at the shank end to +h at the point): two helical flutes cut in
    a cylinder of radius r (twist drills: 30-degree helix; SDS: a steeper 4-cutter double helix; brad point: twist)."""
    m3 = _m3()
    L = 2*h
    if kind == "sds":
        pitch = max(0.028, 7*r)
        cs = (m3.CrossSection.circle(r, n) - m3.CrossSection.circle(0.52*r, 24).translate([0.92*r, 0.])
              - m3.CrossSection.circle(0.52*r, 24).translate([-0.92*r, 0.]))
    else:
        pitch = math.pi*2*r/math.tan(math.radians(30.))
        cs = (m3.CrossSection.circle(r, n) - m3.CrossSection.circle(0.55*r, 24).translate([0.72*r, 0.32*r])
              - m3.CrossSection.circle(0.55*r, 24).translate([-0.72*r, -0.32*r]))
    shank = min(0.25*L, 0.012) if kind != "sds" else min(0.08*L, 0.01)
    fl = L - shank
    turns = fl/pitch
    body = m3.Manifold.extrude(cs, fl, max(16, int(turns*48)), 360.*turns).translate([0., 0., -h + shank])
    body = body + m3.Manifold.cylinder(shank + 1e-4, r*0.995, r*0.995, n).translate([0., 0., -h])
    M = Model()
    M.add(body, "sds_bit" if kind == "sds" else "hss_bit", "flutes", crease=55)
    return M.parts()


# ====================================================================================== fasteners
def hex_head(half):
    """ISO 4017 hex head along x (box collider x thickness, y/z across flats); flats normal to z so it rests flat."""
    af = 2*min(half[1], half[2])
    M = Model()
    Rh = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])      # prism z -> -x (outwards), flat normals -> z
    head = hexagon((half[0], 0, 0), (-half[0], 0, 0), af, R=Rh)
    chamfer = lathe([(0., -half[0]), (0.5*af*1.16, -half[0]), (0.5*af*1.16, half[0] - 0.0009),
                     (0.5*af*0.96, half[0]), (0., half[0])], (0, 0, 0), (-1, 0, 0), 48)
    M.add(head ^ chamfer, "zinc", "hex_head", crease=35)
    return M.parts()


def bolt_shank(r, h, thread_len=None, pitch=0.00125):
    """Threaded bolt shank in its cylinder frame (z from -h at the head to +h at the end)."""
    M = Model()
    L = 2*h
    tl = L if thread_len is None else min(thread_len, L)
    if tl < L:
        M.add(cyl((0, 0, -h), (0, 0, h - tl + 1e-4), r*0.985, r*0.985, 32), "zinc", "shank", smooth=True)
    M.add(threaded(r, pitch, h - tl, h, 28), "zinc", "thread", crease=70)
    return M.parts()


def carriage_dome(r, h):
    """ASME B18.5 carriage bolt head: a low dome (cylinder frame z from -h, the dome top, to +h, under the head)."""
    M = Model()
    prof = [(0., -h)]
    for k in range(13):
        a = k/12*math.pi/2
        prof.append((r*math.sin(a), -h + 2*h*(1 - math.cos(a))*0.95))
    prof += [(r, h), (0., h)]
    prof = sorted(set(prof), key=lambda p: (p[1], -p[0]))
    prof = [(0., -h)] + [(r*math.sin(k/12*math.pi/2), h - 2*h*math.cos(k/12*math.pi/2)) for k in range(1, 13)] + \
           [(r, h), (0., h)]
    M.add(lathe(prof, (0, 0, 0), (0, 0, 1), 64), "zinc", "dome", smooth=True)
    return M.parts()


def hex_nut_x(half):
    """Hex nut along x on tools.Parts.hex_prism's first box (x half length, z half across flats), both faces chamfered;
    flats normal to z as the prism's."""
    h, af = half[0], 2*half[2]
    r = af/math.sqrt(3.)
    M = Model()
    Rh = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    nut = hexagon((h, 0, 0), (-h, 0, 0), af, R=Rh)
    ch = lathe([(0., -h), (0.5*af*0.97, -h), (r*1.05, -h + 0.0012), (r*1.05, h - 0.0012), (0.5*af*0.97, h), (0., h)],
               (0, 0, 0), (1, 0, 0), 48)
    M.add(nut ^ ch, "zinc", "nut", crease=35)
    return M.parts()


def _anchor_nut(half):
    """The wedge anchor's whole nut, drawn on the box of its lower hex (parts.WedgeAnchor)."""
    from . import params as P
    hc = P.NUT_CHAMFER_H_M
    return transform_parts(hex_nut_x((half[0] + 0.5*hc, half[1], half[2])), None, (-0.5*hc, 0., 0.))


def punch_point_round(r, h):
    """The centre punch's point on its capsule collider (cylinder frame z, half length h without the caps): the rod
    from 8 mm inside the barrel to the hemispherical tip at h + r."""
    M = Model()
    prof = [(0., -h - 0.008), (r, -h - 0.008), (r, h)]
    prof += [(r*math.cos(a), h + r*math.sin(a)) for a in np.linspace(0., math.pi/2, 9)[1:]]
    M.add(lathe(prof, (0, 0, 0), (0, 0, 1), 24), "punch_tip", "point", smooth=True)
    return M.parts()


def hex_nut(r, h):
    """Hex nut in a cylinder frame (z axis), circumradius r, both faces chamfered."""
    af = r*math.sqrt(3.)
    M = Model()
    nut = hexagon((0, 0, -h), (0, 0, h), af)
    ch = lathe([(0., -h), (0.5*af*0.97, -h), (r*1.05, -h + 0.0012), (r*1.05, h - 0.0012), (0.5*af*0.97, h), (0., h)],
               (0, 0, 0), (0, 0, 1), 48)
    M.add(nut ^ ch, "zinc", "nut", crease=35)
    return M.parts()


def washer(r_out, h, r_in):
    M = Model()
    M.add(lathe([(r_in, -h), (r_out - 0.0004, -h), (r_out, -h + 0.0004), (r_out, h - 0.0004), (r_out - 0.0004, h),
                 (r_in, h)], (0, 0, 0), (0, 0, 1), 64), "zinc", "washer", crease=40)
    return M.parts()


def anchor_shank(r, h, thread_len=None):
    """KB-TZ2 shank in its cylinder frame (z from -h at the washer to +h at the tip): thread to the stub (thread_len of
    it on the shank: the anchor's thread length less the stub; default 42 %), smooth body, the expansion cone under the
    clip at the tip."""
    M = Model()
    L = 2*h
    tl = 0.42*L if thread_len is None else float(np.clip(thread_len, 0.002, L - 0.03))
    M.add(threaded(r, 0.00195, -h, -h + tl, 28, chamfer=False), "zinc", "thread", crease=70)
    M.add(cyl((0, 0, -h + tl - 1e-4), (0, 0, h - 0.026), r*0.97, r*0.97, 32), "zinc", "body", smooth=True)
    cone = lathe([(0., h - 0.026), (r*0.97, h - 0.026), (r*0.74, h - 0.024), (r*0.74, h - 0.010), (r*0.99, h - 0.0025),
                  (r*0.92, h), (0., h)], (0, 0, 0), (0, 0, 1), 32)
    M.add(cone, "zinc", "cone", smooth=True)
    return M.parts()


def anchor_clip(r, h):
    """Expansion clip: a slit stainless sleeve with gripping undulations."""
    m3 = _m3()
    M = Model()
    sl = ring((0, 0, -h), (0, 0, h), r, r - 0.0011, 40) - box((r, 0., 0.), (0.004, 0.0012, 2*h + 0.002))
    for k in range(3):
        z = -h + 0.004 + 0.006*k
        sl = sl + (ring((0, 0, z), (0, 0, z + 0.0012), r + 0.0002, r - 0.0011, 40)
                   - box((r, 0., z), (0.004, 0.0012, 0.004)))
    M.add(sl, "steel_bright", "clip", crease=40)
    return M.parts()


def anchor_stub(r, h):
    M = Model()
    M.add(threaded(r, 0.00195, -h, h, 28), "zinc", "stub_thread", crease=70)
    return M.parts()


# ====================================================================================== cleaning tools, hand tools
def nozzle_grip(r, h):
    """Vacuum wand handle: contoured rubber grip with ribs (cylinder frame, +z towards the tube)."""
    M = Model()
    prof = [(0., -h), (r*0.92, -h), (r, -h + 0.006)]
    for k in range(10):
        z = -h + 0.014 + 0.0085*k
        prof += [(r*0.94, z - 0.0025), (r*0.995, z), (r*0.94, z + 0.0025)]
    prof += [(r*0.90, h - 0.020), (r*0.70, h - 0.004), (r*0.55, h), (0., h)]
    M.add(lathe(prof, (0, 0, 0), (0, 0, 1), 48), "hose_black", "grip", smooth=True)
    return M.parts()


def nozzle_hose(r, h):
    """Corrugated hose cuff on the wand's rear."""
    M = Model()
    M.add(cyl((0, 0, h - 0.012), (0, 0, h), r, r*0.95, 48), "hose_black", "cuff", smooth=True)
    M.add(sweep([(0, 0, -h), (0, 0, h - 0.012)], r*0.90, corrugate=(0.0012, 0.0045), n=32, step=0.0006),
          "hose_grey", "hose", smooth=True)
    return M.parts()


def nozzle_tube(r, h):
    """Steel wand tube with a ferrule at the handle and a chamfered narrow tip (z +h at the tip)."""
    M = Model()
    M.add(ring((0, 0, -h), (0, 0, h - 0.002), r, r - 0.0007, 32), "steel_bright", "tube", smooth=True)
    M.add(ring((0, 0, h - 0.002), (0, 0, h), r - 0.0003, r - 0.0009, 32), "steel_bright", "tip", smooth=True)
    M.add(cyl((0, 0, -h), (0, 0, -h + 0.016), r + 0.0022, r + 0.0012, 32), "hose_black", "ferrule", smooth=True)
    return M.parts()


def _face_axis(lo, hi, p):
    """Outward unit normal of the face of box [lo, hi] nearest to point p, and that face's index (0..5)."""
    lo, hi, p = (np.asarray(x, float) for x in (lo, hi, p))
    d = np.concatenate([p - lo, hi - p])
    k = int(np.argmin(d))
    n = np.zeros(3)
    n[k % 3] = -1. if k < 3 else 1.
    return n, k


# HEPA dust extractor: DeWalt DWV012 class (10 gal wet/dry HEPA, 23-1/2 x 18-1/2 x 27-1/4 in, 1-1/4 in x 15 ft hose,
# distributor spec sheet, https://www.hanessupply.com/dewalt-dwv012). Default envelope when no collider is built.
VACUUM_SIZE_M = (0.597, 0.470, 0.692)
VACUUM_HOSE = dict(length_m=4.57, r_m=0.0195)          # 15 ft of 1-1/4 in hose, 39 mm OD


def shop_vacuum(lo, hi, port=None, switch=None, rocker=True):
    """HEPA dust extractor (DWV012 class) filling the box [lo, hi] of its body frame (z up): a black tank on two rear
    wheels and two front swivel casters, a yellow motor head latched on top, a telescoping handle at the rear, the hose
    inlet cuff at `port` and (when given) the power switch at `switch` on the face nearest each (its bezel only when
    `rocker` is False: the rocker is then drawn on its own body). The front is the face carrying the port. Visual
    only."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = 0.5*(lo + hi)
    port_p = np.asarray(port, float) if port is not None else np.array([hi[0], c[1], lo[2] + 0.55*(hi[2] - lo[2])])
    n_front, _ = _face_axis(lo, hi, port_p)
    if abs(n_front[2]) > 0.5:
        n_front = np.array([1., 0., 0.])
    yaw = math.atan2(n_front[1], n_front[0])
    R = _rz(yaw)
    # canonical frame: base centre at the origin, front along +x
    ext = np.abs(R.T @ (hi - lo))
    L, W, H = ext
    M = Model()
    m3 = _m3()
    s = H/0.692
    zb = 0.105*s                                            # tank bottom
    zt = 0.60*H                                             # tank rim
    zh = 0.86*H                                             # head top
    # tank: tapered (narrower at the bottom), rounded, with a moulded rim and two vertical ribs per side
    t_lo = m3.Manifold.batch_hull([rbox((0., 0., zb + 0.01), (L - 0.09, W - 0.08, 0.02), 0.045, 10),
                                   rbox((0., 0., zt - 0.01), (L - 0.03, W - 0.03, 0.02), 0.055, 10)])
    M.add(t_lo, "vac_tank", "tank", crease=40)
    M.add(rbox((0., 0., zt - 0.012), (L - 0.018, W - 0.018, 0.024), 0.05, 10), "vac_tank", "tank_rim", crease=40)
    for sy in (-1, 1):
        for xr in (-0.18*L, 0.18*L):
            M.add(rbox((xr, sy*(0.5*W - 0.022), 0.5*(zb + zt)), (0.03, 0.012, zt - zb - 0.06), 0.005, 6), "vac_tank",
                  "tank_rib", crease=40)
    # head: yellow motor housing with a black top deck, exhaust slots at the rear, two side latches
    M.add(rbox((0., 0., 0.5*(zt + zh)), (L - 0.012, W - 0.012, zh - zt), 0.05, 10), "hammer_yellow", "head", crease=40)
    M.add(rbox((0.02*L, 0., zh - 0.006), (L - 0.07, W - 0.07, 0.022), 0.03, 10), "vac_tank", "head_deck", crease=40)
    for k in range(7):
        z = zt + 0.025 + k*0.012
        if z > zh - 0.02:
            break
        M.add(box((-0.5*L + 0.002, 0., z), (0.006, 0.45*W, 0.005)), "vent_dark", "exhaust")
    for sy in (-1, 1):
        M.add(rbox((0.05*L, sy*(0.5*W - 0.004), zt + 0.005), (0.07, 0.016, 0.075), 0.006, 6), "vac_tank", "latch",
              crease=40)
    # HEPA label on each side of the head
    for sy in (-1, 1):
        M.add(box((-0.08*L, sy*(0.5*W - 0.0045), 0.5*(zt + zh)), (0.16, 0.002, 0.05)), "decal_white", "label")
    # telescoping handle at the rear: two tubes and a grip
    hx = -0.5*L + 0.045
    for sy in (-1, 1):
        M.add(cyl((hx, sy*0.14*W/0.47, zh - 0.06), (hx, sy*0.14*W/0.47, H - 0.018), 0.011, 0.011, 20), "anodized_black",
              "handle_tube", smooth=True)
    M.add(sweep([(hx, -0.16*W/0.47, H - 0.04), (hx, -0.16*W/0.47, H - 0.016), (hx, 0.16*W/0.47, H - 0.016),
                 (hx, 0.16*W/0.47, H - 0.04)], 0.014, bend=0.02, n=20), "rubber_dust", "handle_grip", smooth=True)
    # wheels: two rear wheels on an axle, two front swivel casters
    rw = 0.075*s
    for sy in (-1, 1):
        yw = sy*(0.5*W - 0.032)
        tread = lathe([(rw*0.55, -0.018), (rw - 0.006, -0.018), (rw, -0.012), (rw, 0.012), (rw - 0.006, 0.018),
                       (rw*0.55, 0.018)], (-0.5*L + rw + 0.01, yw, rw), (0., 1., 0.), 40)
        hub = lathe([(0., -0.02), (rw*0.58, -0.02), (rw*0.58, 0.02), (0., 0.02)], (-0.5*L + rw + 0.01, yw, rw),
                    (0., 1., 0.), 32)
        M.add(tread, "caster_tread", "wheel", smooth=True)
        M.add(hub, "caster_hub", "wheel_hub", smooth=True)
    M.add(cyl((-0.5*L + rw + 0.01, -0.5*W + 0.012, rw), (-0.5*L + rw + 0.01, 0.5*W - 0.012, rw), 0.008, 0.008, 16),
          "zinc", "axle", smooth=True)
    for sy in (-1, 1):
        M.extend(transform_parts(caster(0.035*s, zb + 0.01, braked=False, width=0.022), None,
                                 (0.5*L - 0.085, sy*(0.5*W - 0.085), 0.)))
    parts = M.parts(R, (c[0], c[1], lo[2]))
    # the hose inlet cuff at the port (world of this body frame), pointing out of its face
    P2 = Model()
    n_p, _ = _face_axis(lo, hi, port_p)
    P2.add(cyl(port_p - n_p*0.015, port_p + n_p*0.035, 0.030, 0.026, 40), "rubber_dust", "inlet_cuff", smooth=True)
    P2.add(ring(port_p - n_p*0.016, port_p - n_p*0.004, 0.042, 0.029, 40), "vac_tank", "inlet_boss", smooth=True)
    if switch is not None:
        P2.extend(rocker_switch_static(lo, hi, switch, rocker))
    return parts + P2.parts()


def drum_vacuum(tank, head, inlet=None, pivot=None, rocker_size=None):
    """Wet/dry drum vacuum with a HEPA filter (RIDGID RV2400HF class: 14 gal), drawn on the unit's own colliders (unit
    frame: z up, x towards the inlet, origin on the floor under the tank axis). tank, head: (radius, z0, z1) of the
    tank and power-head colliders; inlet: (outer end, axis, radius) of the inlet collider; pivot / rocker_size: the
    power switch's pivot and rocker box, around which the head carries a raised bezel (the rocker itself is drawn on
    its own body, rocker_switch). A grey ribbed drum on a five-caster dolly, latches at the head joint, a black power
    head with a carry handle and exhaust grille, a black inlet cuff, a HEPA label. Visual only."""
    M = Model()
    m3 = _m3()
    R, z0, z1 = tank
    Rh, h0, h1 = head
    # caster dolly and casters
    zc = 0.075
    M.add(ring((0., 0., zc - 0.012), (0., 0., zc + 0.010), R + 0.006, R - 0.035, 64), "vac_tank", "dolly", smooth=True)
    for k in range(5):
        a = 2*math.pi*k/5
        M.extend(transform_parts(caster(0.030, zc - 0.012, braked=False, width=0.020), _rz(a + math.pi),
                                 ((R - 0.045)*math.cos(a), (R - 0.045)*math.sin(a), 0.)))
    # drum: tapered, rounded bottom edge, three moulded hoops
    prof = [(0., zc), (0.88*R, zc), (0.92*R, zc + 0.02), (R - 0.004, z1 - 0.03), (R, z1 - 0.012), (R, z1), (0., z1)]
    M.add(lathe(prof, (0, 0, 0), (0, 0, 1), 72), "plastic_grey", "drum", smooth=True)
    for zf in (0.30, 0.55, 0.80):
        z = zc + zf*(z1 - zc - 0.03)
        rr = 0.92*R + (R - 0.004 - 0.92*R)*zf
        M.add(ring((0., 0., z - 0.006), (0., 0., z + 0.006), rr + 0.005, rr - 0.01, 72), "plastic_grey", "drum_hoops",
              smooth=True)
    # two side latches over the drum / head joint
    for sy in (-1, 1):
        M.add(rbox((0., sy*(R + 0.004), z1 - 0.005), (0.06, 0.016, 0.07), 0.005, 6), "vac_tank", "latches", crease=40)
    # power head: black, rounded top edge, a carry handle across it behind the switch, exhaust slots at the back
    hp = [(0., h0), (Rh, h0), (Rh, h1 - 0.025), (Rh - 0.012, h1 - 0.006), (Rh - 0.035, h1), (0., h1)]
    M.add(lathe(hp, (0, 0, 0), (0, 0, 1), 72), "vac_tank", "head", smooth=True)
    M.add(ring((0., 0., h0 - 0.004), (0., 0., h0 + 0.010), R + 0.003, Rh - 0.02, 72), "vac_tank", "head_rim", smooth=True)
    hx = -0.35*Rh
    M.add(sweep([(hx, -0.55*Rh, h1 - 0.004), (hx, -0.45*Rh, h1 + 0.055), (hx, 0.45*Rh, h1 + 0.055),
                 (hx, 0.55*Rh, h1 - 0.004)], 0.012, bend=0.04, n=20), "vac_tank", "handle", smooth=True)
    for k in range(6):
        a = math.pi + (k - 2.5)*0.09
        c = np.array([(Rh - 0.002)*math.cos(a), (Rh - 0.002)*math.sin(a), 0.5*(h0 + h1) - 0.01])
        M.add(_tf(box((0., 0., 0.), (0.006, 0.012, 0.05)), _rz(a), c), "vent_dark", "exhaust")
    # HEPA label on the drum's side
    M.add(_tf(box((0., 0., 0.), (0.004, 0.12, 0.05)), _rz(math.radians(60.)),
              ((R - 0.001)*math.cos(math.radians(60.)), (R - 0.001)*math.sin(math.radians(60.)), 0.62*z1)),
          "decal_white", "label")
    if inlet is not None:
        end, ax, ri = inlet
        end, ax = np.asarray(end, float), np.asarray(ax, float)/np.linalg.norm(ax)
        M.add(cyl(end - ax*0.07, end, ri, ri*0.92, 48), "vac_tank", "inlet_cuff", smooth=True)
        M.add(ring(end - ax*0.07, end - ax*0.05, ri + 0.010, ri - 0.002, 48), "vac_tank", "inlet_boss", smooth=True)
    if pivot is not None and rocker_size is not None:
        L_, W_, T_ = rocker_size
        pv = np.asarray(pivot, float)
        frame = (rbox((pv[0], pv[1], pv[2] - 0.004), (L_ + 0.016, W_ + 0.014, 0.014), 0.003, 8)
                 - box((pv[0], pv[1], pv[2] + 0.004), (L_ + 0.002, W_ + 0.002, 0.03)))
        M.add(frame, "vac_tank", "switch_bezel", crease=40)
    return M.parts()


def rocker_switch_static(lo, hi, p, rocker=True):
    """Bezel and rocker of the power switch at p on the face of box [lo, hi] nearest to it (a 40 x 25 mm rocker)."""
    n, _ = _face_axis(lo, hi, p)
    p = np.asarray(p, float)
    up = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    side = np.cross(up, n)
    Rm = np.column_stack([side, up, n])
    M = Model()
    M.add(rbox((0., 0., 0.002), (0.052, 0.036, 0.008), 0.004, 8), "vac_tank", "switch_bezel", crease=40)
    if rocker:
        M.add(_tf(rbox((0., 0., 0.006), (0.040, 0.024, 0.010), 0.003, 8), _rx(math.radians(8.))), "estop_red",
              "switch_rocker", crease=40)
    return M.parts(Rm, p)


def rocker_switch(lo, hi, normal):
    """The power switch's rocker on its own (moving) body, filling the box [lo, hi] of that body's frame; `normal`:
    the outward normal of the face it sits in (body frame). Its two halves lie along the box's longer in-face axis:
    a white "I" on the + half (ON), an "O" on the - half."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.asarray(normal, float)/np.linalg.norm(normal)
    c = 0.5*(lo + hi)
    ext = hi - lo
    M = Model()
    M.add(rbox(c, np.maximum(ext, 0.002), min(0.003, 0.3*float(ext.min())), 8), "switch_red", "switch_rocker",
          crease=40)
    k = int(np.argmax(np.abs(n)))
    inplane = [i for i in range(3) if i != k]
    a_long = max(inplane, key=lambda i: ext[i])
    a_cross = [i for i in inplane if i != a_long][0]
    face = (hi if n[k] > 0 else lo)[k] + 0.0002*np.sign(n[k])
    for sgn, mark in ((1., "I"), (-1., "O")):
        cc = c.copy()
        cc[k] = face
        cc[a_long] += sgn*0.25*ext[a_long]
        size = np.zeros(3)
        size[k] = 0.0003
        if mark == "I":
            size[a_long], size[a_cross] = 0.0015, 0.45*ext[a_cross]
            M.add(box(cc, size), "decal_white", "switch_marks")
        else:
            rr = 0.18*min(ext[a_long], ext[a_cross])
            ax = np.zeros(3)
            ax[k] = 1.
            M.add(ring(cc - ax*0.00015, cc + ax*0.00015, rr, rr - 0.0012, 24), "decal_white", "switch_marks")
    return M.parts()


def test_stand(lo, hi, top_z, tube=0.04):
    """Welded steel test stand under a fixed test block whose footprint is [lo, hi] (x, y) and whose underside is at
    top_z: a 12 mm top plate of the block's footprint with a cut-out under it (the block bears on a rim, so a drill
    exiting the block or a bolt through it meets nothing; nothing sticks out past the block's faces, where falling
    debris would pass through a part that has no collider), four square-tube legs, low stretchers and floor pads."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    M = Model()
    m3 = _m3()
    half = 0.5*(hi - lo)
    rim = min(0.03, 0.25*float(half.min()))
    t = 0.012
    plo, phi = lo.copy(), hi.copy()
    plate = box2((plo[0], plo[1], top_z - t), (phi[0], phi[1], top_z)) - box2((lo[0] + rim, lo[1] + rim, top_z - t - 0.01),
                                                                             (hi[0] - rim, hi[1] - rim, top_z + 0.01))
    M.add(plate, "steel_dark", "stand_plate", crease=40)
    tube = min(tube, 0.4*float(half.min()) + 0.02)
    corners = [(plo[0] + tube/2 + 0.01, plo[1] + tube/2 + 0.01), (phi[0] - tube/2 - 0.01, plo[1] + tube/2 + 0.01),
               (phi[0] - tube/2 - 0.01, phi[1] - tube/2 - 0.01), (plo[0] + tube/2 + 0.01, phi[1] - tube/2 - 0.01)]
    legs = [rbox((x, y, 0.5*(top_z - t) + 0.006), (tube, tube, top_z - t - 0.012), 0.003, 6) for x, y in corners]
    M.add(m3.Manifold.batch_boolean(legs, m3.OpType.Add), "cart_frame", "stand_legs", crease=40)
    zs = min(0.12, 0.45*(top_z - t))
    if zs > 0.05:
        for k in range(4):
            (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
            M.add(box2((min(x0, x1) - 0.4*tube, min(y0, y1) - 0.4*tube, zs - 0.4*tube),
                       (max(x0, x1) + 0.4*tube, max(y0, y1) + 0.4*tube, zs + 0.4*tube)), "cart_frame", "stand_rails",
                  crease=40)
    for x, y in corners:
        M.add(box((x, y, 0.003), (tube + 0.03, tube + 0.03, 0.006)), "steel_dark", "stand_feet", crease=40)
    return M.parts()


def brush_grip(r, h):
    """Hole-brush handle: red moulded grip with finger rings and a flared end cap (cylinder frame, +z to the rod)."""
    M = Model()
    prof = [(0., -h), (r*1.10, -h), (r*1.18, -h + 0.006), (r*1.05, -h + 0.012)]
    for k in range(6):
        z = -h + 0.020 + 0.011*k
        prof += [(r*0.90, z - 0.003), (r*1.0, z), (r*0.90, z + 0.003)]
    prof += [(r*0.85, h - 0.012), (r*0.55, h - 0.002), (r*0.40, h), (0., h)]
    M.add(lathe(prof, (0, 0, 0), (0, 0, 1), 40), "brush_handle", "grip", smooth=True)
    return M.parts()


def brush_rod(r, h):
    """Twisted-wire rod: two wires twisted around each other."""
    M = Model()
    n = 240
    z = np.linspace(-h, h, n)
    for k in range(2):
        th = 2*math.pi*z/0.012 + math.pi*k
        pts = np.stack([0.5*r*np.cos(th), 0.5*r*np.sin(th), z], -1)
        M.add(sweep(pts, 0.55*r, bend=0.0, n=10, cap=True), "steel_bright", "rod_wire", smooth=True)
    return M.parts()


def brush_bristles(r, h):
    """Wire bristle head: a twisted core and radial steel bristles in a spiral, tip rounded."""
    M = Model()
    M.add(cyl((0, 0, -h), (0, 0, h), 0.0014, 0.0014, 12), "steel_bright", "core", smooth=True)
    rng = np.random.default_rng(7)
    vs, fs, off = [], [], 0
    nb = 520
    for k in range(nb):
        z = -h + 0.002 + (2*h - 0.004)*k/nb
        th = 2*math.pi*k*0.381966 + rng.uniform(-0.2, 0.2)
        L = r*(0.92 + 0.08*rng.random())
        taper = min(1., (h - abs(z))/0.006 + 0.3)
        L *= min(1., 0.65 + 0.35*taper)
        d = np.array([math.cos(th), math.sin(th), 0.])
        p0 = np.array([0., 0., z]) + d*0.0012
        p1 = np.array([0., 0., z + rng.uniform(-0.0008, 0.0008)]) + d*L
        e = np.cross(d, [0., 0., 1.])*0.00012
        e2 = np.array([0., 0., 0.00012])
        quad = np.array([p0 - e - e2, p0 + e - e2, p0 + e + e2, p0 - e + e2, p1 - e - e2, p1 + e - e2, p1 + e + e2,
                         p1 - e + e2])
        f = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7], [4, 5, 6],
                      [4, 6, 7]])[:, ::-1]
        vs.append(quad)
        fs.append(f + off)
        off += 8
    M.add((np.concatenate(vs).astype(np.float32), np.concatenate(fs).astype(np.int32)), "brush_wire", "bristles",
          crease=80)
    return M.parts()


def ballpeen_handle(r, h):
    """Hickory handle (cylinder frame, z along the handle, +h at the head): oval section swelling to the butt."""
    zs = np.linspace(-h, h, 60)
    M = Model()

    def rr(TH, Z):
        s = (Z + h)/(2*h)                                    # 0 at the butt, 1 at the head
        bump = np.exp(-((s - 0.03)/0.04)**2)
        a = r*(0.80 - 0.10*s + 0.05*bump)                    # frame x = body -z: the vertical semi-axis
        b = r*(1.02 - 0.20*s + 0.08*bump)                    # frame y = body y: along the head (the oval's major axis)
        return a*b/np.sqrt((b*np.cos(TH))**2 + (a*np.sin(TH))**2)
    v, f = param_cyl(rr, -h, h - 0.002, 40, 80)
    M.add((v, f), "hickory", "handle", smooth=True)
    return M.parts()


def ballpeen_head(r, h):
    """Forged ball-peen head (cylinder frame: z along the head, +h = the flat striking face, -h = the ball peen); the
    handle eye at the middle with its wedge on top (the handle enters along the frame's -x). The flat face is at +h
    because that is the end the world strikes with (parts.PEEN_FACE, the hammer frame's +y). The collider is one
    plain cylinder of the head's length and mass (drawn richer than it collides)."""
    M = Model()
    flip = lambda m: m.mirror([0., 0., 1.])
    face = lathe([(0., -h), (r*0.86, -h), (r*0.98, -h + 0.002), (r, -h + 0.006), (r, -0.012), (r*1.05, -0.006),
                  (r*1.05, 0.0), (0., 0.0)], (0, 0, 0), (0, 0, 1), 48)
    cheeks = rbox((0., 0., -0.004), (2*r*1.05, 2*r*0.92, 0.030), 0.004)
    ball = sphere((0, 0, h - r*0.92), r*0.92, 40)
    peen = lathe([(0., 0.0), (r*1.0, 0.0), (r*0.92, 0.010), (r*0.66, 0.022), (r*0.62, h - r*1.4), (0., h - r*1.4)],
                 (0, 0, 0), (0, 0, 1), 48) + ball
    M.add(flip(face + cheeks + peen), "steel_dark", "head", crease=40)
    M.add(flip(cyl((0, 0, -h - 0.0002), (0, 0, -h + 0.0008), r*0.84, r*0.84, 48)), "steel_bright", "striking_face",
          smooth=True)
    # the handle's end through the eye, flush with the head's top, split by its steel wedge (frame -x = the handle)
    eye = _m3().Manifold.cylinder(0.004, 1., 1., 32).scale([0.0105, 0.0058, 1.])
    M.add(flip(_tf(eye, _rot_z_to((1., 0., 0.)), (r*1.05 - 0.0034, 0., -0.004))), "hickory", "handle_end", smooth=True)
    M.add(flip(box((r*1.05 + 0.0002, 0., -0.004), (0.0008, 0.0012, 0.019))), "steel_dark", "wedge")
    return M.parts()


def punch_barrel(r, h):
    """Automatic centre punch body: diamond-knurled grip, smooth collar and a tapered nose (+h toward the point)."""
    M = Model()

    def rr(TH, Z):
        s = (Z + h)/(2*h)
        knurl = (s > 0.08) & (s < 0.72)
        a = np.sin(TH*24 + Z/0.0016*math.pi)
        b = np.sin(TH*24 - Z/0.0016*math.pi)
        R = np.where(knurl, r - 0.00035*(1 - np.abs(a*b))**0.5, r*0.985)
        nose = np.clip((s - 0.80)/0.20, 0., 1.)
        R = np.where(s > 0.80, r*(1. - 0.50*nose), R)
        return R
    M.add(param_cyl(rr, -h, h, 96, 200), "punch_knurl", "barrel", crease=75)
    M.add(ring((0, 0, -h + 0.42*2*h), (0, 0, -h + 0.42*2*h + 0.0015), r + 0.0002, r - 0.001, 48), "steel_dark",
          "groove", smooth=True)
    return M.parts()


def punch_cap(r, h):
    M = Model()

    def rr(TH, Z):
        s = (Z + h)/(2*h)
        R = r - 0.0003*(np.sin(TH*30) > 0)
        return np.where(s < 0.25, r*(0.75 + s), R)
    M.add(param_cyl(rr, -h, h, 60, 30), "steel_dark", "cap", crease=60)
    return M.parts()


def punch_point(r, h):
    M = Model()
    # the point emerges from the nose: drawn 8 mm back into the barrel (the collider starts 5 mm ahead of the nose)
    M.add(lathe([(0., -h - 0.008), (r, -h - 0.008), (r, h - 0.0016), (0., h)], (0, 0, 0), (0, 0, 1), 24), "punch_tip",
          "point", smooth=True)
    return M.parts()


# ====================================================================================== the robot cart
def robot_cart(half, center, deck_z, mounts, tray, yaw):
    """Mobile robot cart drawn in its collider's frame (world axes, origin at the collider centre). half/center: the
    collider; mounts: [(x, y)] arm bases (world); tray: ((x, y), (hx, hy)) the rubber tool tray (world, physical, drawn
    by its own shape); yaw: the arms' facing (the cart's front = the work side). Welded 40 mm square-tube frame,
    louvred side and front sheet panels, aluminium tread-plate deck, 15 mm machined mounting plates under the arm
    bases, a black anodised tray frame under the tray, swivel casters with brakes, a floor lock, a push handle on the
    back, an e-stop on a short post, two 19-inch arm controllers on the lower shelf with corrugated conduits up
    through the deck to each arm base."""
    m3 = _m3()
    Rw = _rz(yaw)
    Rl = Rw.T
    c = np.asarray(center, float)
    hw = np.asarray(half, float)
    hl = np.abs(Rl @ hw)
    hf, hs = float(hl[0]), float(hl[1])
    H = float(deck_z)
    loc = lambda p: Rl @ (np.array([p[0], p[1], 0.]) - np.array([c[0], c[1], 0.]))
    ml = [loc(m)[:2] for m in mounts]
    (tx, ty), (thx, thy) = tray
    tc = loc((tx, ty))[:2]
    th = np.abs(Rl @ np.array([thx, thy, 0.]))[:2]
    M = Model()
    a = 0.040                                     # square tube
    z_c = 0.128                                   # caster top plate
    deck0, deck1 = H - 0.020, H - 0.015           # 5 mm tread plate under the 15 mm plates
    zt0, zt1 = deck0 - a, deck0
    zb0, zb1 = z_c, z_c + a
    frame = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            frame.append(rbox((sx*(hf - a/2), sy*(hs - a/2), 0.5*(z_c + zt1)), (a, a, zt1 - z_c), 0.004, 8))
    for z0, z1 in ((zb0, zb1), (zt0, zt1)):
        for sy in (-1, 1):
            frame.append(rbox((0., sy*(hs - a/2), 0.5*(z0 + z1)), (2*hf - 2*a + 0.002, a, a), 0.004, 8))
        for sx in (-1, 1):
            frame.append(rbox((sx*(hf - a/2), 0., 0.5*(z0 + z1)), (a, 2*hs - 2*a + 0.002, a), 0.004, 8))
    # mid rail on the back face (shelf stiffener) and weld fillets at the post/rail joints
    frame.append(rbox((-(hf - a/2), 0., 0.50), (a, 2*hs - 2*a + 0.002, 0.030), 0.004, 8))
    for sx in (-1, 1):
        for sy in (-1, 1):
            for z in (zb1, zt0):
                frame.append(sphere((sx*(hf - a), sy*(hs - a), z), 0.0035, 8))
    M.add(m3.Manifold.batch_boolean(frame, m3.OpType.Add), "cart_frame", "frame", crease=40)
    # louvred sheet panels on both sides and the front
    def louvred(width, height, cols, rows, z_top):
        pan = box((0., 0., 0.), (width, 0.0015, height))
        hoods = []
        cuts = []
        for ci in range(cols):
            x = (ci - (cols - 1)/2)*(width/(cols + 0.6))
            for ri in range(rows):
                z = z_top - 0.03 - ri*0.026
                cuts.append(box((x, 0., z), (width/(cols + 0.6)*0.72, 0.004, 0.006)))
                hood = cyl((x - width/(cols + 0.6)*0.36, 0., z + 0.003), (x + width/(cols + 0.6)*0.36, 0., z + 0.003),
                           0.0075, 0.0075, 24) - cyl((x - 0.2, 0., z + 0.003), (x + 0.2, 0., z + 0.003), 0.0060,
                                                     0.0060, 24)
                hood = hood ^ box((x, -0.006, z + 0.006), (width/(cols + 0.6)*0.72, 0.012, 0.012))
                hoods.append(hood)
        pan = pan - m3.Manifold.batch_boolean(cuts, m3.OpType.Add)
        return pan + m3.Manifold.batch_boolean(hoods, m3.OpType.Add)
    pw, ph = 2*hf - 2*a, zt0 - zb1
    side = louvred(pw - 0.004, ph - 0.004, 2, 6, 0.5*ph - 0.02)
    for sy in (-1, 1):
        Rp = np.eye(3) if sy < 0 else _rz(math.pi)
        M.add(_tf(side, Rp, (0., sy*(hs - 0.004), 0.5*(zb1 + zt0))), "cart_panel", "side_panels", crease=40)
    fw = 2*hs - 2*a
    front = louvred(fw - 0.004, ph - 0.004, 3, 3, 0.5*ph - 0.02)
    M.add(_tf(front, _rz(math.pi/2), (hf - 0.004, 0., 0.5*(zb1 + zt0))), "cart_panel", "front_panel", crease=40)
    # panel rivets
    for sy in (-1, 1):
        for x in np.linspace(-hf + a + 0.02, hf - a - 0.02, 5):
            for z in (zb1 + 0.012, zt0 - 0.012):
                M.add(m3.Manifold.sphere(0.003, 10).scale([1., 0.5, 1.]).translate([x, sy*(hs - 0.0045), z]),
                      "zinc", "rivets", smooth=True)
    # lower shelf (sheet with a turned-up lip at the back)
    M.add(box2((-hf + a, -hs + a, zb1), (hf - a, hs - a, zb1 + 0.002)), "cart_frame", "shelf")
    M.add(box2((-hf + a, -hs + a, zb1), (-hf + a + 0.002, hs - a, zb1 + 0.025)), "cart_frame", "shelf_lip")
    # deck: aluminium tread plate
    M.add(rbox2((-hf, -hs, deck0), (hf, hs, deck1), 0.0015, 6), "tread_alu", "deck", uv=1., crease=30)
    # arm mounting plates, cap screws and the cable grommets behind each base
    for (mx, my) in ml:
        x0, x1 = mx - 0.190, min(mx + 0.050, hf)
        M.add(rbox2((x0, my - 0.105, deck1), (x1, my + 0.105, H), 0.0015, 6), "alu_machined", "mount_plates", crease=30)
        for by in (-0.085, 0.0, 0.085):
            bx = mx - 0.172
            head = cyl((bx, my + by, H), (bx, my + by, H + 0.010), 0.008, 0.008, 32)
            sock = hexagon((bx, my + by, H + 0.0045), (bx, my + by, H + 0.011), 0.008)
            M.add(head - sock, "black_oxide", "cap_screws", crease=40)
        gx, gy = mx - 0.215, my + 0.03
        M.add(ring((gx, gy, deck1 - 0.0005), (gx, gy, deck1 + 0.004), 0.021, 0.0135, 40), "rubber_dust", "grommets",
              smooth=True)
        # conduit: from under the deck up through the grommet into the base's rear connector
        M.add(sweep([(gx, gy, deck1 - 0.02), (gx, gy, H + 0.060), (mx - 0.150, gy, H + 0.060)], 0.0115, bend=0.03,
                    n=20, corrugate=(0.0009, 0.0042)), "conduit_black", "conduit", smooth=True)
        M.add(cyl((mx - 0.163, gy, H + 0.060), (mx - 0.152, gy, H + 0.060), 0.0145, 0.0145, 24), "black_oxide",
              "conduit_glands", smooth=True)
    # tray frame (black anodised) under the rubber tray
    M.add(rbox2((tc[0] - th[0] - 0.004, tc[1] - th[1] - 0.004, deck1), (tc[0] + th[0] + 0.004, tc[1] + th[1] + 0.004, H),
                0.002, 6), "anodized_black", "tray_frame", crease=30)
    # two 19-inch arm controllers (2U, 483 x 355 x 89 mm) facing the open back, on rack rails
    xf0 = -hf + 0.070
    for k in range(min(2, len(ml))):
        z0 = zb1 + 0.002 + k*0.093
        M.add(rbox2((xf0 + 0.004, -0.222, z0), (xf0 + 0.355, 0.222, z0 + 0.089), 0.003, 6), "controller_black",
              "controller_body", crease=30)
        M.add(rbox2((xf0, -0.2415, z0 + 0.0005), (xf0 + 0.004, 0.2415, z0 + 0.0885), 0.001, 6), "controller_front",
              "controller_front", crease=30)
        for sy in (-1, 1):
            M.add(sweep([(xf0, sy*0.212, z0 + 0.016), (xf0 - 0.032, sy*0.212, z0 + 0.016),
                         (xf0 - 0.032, sy*0.212, z0 + 0.073), (xf0, sy*0.212, z0 + 0.073)], 0.0045, bend=0.010, n=12),
                  "alu_machined", "rack_handles", smooth=True)
            for z in (z0 + 0.012, z0 + 0.077):
                M.add(cyl((xf0, sy*0.232, z), (xf0 - 0.003, sy*0.232, z), 0.0035, 0.0035, 16), "zinc", "rack_screws")
        for j in range(9):
            y = -0.150 + 0.026*j
            M.add(box2((xf0 - 0.0006, y, z0 + 0.018), (xf0 + 0.001, y + 0.017, z0 + 0.071)), "vent_dark", "vents")
        M.add(rbox2((xf0 - 0.003, 0.120, z0 + 0.030), (xf0, 0.150, z0 + 0.058), 0.0015, 6), "switch_red", "switches")
        for j, mat in enumerate(("led_green", "led_green", "led_amber")):
            M.add(box2((xf0 - 0.0015, 0.168 + 0.009*j, z0 + 0.060), (xf0, 0.173 + 0.009*j, z0 + 0.065)), mat, "leds")
        M.add(box2((xf0 - 0.0008, -0.200, z0 + 0.034), (xf0, -0.160, z0 + 0.054)), "decal_white", "labels")
        # conduits from the controller's rear up to the arm's grommet
        mx, my = ml[k]
        gx, gy = mx - 0.215, my + 0.03
        yr = np.sign(my)*0.12
        zc = z0 + 0.045
        M.add(sweep([(xf0 + 0.355, yr, zc), (xf0 + 0.40, yr, zc), (gx + 0.02, gy, zc + 0.12), (gx, gy, deck0 - 0.08),
                     (gx, gy, deck1 - 0.01)], 0.0115, bend=0.05, n=20, corrugate=(0.0009, 0.0042)), "conduit_black",
              "conduit", smooth=True)
    for sy in (-1, 1):
        M.add(box2((xf0 - 0.002, sy*0.2415 - 0.010, zb1), (xf0 + 0.018, sy*0.2415 + 0.010, zt0)), "zinc", "rack_rails")
    # power distribution strip (1U) above the controllers
    zp = zb1 + 0.002 + 2*0.093
    M.add(rbox2((xf0 + 0.004, -0.222, zp), (xf0 + 0.20, 0.222, zp + 0.044), 0.003, 6), "controller_black", "pdu")
    for j in range(6):
        y = -0.17 + 0.065*j
        M.add(rbox2((xf0 - 0.001, y - 0.016, zp + 0.008), (xf0 + 0.004, y + 0.016, zp + 0.036), 0.002, 6),
              "decal_white", "pdu_sockets")
    # push handle on the back (away from the work) with a foam grip, e-stop on a short post at the back-left corner
    hb = -hf - 0.045
    M.add(sweep([(-hf + 0.002, hs - a/2, H - 0.11), (hb, hs - a/2, H - 0.11), (hb, hs - a/2, H + 0.16),
                 (hb, -(hs - a/2), H + 0.16), (hb, -(hs - a/2), H - 0.11), (-hf + 0.002, -(hs - a/2), H - 0.11)],
                0.0127, bend=0.035, n=24), "cart_frame", "handle", smooth=True)
    M.add(sweep([(hb, 0.30, H + 0.16), (hb, -0.30, H + 0.16)], 0.0172, bend=0.0, n=24), "rubber_grey", "handle_grip",
          smooth=True)
    for sy in (-1, 1):
        M.add(rbox2((-hf - 0.004, sy*(hs - a/2) - 0.022, H - 0.135), (-hf + 0.001, sy*(hs - a/2) + 0.022, H - 0.085),
                    0.002, 6), "cart_frame", "handle_brackets")
    ex, ey = -hf + 0.050, hs - 0.050
    ez = H + 0.065                                # box bottom on a 65 mm post
    M.add(cyl((ex, ey, deck1), (ex, ey, deck1 + 0.004), 0.030, 0.030, 32), "cart_frame", "estop_post")
    M.add(cyl((ex, ey, deck1 + 0.004), (ex, ey, ez), 0.0125, 0.0125, 24), "cart_frame", "estop_post", smooth=True)
    M.add(rbox2((ex - 0.036, ey - 0.036, ez), (ex + 0.036, ey + 0.036, ez + 0.073), 0.007), "estop_yellow",
          "estop_box", crease=40)
    M.add(cyl((ex, ey, ez + 0.073), (ex, ey, ez + 0.0735), 0.031, 0.031, 40), "estop_yellow", "estop_ring")
    M.add(cyl((ex, ey, ez + 0.073), (ex, ey, ez + 0.088), 0.013, 0.013, 32), "estop_red", "estop_button", smooth=True)
    M.add(lathe([(0., 0.), (0.021, 0.), (0.0215, 0.006), (0.017, 0.013), (0., 0.0155)], (ex, ey, ez + 0.085),
                (0, 0, 1), 40), "estop_red", "estop_button", smooth=True)
    M.add(cyl((ex + 0.036, ey, ez + 0.020), (ex + 0.046, ey, ez + 0.020), 0.006, 0.006, 16), "black_oxide", "estop_gland")
    # swivel casters with brakes
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx*(hf - 0.060), sy*(hs - 0.060)
            M.extend(transform_parts(caster(0.050, z_c, braked=True), _rz(math.pi if sx > 0 else 0.), (cx, cy, 0.)))
    # floor lock at the back centre (engaged: its rubber foot is down on the floor)
    lx = -hf + 0.030
    M.add(rbox2((lx - 0.030, -0.035, 0.050), (lx + 0.020, 0.035, z_c), 0.004), "zinc", "floor_lock", crease=40)
    M.add(cyl((lx - 0.005, 0., 0.014), (lx - 0.005, 0., 0.052), 0.010, 0.010, 24), "steel_bright", "floor_lock_pin",
          smooth=True)
    M.add(cyl((lx - 0.005, 0., 0.0), (lx - 0.005, 0., 0.014), 0.028, 0.026, 32), "rubber_grey", "floor_lock_foot",
          smooth=True)
    M.add(rbox2((lx - 0.075, -0.025, 0.090), (lx - 0.028, 0.025, 0.098), 0.002, 6), "estop_red", "floor_lock_pedal")
    # parts were built about the cart's floor centre with x forward: into the collider frame
    return M.parts(Rw, (0., 0., -c[2]))


def caster(r, h_top, braked=True, width=0.032):
    """Swivel caster about the origin (swivel axis = z), wheel trailing towards -x; top plate under z = h_top."""
    M = Model()
    off = -0.032 if r > 0.04 else -0.024
    M.add(rbox2((-0.045, -0.038, h_top - 0.006), (0.045, 0.038, h_top), 0.002, 6), "zinc", "caster_plate")
    M.add(cyl((0, 0, h_top - 0.016), (0, 0, h_top - 0.006), 0.032*r/0.05, 0.032*r/0.05, 32), "zinc", "caster_race",
          smooth=True)
    yoke = rbox2((off - 0.022, -0.026*width/0.032, h_top - 0.021), (0.018, 0.026*width/0.032, h_top - 0.016), 0.002, 6)
    legs = []
    for sy in (-1, 1):
        yy = sy*(0.5*width + 0.005)
        legs.append(_m3().Manifold.batch_hull([box((0.0, yy, h_top - 0.019), (0.030, 0.004, 0.004)),
                                               box((off, yy, r), (0.030, 0.004, 0.026))]))
    M.add(yoke + legs[0] + legs[1], "zinc", "caster_fork", crease=40)
    tread = lathe([(r*0.62, -0.5*width), (r - 0.005, -0.5*width), (r, -0.5*width + 0.006), (r, 0.5*width - 0.006),
                   (r - 0.005, 0.5*width), (r*0.62, 0.5*width)], (off, 0., r), (0., 1., 0.), 48)
    hub = lathe([(0., -0.5*width - 0.002), (r*0.64, -0.5*width - 0.002), (r*0.64, 0.5*width + 0.002),
                 (0., 0.5*width + 0.002)], (off, 0., r), (0., 1., 0.), 40)
    M.add(tread, "caster_tread", "caster_tread", smooth=True)
    M.add(hub, "caster_hub", "caster_hub", smooth=True)
    for sy in (-1, 1):
        M.add(hexagon((off, sy*(0.5*width + 0.007), r), (off, sy*(0.5*width + 0.012), r), 0.010), "zinc", "caster_nuts")
    if braked:
        M.add(rbox2((off - 0.034, -0.016, r + 0.020), (off - 0.012, 0.016, r + 0.026), 0.002, 6), "estop_red",
              "caster_brake")
        M.add(box2((off - 0.014, -0.004, r + 0.018), (off, 0.004, h_top - 0.021)), "zinc", "caster_brake_arm")
    return M.parts()


# ====================================================================================== kit trolley
def kit_trolley(half):
    """Two-shelf steel service trolley in its collider's frame (origin at the centre): 25 mm tube posts, a top shelf
    under the rubber kit tray, a lipped lower shelf with an anchor box and a dust-bag roll, 75 mm casters, a push handle
    at the +x end below the tray."""
    hx, hy, hz = half
    M = Model()
    m3 = _m3()
    zc = 0.098
    a = 0.025
    top = 2*hz
    posts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            posts.append(rbox((sx*(hx - a/2), sy*(hy - a/2), 0.5*(zc + top - 0.015)), (a, a, top - 0.015 - zc), 0.003, 8))
    M.add(m3.Manifold.batch_boolean(posts, m3.OpType.Add), "trolley_grey", "posts", crease=40)
    M.add(rbox2((-hx, -hy, top - 0.016), (hx, hy, top), 0.002, 6) - box2((-hx + 0.004, -hy + 0.004, top - 0.020),
                                                                          (hx - 0.004, hy - 0.004, top - 0.003)),
          "trolley_grey", "top_shelf", crease=40)
    zl = 0.22
    lower = (rbox2((-hx, -hy, zl - 0.002), (hx, hy, zl + 0.022), 0.002, 6)
             - box2((-hx + 0.0015, -hy + 0.0015, zl), (hx - 0.0015, hy - 0.0015, zl + 0.03)))
    M.add(lower, "trolley_grey", "lower_shelf", crease=40)
    for sx in (-1, 1):
        for sy in (-1, 1):
            M.extend(transform_parts(caster(0.0375, zc, braked=sx > 0, width=0.025), _rz(0. if sx > 0 else math.pi),
                                     (sx*(hx - 0.045), sy*(hy - 0.050), 0.)))
    M.add(sweep([(hx - 0.002, 0.16, top - 0.050), (hx + 0.045, 0.16, top - 0.050), (hx + 0.045, -0.16, top - 0.050),
                 (hx - 0.002, -0.16, top - 0.050)], 0.011, bend=0.025, n=20), "steel_bright", "handle", smooth=True)
    # stock on the lower shelf
    M.add(rbox2((-hx + 0.03, -0.20, zl + 0.001), (-hx + 0.18, -0.07, zl + 0.091), 0.002, 6), "anchor_box_red", "box_a",
          crease=40)
    M.add(box2((-hx + 0.03, -0.1355, zl + 0.040), (-hx + 0.18, -0.1345, zl + 0.070)) .translate([0., -0.065, 0.]),
          "decal_white", "box_label")
    M.add(rbox2((0.0, -0.06, zl + 0.001), (0.16, 0.15, zl + 0.071), 0.002, 6), "cardboard", "box_b", crease=40)
    M.add(cyl((-0.10, 0.12, zl + 0.045), (0.05, 0.12, zl + 0.045), 0.044, 0.044, 32), "bag_clear", "dust_bags",
          smooth=True)
    return M.parts(None, (0., 0., -hz))


# ====================================================================================== steel workbench
def workbench(bx, by, top, legs=None, shelf_z=(0.19, 0.21), drawer_end=1):
    """Heavy-duty steel workbench in world coordinates: 40 mm steel top with a folded edge, 50 mm box-tube legs with
    levelling feet, 40 mm aprons and stretchers, a lower shelf, and a two-drawer unit hung under the end `drawer_end`
    (+1: the +y end) of the top. bx, by: the top's x and y ranges; legs: [(x, y)] leg centres (50 mm square)."""
    m3 = _m3()
    M = Model()
    x0, x1 = bx
    y0, y1 = by
    legs = legs or [(x0 + 0.05, y0 + 0.05), (x0 + 0.05, y1 - 0.05), (x1 - 0.05, y0 + 0.05), (x1 - 0.05, y1 - 0.05)]
    lx0, lx1 = min(p[0] for p in legs), max(p[0] for p in legs)
    ly0, ly1 = min(p[1] for p in legs), max(p[1] for p in legs)
    t0 = top - 0.040
    # top: a 3 mm steel plate on a folded 40 mm edge
    M.add(rbox2((x0, y0, top - 0.004), (x1, y1, top), 0.0015, 6), "bench_top_steel", "top_plate", uv=1., crease=30)
    edge = rbox2((x0, y0, t0), (x1, y1, top - 0.004), 0.002, 6) - box2((x0 + 0.003, y0 + 0.003, t0 - 0.01),
                                                                        (x1 - 0.003, y1 - 0.003, top - 0.006))
    M.add(edge, "bench_blue", "top_edge", uv=1., crease=30)
    frame = []
    za0, za1 = t0 - 0.040, t0
    for (lx, ly) in legs:
        frame.append(rbox((lx, ly, 0.5*(0.040 + t0)), (0.050, 0.050, t0 - 0.040), 0.004, 8))
    for z0, z1 in ((za0, za1), (shelf_z[0] - 0.040, shelf_z[0])):
        for lx in (lx0, lx1):
            frame.append(rbox((lx, 0.5*(ly0 + ly1), 0.5*(z0 + z1)), (0.040, ly1 - ly0 - 0.048, z1 - z0), 0.004, 8))
        for ly in (ly0, ly1):
            frame.append(rbox((0.5*(lx0 + lx1), ly, 0.5*(z0 + z1)), (lx1 - lx0 - 0.048, 0.040, z1 - z0), 0.004, 8))
    M.add(m3.Manifold.batch_boolean(frame, m3.OpType.Add), "bench_blue", "frame", uv=1., crease=40)
    for (lx, ly) in legs:
        M.add(cyl((lx, ly, 0.035), (lx, ly, 0.040), 0.024, 0.024, 6), "black_oxide", "leg_caps")
        M.add(hexagon((lx, ly, 0.026), (lx, ly, 0.035), 0.024), "zinc", "feet_nuts")
        M.add(cyl((lx, ly, 0.012), (lx, ly, 0.035), 0.008, 0.008, 16), "zinc", "feet_studs", smooth=True)
        M.add(lathe([(0., 0.), (0.030, 0.), (0.030, 0.006), (0.020, 0.013), (0., 0.013)], (lx, ly, 0.), (0, 0, 1), 40),
              "rubber_grey", "feet_pads", smooth=True)
    # lower shelf: steel sheet with a turned-down edge (the collider shelf)
    M.add(rbox2((x0 + 0.03, y0 + 0.03, shelf_z[0]), (x1 - 0.03, y1 - 0.03, shelf_z[1]), 0.002, 6), "bench_blue", "shelf",
          uv=1., crease=30)
    # drawer unit under one end, opening to the front (-x)
    if drawer_end:
        dy0, dy1 = (ly1 - 0.50, ly1 - 0.03) if drawer_end > 0 else (ly0 + 0.03, ly0 + 0.50)
        dx0, dx1 = lx0 + 0.035, lx1 - 0.030
        dz0, dz1 = za0 - 0.30, za0
        M.add(rbox2((dx0 + 0.010, dy0, dz0), (dx1, dy1, dz1), 0.002, 6), "bench_blue", "drawer_case", uv=1., crease=30)
        for k in range(2):
            z0 = dz0 + 0.006 + k*0.147
            z1 = z0 + 0.141
            M.add(rbox2((dx0, dy0 + 0.005, z0), (dx0 + 0.012, dy1 - 0.005, z1), 0.002, 6), "drawer_grey", "drawer_fronts",
                  crease=30)
            zm = 0.5*(z0 + z1) + 0.035
            M.add(sweep([(dx0, 0.5*(dy0 + dy1) - 0.11, zm), (dx0 - 0.028, 0.5*(dy0 + dy1) - 0.09, zm),
                         (dx0 - 0.028, 0.5*(dy0 + dy1) + 0.09, zm), (dx0, 0.5*(dy0 + dy1) + 0.11, zm)], 0.006, bend=0.012,
                        n=14), "alu_machined", "drawer_pulls", smooth=True)
            M.add(box2((dx0 - 0.001, 0.5*(dy0 + dy1) - 0.04, zm - 0.065), (dx0, 0.5*(dy0 + dy1) + 0.04, zm - 0.045)),
                  "decal_white", "drawer_labels")
    return M.parts()


def press_cabinet(half):
    """Steel cabinet stand under the drill press, in its collider's frame (origin at the centre, local +x = the front
    facing the operator): 8 mm top plate, folded-sheet body, recessed plinth, a front door with a recessed swing
    handle, key lock, hinges and louvres."""
    hx, hy, hz = half
    M = Model()
    M.add(rbox2((-hx, -hy, hz - 0.008), (hx, hy, hz), 0.002, 6), "steel_dark", "top_plate", crease=30)
    body = rbox2((-hx + 0.004, -hy + 0.003, -hz + 0.055), (hx - 0.004, hy - 0.003, hz - 0.008), 0.004, 8)
    M.add(body, "bench_blue", "body", uv=1., crease=30)
    M.add(box2((-hx + 0.03, -hy + 0.02, -hz), (hx - 0.045, hy - 0.02, -hz + 0.056)), "black_oxide", "plinth")
    dz0, dz1 = -hz + 0.065, hz - 0.020
    door = rbox2((hx - 0.006, -hy + 0.010, dz0), (hx, hy - 0.010, dz1), 0.002, 6)
    recess = box2((hx - 0.0045, hy - 0.060, 0.10), (hx + 0.001, hy - 0.030, 0.22))
    louv = []
    m3 = _m3()
    for k in range(6):
        z = dz0 + 0.04 + 0.022*k
        louv.append(box2((hx - 0.0045, -hy + 0.06, z), (hx + 0.001, hy - 0.10, z + 0.008)))
    door = door - recess - m3.Manifold.batch_boolean(louv, m3.OpType.Add)
    M.add(door, "drawer_grey", "door", crease=30)
    M.add(box2((hx - 0.0046, hy - 0.060, 0.10), (hx - 0.0040, hy - 0.030, 0.22)), "vent_dark", "recess_back")
    for k in range(6):
        z = dz0 + 0.04 + 0.022*k
        M.add(box2((hx - 0.0046, -hy + 0.06, z), (hx - 0.004, hy - 0.10, z + 0.008)), "vent_dark", "louvre_backs")
    M.add(rbox2((hx - 0.004, hy - 0.056, 0.106), (hx - 0.0005, hy - 0.034, 0.200), 0.003, 6), "black_oxide", "handle")
    M.add(cyl((hx, hy - 0.045, 0.245), (hx + 0.003, hy - 0.045, 0.245), 0.008, 0.008, 24), "steel_bright", "lock",
          smooth=True)
    M.add(box2((hx + 0.0025, hy - 0.0465, 0.239), (hx + 0.0031, hy - 0.0435, 0.251)), "vent_dark", "lock_slot")
    for z in (dz0 + 0.06, dz1 - 0.06):
        M.add(cyl((hx + 0.003, -hy + 0.006, z - 0.03), (hx + 0.003, -hy + 0.006, z + 0.03), 0.004, 0.004, 16),
              "steel_dark", "hinges", smooth=True)
    M.add(box2((hx - 0.0005, -0.05, dz1 - 0.07), (hx + 0.0003, 0.05, dz1 - 0.04)), "decal_white", "door_label")
    return M.parts()


def strap_clamp_strap(toe_half):
    """Plain strap clamp (TE-CO 30402) drawn on its toe collider (box frame: x towards the toe's tip): the whole strap
    with its stud slot and its tapered nose. clamps.StrapClamp builds the same outline from four boxes (toe, two rails,
    tail); only the nose's taper is not in the colliders (they keep the full height to the tip)."""
    from . import params as P
    L, W, H = P.HOLD_DOWN_STRAP_M
    sl, sw = P.HOLD_DOWN_SLOT_LEN_M, P.HOLD_DOWN_SLOT_W_M
    x1 = float(toe_half[0])                          # the toe's tip
    x0 = x1 - L
    xc = 0.5*(x0 + x1)
    zc = -0.5*P.HOLD_DOWN_BEARING_M[1]               # the strap's mid-plane: the toe box stands back from the underside
    body = rbox2((x0, -0.5*W, zc - 0.5*H), (x1 - 0.022, 0.5*W, zc + 0.5*H), 0.0012, 6)
    nose = rbox2((x1 - 0.004, -0.5*W, zc - 0.5*H), (x1, 0.5*W, zc - 0.05*H), 0.0012, 6)
    strap = _m3().Manifold.batch_hull([body, nose])
    slot = rbox2((xc - 0.5*sl, -0.5*sw, zc - H), (xc + 0.5*sl, 0.5*sw, zc + H), 0.499*sw, 12)
    M = Model()
    M.add(strap - slot, "black_oxide", "strap", crease=35)
    return M.parts()


def strap_clamp_washer(half):
    """Spherical washer under the clamp nut (TE-CO 42742), drawn round on its collider (a square of the same area;
    box frame, z up): the dished seat on top."""
    from . import params as P
    r_out, t = 0.5*P.HOLD_DOWN_WASHER_M[0], 2*float(half[2])
    r_in = 0.5*0.438*0.0254
    M = Model()
    M.add(lathe([(r_in, -0.5*t), (r_out - 0.0005, -0.5*t), (r_out, -0.5*t + 0.0005), (r_out, 0.5*t),
                 (r_in + 0.002, 0.5*t - 0.0012), (r_in, 0.5*t - 0.0012)], (0, 0, 0), (0, 0, 1), 48), "black_oxide",
          "washer", crease=40)
    return M.parts()


def strap_clamp_nut(r, h):
    """Spherical flange nut (TE-CO 41803) drawn on its flange collider (cylinder frame, z up): the flange with its
    rounded seat and the hexagon above it (the hexagon's own collider is hidden)."""
    from . import params as P
    _, nut_h, _, af = P.HOLD_DOWN_NUT_M
    top = -h + nut_h
    rc = af/math.sqrt(3.)
    M = Model()
    M.add(lathe([(0., -h - 0.0008), (0.55*r, -h - 0.0008), (r, -h + 0.0010), (r, h - 0.0004), (r - 0.0012, h), (0., h)],
                (0, 0, 0), (0, 0, 1), 48), "black_oxide", "flange", crease=40)
    nut = hexagon((0, 0, h), (0, 0, top), af)
    ch = lathe([(0., h - 0.001), (rc*1.05, h - 0.001), (rc*1.05, top - 0.0012), (0.5*af*0.97, top), (0., top)],
               (0, 0, 0), (0, 0, 1), 48)
    M.add(nut ^ ch, "black_oxide", "nut", crease=35)
    return M.parts()


# ====================================================================================== press vise, press chuck
def vise_bed(half, side):
    """Cast way rail of the drill-press vise (box frame), with its slotted mounting ear outside; the bed0 rail also
    carries the screw's end bracket (the bed rails are at vise y = -+0.04; side = the rail's vise-y sign)."""
    hx, hy, hz = half
    M = Model()
    rail = rbox((0., 0., 0.), (2*hx, 2*hy, 2*hz), 0.003, 8)
    ey = side*(hy + 0.010)
    ear = rbox((0., ey, -hz + 0.006), (0.060, 0.030, 0.012), 0.004, 8)
    slot = rbox((0., ey + side*0.004, -hz + 0.006), (0.030, 0.0085, 0.03), 0.004, 8)
    M.add((rail + ear) - slot, "vise_blue", "rail", crease=40)
    if side < 0:
        # end bracket across both rails at the crank end, bored for the lead screw (axis at vise z 0.035)
        yc = -side*0.04
        br = rbox((hx - 0.010, yc, 0.0225 - 0.015), (0.020, 0.090, 0.045), 0.004, 8)
        bore = cyl((hx - 0.03, yc, 0.035 - 0.015), (hx + 0.01, yc, 0.035 - 0.015), 0.0075, 0.0075, 24)
        M.add(br - bore, "vise_blue", "end_bracket", crease=40)
    return M.parts()


def vise_jaw(half, moving):
    """Cast vise jaw (box frame): chamfered top edges, a gusset on its back, and on the moving jaw skirts that ride
    over the ways."""
    hx, hy, hz = half
    M = Model()
    jaw = rbox((0., 0., 0.), (2*hx, 2*hy, 2*hz), 0.004, 8)
    back = 1. if moving else -1.                     # the back faces away from the clamping face
    for sy in (-1, 1):
        g = _m3().Manifold.batch_hull([box((back*(hx - 0.004), sy*(hy - 0.012), -hz + 0.004), (0.008, 0.010, 0.008)),
                                       box((back*(hx + 0.010), sy*(hy - 0.012), -hz + 0.004), (0.004, 0.010, 0.008)),
                                       box((back*(hx - 0.004), sy*(hy - 0.012), hz - 0.010), (0.008, 0.010, 0.006))])
        jaw = jaw + g
    if moving:
        for sy in (-1, 1):
            jaw = jaw + rbox((0., sy*(hy + 0.004), -hz - 0.006), (2*hx, 0.010, 0.024), 0.003, 8)
    M.add(jaw, "vise_blue", "jaw", crease=40)
    return M.parts()


def jaw_plate(half):
    """Hardened jaw plate with horizontal serrations and two cap screws."""
    hx, hy, hz = half
    M = Model()
    pl = box((0., 0., 0.), (2*hx, 2*hy, 2*hz))
    for k in range(9):
        z = -hz + 0.003 + k*(2*hz - 0.006)/8
        pl = pl - _tf(box((0., 0., 0.), (0.0012, 2*hy + 0.01, 0.0012)), _ry(math.pi/4), (hx, 0., z)) \
                - _tf(box((0., 0., 0.), (0.0012, 2*hy + 0.01, 0.0012)), _ry(math.pi/4), (-hx, 0., z))
    M.add(pl, "steel_dark", "plate", crease=30)
    for sy in (-1, 1):
        for sx in (-1, 1):
            M.add(cyl((sx*hx, sy*hy*0.7, 0.002), (sx*(hx + 0.0006), sy*hy*0.7, 0.002), 0.0035, 0.0035, 16),
                  "black_oxide", "screws")
    return M.parts()


def vise_screw(r, h):
    """Lead screw hub at the end bracket with the screw running back under the jaw (cylinder frame z = vise x)."""
    M = Model()
    M.add(lathe([(0., -h), (r*0.9, -h), (r, -h + 0.002), (r, h - 0.002), (r*0.9, h), (0., h)], (0, 0, 0), (0, 0, 1), 32),
          "steel_dark", "hub", smooth=True)
    M.add(threaded(0.0065, 0.0025, -h - 0.105, -h + 0.001, 24), "steel_bright", "lead_screw", crease=70)
    return M.parts()


def crank_arm(r, h):
    """Forged crank lever (capsule frame: z from the hub up to the knob)."""
    M = Model()
    M.add(sweep([(0., 0., -h - 0.004), (0., 0., h + 0.004)], r, bend=0., n=16), "steel_bright", "lever", smooth=True)
    M.add(cyl((0., 0., -h - 0.010), (0., 0., -h + 0.010), r*1.9, r*1.9, 24), "steel_bright", "boss", smooth=True)
    return M.parts()


def crank_knob(r, h):
    """Revolving handle grip on the crank (cylinder frame along the handle)."""
    M = Model()
    prof = [(0., -h), (r*0.6, -h), (r*0.9, -h + 0.004), (r, -h + 0.012), (r*0.92, 0.), (r, h - 0.010), (r*0.85, h - 0.002),
            (r*0.5, h), (0., h)]
    M.add(lathe(prof, (0, 0, 0), (0, 0, 1), 40), "knob_black", "grip", smooth=True)
    return M.parts()


def jacobs_chuck(r, h):
    """Keyed 1/2 in drill chuck (cylinder frame, z up; the nose and jaws at -h): arbor collar, knurled sleeve with
    gear teeth for the key, three key holes, tapered nose with the jaws converging on the bit."""
    M = Model()

    def rr(TH, Z):
        s = (Z + h)/(2*h)                       # 0 at the nose, 1 at the top
        R = np.where(s < 0.32, r*(0.55 + 0.45*s/0.32), r)
        knurl = (s > 0.45) & (s < 0.92)
        R = np.where(knurl, r - 0.0004*(np.sin(TH*36) > 0), R)
        R = np.where(s > 0.95, r*0.82, R)
        return R
    M.add(param_cyl(rr, -h, h, 72, 90), "chuck_black", "body", crease=60)
    for k in range(3):
        a = 2*math.pi*k/3 + math.pi/6
        d = np.array([math.cos(a), math.sin(a), 0.])
        M.add(cyl(d*(r - 0.003) + np.array([0, 0, -h + 0.45*2*h]), d*(r + 0.0004) + np.array([0, 0, -h + 0.45*2*h]),
                  0.0032, 0.0032, 16), "vent_dark", "key_holes")
        jaw = cyl((0.006*math.cos(a + math.pi/3), 0.006*math.sin(a + math.pi/3), -h + 0.006),
                  (0.0042*math.cos(a + math.pi/3), 0.0042*math.sin(a + math.pi/3), -h - 0.003), 0.0024, 0.0020, 12)
        M.add(jaw, "steel_bright", "jaws", smooth=True)
    return M.parts()


def knurled_nut(r, h):
    M = Model()
    M.add(param_cyl(lambda TH, Z: r - 0.0005*(np.sin(TH*20) > 0) - 0.0008*(np.abs(Z) > h - 0.0012), -h, h, 80, 12),
          "steel_bright", "nut", crease=60)
    return M.parts()


# ====================================================================================== the registry
STATIC = {"cart_body": "cart", "kit_trolley": "trolley", "bench_top": "bench", "bench_leg0": "hide",
          "bench_leg1": "hide", "bench_leg2": "hide", "bench_leg3": "hide", "bench_shelf": "hide"}
# (owner kind, label suffix) -> builder(r_or_half, h) for cylinders / boxes in the shape's own frame
BY_SUFFIX = {
    ("RotaryHammer", "chuck_sleeve"): lambda sc, s: _hammer_chuck(s[0], s[1]),
    ("RotaryHammer", "trigger_blade"): lambda sc, s: _hammer_trigger(s),
    ("RotaryHammer", "mode_dial"): lambda sc, s: _hammer_dial(s[0], s[1]),
    ("RotaryHammer", "mode_pointer"): lambda sc, s: [],
    ("RotaryHammer", "depth_rod_bar"): lambda sc, s: _depth_rod(s[0], s[1]),
    ("bolt", "hex"): lambda sc, s: hex_head(s),
    # tools.Parts.hex_prism: three boxes turned 60 deg apart; the first carries the drawing, the others are hidden
    ("bolt", "hex_head"): lambda sc, s: hex_head((s[0], s[2], s[2])),
    ("bolt", "hex_head1"): lambda sc, s: [],
    ("bolt", "hex_head2"): lambda sc, s: [],
    # parts.WedgeAnchor's nut: the hex below its chamfered top carries the drawing of the whole nut (the collider's
    # box is shorter than the nut by NUT_CHAMFER_H_M and stands that much nearer the washer); the top's boxes are hidden
    ("anchor", "hex_nut"): lambda sc, s: _anchor_nut(s),
    ("anchor", "hex_nut1"): lambda sc, s: [],
    ("anchor", "hex_nut2"): lambda sc, s: [],
    ("anchor", "hex_nut_top"): lambda sc, s: [],
    ("anchor", "hex_nut_top1"): lambda sc, s: [],
    ("anchor", "hex_nut_top2"): lambda sc, s: [],
    ("bolt", "shank_body"): lambda sc, s: bolt_shank(s[0], s[1], None, 0.00125),
    ("carriage_bolt", "dome"): lambda sc, s: carriage_dome(s[0], s[1]),
    ("carriage_bolt", "shank_body"): lambda sc, s: bolt_shank(s[0], s[1], 0.040, 0.00195),
    ("anchor", "nut"): lambda sc, s: hex_nut(s[0], s[1]),
    ("anchor", "washer"): lambda sc, s: washer(s[0], s[1], 0.0068),
    ("anchor", "stub"): lambda sc, s: anchor_stub(s[0], s[1]),
    ("anchor", "shank_body"): lambda sc, s: anchor_shank(s[0], s[1]),
    ("anchor", "clip"): lambda sc, s: anchor_clip(s[0], s[1]),
    ("nozzle", "grip"): lambda sc, s: nozzle_grip(s[0], s[1]),
    ("nozzle", "hose_stub"): lambda sc, s: nozzle_hose(s[0], s[1]),
    ("nozzle", "tube_body"): lambda sc, s: nozzle_tube(s[0], s[1]),
    ("brush", "grip"): lambda sc, s: brush_grip(s[0], s[1]),
    ("brush", "rod_body"): lambda sc, s: brush_rod(s[0], s[1]),
    ("brush", "bristles"): lambda sc, s: brush_bristles(s[0], s[1]),
    ("ballpeen", "handle"): lambda sc, s: ballpeen_handle(s[0], s[1]),
    ("ballpeen", "head"): lambda sc, s: ballpeen_head(s[0], s[1]),
    ("CentrePunch", "barrel"): lambda sc, s: punch_barrel(s[0], s[1]),
    ("CentrePunch", "cap"): lambda sc, s: punch_cap(s[0], s[1]),
    ("CentrePunch", "point_rod"): lambda sc, s: punch_point_round(s[0], s[1]),
    # clamps.StrapClamp: the strap is drawn on its toe box, washer and nut on their own bodies; stud and heel block
    # are drawn as their colliders (a rod and a block)
    ("StrapClamp", "toe"): lambda sc, s: strap_clamp_strap(s),
    ("StrapClamp", "tail"): lambda sc, s: [],
    ("StrapClamp", "rail0"): lambda sc, s: [],
    ("StrapClamp", "rail1"): lambda sc, s: [],
    ("StrapClamp", "washer_pad"): lambda sc, s: strap_clamp_washer(s),
    ("StrapClamp", "nut_flange"): lambda sc, s: strap_clamp_nut(s[0], s[1]),
    ("StrapClamp", "nut_hex"): lambda sc, s: [],
    ("Vise", "bed0"): lambda sc, s: vise_bed(s, -1.),
    ("Vise", "bed1"): lambda sc, s: vise_bed(s, 1.),
    ("Vise", "fixed_jaw"): lambda sc, s: vise_jaw(s, False),
    ("Vise", "jaw_block"): lambda sc, s: vise_jaw(s, True),
    ("Vise", "fixed_plate"): lambda sc, s: jaw_plate(s),
    ("Vise", "jaw_plate"): lambda sc, s: jaw_plate(s),
    ("Vise", "screw_hub"): lambda sc, s: vise_screw(s[0], s[1]),
    ("Vise", "crank_arm"): lambda sc, s: crank_arm(s[0], s[1]),
    ("Vise", "crank_knob"): lambda sc, s: crank_knob(s[0], s[1]),
    ("DrillPress", "chuck_body"): lambda sc, s: jacobs_chuck(s[0], s[1]),
    ("DrillPress", "depth_nut_body"): lambda sc, s: knurled_nut(s[0], s[1]),
}


def _owner(sc, label):
    """(kind, suffix) of the tool or part a shape label belongs to (longest name prefix wins)."""
    best = None
    for name, t in getattr(sc, "tools", {}).items():
        if label.startswith(name + "_") and (best is None or len(name) > len(best[0])):
            best = (name, type(t).__name__)
    for name, p in getattr(sc, "parts", {}).items():
        if label.startswith(name + "_") and (best is None or len(name) > len(best[0])):
            best = (name, p.get("kind"))
    if best is None:
        return None, None, label
    return best[0], best[1], label[len(best[0]) + 1:]


def shape_visual(sc, i, label, material, scale, tf):
    """Replacement appearance of collision shape i (render side): None draws the primitive as before, [] hides it,
    otherwise a list of parts in the shape's own frame (the frame its transform `tf` places)."""
    try:
        L = sc.layout
        kind = STATIC.get(label)
        if kind == "hide":
            return []
        if kind == "cart":
            mounts = [v["xyz"][:2] for v in L["mounts"].values()]
            yaw = float(next(iter(L["mounts"].values()))["yaw"])
            return robot_cart(L["cart"]["half"], L["cart"]["center"], L["cart"]["center"][2] + L["cart"]["half"][2],
                              mounts, L["cart"]["tray"], yaw)
        if kind == "trolley":
            return kit_trolley(tuple(float(x) for x in scale))
        if kind == "bench":
            c = np.asarray(tf[:3], float)
            hb = np.asarray(scale, float)
            from . import params as P
            parts = workbench((c[0] - hb[0], c[0] + hb[0]), (c[1] - hb[1], c[1] + hb[1]), P.BENCH_TOP_Z)
            return transform_parts(parts, None, -c)
        if material == "stand_grey" and label.endswith("_stand"):
            return press_cabinet(tuple(float(x) for x in scale))
        name, owner, suffix = _owner(sc, label)
        if suffix == "bit_body" or label.endswith("_bit_body"):
            it = next((x for x in getattr(sc, "items", {}).values() if label == x.name + "_body"), None)
            bt = getattr(it, "bit_type", "hss") if it is not None else ("sds" if material == "sds_bit" else "hss")
            return fluted_bit(float(scale[0]), float(scale[1]), "sds" if bt == "sds" else "hss")
        if owner == "ImpactWrench":
            if suffix == "head":                          # the whole housing, drawn in the head collider's frame
                from scipy.spatial.transform import Rotation
                Rt = Rotation.from_quat(np.asarray(tf[3:], float)).as_matrix()
                return transform_parts(impact_wrench(), Rt.T, -Rt.T @ np.asarray(tf[:3], float))
            if suffix in ("grip", "battery") or suffix.startswith(("socket_wall", "socket_seat", "socket_lead", "socket_rim")):
                return []
            if suffix == "socket_barrel":
                return impact_socket(float(scale[0]), float(scale[1]))
            if suffix == "trigger_blade":
                return impact_trigger(tuple(float(x) for x in scale))
        if owner == "anchor" and suffix == "shank_body":
            from . import params as P
            spec = sc.parts.get(name, {})
            L_, seat = spec.get("length"), spec.get("seat")
            if L_ is not None and seat is not None:
                letter = "L" if L_ > 0.15 else "G"
                return anchor_shank(float(scale[0]), float(scale[1]), P.KBTZ2_THREAD_LEN_M[letter] - (L_ - seat))
        fn = BY_SUFFIX.get((owner, suffix))
        if fn is not None:
            return fn(sc, tuple(float(x) for x in scale))
    except Exception as exc:                       # a failed model never breaks a render: draw the primitive
        import sys
        print(f"visual_models: {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return None


# ====================================================================================== room furniture and services
def pegboard(y0, y1, z0, z1, x_wall, pitch=0.0254, hole_r=0.0032):
    """Tempered-hardboard pegboard (1/4 in holes on a 1 in grid) on 19 mm furring strips against a wall whose face
    is x = x_wall (the board faces -x), framed with wood trim. Returns (parts, board front x)."""
    M = Model()
    xb = x_wall - 0.019
    xf = xb - 0.0064
    M.add(box2((xf, y0, z0), (xb, y1, z1)), "pegboard", "board")
    for z in (z0 + 0.02, 0.5*(z0 + z1), z1 - 0.02):
        M.add(box2((xb, y0 + 0.01, z - 0.019), (x_wall, y1 - 0.01, z + 0.019)), "wood_frame", "furring")
    trim = 0.032
    for (a, b) in (((xf - 0.012, y0 - 0.004, z0 - 0.004), (xf, y1 + 0.004, z0 + trim)),
                   ((xf - 0.012, y0 - 0.004, z1 - trim), (xf, y1 + 0.004, z1 + 0.004)),
                   ((xf - 0.012, y0 - 0.004, z0 + trim), (xf, y0 + trim, z1 - trim)),
                   ((xf - 0.012, y1 - trim, z0 + trim), (xf, y1 + 0.004, z1 - trim))):
        M.add(rbox2(a, b, 0.002, 6), "wood_frame", "trim")
    # holes: dark discs a hair in front of the board
    ys = np.arange(y0 + trim + 0.012, y1 - trim - 0.006, pitch)
    zs = np.arange(z0 + trim + 0.012, z1 - trim - 0.006, pitch)
    n = 8
    ang = 2*math.pi*np.arange(n)/n
    disc = np.stack([np.zeros(n), hole_r*np.cos(ang), hole_r*np.sin(ang)], -1)
    vs, fs = [], []
    k = 0
    for y in ys:
        for z in zs:
            c = np.array([xf - 0.0002, y, z])
            vs.append(np.vstack([c, c + disc]))
            fs.append(np.array([[k, k + 1 + j, k + 1 + (j + 1) % n] for j in range(n)]))
            k += n + 1
    M.add((np.concatenate(vs).astype(np.float32), np.concatenate(fs).astype(np.int32)), "peg_hole", "holes")
    return M.parts(), xf


def peg_hook(x_front, y, z, length=0.055, up=0.012, r=0.0022, double=False):
    """Steel pegboard hook from its hole at (x_front, y, z) out along -x with an upturned tip."""
    M = Model()
    pts = [(x_front + 0.005, y, z + 0.006), (x_front + 0.005, y, z), (x_front - length, y, z),
           (x_front - length - 0.004, y, z + up)]
    M.add(sweep(pts, r, bend=0.004, n=10), "steel_bright", "hook", smooth=True)
    if double:
        M.add(sweep([(x_front + 0.005, y, z + 0.0254), (x_front - 0.004, y, z + 0.0254), (x_front - 0.010, y, z + 0.002)],
                    r, bend=0.004, n=10), "steel_bright", "hook", smooth=True)
    return M.parts()


def strut(a, b, size=0.041, mat="emt_galv"):
    """41 mm strut channel (Unistrut) from a to b, open side facing its local +y (slot with lips)."""
    m3 = _m3()
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    s = size
    outline = np.array([[-s/2, -s/2], [s/2, -s/2], [s/2, s/2], [0.011, s/2], [0.011, s/2 - 0.007], [s/2 - 0.0025,
                        s/2 - 0.007], [s/2 - 0.0025, -s/2 + 0.0025], [-s/2 + 0.0025, -s/2 + 0.0025],
                        [-s/2 + 0.0025, s/2 - 0.007], [-0.011, s/2 - 0.007], [-0.011, s/2], [-s/2, s/2]])
    man = m3.Manifold.extrude(m3.CrossSection([outline]), L)
    M = Model()
    M.add(_tf(man, _rot_z_to(d/L), a), mat, "strut", crease=30)
    return M.parts()


def pipe_run(points, r, mat, bend=None, flanges=(), smooth=True):
    """A pipe along a polyline with elbows (bend radius 1.5 d), optional couplings at `flanges` (points)."""
    M = Model()
    M.add(sweep(points, r, bend=bend if bend is not None else 3*r, n=28), mat, "pipe", smooth=True)
    P = [np.asarray(p, float) for p in points]
    for p in flanges:
        p = np.asarray(p, float)
        # the run direction at p: the segment containing it
        d = None
        for k in range(len(P) - 1):
            u = P[k + 1] - P[k]
            t = float((p - P[k]) @ u/(u @ u))
            if -1e-6 <= t <= 1 + 1e-6 and np.linalg.norm(P[k] + t*u - p) < 1e-4:
                d = u/np.linalg.norm(u)
                break
        if d is None:
            continue
        M.add(cyl(p - d*0.025, p + d*0.025, r*1.18, r*1.18, 32), mat, "couplings", smooth=True)
    return M.parts()


def pipe_clamp(p, r, axis, wall_dir, mat="emt_galv"):
    """Strut pipe clamp around a pipe at p (axis along `axis`), its bolt ears towards `wall_dir`."""
    M = Model()
    p = np.asarray(p, float)
    a = np.asarray(axis, float)/np.linalg.norm(axis)
    M.add(ring(p - a*0.013, p + a*0.013, r + 0.0035, r + 0.0005, 32), mat, "clamp", smooth=True)
    w = np.asarray(wall_dir, float)/np.linalg.norm(wall_dir)
    side = np.cross(a, w)
    for sgn in (-1, 1):
        c = p + w*(r + 0.006) + side*sgn*(r + 0.004)
        M.add(cyl(c - w*0.006, c + w*0.006, 0.0045, 0.0045, 12), "zinc", "clamp_bolts", smooth=True)
    return M.parts()


def gate_valve(p, axis, r, stem_dir=(0., 0., -1.), mat="pipe_red"):
    """Flanged gate valve on a pipe of radius r at p, its bonnet and handwheel along `stem_dir`."""
    M = Model()
    p = np.asarray(p, float)
    a = np.asarray(axis, float)/np.linalg.norm(axis)
    s = np.asarray(stem_dir, float)/np.linalg.norm(stem_dir)
    for sgn in (-1, 1):
        M.add(cyl(p + a*sgn*0.07, p + a*sgn*0.09, r*1.9, r*1.9, 32), mat, "flanges", smooth=True)
        for k in range(6):
            ang = 2*math.pi*k/6
            u = np.cross(a, s)
            q = p + a*sgn*0.08 + (math.cos(ang)*s + math.sin(ang)*u)*r*1.6
            M.add(cyl(q - a*0.014, q + a*0.014, 0.006, 0.006, 6), "zinc", "flange_bolts")
    M.add(cyl(p - a*0.07, p + a*0.07, r*1.35, r*1.35, 32), mat, "body", smooth=True)
    M.add(cyl(p, p + s*r*3.2, r*0.9, r*0.7, 32), mat, "bonnet", smooth=True)
    M.add(cyl(p + s*r*3.2, p + s*r*4.6, 0.008, 0.008, 12), "steel_bright", "stem", smooth=True)
    hw = p + s*r*4.2
    M.add(ring(hw - s*0.006, hw + s*0.006, r*2.0, r*2.0 - 0.012, 40), "pipe_red", "handwheel", smooth=True)
    u = np.cross(a, s)
    for k in range(4):
        ang = math.pi*k/4
        d = math.cos(ang)*a + math.sin(ang)*u
        M.add(cyl(hw - d*(r*2.0 - 0.008), hw + d*(r*2.0 - 0.008), 0.004, 0.004, 10), "pipe_red", "spokes")
    return M.parts()


def duct(x0, x1, y, z_top, w=0.50, h=0.30, joints=1.5, registers=()):
    """Galvanised rectangular supply duct along x hung under the ceiling (top at z_top): TDC flanges at the joints,
    trapeze hangers on threaded rods, supply registers (grilles) on its sides at the given x positions."""
    M = Model()
    z0 = z_top - h
    M.add(box2((x0, y - w/2, z0), (x1, y + w/2, z_top)), "duct_galv", "duct", crease=30)
    xs = np.arange(x0 + joints, x1 - 0.1, joints)
    for x in xs:
        M.add(rbox2((x - 0.012, y - w/2 - 0.018, z0 - 0.018), (x + 0.012, y + w/2 + 0.018, z_top), 0.003, 6), "duct_galv",
              "flanges", crease=30)
        for sgn in (-1, 1):
            for zz in (z0 - 0.009, z_top - 0.03):
                M.add(cyl((x, y + sgn*(w/2 + 0.018), zz), (x, y + sgn*(w/2 + 0.004), zz), 0.006, 0.006, 8), "zinc",
                      "cleats")
    for x in np.arange(x0 + 0.5*joints, x1, joints):
        M.extend(strut((x, y - w/2 - 0.08, z0 - 0.0205), (x, y + w/2 + 0.08, z0 - 0.0205)))
        for sgn in (-1, 1):
            M.add(cyl((x, y + sgn*(w/2 + 0.05), z0 - 0.04), (x, y + sgn*(w/2 + 0.05), z_top + 0.02), 0.0048, 0.0048,
                      10), "zinc", "hanger_rods", smooth=True)
            M.add(hexagon((x, y + sgn*(w/2 + 0.05), z0 - 0.050), (x, y + sgn*(w/2 + 0.05), z0 - 0.041), 0.013), "zinc",
                  "hanger_nuts")
    for (x, side) in registers:
        yf = y + side*(w/2)
        M.add(rbox2((x - 0.17, yf - 0.002 if side < 0 else yf, z0 + 0.06), (x + 0.17, yf + 0.002 if side > 0 else yf,
                                                                             z_top - 0.06), 0.003, 6), "white_paint",
              "register_frame")
        for k in range(8):
            zz = z0 + 0.075 + k*0.019
            M.add(box2((x - 0.15, yf + side*0.0022 - 0.0008, zz), (x + 0.15, yf + side*0.0022 + 0.0008, zz + 0.012)),
                  "white_paint", "register_vanes")
        M.add(box2((x - 0.155, yf + side*0.0005 - 0.0005, z0 + 0.07), (x + 0.155, yf + side*0.0005 + 0.0005, z_top - 0.07)),
              "vent_dark", "register_back")
    return M.parts()


def junction_box(p, normal, up=(0., 0., 1.), size=0.102, depth=0.054, mat="box_grey"):
    """4 in square steel junction box on a wall (back at p, facing `normal`) with a flat cover and two screws."""
    M = Model()
    n = np.asarray(normal, float)/np.linalg.norm(normal)
    u = np.asarray(up, float)
    s = np.cross(u, n)
    R = np.column_stack([s, u, n])
    M.add(rbox((0., 0., depth/2), (size, size, depth), 0.004, 8), mat, "box", crease=35)
    M.add(rbox((0., 0., depth + 0.0015), (size + 0.006, size + 0.006, 0.003), 0.004, 8), mat, "cover", crease=35)
    for sy in (-1, 1):
        M.add(cyl((sy*0.032, sy*0.032, depth + 0.003), (sy*0.032, sy*0.032, depth + 0.006), 0.004, 0.004, 12), "zinc",
              "screws", smooth=True)
    return M.parts(R, p)


def disconnect_switch(p, normal, mat="box_grey"):
    """Safety disconnect switch (30 A, NEMA 1): a grey enclosure with an operating handle on its right side."""
    M = Model()
    n = np.asarray(normal, float)/np.linalg.norm(normal)
    u = np.array([0., 0., 1.])
    s = np.cross(u, n)
    R = np.column_stack([s, u, n])
    M.add(rbox((0., 0., 0.07), (0.24, 0.38, 0.14), 0.006, 8), mat, "enclosure", crease=35)
    M.add(rbox((0., -0.005, 0.141), (0.226, 0.355, 0.004), 0.004, 8), mat, "door", crease=35)
    M.add(rbox((0.128, 0.02, 0.09), (0.016, 0.05, 0.05), 0.004, 8), "black_oxide", "handle_base")
    M.add(rbox((0.150, 0.06, 0.09), (0.03, 0.12, 0.022), 0.006, 8), "estop_red", "handle")
    M.add(box((0., 0.10, 0.1435), (0.10, 0.05, 0.001)), "decal_white", "label")
    M.add(box((0., 0.10, 0.1440), (0.09, 0.012, 0.0006)), "estop_red", "label_band")
    return M.parts(R, p)


def outlet_strip(p0, p1, normal, n_out=4):
    """Surface-mounted aluminium multi-outlet strip from p0 to p1 on a wall facing `normal`."""
    M = Model()
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    d /= L
    n = np.asarray(normal, float)/np.linalg.norm(normal)
    u = np.cross(n, d)
    R = np.column_stack([d, u, n])
    M.add(rbox((L/2, 0., 0.02), (L, 0.07, 0.04), 0.004, 8), "alu_machined", "strip", crease=35)
    for k in range(n_out):
        x = L*(k + 0.5)/n_out
        M.add(rbox((x, 0., 0.0405), (0.045, 0.06, 0.002), 0.003, 6), "decal_white", "outlets")
        for sy in (-1, 1):
            M.add(box((x - 0.009, sy*0.012, 0.0418), (0.0018, 0.008, 0.001)), "vent_dark", "outlet_slots")
            M.add(box((x + 0.009, sy*0.012, 0.0418), (0.0018, 0.008, 0.001)), "vent_dark", "outlet_slots")
            M.add(cyl((x, sy*0.012 - sy*0.010, 0.0412), (x, sy*0.012 - sy*0.010, 0.0420), 0.0022, 0.0022, 10),
                  "vent_dark", "outlet_slots")
    return M.parts(R, p0)


def steel_door(x0, x1, y_wall, h=2.10, normal=1.):
    """Hollow-metal door and frame in a wall at y = y_wall (the room is on the `normal` side): 50 mm frame faces,
    a 44 mm leaf with a lever handle, an overhead closer, an aluminium kick plate and a sign plate."""
    M = Model()
    s = float(normal)
    f = 0.05
    out = y_wall + s*0.012
    for (a, b) in (((x0 - f, y_wall - 0.01, 0.), (x0, out, h + f)), ((x1, y_wall - 0.01, 0.), (x1 + f, out, h + f)),
                   ((x0, y_wall - 0.01, h), (x1, out, h + f))):
        M.add(rbox2(np.minimum(a, b), np.maximum(a, b), 0.003, 6), "door_grey", "frame", crease=35)
    leaf0, leaf1 = y_wall - 0.002, y_wall + s*0.004
    M.add(rbox2((x0 + 0.003, min(leaf0, leaf1), 0.010), (x1 - 0.003, max(leaf0, leaf1), h - 0.003), 0.003, 6),
          "door_grey", "leaf", crease=35)
    yk = y_wall + s*0.0055
    M.add(box2((x0 + 0.02, min(yk, y_wall + s*0.004), 0.015), (x1 - 0.02, max(yk, y_wall + s*0.004), 0.25)),
          "alu_machined", "kick_plate")
    xh = x1 - 0.075
    yh = y_wall + s*0.004
    M.add(rbox2((xh - 0.025, min(yh, yh + s*0.012), 0.98), (xh + 0.025, max(yh, yh + s*0.012), 1.10), 0.006, 8),
          "steel_bright", "lever_rose")
    M.add(sweep([(xh, yh + s*0.012, 1.04), (xh, yh + s*0.060, 1.04), (xh - 0.13, yh + s*0.066, 1.04)], 0.0095,
                bend=0.02, n=18), "steel_bright", "lever", smooth=True)
    M.add(rbox2((x0 + 0.10, min(yh, yh + s*0.06), h - 0.12), (x0 + 0.38, max(yh, yh + s*0.06), h - 0.05), 0.008),
          "white_paint", "closer")
    M.add(sweep([(x0 + 0.36, yh + s*0.05, h - 0.06), (x0 + 0.52, yh + s*0.14, h + 0.02),
                 (x0 + 0.30, yh + s*0.02, h + 0.035)], 0.007, bend=0.01, n=10), "black_oxide", "closer_arm", smooth=True)
    M.add(box2((0.5*(x0 + x1) - 0.14, min(yh, yh + s*0.002), 1.45), (0.5*(x0 + x1) + 0.14, max(yh, yh + s*0.002), 1.65)),
          "safety_yellow", "sign")
    M.add(box2((0.5*(x0 + x1) - 0.12, min(yh, yh + s*0.0025), 1.47), (0.5*(x0 + x1) + 0.12, max(yh, yh + s*0.0025), 1.51)),
          "vent_dark", "sign_text")
    M.add(box2((0.5*(x0 + x1) - 0.12, min(yh, yh + s*0.0025), 1.53), (0.5*(x0 + x1) + 0.12, max(yh, yh + s*0.0025), 1.63)),
          "vent_dark", "sign_text")
    return M.parts()


def roller_cabinet(c, yaw, w=0.70, d=0.46, h=0.84, mat="chest_red"):
    """Rolling tool cabinet (drawers on the local -y face, which faces `yaw` rotated), top at z = h."""
    M = Model()
    zc = 0.11
    M.add(rbox2((-w/2, -d/2, zc), (w/2, d/2, h - 0.012), 0.004, 8), mat, "body", crease=35)
    M.add(rbox2((-w/2 - 0.005, -d/2 - 0.005, h - 0.012), (w/2 + 0.005, d/2 + 0.005, h), 0.003, 6), "rubber_grey",
          "top_mat")
    hs = [0.08, 0.08, 0.10, 0.12, 0.18, 0.11]
    z = h - 0.03
    for k, dh in enumerate(hs):
        z0 = z - dh
        M.add(rbox2((-w/2 + 0.025, -d/2 - 0.012, z0 + 0.004), (w/2 - 0.025, -d/2, z - 0.004), 0.003, 6), mat, "drawers",
              crease=35)
        M.add(rbox2((-w/2 + 0.03, -d/2 - 0.026, z - 0.022), (w/2 - 0.03, -d/2 - 0.012, z - 0.012), 0.003, 6),
              "alu_machined", "pulls", crease=35)
        z = z0
    M.add(rbox2((w/2, -0.10, 0.62), (w/2 + 0.035, 0.10, 0.66), 0.006, 8), "alu_machined", "side_handle")
    for sx in (-1, 1):
        for sy in (-1, 1):
            M.extend(transform_parts(caster(0.05, zc, braked=sy < 0), _rz(math.pi/2), (sx*(w/2 - 0.06), sy*(d/2 - 0.06),
                                                                                    0.)))
    return M.parts(_rz(yaw), c)


def wall_base(segments, h=0.10, t=0.003, mat="rubber_grey"):
    """Rubber cove base along walls: segments [(p0, p1, inward normal)] at floor level."""
    M = Model()
    for p0, p1, n in segments:
        p0, p1, n = np.asarray(p0, float), np.asarray(p1, float), np.asarray(n, float)
        lo = np.minimum(p0, p1 + n*t)
        hi = np.maximum(p0, p1 + n*t)
        hi[2] = h
        M.add(box2(lo, hi), mat, "cove_base")
    return M.parts()


def anchor_boxes(c, nx, ny, size=(0.095, 0.065, 0.075), yaw=0., colors=("anchor_box_red", "cardboard")):
    """A packed layer of small product boxes (anchor and fastener boxes) at c (bottom centre)."""
    M = Model()
    k = 0
    for i in range(nx):
        for j in range(ny):
            x = (i - (nx - 1)/2)*(size[0] + 0.004)
            y = (j - (ny - 1)/2)*(size[1] + 0.004)
            mat = colors[(i + 2*j) % len(colors)]
            M.add(rbox((x, y, size[2]/2), size, 0.002, 6), mat, "boxes", crease=35)
            M.add(box((x, y - size[1]/2 - 0.0003, size[2]*0.55), (size[0]*0.7, 0.0008, size[2]*0.35)), "decal_white",
                  "box_labels")
            k += 1
    return M.parts(_rz(yaw), c)
