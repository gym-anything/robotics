"""Mode tables of every vibrating part, built from published measurements and cached in percussion/data/.

Bars (Rucz et al. 2021): 1-D Timoshenko vertical and horizontal bending and St Venant torsion of the
undercut rosewood bar; the undercut (flat half-length s, centre thickness h_min, 62.5 mm grindstone arcs) is solved per
bar so the model's f1 and f2 equal Rucz's measured partials, with Rucz's per-bar E scale; the measured f3 replaces the
model's (shape from the model). Modes to 20 kHz.

Drumhead (Remo Ambassador Coated 14", Wagner 2006): circular-membrane modes (m, n) in both orientations
with the film's bending stiffness, air loading from the King (Hankel-transform) integral of a baffled face (outer face
baffled; inside: the open 5-in shell as an air plug for m = 0 modes, a second baffled face for m >= 1), solved at the
loaded frequency; the radiation resistance gives each mode's radiation damping.

Drumstick (Vic Firth 5A): tapered hickory Timoshenko beam; rigid mass properties and the first bending
modes of the free-free stick (a loose grip; Wagner's in-hand measurement 400/1000/1730 Hz matches free-free -
a pin at the fulcrum would give 300 Hz).

`python -m percussion.modes --build` rebuilds every table (minutes, BLAS pinned to two threads); the environment only
reads the cached .npz files, whose SHA-256 is recorded in data/MANIFEST.json.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")              # shared 16-CPU sandbox: never let BLAS take every core
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

from . import params as P

DATA = Path(__file__).resolve().parent/"data"
NEL_BAR = 160


# ====================================================================================================== bars
def bar_profile(L, s, hmin, n=NEL_BAR):
    """Node x (0..L) and element thickness h of an undercut bar: flat of half-length s at h_min, stone arcs, ends H0."""
    H0, R = P.BAR_THICK_M, P.BAR_STONE_R_M
    x = np.linspace(0, L, n + 1)
    xm = 0.5*(x[1:] + x[:-1])
    u = np.abs(xm - L/2) - s
    d = H0 - hmin
    umax = np.sqrt(R**2 - (R - d)**2)
    h = np.where(u <= 0, hmin, np.where(u >= umax, H0, hmin + R - np.sqrt(np.maximum(R**2 - u**2, 0))))
    return x, np.minimum(h, H0)


def _bending(x, h, esc, horizontal=False, n_modes=14):
    from .fem import timoshenko_modes
    b = P.BAR_WIDTH_M
    A = b*h
    I = (h*b**3/12) if horizontal else (b*h**3/12)
    G = P.BAR_G_LT if horizontal else P.BAR_G_RL
    return timoshenko_modes(x, esc*P.BAR_E_L*I, P.BAR_SHEAR_K*esc*G*A, P.BAR_RHO*A, P.BAR_RHO*I, n_modes=n_modes)


def _torsion(x, h, esc, n_modes=6):
    from .fem import torsion_modes, rect_torsion_beta
    b = P.BAR_WIDTH_M
    J = np.array([rect_torsion_beta(b, hh)*max(b, hh)*min(b, hh)**3 for hh in h])
    Ip = b*h*(b**2 + h**2)/12
    return torsion_modes(x, esc*P.BAR_G_LT*J, P.BAR_RHO*Ip, n_modes=n_modes)


def build_bar(note):
    from scipy.optimize import least_squares
    L = P.bar_length(note)
    esc = P.BAR_E_SCALE[note]
    m1, m2, m3 = P.BAR_PARTIALS[note]
    H0, R = P.BAR_THICK_M, P.BAR_STONE_R_M

    def resid(p):
        x, h = bar_profile(L, *p)
        f, _, _ = _bending(x, h, esc, n_modes=3)
        return [f[0]/m1 - 1, f[1]/m2 - 1]

    best = None
    smax = max(0.0, L/2 - np.sqrt(R**2 - (R - (H0 - 0.004))**2) - 0.002)
    for s0 in (0.0, 0.03):
        for h0 in (0.006, 0.009):
            r = least_squares(resid, [min(s0, smax), h0], bounds=([0.0, 0.003], [smax + 1e-9, 0.0155]),
                              xtol=1e-10, ftol=1e-12)
            if best is None or r.cost < best.cost:
                best = r
    s, hmin = best.x
    x, h = bar_profile(L, s, hmin)
    fv, Wv, _ = _bending(x, h, esc)
    fh, Wh, _ = _bending(x, h, esc, horizontal=True, n_modes=8)
    ft, Wt = _torsion(x, h, esc)
    fv = fv.copy()
    fv[:3] = (m1, m2, m3)                       # measured partials (shape from the model)
    keepv, keeph, keept = fv < P.BAR_MODE_FMAX_HZ, fh < P.BAR_MODE_FMAX_HZ, ft < P.BAR_MODE_FMAX_HZ
    # rigid mass properties in the bar frame: origin at mid-length, mid-width, mid full thickness; top face z = +H0/2
    xm = 0.5*(x[1:] + x[:-1]) - L/2
    dx = np.diff(x)
    b = P.BAR_WIDTH_M
    zc = H0/2 - h/2
    dm = P.BAR_RHO*b*h*dx
    mass = float(dm.sum())
    zcom = float((dm*zc).sum()/mass)
    Ixx = float((dm*((b**2 + h**2)/12 + (zc - zcom)**2)).sum())
    Iyy = float((dm*(h**2/12 + xm**2 + (zc - zcom)**2)).sum() + 0.)
    Izz = float((dm*(b**2/12 + xm**2)).sum())
    sgn = np.sign(Wv[:, 0])
    nodes = [float(x[i] - Wv[i, 0]*(x[i + 1] - x[i])/(Wv[i + 1, 0] - Wv[i, 0])) - L/2
             for i in np.where(sgn[:-1]*sgn[1:] < 0)[0]]
    return dict(note=note, length=L, flat_half=float(s), h_min=float(hmin), e_scale=esc, fit_residual=best.fun.tolist(),
                x=x - L/2, h=h, f_vert=fv[keepv], W_vert=Wv[:, keepv], f_horiz=fh[keeph], W_horiz=Wh[:, keeph],
                f_tors=ft[keept], W_tors=Wt[:, keept], f_model_vert=_bending(x, h, esc, n_modes=4)[0],
                mass=mass, com=np.array([0., 0., zcom]), inertia=np.array([Ixx, Iyy, Izz]), nodes=np.array(nodes))


# ====================================================================================================== membrane
def _phit(m, alpha, kappa, A):
    """Hankel transform of J_m(alpha r) over [0, A] (J_m(alpha A) = 0)."""
    from scipy.special import jv, jvp
    kappa = np.asarray(kappa, float)
    num = A*alpha*jv(m, kappa*A)*jvp(m, alpha*A)
    den = kappa**2 - alpha**2
    near = np.abs(den) < 1e-7*alpha**2
    return np.where(near, 0.5*A**2*jvp(m, alpha*A)**2, num/np.where(near, 1., den))


def _baffled_Z(m, alpha, omega, A):
    """(added mass kg, radiation resistance kg/s) of mode (m, alpha) per unit modal velocity, one baffled face (King)."""
    k = omega/P.C_AIR
    cm = 2*np.pi if m == 0 else np.pi
    th = np.linspace(0, np.pi/2, 1201)[1:-1]
    kr = k*np.sin(th)
    R = cm*P.RHO_AIR*omega*np.trapezoid(_phit(m, alpha, kr, A)**2*k*np.sin(th), th)
    umax = np.arccosh(max(1.0001, 400/(k*A)))
    u = np.linspace(0, umax, 8001)[1:]
    ka = k*np.cosh(u)
    madd = cm*P.RHO_AIR*np.trapezoid(_phit(m, alpha, ka, A)**2*ka, u)
    return madd, R


def build_membrane(tension_name):
    from scipy.special import jv, jn_zeros
    T = P.DRUM_TENSIONS[tension_name]
    A, sigma = P.DRUM_RADIUS_M, P.DRUM_SIGMA
    D = P.DRUM_E_FILM*P.DRUM_FILM_M**3/(12*(1 - P.DRUM_NU_FILM**2))
    rows = []
    for m in range(0, 80):
        zs = jn_zeros(m, 60)
        if np.sqrt(T/sigma)*zs[0]/A/(2*np.pi) > P.DRUM_MODE_FMAX_HZ:
            break
        for n, z in enumerate(zs, start=1):
            alpha = z/A
            f_vac = np.sqrt((T*alpha**2 + D*alpha**4)/sigma)/(2*np.pi)
            if f_vac > P.DRUM_MODE_FMAX_HZ:
                break
            cm = 2*np.pi if m == 0 else np.pi
            M0 = sigma*cm*0.5*A**2*jv(m + 1, alpha*A)**2
            Kmod = (T*alpha**2 + D*alpha**4)*M0/sigma
            w = np.sqrt(Kmod/M0)
            for _ in range(30):
                ma, Ra = _baffled_Z(m, alpha, w, A)
                if m == 0:
                    vol = 2*np.pi*A*jv(1, alpha*A)/alpha
                    plug = P.RHO_AIR*(P.DRUM_SHELL_DEPTH_M + 0.61*A)*vol**2/(np.pi*A**2)
                    madd, R = ma + plug, Ra
                else:
                    madd, R = 2*ma, 2*Ra
                wn = np.sqrt(Kmod/(M0 + madd))
                if abs(wn - w) < 1e-7*w:
                    w = wn
                    break
                w = wn
            zeta_rad = R/(2*w*(M0 + madd))
            for orient in ((0,) if m == 0 else (0, 1)):
                rows.append((m, n, orient, alpha, w/(2*np.pi), f_vac, M0, madd, zeta_rad))
    a = np.array(rows)
    return dict(tension=T, sigma=sigma, radius=A, m=a[:, 0].astype(int), n=a[:, 1].astype(int),
                orient=a[:, 2].astype(int), alpha=a[:, 3], f=a[:, 4], f_vacuo=a[:, 5], M0=a[:, 6], madd=a[:, 7],
                zeta_rad=a[:, 8])


# ====================================================================================================== drumstick
def stick_geometry(n=200):
    s = P.STICK
    x = np.linspace(0, s["L"], n + 1)
    xm = 0.5*(x[1:] + x[:-1])
    xt0 = s["L"] - s["taper"]
    xn = s["L"] - s["tip"]
    d = np.where(xm <= xt0, s["D"], s["D"] + (s["dn"] - s["D"])*(xm - xt0)/(xn - xt0))
    u = np.clip((xm - xn)/s["tip"], 0, 1)
    ell = s["dt"]*np.sqrt(np.clip(1 - ((u - 0.45)/0.55)**2, 0.0625, 1))
    tipd = np.where(u < 0.45, np.maximum(ell, s["dn"]), ell)
    d = np.where(xm >= xn, tipd, d)
    return x, xm, d


def build_stick():
    from scipy.optimize import brentq
    from .fem import timoshenko_modes
    s = P.STICK
    x, xm, d = stick_geometry()
    A = np.pi*d**2/4
    I = np.pi*d**4/64
    le = np.diff(x)

    def mass(rho):
        return rho*np.sum(A*le)
    rho = brentq(lambda r: mass(r) - s["mass"], 300, 1500)
    m = mass(rho)
    xc = float(np.sum(rho*A*le*xm)/m)
    I_t = float(np.sum(rho*A*le*(xm - xc)**2) + np.sum(rho*I*le))      # transverse, about the COM
    I_a = float(np.sum(rho*A*le*d**2/8))                              # axial
    xf = s["L"]*P.STICK_FULCRUM_FRAC
    E = P.STICK_E
    # free-free elastic modes (orthogonal to the rigid body the world simulates): the 5B model gives 410/978/1706 Hz
    # against Wagner's in-hand measurement 400/1000/1730 Hz; a pin at the fulcrum would give 300 Hz
    f, W, _ = timoshenko_modes(x, E*I, 0.9*P.STICK_G_RATIO*E*A, rho*A, rho*I, n_modes=P.STICK_MODES, fmin_hz=60.0)
    f, W = f[:P.STICK_MODES], W[:, :P.STICK_MODES]
    tip_x = s["L"] - s["tip"]*0.5
    return dict(rho=rho, mass=m, com_x=xc, I_transverse=I_t, I_axial=I_a, fulcrum_x=xf, x=x, d=np.r_[d, d[-1]],
                f=f, W=W, tip_x=tip_x, tip_r=s["dt"]/2)


# ====================================================================================================== cache
def _key(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _inputs(kind, name=None):
    if kind == "bar":
        return dict(kind=kind, note=name, rho=P.BAR_RHO, E=P.BAR_E_L, GRL=P.BAR_G_RL, GLT=P.BAR_G_LT, b=P.BAR_WIDTH_M,
                    h=P.BAR_THICK_M, R=P.BAR_STONE_R_M, L=P.bar_length(name), esc=P.BAR_E_SCALE[name],
                    f=P.BAR_PARTIALS[name], fmax=P.BAR_MODE_FMAX_HZ, nel=NEL_BAR, v=3)
    if kind == "membrane":
        return dict(kind=kind, T=P.DRUM_TENSIONS[name], sigma=P.DRUM_SIGMA, a=P.DRUM_RADIUS_M, E=P.DRUM_E_FILM,
                    nu=P.DRUM_NU_FILM, h=P.DRUM_FILM_M, depth=P.DRUM_SHELL_DEPTH_M, fmax=P.DRUM_MODE_FMAX_HZ, v=2)
    return dict(kind=kind, stick=P.STICK, E=P.STICK_E, G=P.STICK_G_RATIO, frac=P.STICK_FULCRUM_FRAC, n=P.STICK_MODES, v=3)


def _path(kind, name=None):
    return DATA/(f"{kind}_{name}.npz" if name else f"{kind}.npz")


def _load(kind, name, builder, rebuild=False):
    path = _path(kind, name)
    key = _key(_inputs(kind, name))
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=False)
        if str(z["_key"]) == key:
            return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files if k != "_key"}
        raise RuntimeError(f"{path.name} was built from other parameters (key {z['_key']} != {key}); "
                           "run python -m percussion.modes --build")
    if not rebuild:
        raise FileNotFoundError(f"{path} missing; run python -m percussion.modes --build")
    d = builder(name) if name else builder()
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(path, _key=key, **{k: np.asarray(v) for k, v in d.items()})
    return _load(kind, name, builder)


@lru_cache(maxsize=None)
def bar(note):
    return _load("bar", note, build_bar)


@lru_cache(maxsize=None)
def membrane(tension_name):
    return _load("membrane", tension_name, build_membrane)


@lru_cache(maxsize=None)
def stick():
    return _load("stick", None, build_stick)


def build_all():
    notes = sorted(set(P.XYLO_NOTES) | set(P.RESONATOR_BAR_NOTES) | {"F#4", "Bb4", "F#5"})
    out = {}
    for n in notes:
        d = _load("bar", n, build_bar, rebuild=True)
        out[f"bar_{n}"] = dict(L=d["length"], s=d["flat_half"], hmin=d["h_min"], mass=d["mass"],
                               f_vert=np.round(d["f_vert"][:5], 1).tolist(), f_tors=np.round(d["f_tors"][:2], 1).tolist())
        print(n, out[f"bar_{n}"], flush=True)
    d = _load("stick", None, build_stick, rebuild=True)
    out["stick"] = dict(mass=d["mass"], com_x=d["com_x"], f=np.round(d["f"], 1).tolist())
    print("stick", out["stick"], flush=True)
    for t in P.DRUM_TENSIONS:
        d = _load("membrane", t, build_membrane, rebuild=True)
        i = {(int(m), int(n)): k for k, (m, n, o) in enumerate(zip(d["m"], d["n"], d["orient"])) if o == 0}
        out[f"membrane_{t}"] = dict(modes=int(len(d["f"])), f01=d["f"][i[(0, 1)]], f11=d["f"][i[(1, 1)]],
                                    f21=d["f"][i[(2, 1)]], zeta01=d["zeta_rad"][i[(0, 1)]])
        print(t, out[f"membrane_{t}"], flush=True)
    man = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob("*.npz"))}
    (DATA/"MANIFEST.json").write_text(json.dumps(dict(files=man, summary=out), indent=1, default=float))
    return out


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        build_all()
