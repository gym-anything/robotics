"""Audio-rate modal structures: exact complex-mode propagation of every vibrating part.

A structure is a linear system  q'' + C q' + K q = Phi^T f  in unit-mass modal coordinates q (n DOFs: bar bending and
torsion modes, membrane modes, Helmholtz air plugs). In state space x = [q, q'], x' = A x + B f with
A = [[0, I], [-K, -C]]; its complex modes (eigenvalues lam with Im > 0, right eigenvectors Phi_+, rows Psi_+ of the
inverse) give  x = 2 Re(Phi_+ z)  and  z' = lam z + Psi_+ B f.  With the force held constant over one audio sample
(zero-order hold) the propagation is exact:  z <- E z + G (Psi_+ B f),  E = exp(lam dt),  G = (E - 1)/lam.

Uncoupled structures (the drumhead, a bar without resonator, the drumstick) use the closed form per DOF
(lam = -zeta w + i w_d, Phi_q = 1, Psi_+ B = 1/(2 i w_d)); coupled ones (the xylophone: 13 bars and the six box
resonators that couple to their bending modes) use a numerical eigendecomposition once at construction.
A contact at a point is a shape vector b (n,) over the DOFs: the surface displacement along the contact normal is
u = b.q = 2 Re(c . z) with c = b Phi_q, and a normal force F on the structure enters as the modal force -F b
(w = Psi_+ B b). The analytic velocity of DOF i is 2 (Phi_q lam z)_i, from which the acoustics radiate.
"""
import numpy as np

from . import params as P


class Structure:
    def __init__(self, name, omega, zeta, *, K_extra=None, dof_labels=None, dt=1.0/P.AUDIO_FS):
        """omega, zeta: (n,) natural frequencies (rad/s) and damping ratios of the unit-mass DOFs; K_extra: (n, n)
        symmetric stiffness coupling added to diag(omega^2) (None: uncoupled, closed form)."""
        self.name = name
        self.omega = np.asarray(omega, float)
        self.zeta = np.asarray(zeta, float)
        self.n = len(self.omega)
        self.dt = dt
        self.labels = list(dof_labels or [])
        self.coupled = K_extra is not None
        if not self.coupled:
            wd = self.omega*np.sqrt(1 - self.zeta**2)
            self.lam = -self.zeta*self.omega + 1j*wd
            self.psi = 1.0/(2j*wd)                              # Psi_+ B per DOF
            self.K = self.omega**2
        else:
            K = np.diag(self.omega**2) + np.asarray(K_extra, float)
            # damping stays modal (diag 2 zeta w) on the uncoupled DOFs, as measured/estimated per mode
            C = np.diag(2*self.zeta*self.omega)
            n = self.n
            A = np.block([[np.zeros((n, n)), np.eye(n)], [-K, -C]])
            lam, V = np.linalg.eig(A)
            sel = np.flatnonzero(lam.imag > 0)
            if len(sel) != n:
                raise ValueError(f"{name}: {2*n - 2*len(sel)} overdamped modes; a coupled structure must be underdamped")
            Vinv = np.linalg.inv(V)
            order = sel[np.argsort(lam[sel].imag)]
            self.lam = lam[order]
            self.Phi_q = V[:n, order]                          # (n, nc)
            self.Psi = Vinv[order, :]                          # (nc, 2n)
            self.Psi_v = self.Psi[:, n:]                       # (nc, n)
            self.K = K
        self.nc = len(self.lam)
        self.E = np.exp(self.lam*dt)
        self.G = (self.E - 1)/self.lam
        self.Eh = np.exp(self.lam*dt/2)
        self.Gh = (self.Eh - 1)/self.lam
        k = np.arange(1, P.SAMPLES_PER_TICK + 1)
        self.Epow = np.exp(np.outer(self.lam, k*dt))           # (nc, 48): free propagation over one world tick
        self.z = np.zeros(self.nc, complex)
        self.radiators = None                                  # acoustics.Radiation (set by the acoustics)
        self.silent_tol = 1e-16                                # J: below this the structure is skipped

    # ------------------------------------------------------------------ contact maps
    def out_row(self, b):
        """c with u = 2 Re(c.z) the displacement along the contact normal for shape vector b."""
        b = np.asarray(b, float)
        return b.astype(complex) if not self.coupled else b @ self.Phi_q

    def in_col(self, b):
        """w with z += G (w (-F)) for a normal force F pushing the surface away from the striker."""
        b = np.asarray(b, float)
        return self.psi*b if not self.coupled else self.Psi_v @ b

    def compliance(self, c, w, half=True):
        """Displacement response 2 Re(sum c G w) to a unit (-1 sign included by the caller) force held one (half)
        sample: the structure's share of the contact's instantaneous compliance."""
        return float(2*np.real(np.sum(c*(self.Gh if half else self.G)*w)))

    def displacement(self, c, half=False):
        return float(2*np.real(np.dot(c, (self.Eh*self.z) if half else self.z)))

    # ------------------------------------------------------------------ propagation
    def free_block(self):
        """Advance one world tick without contact forces; returns (active mode indices, Z (n_active, 48) of their states
        at each sample's end). Modes whose velocity amplitude has decayed 140 dB below the strongest one (or below
        1e-13 m/s) are set to zero first: inaudible (the microphone's self-noise is ~100 dB below the levels that
        matter) and it keeps a ringing 810-mode drumhead from costing its full width for seconds."""
        a = np.abs(self.z)*np.abs(self.lam)
        amax = float(a.max()) if len(a) else 0.0
        dead = a <= max(1e-7*amax, 1e-13)
        if dead.any():
            self.z[dead] = 0
        act = np.flatnonzero(~dead)
        Z = self.Epow[act]*self.z[act, None]
        self.z[act] = Z[:, -1]
        return act, Z

    def step(self, wF=None):
        """One audio sample with modal input wF (= sum over contacts of w * (-F))."""
        self.z = self.E*self.z if wF is None else self.E*self.z + self.G*wF
        return self.z

    def step_radiating(self, wF=None):
        """step(), returning the part of the state that radiates: z minus every mode's quasi-static response to the force
        held over the sample, z_p = -wF/lam (the fixed point of z <- E z + G wF). z_p carries the static deflection
        (displacement K^-1 b F, velocity exactly 0), so the far field of a resting or pressing load is zero, while the
        per-mode transfer H(omega_j) - valid for the oscillating part only - applied to z_p's analytic velocity -2 wF
        gives a constant pressure (+0.058 Pa under a mallet lying on the head, up to 12 Pa under a pressed hand). Free ringing (wF = 0) is unchanged; during a stroke only the force-following part is removed."""
        z = self.step(wF)
        return z if wF is None else z + wF/self.lam

    # ------------------------------------------------------------------ physics
    def physical(self, z=None):
        z = self.z if z is None else z
        if not self.coupled:
            return 2*np.real(z), 2*np.real(self.lam*z)
        return 2*np.real(self.Phi_q @ z), 2*np.real(self.Phi_q @ (self.lam*z))

    def energy(self, z=None):
        q, v = self.physical(z)
        pot = 0.5*np.sum(self.K*q**2) if not self.coupled else 0.5*q @ self.K @ q
        return float(0.5*np.sum(v**2) + pot)

    def dof_energy(self, z=None):
        """Per-DOF energy (diagonal stiffness part): for mode-share measures."""
        q, v = self.physical(z)
        return 0.5*(v**2 + self.omega**2*q**2)

    def silent(self):
        """No mode left: free_block zeroes modes as they decay below its audibility floor, so a struck structure
        becomes silent by itself (the energy form cost ~1 ms per tick on the coupled xylophone)."""
        return not np.any(self.z)

    def from_physical(self, q, v):
        """Complex modal state for physical DOF displacements q and velocities v."""
        q, v = np.asarray(q, float), np.asarray(v, float)
        if not self.coupled:
            lc = np.conj(self.lam)
            return (lc*q - v)/(lc - self.lam)
        return self.Psi @ np.r_[q, v]

    def transfer_from(self, old):
        """Take over the vibration of `old` (same DOFs, other coupling): physical state preserved exactly."""
        q, v = old.physical()
        self.z = self.from_physical(q, v)

    def state(self):
        return self.z.copy()

    def set_state(self, z):
        self.z = np.asarray(z, complex).copy()


# ====================================================================================================== builders
def bar_dofs(bar):
    """(omega, zeta, labels, kinds) of one bar's modes: vertical bending, horizontal bending, torsion."""
    f = np.r_[bar["f_vert"], bar["f_horiz"], bar["f_tors"]]
    kinds = ["v"]*len(bar["f_vert"]) + ["h"]*len(bar["f_horiz"]) + ["t"]*len(bar["f_tors"])
    w = 2*np.pi*f
    # Rucz's viscous law with the measured constant loss-factor floor (params.BAR_LOSS_FACTOR, Hill 2011): the viscous
    # law alone rings the low bars 2-3x longer than measured rosewood bars
    zeta = np.maximum(P.BAR_ETA_S*w/2, P.BAR_LOSS_FACTOR/2)
    labels = [f"{bar['note']}:{k}{i}" for i, k in enumerate(kinds)]
    return w, zeta, labels, kinds


def bar_shape(bar, x, y, z, n):
    """Shape vector of one bar's modes for a contact at (x, y, z) (bar frame) with normal n (bar frame, pointing from
    the bar toward the striker): vertical w(x) n_z + horizontal v(x) n_y + torsion phi(x) (y n_z - z n_y)."""
    xs = bar["x"]
    wv = np.array([np.interp(x, xs, bar["W_vert"][:, k]) for k in range(bar["W_vert"].shape[1])])
    wh = np.array([np.interp(x, xs, bar["W_horiz"][:, k]) for k in range(bar["W_horiz"].shape[1])])
    wt = np.array([np.interp(x, xs, bar["W_tors"][:, k]) for k in range(bar["W_tors"].shape[1])])
    return np.r_[wv*n[2], wh*n[1], wt*(y*n[2] - z*n[1])]


def helmholtz(res_index):
    """Helmholtz air plug of a xylophone resonator as installed: Rucz's measured bare frequency lowered by the measured
    detuning with all bars in place (Table IV); radiation damping k S/(4 pi l) x calibration + wall loss."""
    V, S, leff, f_bare, detune = P.RESONATORS[res_index]
    V, S = V*1e-6, S*1e-4
    f = f_bare*(1 - detune/100)
    w = 2*np.pi*f
    l_inst = S*P.C_AIR**2/(V*w**2)                               # effective neck length that gives the installed f
    m_a = P.RHO_AIR*S*l_inst
    k = w/P.C_AIR
    zeta = P.RESONATOR_RAD_FACTOR*k*S/(4*np.pi*l_inst) + P.RESONATOR_WALL_ZETA
    return dict(V=V, S=S, l_eff=l_inst, m_a=m_a, omega=w, zeta=zeta, k_c=P.RHO_AIR*P.C_AIR**2/V, f=f)


def build_xylophone(notes, with_resonators=True, removed=()):
    """Coupled structure of the xylophone's bars (in `notes` order) and its six box resonators. `removed`: notes whose
    bars are off the instrument (their coupling to the resonator is dropped; they keep vibrating as free bars)."""
    from . import modes
    omega, zeta, labels, slices, bars = [], [], [], {}, {}
    for note in notes:
        b = modes.bar(note)
        w, z, lab, kinds = bar_dofs(b)
        slices[note] = slice(len(omega), len(omega) + len(w))
        omega += list(w); zeta += list(z); labels += lab
        bars[note] = dict(data=b, kinds=np.array(kinds))
    res = []
    if with_resonators:
        for r_i, covered in enumerate(P.RESONATOR_BARS):
            h = helmholtz(r_i)
            h["dof"] = len(omega)
            h["bars"] = [n for n in covered if n in notes]
            res.append(h)
            omega.append(h["omega"]); zeta.append(h["zeta"]); labels.append(f"resonator{r_i + 1}")
    n = len(omega)
    Kx = np.zeros((n, n))
    for r_i, h in enumerate(res):
        # E_pot = 1/2 k_c (S xi_in - sum_bars A_b <psi_b> q_b)^2  (bar moving down pushes air into the neck)
        a = np.zeros(n)
        a[h["dof"]] = h["S"]/np.sqrt(h["m_a"])
        nb = len(P.RESONATOR_BARS[r_i])
        l_open = h["S"]/(nb*P.BAR_PITCH_M)                       # opening length under each bar (rounded rectangle)
        for note in h["bars"]:
            if note in removed:
                continue
            b = bars[note]["data"]
            area = P.RESONATOR_COUPLING_ETA*P.BAR_WIDTH_M*l_open
            sel = np.abs(b["x"]) <= l_open/2
            psi = b["W_vert"][sel].mean(axis=0)                 # mean vertical mode shape over the opening
            sl = slices[note]
            idx = np.arange(sl.start, sl.stop)[bars[note]["kinds"] == "v"]
            a[idx] = -area*psi
        Kx += h["k_c"]*np.outer(a, a)
        Kx[h["dof"], h["dof"]] -= h["k_c"]*a[h["dof"]]**2
        # (the plug's own k_c S^2/m_a is already omega_a^2; the bars' added air stiffness k_c A^2 psi^2 is ~2e-4 of
        # their omega^2 and is kept: the measured partials are on the instrument)
    s = Structure("xylophone", omega, zeta, K_extra=Kx if res else None, dof_labels=labels)
    s.bars, s.slices, s.resonators = bars, slices, res
    return s


def resonator_bar_box(note):
    """Geometry of a resonator bar's own box (params.RESONATOR_BAR_DATASHEET): outer length/width/height (m), wall
    (derived from the listed mass), inner volume V, and the slot (25 mm wide, its length solved so that the box with
    its physical neck l_eff = wall + 2 slot_end_correction(w, l) resonates at the bar's f1)."""
    return _box(note)


def slot_end_correction(a, b, n=240):
    """End correction (m) of one side of an a x b rectangular opening: the low-frequency radiation mass of a
    uniform-velocity rectangular piston in a baffle, M = rho/(2 pi) int int dS dS'/|r - r'|, over rho S (the
    circle's 8a/(3 pi) = 0.85 a in the same form; a square gives 0.99 of its equal-area circle, a 25 x 120 mm slot
    0.84). The inner integral over the rectangle is analytic; the outer one a midpoint rule."""
    def F(U, V):
        U = np.maximum(U, 1e-15)
        V = np.maximum(V, 1e-15)
        R = np.sqrt(U*U + V*V)
        return U*np.log((V + R)/U) + V*np.log((U + R)/V)
    x = (np.arange(n) + 0.5)/n*a
    y = (np.arange(n) + 0.5)/n*b
    X, Y = np.meshgrid(x, y)
    return float((F(a - X, b - Y) + F(X, b - Y) + F(a - X, Y) + F(X, Y)).mean()/(2*np.pi))


def _box_impl(note, slot_w=None, end="rectangle"):
    from . import modes
    D = P.RESONATOR_BAR_DATASHEET
    b = modes.bar(note)
    L = D["length_c1"] - (D["length_c1"] - D["length_h1"])*P.BAR_SEMITONE[note]/11
    W, H = D["width"], D["height"]
    t = (D["mass"] - float(b["mass"]))/(P.MAPLE_RHO*2*(L*W + L*H + W*H))
    V = (L - 2*t)*(W - 2*t)*(H - 2*t)
    f = float(b["f_vert"][0])
    w_s = P.RESONATOR_BAR_SLOT_W_M if slot_w is None else slot_w
    # the end correction: the slot's own (rectangular piston), or - for the sensitivity study only - an
    # equal-area circle's 1.7 a_eq or another factor x a_eq
    corr = (lambda ls: 2*slot_end_correction(w_s, ls)) if end == "rectangle" else \
        (lambda ls: float(end)*np.sqrt(w_s*ls/np.pi))
    target = V*(2*np.pi*f/P.C_AIR)**2                         # S / l_eff of a box tuned to f
    lo, hi = 1e-4, 0.9*(L - 2*t)
    for _ in range(50):                                        # S/l_eff grows with the slot length: bisection
        ls = 0.5*(lo + hi)
        S = w_s*ls
        if S/(t + corr(ls)) < target:
            lo = ls
        else:
            hi = ls
    S = w_s*ls
    return dict(note=note, length=L, width=W, height=H, wall=t, V=V, slot_w=w_s, slot_l=ls, S=S,
                l_eff=t + corr(ls), f=f, mass=D["mass"] - float(b["mass"]), end_correction=end,
                slot_share_of_box=ls/(L - 2*t), box_length_wavelengths=L*f/P.C_AIR)


_BOX = {}


def _box(note):
    if note not in _BOX:
        _BOX[note] = _box_impl(note)
    return _BOX[note]


_BEAT = {}


def resonator_bar_beat_s(note):
    """Period (s) of the energy exchange between a resonator bar's mode 1 and its box: 1/|f_a - f_b| of the two coupled
    complex modes nearest the bar's f1 (the bar's ring 'breathes' with this period)."""
    if note not in _BEAT:
        st = build_bar_structure(note, resonator=True)
        f = st.lam.imag/(2*np.pi)
        i = np.argsort(np.abs(f - st.resonators[0]["f"]))[:2]
        _BEAT[note] = float(1.0/abs(f[i[0]] - f[i[1]]))
    return _BEAT[note]


def build_bar_structure(note, resonator=None, box=None):
    """A single resonator bar (Studio 49 KB/AX class): the bar over its own box resonator (resonator_bar_box: datasheet
    size, slot tuned to the bar's f1 with its physical neck), or a free bar (resonator=None). `box`: another box
    geometry (the sensitivity study)."""
    from . import modes
    b = modes.bar(note)
    w, z, lab, kinds = bar_dofs(b)
    omega, zeta, labels = list(w), list(z), list(lab)
    slices = {note: slice(0, len(w))}
    bars = {note: dict(data=b, kinds=np.array(kinds))}
    res = []
    if resonator is not None:
        bx = box or resonator_bar_box(note)
        V, S, l = bx["V"], bx["S"], bx["l_eff"]
        wa = P.C_AIR*np.sqrt(S/(V*l))                            # = 2 pi f1 (the slot was solved for it)
        h = dict(V=V, S=S, l_eff=l, m_a=P.RHO_AIR*S*l, omega=wa, k_c=P.RHO_AIR*P.C_AIR**2/V, f=wa/(2*np.pi),
                 zeta=P.RESONATOR_RAD_FACTOR*(wa/P.C_AIR)*S/(4*np.pi*l) + P.RESONATOR_WALL_ZETA, dof=len(omega),
                 bars=[note], box=bx)
        omega.append(wa); zeta.append(h["zeta"]); labels.append("resonator")
        res.append(h)
        n = len(omega)
        a = np.zeros(n)
        a[h["dof"]] = S/np.sqrt(h["m_a"])
        l_open = bx["slot_l"]
        sel = np.abs(b["x"]) <= l_open/2
        idx = np.flatnonzero(np.array(kinds) == "v")
        a[idx] = -P.RESONATOR_COUPLING_ETA*bx["slot_w"]*l_open*b["W_vert"][sel].mean(axis=0)
        Kx = h["k_c"]*np.outer(a, a)
        Kx[h["dof"], h["dof"]] = 0.
    s = Structure(f"resonator_bar_{note}", omega, zeta, K_extra=Kx if res else None, dof_labels=labels)
    s.bars, s.slices, s.resonators = bars, slices, res
    return s


def build_drumhead(tension_name):
    from . import modes
    d = modes.membrane(tension_name)
    w = 2*np.pi*d["f"]
    zeta = d["zeta_rad"] + P.DRUM_ZETA_INTERNAL
    s = Structure(f"drumhead_{tension_name}", w, zeta,
                  dof_labels=[f"({m},{n}){'cs'[o]}" for m, n, o in zip(d["m"], d["n"], d["orient"])])
    s.membrane = d
    s.M_total = d["M0"] + d["madd"]
    s.norm = 1.0/np.sqrt(s.M_total)
    return s


def membrane_shape(s, r, theta, patch_r):
    """Mode shapes of the drumhead averaged over a contact disc of radius patch_r centred at (r, theta)."""
    from scipy.special import jv
    d = s.membrane
    k = 7
    g = np.linspace(-patch_r, patch_r, k)
    X, Y = np.meshgrid(g, g)
    inside = X**2 + Y**2 <= patch_r**2 + 1e-15
    cx, cy = r*np.cos(theta), r*np.sin(theta)
    px, py = cx + X[inside], cy + Y[inside]
    rr = np.minimum(np.hypot(px, py), d["radius"])
    th = np.arctan2(py, px)
    m = d["m"][:, None]
    ang = np.where(d["orient"][:, None] == 0, np.cos(m*th[None, :]), np.sin(m*th[None, :]))
    vals = jv(m, d["alpha"][:, None]*rr[None, :])*ang
    return vals.mean(axis=1)*s.norm


def membrane_residual(s, r, patch_r, b):
    """Quasi-static compliance (m/N) of the membrane modes not retained at a patch of radius patch_r at radius r:
    exact patch-averaged static membrane Green's function minus the retained modes' static sum."""
    a = s.membrane["radius"]
    T = float(s.membrane["tension"])
    r = min(r, a - 2*patch_r)
    g = (np.log(1/patch_r) + 0.25 + np.log((a**2 - r**2)/a))/(2*np.pi*T)
    return max(0.0, g - float(np.sum(b**2/s.omega**2)))


def build_stick_modes(name):
    """A drumstick's free-free bending modes in its two transverse planes (local y and z)."""
    from . import modes
    st = modes.stick()
    w = 2*np.pi*np.r_[st["f"], st["f"]]
    s = Structure(name, w, np.full(len(w), P.STICK_MODE_ZETA),
                  dof_labels=[f"bend{i + 1}{ax}" for ax in "yz" for i in range(len(st["f"]))])
    s.stick = st
    return s


def stick_shape(s, x_from_butt, n_local):
    """Shape vector for a contact at x_from_butt with normal n_local (stick frame, from the struck surface toward the
    stick): the stick's elastic displacement toward the surface, -W(x) (n_y, n_z)."""
    st = s.stick
    W = np.array([np.interp(x_from_butt, st["x"], st["W"][:, k]) for k in range(st["W"].shape[1])])
    return -np.r_[W*n_local[1], W*n_local[2]]
