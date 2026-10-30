"""Sound at the microphones: per-mode radiation, delays, first-order room reflections, a diffuse
reverberant tail, the microphone model and IEC 61672 level meters.

Radiation (frequency-domain transfer per DOF, evaluated per complex mode; p(t) = Re sum_j R_j z_j(t - tau)):
  * drumhead: Rayleigh integral of a baffled circular-membrane mode, far field: p = (i w rho / R) i^m cos(m Phi)
    Phit_m(alpha, k sin Theta) / sqrt(M) v  (the King/Hankel transform); below the head plane the open
    shell's inner face radiates the opposite sign (approximation);
  * bars: a line of point-force dipoles: each element of length dx exerts on the air the force (added + displaced air
    mass) x acceleration, mu = rho pi (a_y^2 + a_y a_z) for vertical bending (elliptic-cylinder added mass of the
    31 x 16 mm section), p = F cos(theta) (ik + 1/R) e^{-ikR} / (4 pi R); torsion (no net force) does not radiate;
  * resonators: monopole of the neck's volume velocity, p = i w rho U e^{-ikR} / (4 pi R).
Room: image sources of the six room surfaces and of the table top (when the source is above it), each with the
octave-band reflection amplitude sqrt(1 - alpha); a late reverberant tail (exponentially decaying noise per octave
band with the Sabine T60 of the room, energy 16 pi (1 - a)/(S a) relative to the 1 m direct sound) by block FFT
convolution starting 1024 samples (21 ms) after the direct sound.
Microphone: omnidirectional, flat 20 Hz - 20 kHz, clipping at 133 dB SPL (miniDSP UMIK-1 class [S39]).
"""
import math
import numpy as np

from . import params as P

BLOCK = 1024                         # late-reverb block (samples); the tail starts at this delay


# ====================================================================================================== room
def room_t60():
    """Octave-band (Sabine) reverberation time and mean absorption of the therapy room."""
    R = P.ROOM
    Lx, Ly, H = R["x1"] - R["x0"], R["y1"] - R["y0"], R["h"]
    V = Lx*Ly*H
    win = 1.6*1.4                                            # window area on the right wall (dressing.py)
    walls = 2*(Lx + Ly)*H - win
    S = 2*Lx*Ly + walls + win
    A = (Lx*Ly*np.array(P.ABSORPTION["floor"]) + Lx*Ly*np.array(P.ABSORPTION["ceiling"]) +
         walls*np.array(P.ABSORPTION["walls"]) + win*np.array(P.ABSORPTION["window"]))
    return dict(V=V, S=S, A=A, alpha=A/S, t60=0.161*V/A)


def _alpha_at(f, table):
    lf = np.log2(np.clip(np.asarray(f, float), P.OCTAVES_HZ[0], P.OCTAVES_HZ[-1]))
    return np.interp(lf, np.log2(P.OCTAVES_HZ), table)


def image_points(X, source_above_table=False):
    """[(label, point, absorption table or None)] of the direct microphone position and its first-order images."""
    R = P.ROOM
    X = np.asarray(X, float)
    out = [("direct", X, None)]
    out.append(("floor", X*(1, 1, -1), P.ABSORPTION["floor"]))
    out.append(("ceiling", np.array([X[0], X[1], 2*R["h"] - X[2]]), P.ABSORPTION["ceiling"]))
    out.append(("wall_x0", np.array([2*R["x0"] - X[0], X[1], X[2]]), P.ABSORPTION["walls"]))
    out.append(("wall_x1", np.array([2*R["x1"] - X[0], X[1], X[2]]), P.ABSORPTION["window"]))
    out.append(("wall_y0", np.array([X[0], 2*R["y0"] - X[1], X[2]]), P.ABSORPTION["walls"]))
    out.append(("wall_y1", np.array([X[0], 2*R["y1"] - X[1], X[2]]), P.ABSORPTION["walls"]))
    if source_above_table:
        out.append(("table", np.array([X[0], X[1], 2*P.TABLE["top_z"] - X[2]]), (0.05,)*6))
    return out


def late_tail(seed, fs=P.AUDIO_FS):
    """Late reverberation impulse response (pressure per unit 1-m direct pressure), starting at BLOCK samples."""
    rt = room_t60()
    rng = np.random.default_rng(seed)
    n = int(1.2*float(rt["t60"].max())*fs)
    t = np.arange(n)/fs
    from scipy.signal import butter, sosfilt
    h = np.zeros(n)
    ramp = np.clip((np.arange(n) - BLOCK)/(0.01*fs), 0, 1)     # the tail fades in over 10 ms after its start
    for b, fc in enumerate(P.OCTAVES_HZ):
        lo, hi = fc/math.sqrt(2), min(fc*math.sqrt(2), 0.45*fs)
        sos = butter(2, [lo, hi], btype="band", fs=fs, output="sos")
        x = sosfilt(sos, rng.standard_normal(n))*np.exp(-6.91*t/rt["t60"][b])*ramp
        # pass-band energy gain of this band's tail: 16 pi (1 - a)/(S a) relative to the 1-m direct sound (diffuse
        # field of a room with Sabine absorption); for a band of width B the FIR's sum h^2 = gain^2 * 2B/fs
        target = 16*math.pi*(1 - rt["alpha"][b])/(rt["S"]*rt["alpha"][b])
        h += x*math.sqrt(target*2*(hi - lo)/fs/float(np.sum(x**2)))
    h[:BLOCK] = 0.
    return h                                                   # discrete convolution: y_n = sum_k h_k x_{n-k}


# ====================================================================================================== transfer functions
def drum_transfer(s, R_drum, p_drum, points, omega):
    """(rows, nc) complex transfer of every membrane mode to each point (world), with the integer-sample-free delay
    phase removed (returned separately as distances)."""
    from .modes import _phit
    d = s.membrane
    pts = (np.asarray(points, float) - p_drum) @ R_drum                 # into the drum frame (z = head normal)
    Rx = np.linalg.norm(pts, axis=1)
    cosT = pts[:, 2]/Rx
    below = cosT < 0
    sinT = np.sqrt(np.maximum(0., 1 - cosT**2))
    Phi = np.arctan2(pts[:, 1], pts[:, 0])
    k = omega/P.C_AIR
    m = d["m"]
    ang = np.where(d["orient"][None, :] == 0, np.cos(m[None, :]*Phi[:, None]), np.sin(m[None, :]*Phi[:, None]))
    H = np.empty((len(pts), len(omega)), complex)
    for r in range(len(pts)):
        ph = np.zeros(len(omega))
        for mm in np.unique(m):
            sel = m == mm
            ph[sel] = _phit(int(mm), d["alpha"][sel], k[sel]*sinT[r], d["radius"])
        H[r] = (1j*omega*P.RHO_AIR/Rx[r])*(1j**m)*ang[r]*ph*s.norm*(-1.0 if below[r] else 1.0)
    return H, Rx


def bar_transfer(bar, R_bar, p_bar, points, omega_v, omega_h):
    """(rows, n_v + n_h) transfer of a bar's vertical and horizontal bending modes (velocity -> pressure) to points."""
    x = bar["x"]
    h = np.r_[bar["h"], bar["h"][-1]]
    dx = np.gradient(x)
    ay, az = P.BAR_WIDTH_M/2, h/2
    mu_v = P.RHO_AIR*math.pi*(ay**2 + ay*az)*dx
    mu_h = P.RHO_AIR*math.pi*(az**2 + ay*az)*dx
    pts = np.asarray(points, float)
    elems = p_bar[None, :] + np.outer(x, R_bar[:, 0])                  # element positions (world)
    out = np.zeros((len(pts), len(omega_v) + len(omega_h)), complex)
    dist = np.linalg.norm(pts - p_bar[None, :], axis=1)
    for r, X in enumerate(pts):
        rv = X[None, :] - elems
        Re = np.linalg.norm(rv, axis=1)
        cz = (rv @ R_bar[:, 2])/Re
        cy = (rv @ R_bar[:, 1])/Re
        for kind, (om, W, mu, cth) in enumerate(((omega_v, bar["W_vert"], mu_v, cz), (omega_h, bar["W_horiz"], mu_h, cy))):
            if not len(om):
                continue
            k = om/P.C_AIR
            g = (1j*k[None, :] + 1/Re[:, None])*cth[:, None]*np.exp(-1j*np.outer(Re - dist[r], k))/(4*math.pi*Re[:, None])
            Hk = (1j*om)*np.sum((mu[:, None]*W[:, :len(om)])*g, axis=0)
            if kind == 0:
                out[r, :len(om)] = Hk
            else:
                out[r, len(omega_v):] = Hk
    return out, dist


def monopole_transfer(p_src, points, omega, gain):
    pts = np.asarray(points, float)
    Rx = np.linalg.norm(pts - p_src[None, :], axis=1)
    return (1j*omega[None, :]*P.RHO_AIR*gain)/(4*math.pi*Rx[:, None])*np.ones((len(pts), 1)), Rx


# ====================================================================================================== radiation of one structure
class Radiation:
    """Row coefficients R (rows, nc) and integer delays of one structure toward every microphone row."""

    def __init__(self, R, delays, mic_of_row, send_rows):
        self.R, self.delays, self.mic_of_row, self.send_rows = R, delays, mic_of_row, send_rows


class Acoustics:
    def __init__(self, mics, seed=0, noise=False):
        """mics: {name: position (world)}."""
        self.mics = {k: np.asarray(v, float) for k, v in mics.items()}
        self.names = list(self.mics)
        self.fs = P.AUDIO_FS
        self.cap = 1 << 20
        self.buf = {k: np.zeros(self.cap) for k in self.names}         # pressure (Pa) per sample, from sample 0
        self.send = {k: np.zeros(self.cap) for k in self.names}        # reverb input (1-m direct pressure)
        self.n = 0                                                    # samples finalised (the current time)
        self.reverb_done = {k: 0 for k in self.names}
        self.tail = {k: late_tail(seed*31 + i) for i, k in enumerate(self.names)}
        self._fft = {}
        self.noise = noise
        self.rng = np.random.default_rng(seed + 7)
        self.peak_pa = 0.0
        self.clipped = 0

    # -------------------------------------------------------------- buffers
    def _ensure(self, n):
        if n <= self.cap:
            return
        new = 1 << int(math.ceil(math.log2(n + 1)))
        for d in (self.buf, self.send):
            for k in d:
                a = np.zeros(new)
                a[:self.cap] = d[k]
                d[k] = a
        self.cap = new

    def rows_for(self, source_point, source_above_table=False):
        """Row points (world) and their bookkeeping for a source near source_point."""
        pts, mic, alpha, labels = [], [], [], []
        for k, X in self.mics.items():
            for lab, pt, a in image_points(X, source_above_table):
                pts.append(pt); mic.append(k); alpha.append(a); labels.append(lab)
        return np.array(pts), mic, alpha, labels

    def radiation(self, struct, H_rows, dist, mic, alpha, labels, omega_j):
        """Combine per-row complex transfer at the complex-mode frequencies with reflection gains and fractional
        delays into a Radiation for `struct` (H_rows: (rows, nc) per complex mode of the analytic velocity)."""
        rows = len(dist)
        delays = np.round(dist/P.C_AIR*self.fs).astype(int)
        frac = dist/P.C_AIR - delays/self.fs
        R = H_rows*np.exp(-1j*np.outer(frac, omega_j))
        for r in range(rows):
            if alpha[r] is not None:
                R[r] *= np.sqrt(1 - _alpha_at(omega_j/(2*math.pi), alpha[r]))
        send = [r for r in range(rows) if labels[r] == "direct"]
        Rs = np.array([R[r]*dist[r] for r in send])
        return Radiation(np.vstack([R, Rs]), np.r_[delays, delays[send]], list(mic) + [mic[r] for r in send],
                         [False]*rows + [True]*len(send))

    # -------------------------------------------------------------- per tick
    def add_block(self, rad, Z, start, idx=None):
        """Add a structure's contribution for the 48 samples starting at `start` (Z: (nc, 48) complex states, or the
        rows `idx` of the modes only)."""
        if idx is not None and not len(idx):
            return
        Y = np.real((rad.R if idx is None else rad.R[:, idx]) @ Z)
        L = Z.shape[1]
        self._ensure(start + L + int(rad.delays.max()) + 2)
        for r in range(len(rad.delays)):
            i = start + int(rad.delays[r])
            (self.send if rad.send_rows[r] else self.buf)[rad.mic_of_row[r]][i:i + L] += Y[r]

    def add_signal(self, mic, signal, start, gain=1.0):
        self._ensure(start + len(signal) + 1)
        self.buf[mic][start:start + len(signal)] += gain*np.asarray(signal)

    def add_point_source(self, position, p_1m, start):
        """A loudspeaker: on-axis pressure at 1 m p_1m(t) (Pa) starting at sample `start`, radiated omni with the
        room's image sources and tail."""
        pts, mic, alpha, labels = self.rows_for(position)
        dist = np.linalg.norm(pts - np.asarray(position)[None, :], axis=1)
        for r in range(len(dist)):
            d = int(round(dist[r]/P.C_AIR*self.fs))
            g = 1.0/dist[r]*(1.0 if alpha[r] is None else math.sqrt(1 - _alpha_at(1000., alpha[r])))
            self.add_signal(mic[r], p_1m, start + d, g)
            if labels[r] == "direct":
                self._ensure(start + d + len(p_1m) + 1)
                self.send[mic[r]][start + d:start + d + len(p_1m)] += p_1m
        return len(p_1m)

    def advance(self, n_samples):
        """The current time moves on by n_samples; complete reverb blocks are convolved into the future."""
        self.n += n_samples
        self._ensure(self.n + BLOCK)
        for k in self.names:
            while self.reverb_done[k] + BLOCK <= self.n:
                b0 = self.reverb_done[k]
                x = self.send[k][b0:b0 + BLOCK]
                if np.any(x):
                    h = self.tail[k]
                    nfft = 1 << int(math.ceil(math.log2(len(h) + BLOCK)))
                    key = (k, nfft)
                    if key not in self._fft:
                        self._fft[key] = np.fft.rfft(h, nfft)
                    y = np.fft.irfft(np.fft.rfft(x, nfft)*self._fft[key], nfft)[:len(h) + BLOCK]
                    self._ensure(b0 + len(y) + 1)
                    self.buf[k][b0:b0 + len(y)] += y
                self.reverb_done[k] = b0 + BLOCK

    def signal(self, mic, t0=0, t1=None):
        """Microphone pressure (Pa) for samples [t0, t1) (t1 <= current time), with the microphone's clipping."""
        t1 = self.n if t1 is None else min(t1, self.n)
        x = self.buf[mic][t0:t1].copy()
        if self.noise:
            x += self.background(t0, t1)
        lim = P.P_REF*10**(P.MIC_MAX_SPL_DB/20)*math.sqrt(2)
        return np.clip(x, -lim, lim)

    def background(self, t0, t1):
        """Room background noise at ROOM_NOISE_DBA (pink-ish, reproducible per sample index)."""
        rng = np.random.default_rng(int(t0) + 99991)
        w = rng.standard_normal(t1 - t0)
        from scipy.signal import lfilter
        y = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], w)
        target = P.P_REF*10**(P.ROOM_NOISE_DBA/20)
        return y/max(float(np.std(y)), 1e-12)*target

    def state(self):
        return dict(n=self.n, reverb_done=dict(self.reverb_done),
                    buf={k: v[:self.n + 4*BLOCK + 48000].copy() for k, v in self.buf.items()},
                    send={k: v[:self.n + 4*BLOCK].copy() for k, v in self.send.items()})

    def set_state(self, s):
        self.n = s["n"]
        self.reverb_done = dict(s["reverb_done"])
        for d, src in ((self.buf, s["buf"]), (self.send, s["send"])):
            for k in d:
                d[k][:] = 0.
                d[k][:len(src[k])] = src[k]


_CLICK = {}


def click_signal(kind, E_rad, fs=P.AUDIO_FS):
    """1-m pressure (Pa) of a rigid-part click radiating E_rad joules omnidirectionally: authored damped partials
    (params.RIGID_CLICK_SPECTRA) scaled so that the radiated energy 4 pi (1 m)^2/(rho c) * integral p_1m^2 dt = E_rad."""
    if kind not in _CLICK:
        parts = P.RIGID_CLICK_SPECTRA[kind]
        n = int(fs*1.2*max(t60 for _, t60, _ in parts))
        t = np.arange(n)/fs
        x = sum(a*np.exp(-6.91*t/t60)*np.sin(2*math.pi*f*t) for f, t60, a in parts)
        _CLICK[kind] = x/math.sqrt(np.sum(x**2)/fs)                      # unit integral of p^2
    return _CLICK[kind]*math.sqrt(E_rad*P.RHO_AIR*P.C_AIR/(4*math.pi))


# ====================================================================================================== level meters
def a_weighting_sos(fs=P.AUDIO_FS):
    """IEC 61672-1 A-weighting as a digital filter (bilinear transform of the analog prototype, normalised to 0 dB at
    1 kHz)."""
    from scipy.signal import bilinear_zpk, zpk2sos, freqz_zpk
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    z = [0, 0, 0, 0]
    p = [-2*np.pi*f4, -2*np.pi*f4, -2*np.pi*f1, -2*np.pi*f1, -2*np.pi*f2, -2*np.pi*f3]
    k = (2*np.pi*f4)**2
    zd, pd, kd = bilinear_zpk(z, p, k, fs)
    w, h = freqz_zpk(zd, pd, kd, worN=[1000.0], fs=fs)
    return zpk2sos(zd, pd, kd/abs(h[0]))


_A = {}


def level_fast(x, fs=P.AUDIO_FS, weighting="A", tau=0.125):
    """L_AF(t) in dB re 20 uPa: A-weighted, exponential (fast, 125 ms) time weighting (IEC 61672)."""
    from scipy.signal import sosfilt, lfilter
    if weighting == "A":
        if fs not in _A:
            _A[fs] = a_weighting_sos(fs)
        x = sosfilt(_A[fs], x)
    a = 1 - math.exp(-1/(tau*fs))
    e = lfilter([a], [1, -(1 - a)], np.asarray(x)**2)
    return 10*np.log10(np.maximum(e, 1e-30)/P.P_REF**2)


def policy_audio(x, n_out=8000, fs=P.AUDIO_FS):
    """The policy's microphone observation from the last window x (Pa at fs; 0.75 s): the last n_out samples at 16 kHz
    and L_AF now, L_Aeq over the last 100 ms and the peak of the last 100 ms. The A-weighting filter starts from the
    steady state of the window's first sample (no start-up step at the window's edge), L_AF's 125 ms exponential has
    0.75 s to forget its zero start (e^-6), and L_Aeq(100 ms) is the mean square of the last 100 ms of the same weighted
    signal (a 100 ms filter started from zero turns any offset at its start into a 'level')."""
    from scipy.signal import resample_poly, sosfilt, sosfilt_zi, lfilter
    y = resample_poly(x, 1, 3, padtype="line")[-n_out:]
    n100 = int(0.1*fs)
    laf = leq = 0.0
    if np.any(x):
        if fs not in _A:
            _A[fs] = a_weighting_sos(fs)
        ya, _ = sosfilt(_A[fs], x, zi=sosfilt_zi(_A[fs])*x[0])
        a = 1 - math.exp(-1/(0.125*fs))
        e = lfilter([a], [1, -(1 - a)], ya**2)
        laf = 10*math.log10(max(float(e[-1]), 1e-30)/P.P_REF**2)
        leq = 10*math.log10(max(float(np.mean(ya[-n100:]**2)), 1e-30)/P.P_REF**2)
    clean = lambda v: float(v) if math.isfinite(v) else 0.0
    return {"pressure_pa_16k": [float(v) for v in y], "laf_dba": clean(max(laf, 0.0)),
            "laeq_100ms_dba": clean(max(leq, 0.0)), "peak_pa_100ms": float(np.abs(x[-n100:]).max())}


def leq_frames(x, frame, fs=P.AUDIO_FS, weighting="A"):
    """L_Aeq of consecutive frames of `frame` samples."""
    from scipy.signal import sosfilt
    if weighting == "A":
        if fs not in _A:
            _A[fs] = a_weighting_sos(fs)
        x = sosfilt(_A[fs], x)
    n = len(x)//frame
    e = np.mean(np.asarray(x[:n*frame]).reshape(n, frame)**2, axis=1)
    return 10*np.log10(np.maximum(e, 1e-30)/P.P_REF**2)
