"""Paint materials: thixotropic Cross/Bingham rheology with Krieger-Dougherty drying build and
Kubelka-Munk optics.

Rheology (fit to the TA/AZoM waterborne profile and Hester and Squire 1997):
    eta(gd, lam, phi) = KD(phi) [eta_inf + (eta0(lam) - eta_inf) / (1 + (gd / gc)^n)],  eta0 = eta0_min + (eta0_max - eta0_min) lam
    tau_y(lam, phi)   = KD(phi) tau_y_max lam,       KD(phi) = ((1 - phi/phi_m) / (1 - phi0/phi_m))^-2
    dlam/dt = (1 - lam) / t_r  (structure recovery after an applicator sheared it to lam = 0)
Optics: per linear-RGB channel, K/S from the colour of an infinitely thick film, S per micrometre of dry film; a layer
of solids thickness X over a background R_g reflects
    R = (1 - R_g (a - b coth(b S X))) / (a - R_g + b coth(b S X)),  a = 1 + K/S, b = sqrt(a^2 - 1)
Mixtures add K and S by solids volume fraction. Wet water-borne films add binder-haze scattering (haze x S_white per
um of wet solids): wet latex reads lighter and darkens as the milky binder clears (paint-store guidance).
"""

from dataclasses import dataclass, field
import numpy as np

from . import params as P

S_WHITE = P.LATEX["S_um"]


def srgb_to_linear(c):
    c = np.asarray(c, float)/255.0
    return np.where(c <= 0.04045, c/12.92, ((c + 0.055)/1.055)**2.4)


def linear_to_srgb8(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92*x, 1.055*np.power(x, 1/2.4) - 0.055)
    return np.clip(np.round(s*255.0), 0, 255).astype(np.uint8)


def ks_from_rinf(r):
    r = np.clip(np.asarray(r, float), 1e-4, 0.9999)
    return (1 - r)**2/(2*r)


def km_layer(K, S, X, Rg):
    """Kubelka-Munk reflectance of a layer (K, S per um; X um) over background Rg (all broadcastable)."""
    K = np.asarray(K, float); S = np.asarray(S, float); X = np.asarray(X, float); Rg = np.asarray(Rg, float)
    S = np.maximum(S, 1e-9)
    a = 1.0 + K/S
    b = np.sqrt(np.maximum(a*a - 1.0, 1e-12))
    bsx = np.clip(b*S*X, 1e-9, 50.0)
    coth = 1.0/np.tanh(bsx)
    R = (1.0 - Rg*(a - b*coth))/(a - Rg + b*coth)
    clear = X <= 1e-6
    return np.where(clear, Rg, R)


TINT_BASES = P.TINT_BASES


def tint_base(rinf):
    """(base name, scattering relative to the white base) for a wall colour of linear reflectance rinf, by its
    luminance Y (LRV): white >= 0.70, pastel >= 0.40, medium >= 0.15, else deep."""
    Y = float(np.asarray(rinf) @ np.array([0.2126, 0.7152, 0.0722]))
    for name, (y_min, f) in TINT_BASES.items():
        if Y >= y_min:
            return name, f
    return "deep", TINT_BASES["deep"][1]


@dataclass
class Paint:
    name: str
    product: str
    color: str
    rec: dict = field(repr=False)
    climate: dict = None
    K: np.ndarray = field(init=False, repr=False)
    S: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        r = self.rec
        self.rho, self.phi0, self.sigma = r["rho"], r["phi0"], r["sigma"]
        self.rinf = srgb_to_linear(P.COLORS[self.color])
        S = r["S_um"]
        if self.name.split(":")[0] == "latex":
            # tinted wall paint is mixed in the tint base its lightness needs; a base's scattering scales with its TiO2
            # (white ~30 %, pastel 15-20, medium 8-12 % of solids, tint base guides; a deep base carries none, its
            # scattering is the extenders' and colorants', calibrated on coats-to-hide: params.TINT_BASES); S_um is the white base's (E5)
            S = S*tint_base(self.rinf)[1]
        self.S = np.full(3, S)
        self.K = self.S*ks_from_rinf(self.rinf) if S > 0 else np.zeros(3)
        self.j_evap = P.evaporation_rate(r, self.climate)
        self.clear = S <= 0
        # solids density from the as-supplied density: rho = phi0 rho_s + (1 - phi0) rho_vehicle (vehicle ~ water)
        self.rho_solids = (self.rho - (1 - self.phi0)*P.WATER_RHO)/self.phi0

    # ------------------------------------------------------------------ rheology
    def kd(self, phi):
        r = self.rec
        phi = np.minimum(phi, r["phi_m"]*0.999)
        return ((1 - phi/r["phi_m"])/(1 - r["phi0"]/r["phi_m"]))**-2

    def eta(self, gd, lam, phi=None):
        r = self.rec
        e0 = r["eta0_min"] + (r["eta0_max"] - r["eta0_min"])*lam
        base = r["eta_inf"] + (e0 - r["eta_inf"])/(1 + (np.maximum(gd, 0.0)/r["gamma_c"])**r["n"])
        return base*(self.kd(phi) if phi is not None else 1.0)

    def tau_y(self, lam, phi=None):
        return self.rec["tau_y_max"]*lam*(self.kd(phi) if phi is not None else 1.0)

    def to_dict(self):
        return dict(name=self.name, product=self.product, color=self.color, rho=self.rho, phi0=self.phi0,
                    sigma=self.sigma, evaporation_um_s=self.j_evap*1e6)


def make_paint(kind, color, climate=None):
    rec = dict(latex=P.LATEX, acrylic=P.ACRYLIC, enamel=P.ENAMEL, lacquer=P.LACQUER)[kind]
    product = dict(latex="interior latex eg-shel (ProMar 200 class)", acrylic="artist fluid acrylic",
                   enamel="aerosol enamel (Painter's Touch class)", lacquer="aerosol clear lacquer")[kind]
    return Paint(f"{kind}:{color}", product, color, dict(rec), climate)


def stack_reflectance(paints, dry, wet_s, wet_w, Rg):
    """Linear-RGB reflectance of substrate -> dry layer -> wet layer.

    paints: list of Paint; dry, wet_s: (P, ...) solids thickness per paint (m); wet_w: water (m); Rg: (..., 3).
    Returns (..., 3)."""
    dry = np.asarray(dry, float); wet_s = np.asarray(wet_s, float)
    shape = dry.shape[1:]
    out = np.asarray(Rg, float)
    for layer, wet in ((dry, False), (wet_s, True)):
        X = layer.sum(axis=0)*1e6                                        # um of solids
        if not np.any(X > 1e-6):
            continue
        K = np.zeros(shape + (3,)); S = np.zeros(shape + (3,))
        frac = layer/np.maximum(X*1e-6, 1e-15)
        for k, pt in enumerate(paints):
            K += frac[k][..., None]*pt.K
            S += frac[k][..., None]*pt.S
            if wet and pt.rec["haze"] > 0:
                S += frac[k][..., None]*pt.rec["haze"]*S_WHITE
        clear = np.all(S < 1e-9, axis=-1)
        R = km_layer(K, np.maximum(S, 1e-9), X[..., None], out)
        out = np.where(clear[..., None], out, R)
    return out


# ------------------------------------------------------------------ colour difference (evaluation)
_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = _M @ np.ones(3)


def lab(rgb_linear):
    xyz = np.asarray(rgb_linear, float) @ _M.T/_WHITE
    f = np.where(xyz > (6/29)**3, np.cbrt(xyz), xyz/(3*(6/29)**2) + 4/29)
    return np.stack([116*f[..., 1] - 16, 500*(f[..., 0] - f[..., 1]), 200*(f[..., 1] - f[..., 2])], axis=-1)


def delta_e(a, b):
    return np.linalg.norm(lab(a) - lab(b), axis=-1)
