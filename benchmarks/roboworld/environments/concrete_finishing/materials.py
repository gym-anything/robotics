"""Material laws: yield stress against age for every paste, bleeding and evaporation.

Concrete (M6):  tau_b(t) = tau_b0 + A_thix t/s_T + tau_IS (t/(t_IS s_T))^m,  tau_s = r_s tau_b,
with s_T the MASCO set-time factor of the concrete's temperature. Repair mortar takes the same form with its own
start and set time; masonry mortar stiffens by suction (a cubic in its age); sealant and compound keep a constant
yield stress over a task (sealant forms a skin at 2 h). Bleeding is a triangle peaking at a third of its duration
that ends when the bulk reaches BLEED_END_TAU; evaporation is the ACI 308 nomograph's equation
(FHWA-RD-02-099 figure 6). Every function takes numpy arrays of ages and returns arrays.
"""

import math
import numpy as np
from . import params as P


def masco_factor(temp_c):
    """Set-time factor against 70 F from MASCO's final-set table (opened): 1.72 at 10 C ... 0.45 at 32 C."""
    tf = temp_c*1.8 + 32.0
    ks = sorted(P.MASCO)
    return float(np.interp(tf, ks, [P.MASCO[k] for k in ks])/P.MASCO[70])


class Concrete:
    name = "concrete"
    rho = P.RHO_CONCRETE
    mu_bulk, mu_surf = P.MU_BULK, P.MU_SURF
    bleeds = True

    def __init__(self, temp_c=21.1, tau0=P.TAU_B0, t_is=P.T_IS, bleed_mean=P.BLEED_MEAN):
        self.temp = temp_c
        self.s = masco_factor(temp_c)
        self.tau0, self.t_is = tau0, t_is
        self.bleed_mean = bleed_mean
        self.bleed_end = self._time_to(P.BLEED_END_TAU)
        self.initial_set = self._time_to(P.PR_IS/P.N_DEEP)

    def tau_bulk(self, age):
        a = np.maximum(np.asarray(age, float), 0.0)
        return self.tau0 + P.A_THIX*a/self.s + P.TAU_IS*(a/(self.t_is*self.s))**P.SET_M

    def tau_surf(self, age, dens=0.0):
        r0, r1 = P.R_SURF
        return (r0 + (r1 - r0)*np.asarray(dens, float))*self.tau_bulk(age)

    def _time_to(self, tau):
        lo, hi = 0.0, 48*3600.0
        for _ in range(60):
            mid = 0.5*(lo + hi)
            if float(self.tau_bulk(mid)) < tau:
                lo = mid
            else:
                hi = mid
        return hi

    def bleed_rate(self, age):
        """kg/m2/s of bleed water rising to the surface at this age (a triangle with mean bleed_mean over its
        duration, peak at a third of it)."""
        a = np.asarray(age, float)
        te = self.bleed_end
        peak = 2.0*self.bleed_mean/3600.0
        up = peak*a/(te/3.0)
        down = peak*(te - a)/(2.0*te/3.0)
        r = np.where(a < te/3.0, up, down)
        return np.clip(r, 0.0, None)

    def marks(self):
        """Ages (s) of the field marks at this temperature: a boot (90 kg on 0.027 m2) holds, a
        fingertip (20 N on 15 x 20 mm) holds, initial and final set (deep punch factor N_DEEP)."""
        boot = 90*9.81/0.027
        return dict(boot_s=self._time_to(boot/P.NC), finger_s=self._time_to(20.0/(0.015*0.020)/P.NC),
                    initial_set_s=self._time_to(P.PR_IS/P.N_DEEP), final_set_s=self._time_to(P.PR_FS/P.N_DEEP),
                    bleed_end_s=self.bleed_end)


class RepairMortar(Concrete):
    name = "repair_mortar"

    def __init__(self, temp_c=21.1):
        r = P.REPAIR
        super().__init__(temp_c, tau0=r["tau0"], t_is=r["t_is"], bleed_mean=r["bleed"])
        self.rho = r["rho"]
        self.mu_bulk = self.mu_surf = r["mu"]

    def tau_surf(self, age, dens=0.0):
        r0, r1 = P.R_SURF
        return (0.5 + 0.5*np.asarray(dens, float))*self.tau_bulk(age)     # a neat mortar has no weak paste layer to speak of


class MasonryMortar:
    name = "mortar"
    bleeds = False

    def __init__(self, temp_c=21.1):
        m = P.MORTAR
        self.rho, self.mu_bulk, self.mu_surf = m["rho"], m["mu"], m["mu"]
        self.s = masco_factor(temp_c)

    def tau_bulk(self, age):
        m = P.MORTAR
        a = np.maximum(np.asarray(age, float), 0.0)/self.s
        return m["tau0"] + m["tau_ref"]*(a/m["t_ref"])**m["m"]

    def tau_surf(self, age, dens=0.0):
        # Region passes normalized surface consolidation here, not concrete D.
        return self.tau_bulk(age)*(1.0 + P.MORTAR_HARDEN*np.clip(dens, 0., 1.))

    def bleed_rate(self, age):
        return np.zeros_like(np.asarray(age, float))


def consolidate_mortar(state, pressure, distance, tau_bulk, interface):
    """Surface-fabric progress from confined plastic work, not volume deletion.

    Work/area = interface*tau_bulk*ds [J/m2]. The fraction p/(p+tau)
    represents compressive confinement; a lifted/unloaded tool gives no work.
    Characteristic work = tau_bulk*layer*strain [J/m2]. The exponential
    saturates as rearrangeable surface contacts are exhausted. Parameters are
    estimated and sensitivity-tested; this is not a measured porosity law.
    """
    tau = np.maximum(np.asarray(tau_bulk, float), 1.)
    p = max(float(pressure), 0.)
    dose = p/(p+tau)*np.asarray(interface)*max(float(distance), 0.)/(P.MORTAR_LAYER*P.MORTAR_STRAIN)
    return 1. - (1.-np.clip(state, 0., 1.))*np.exp(-dose)


class Sealant:
    name = "sealant"
    bleeds = False

    def __init__(self, temp_c=21.1, skin_s=None):
        s = P.SEALANT
        self.rho, self.mu_bulk, self.mu_surf = s["rho"], s["mu"], s["mu"]
        self.skin = float(skin_s if skin_s is not None else s["skin"])*masco_factor(temp_c)

    def tau_bulk(self, age):
        return np.full(np.shape(age), P.SEALANT["tau0"], float)

    def tau_surf(self, age, dens=0.0):
        return self.tau_bulk(age)*np.where(np.asarray(age) >= self.skin, P.SKIN_FACTOR, 1.)

    def bleed_rate(self, age):
        return np.zeros_like(np.asarray(age, float))


class Compound:
    name = "compound"
    bleeds = False

    def __init__(self, temp_c=21.1):
        c = P.COMPOUND
        self.rho, self.mu_bulk, self.mu_surf = c["rho"], c["mu"], c["mu"]

    def tau_bulk(self, age):
        return np.full(np.shape(age), P.COMPOUND["tau0"], float)

    def tau_surf(self, age, dens=0.0):
        return self.tau_bulk(age)

    def bleed_rate(self, age):
        return np.zeros_like(np.asarray(age, float))


MATERIALS = dict(concrete=Concrete, repair_mortar=RepairMortar, mortar=MasonryMortar, sealant=Sealant, compound=Compound)


def evaporation(concrete_c, air_c, rh, wind):
    """kg/m2/h: ACI 308's nomograph equation as printed in FHWA-RD-02-099 figure 6 (opened)."""
    return 4.88*(0.1113 + 0.04224*wind/0.447)*0.0443*(math.exp(0.0302*(concrete_c*1.8 + 32))
                                                    - rh/100.0*math.exp(0.0302*(air_c*1.8 + 32)))
