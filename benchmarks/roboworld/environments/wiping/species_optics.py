"""Reduced thin-coating appearance from the existing species mass ledger.

Oil is a weak, spectrally selective absorber, not the same brown pigment as
food solids. These optical coefficients are authored estimates for generic
kitchen residue, not measurements of a named sauce/oil. Beer attenuation form:
https://pbr-book.org/4ed/Volume_Scattering/Transmittance
Edible-oil spectral dependence: https://doi.org/10.1364/AO.54.003886
This is a substrate/coating BRDF approximation, not volumetric spectral optics.
No changes to masses, film heights, normal geometry, contact or cleaning score.
"""
import numpy as np

def properties(mass,area,substrate,*,layered=False):
    mass=np.asarray(mass,dtype=float)
    if np.any(mass<0) or area<=0:raise ValueError('Invalid physical coating mass')
    density=mass/area
    water,food,oil=np.moveaxis(density,-1,0)
    food_coverage=-np.expm1(-food/.015)
    # At10mm path, transmission(R,G,B)~(.923,.779,.407); a micrometre-scale
    # oil film is mostly clear. These are estimates, not fitted spectra.
    oil_transmission=np.exp(-(oil/920.)[...,None]*np.array([8.,25.,90.]))
    liquid_height=water/1000.+oil/920.
    wet=-np.expm1(-liquid_height/.000025)
    rgb,rough,metal=substrate
    base=np.asarray(rgb)*oil_transmission
    color=base*(1-food_coverage[...,None])+np.array([.27,.045,.009])*food_coverage[...,None]
    # A separate dielectric layer must not also turn the substrate into a
    # smooth wet metal. Keep the old effective-BRDF edition exactly available.
    bare_rough=np.full_like(wet,rough) if layered else rough*(1-wet)+.055*wet
    food_rough=np.full_like(wet,.55) if layered else .55*(1-wet)+.25*wet
    roughness=bare_rough*(1-food_coverage)+food_rough*food_coverage
    metallic=metal*(1-food_coverage)
    return color,roughness,metallic


def coating(mass,area):
    """Reduced water/oil dielectric layer, not a new physical reservoir.

    25micrometre coverage scale and0.95 maximum weight are optical estimates.
    The renderer's coat IOR is uniform, set separately to the documented
    estimate1.40 for mixed residue, not a claim of a measured spectrum.
    """
    mass=np.asarray(mass,dtype=float)
    if np.any(mass<0) or not np.isfinite(mass).all() or area<=0:
        raise ValueError('Invalid physical coating mass')
    water=mass[...,0]/(area*1000.);oil=mass[...,2]/(area*920.)
    height=water+oil
    weight=.95*(-np.expm1(-height/.000025))
    return weight
