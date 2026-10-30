"""RGB Beer-Lambert transport through the reconstructed liquid interior.

Mitsuba's heterogeneous medium is attached to the dielectric liquid boundary.
This is render-only: it consumes the optical field, never updates chemistry.
"""
import numpy as np


def runtime():
    import mitsuba as mi
    if mi.__version__ != '3.8.0':
        raise RuntimeError('Volume transport requires mitsuba==3.8.0')
    mi.set_variant('cuda_ad_rgb')
    return mi


def medium(field):
    from .liquid_optics import GREEN_NAPERIAN_M_INV_MOLAR_INV
    mi = runtime()
    counts = np.asarray(field['counts_xyz'])
    spacing = field['spacing_m']
    # Grid samples are voxel centres; normalized texture coordinates include
    # half a cell beyond the first/last centre (matching the GPU interpolator).
    transform = mi.ScalarTransform4f().translate(np.asarray(field['origin_m'])-.5*spacing)
    transform = transform.scale(counts*spacing)
    concentration = field['values'].reshape(tuple(counts[::-1]))
    sigma = np.zeros((*concentration.shape, 3), np.float32)
    sigma[..., 1] = concentration*GREEN_NAPERIAN_M_INV_MOLAR_INV
    return mi.load_dict(dict(type='heterogeneous', albedo=0.,
        sigma_t=dict(type='gridvolume', grid=mi.VolumeGrid(sigma), raw=True,
                     accel=False, to_world=transform), sample_emitters=False))
