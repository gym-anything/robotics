"""Finite-difference check of the cloth's compression law: the contact kernel's force, energy and stiffness must
agree with each other and with the KES points the law was fitted to.
Run: python -m pytest tests/environments/sewing/test_contact_gradients.py"""

import numpy as np
import warp as wp

from sewing import cloth as C

F64 = wp.float64


@wp.kernel
def _k_law(pen: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.vec3d), ca: wp.float64, k2: wp.float64,
           cr: wp.float64):
    i = wp.tid()
    f, e, h = C.contact_law(pen[i], ca, k2, cr, wp.float64(0.0))
    out[i] = wp.vec3d(f, e, h)


def test_compression_law_is_c1_and_consistent():
    """The KES exponential law: force = dE/dpen and stiffness = dF/dpen everywhere (exponential range,
    the switch to the stiff penalty, beyond it); force and stiffness continuous at the switch; it reproduces the KES
    points it was fitted to (Tm at 50 gf/cm2, the compression work WC through LC) for muslin."""
    from sewing import params as P
    f = P.FABRICS["muslin_125"]
    a, b = C.compression_law(f["thickness_free"], f["thickness"], f["comp_LC"])
    D = f["thickness_free"] - f["thickness"]
    assert abs(a*np.expm1(b*D) - P.KES_PM) < 1e-6*P.KES_PM                      # Tm at Pm
    W = a*(np.expm1(b*D)/b - D)                                                  # compression work (J/m2)
    assert abs(W/(P.KES_PM*D/2) - f["comp_LC"]) < 1e-6                           # LC = WC / (Pm D / 2)
    assert abs(W - 0.181*9.807e-3*1e-2/1e-4) < 2e-3*W                            # WC 0.181 gf cm/cm2 (Ramaiah)
    A, k2 = 5.4e-6, 2.7e5                                                         # a fine-mesh vertex
    ca, cr = a*A, 2*b                                                             # one surface on a rigid face
    ps = np.log(k2/(ca*cr))/cr
    pen = np.linspace(-0.02e-3, ps + 0.05e-3, 4001)
    out = wp.zeros(len(pen), dtype=wp.vec3d)
    wp.launch(_k_law, len(pen), [wp.array(pen, dtype=wp.float64), out, F64(ca), F64(k2), F64(cr)])
    r = out.numpy()
    dE = np.gradient(r[:, 1], pen)
    dF = np.gradient(r[:, 0], pen)
    inner = slice(2, -2)
    assert np.abs(dE - r[:, 0])[inner].max() < 2e-3*r[:, 0].max()
    assert np.abs(dF - r[:, 2])[inner][pen[inner] > 1e-6].max() < 2e-2*k2
    j = np.argmin(np.abs(pen - ps))
    assert abs(r[j + 1, 0] - r[j - 1, 0]) < 3*k2*(pen[1] - pen[0])               # force continuous at the switch
    assert abs(r[j + 2, 2] - k2) < 1e-6*k2 and abs(r[j - 2, 2] - k2) < 0.01*k2   # stiffness continuous there
    # no compressible surface: the stiff penalty alone
    wp.launch(_k_law, len(pen), [wp.array(pen, dtype=wp.float64), out, F64(0.0), F64(k2), F64(0.0)])
    r0 = out.numpy()
    assert np.allclose(r0[:, 0], k2*np.maximum(pen, 0.0))


