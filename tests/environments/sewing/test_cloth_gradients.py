"""Finite-difference checks of the solver's constraint gradients. A flipped edge-vertex sign in the bending gradient
would give an anti-restoring bending force that amplifies any perturbation.
Run: python -m pytest tests/environments/sewing/test_cloth_gradients.py"""

import numpy as np
import warp as wp

from sewing import cloth as C


@wp.kernel
def _k_dihedral(xs: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d), th: wp.array(dtype=wp.float64)):
    r = C.dihedral(xs[0], xs[1], xs[2], xs[3])
    th[0] = r[0]
    out[0] = r[1]
    out[1] = r[2]
    out[2] = r[3]
    out[3] = r[4]


def _theta(X):
    xs = wp.array(X, dtype=wp.vec3d)
    out = wp.zeros(4, dtype=wp.vec3d)
    th = wp.zeros(1, dtype=wp.float64)
    wp.launch(_k_dihedral, 1, [xs, out, th])
    return float(th.numpy()[0]), out.numpy()


def test_dihedral_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    for _ in range(8):
        X = np.array([[0, 0, 0], [1, 0, 0], [0.4, 0.8, 0], [0.6, -0.7, 0]], float) + rng.normal(0, 0.2, (4, 3))
        t0, G = _theta(X)
        eps = 1e-7
        Gfd = np.zeros((4, 3))
        for i in range(4):
            for j in range(3):
                Xp, Xm = X.copy(), X.copy()
                Xp[i, j] += eps
                Xm[i, j] -= eps
                Gfd[i, j] = (_theta(Xp)[0] - _theta(Xm)[0])/(2*eps)
        assert np.abs(G - Gfd).max() < 1e-6*max(1.0, np.abs(Gfd).max()), (G, Gfd)
        assert np.abs(G.sum(0)).max() < 1e-9          # translation invariance


@wp.kernel
def _k_dihedral32(xs: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3), th: wp.array(dtype=float)):
    r = C.dihedral32(xs[1] - xs[0], xs[2] - xs[0], xs[3] - xs[0])
    th[0] = r[0]
    out[0] = r[1]
    out[1] = r[2]
    out[2] = r[3]
    out[3] = r[4]


def test_dihedral32_matches_float64():
    rng = np.random.default_rng(1)
    for _ in range(8):
        X = (np.array([[0, 0, 0], [1, 0, 0], [0.4, 0.8, 0], [0.6, -0.7, 0]], float) + rng.normal(0, 0.2, (4, 3)))*2.5e-3
        t64, G64 = _theta(X)
        xs = wp.array((X + 0.3).astype(np.float32), dtype=wp.vec3)
        out = wp.zeros(4, dtype=wp.vec3)
        th = wp.zeros(1, dtype=float)
        wp.launch(_k_dihedral32, 1, [xs, out, th])
        assert abs(float(th.numpy()[0]) - t64) < 2e-3
        assert np.abs(out.numpy() - G64).max() < 2e-3*np.abs(G64).max()


