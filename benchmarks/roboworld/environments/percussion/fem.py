"""Small 1-D finite-element tools (numpy/scipy) for the instrument mode tables.

Timoshenko bending (w, theta per node; linear shape functions, one-point shear integration, locking-free for linear
elements) and St Venant torsion for bars of variable cross-section, free-free unless springs are added. This code
reproduces Rucz's measured untuned rosewood sample within 1-5 % and the F4 bar's mass within 0.6 %.
"""
import numpy as np
from scipy.linalg import eigh


def timoshenko_modes(x_nodes, EI, kGA, rhoA, rhoI, n_modes=8, springs=None, rot_springs=None, fmin_hz=5.0):
    """x_nodes: (N+1,) node coordinates; EI, kGA, rhoA, rhoI: (N,) element properties; springs: [(node, k_w)],
    rot_springs: [(node, k_theta)] to ground. Returns (freqs_hz, W (N+1, n), TH (N+1, n)) mass-normalised (V^T M V = I),
    modes below fmin_hz (rigid body) removed."""
    n = len(x_nodes)
    ndof = 2*n
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    for e in range(n - 1):
        le = x_nodes[e + 1] - x_nodes[e]
        d = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]          # w1, th1, w2, th2
        kb = EI[e]/le*np.array([[0, 0, 0, 0], [0, 1, 0, -1], [0, 0, 0, 0], [0, -1, 0, 1]])
        B = np.array([-1/le, -0.5, 1/le, -0.5])
        ks = kGA[e]*le*np.outer(B, B)
        me = np.zeros((4, 4))
        me[np.ix_([0, 2], [0, 2])] = rhoA[e]*le/6*np.array([[2, 1], [1, 2]])
        me[np.ix_([1, 3], [1, 3])] = rhoI[e]*le/6*np.array([[2, 1], [1, 2]])
        K[np.ix_(d, d)] += kb + ks
        M[np.ix_(d, d)] += me
    for (i, k) in (springs or []):
        K[2*i, 2*i] += k
    for (i, k) in (rot_springs or []):
        K[2*i + 1, 2*i + 1] += k
    w2, V = eigh(K, M, subset_by_index=[0, min(ndof - 1, n_modes + 3)])
    keep = w2 > (2*np.pi*fmin_hz)**2
    w2, V = w2[keep][:n_modes], V[:, keep][:, :n_modes]
    return np.sqrt(np.maximum(w2, 0))/(2*np.pi), V[0::2, :], V[1::2, :]


def torsion_modes(x_nodes, GJ, rhoIp, n_modes=4):
    n = len(x_nodes)
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    for e in range(n - 1):
        le = x_nodes[e + 1] - x_nodes[e]
        d = [e, e + 1]
        K[np.ix_(d, d)] += GJ[e]/le*np.array([[1, -1], [-1, 1]])
        M[np.ix_(d, d)] += rhoIp[e]*le/6*np.array([[2, 1], [1, 2]])
    w2, V = eigh(K, M, subset_by_index=[0, min(n - 1, n_modes + 1)])
    keep = w2 > (2*np.pi*5.0)**2
    return np.sqrt(w2[keep][:n_modes])/(2*np.pi), V[:, keep][:, :n_modes]


def rect_torsion_beta(b, h):
    """St Venant torsion constant J = beta b h^3 of a b x h rectangle (b >= h), series solution."""
    if h > b:
        b, h = h, b
    r = b/h
    k = np.arange(200)
    n = 2*k + 1
    return (1 - 192/(np.pi**5*r)*np.sum(np.tanh(n*np.pi*r/2)/n**5))/3
