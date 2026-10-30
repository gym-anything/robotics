"""Drawing the rope handles (render only): each rod of the contact world is drawn as a tube of its own radius swept
along its simulated centre line, with a braided-rope texture coordinate along its length. The tube topology is fixed
when the scene is exported; every frame sends only the swept points and normals (render.py, video.py)."""
import numpy as np

RING = 8             # vertices around the tube


def tube_faces(n, m=RING):
    """Triangles of a tube with n rings of m vertices (open ends; the knots cover them)."""
    f = []
    for i in range(n - 1):
        for j in range(m):
            a, b = i*m + j, i*m + (j + 1) % m
            c, d = a + m, b + m
            f += [(a, c, b), (b, c, d)]
    return np.asarray(f, np.int32)


def tube_uv(n, m=RING, length=1., pitch=.006):
    """Texture coordinates: u around the tube, v along it in braid pitches."""
    u = np.tile(np.arange(m)/m, n)
    v = np.repeat(np.linspace(0., length/pitch, n), m)
    return np.c_[u, v].astype(np.float32)


def tube_points(P, radius, m=RING):
    """Points and outward normals of the tube around polyline P (n x 3), parallel-transport frames (no twist flips)."""
    P = np.asarray(P, float)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    a = np.array([0., 0., 1.]) if abs(T[0, 2]) < .9 else np.array([1., 0., 0.])
    N = np.cross(T[0], a); N /= np.linalg.norm(N)
    ang = 2*np.pi*np.arange(m)/m
    pts, nrm = np.zeros((n*m, 3)), np.zeros((n*m, 3))
    for i in range(n):
        if i:
            N = N - (N @ T[i])*T[i]
            nn = np.linalg.norm(N)
            N = N/nn if nn > 1e-9 else np.cross(T[i], a)/np.linalg.norm(np.cross(T[i], a))
        B = np.cross(T[i], N)
        ring = np.cos(ang)[:, None]*N + np.sin(ang)[:, None]*B
        pts[i*m:(i + 1)*m] = P[i] + radius*ring
        nrm[i*m:(i + 1)*m] = ring
    return pts.astype(np.float32), nrm.astype(np.float32)


def rope_polylines(ipc):
    """{rod index: centre-line points} of every rod in a contact world."""
    x = ipc.x_free()
    return {k: x[sl].copy() for k, sl in enumerate(getattr(ipc, "rod_slices", []))}
