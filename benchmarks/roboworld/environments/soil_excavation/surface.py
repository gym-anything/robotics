"""Render surface of the soil particles (visual only; physics never reads it).

wine_pouring/surface.py's recipe (Zhu & Bridson 2005 implicit surface over the particle centres, Warp GPU marching
cubes, Taubin smoothing, area-weighted normals), with soil-specific centre smoothing (Yu & Turk, lambda 0.5: a soil
surface keeps its lumps; wine used 0.9 for a liquid), then per-vertex attributes interpolated from the nearest
particles (k = 8 within 2.5 h, weights (1 - d/R)^2):

  color   linear RGB: the particle's material colour (params.MATERIALS, moist), a per-particle clod variation (+-10 %
          lightness from a hash of the particle id), the surface crust of particles that started within 15 mm of grade
          (+12 %, drier), moisture (dry +15 %, wet -15 %) and paint (white / purple marks sprayed on the top layer);
          the hardpan's crust is cracked (crack_factor: a Voronoi pattern on the bank positions);
  turf, mulch  material shares that blend the lawn and mulch textures in the soil material (soil_surfaces.mdl).

Everything follows the recorded particle state: dug soil carries its paint and turf onto the tarp.
"""

import numpy as np
from . import params as P
from . import materials as M

PAINT = {1: (0.92, 0.92, 0.90), 2: (0.45, 0.18, 0.62)}           # white, APWA purple (sRGB)


def srgb_to_linear(c):
    c = np.asarray(c, float)/255.0 if np.max(c) > 1.0 else np.asarray(c, float)
    return np.where(c <= 0.04045, c/12.92, ((c + 0.055)/1.055)**2.4)


MAT_LINEAR = np.array([srgb_to_linear(P.MATERIALS[n]["color"]) for n in M.NAMES])
TURF, MULCH, HARDPAN = M.NAMES.index("turf"), M.NAMES.index("mulch"), M.NAMES.index("hardpan")
# The 40 mm turf material includes roots and brown earth. Grass is only its thin
# original surface, not a green texture throughout every excavated root-mat clod.
MAT_LINEAR[TURF] = MAT_LINEAR[M.NAMES.index("topsoil")]


def crack_factor(xy, cell=0.12, width=0.0025, seed=7):
    """Desiccation cracks of a dry clay crust (render only): a Voronoi pattern of polygons about `cell` across (dry
    compacted clay paths crack into 5-15 cm plates, cracks a few mm wide) whose edges are darkened over `width` on each
    side (the soil surface's vertices are ~5 mm apart). xy are bank positions, so the cracks move with the soil a tool
    breaks up."""
    xy = np.asarray(xy, float)
    g = np.floor(xy/cell).astype(np.int64)
    best1 = np.full(len(xy), np.inf); best2 = np.full(len(xy), np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            c = g + (dx, dy)
            h = (c[:, 0]*73856093 ^ c[:, 1]*19349663 ^ seed) & 0xFFFFFF
            jx = ((h*2654435761) % 1000003)/1000003.0
            jy = ((h*40503) % 999983)/999983.0
            sx, sy = (c[:, 0] + jx)*cell, (c[:, 1] + jy)*cell
            d = np.hypot(xy[:, 0] - sx, xy[:, 1] - sy)
            best2 = np.where(d < best1, best1, np.minimum(best2, d))
            best1 = np.minimum(best1, d)
    edge = 0.5*(best2 - best1)                       # distance to the bisector between the two nearest seeds
    return np.where(edge < width, 0.45, np.where(edge < 1.8*width, 0.85, 1.0))


def particle_colors(mat, pos0, grade, paint=None, moisture="moist"):
    n = len(mat)
    c = MAT_LINEAR[mat].copy()
    idx = np.arange(n, dtype=np.uint64)
    hsh = ((idx*np.uint64(2654435761)) % np.uint64(1000003)).astype(float)/1000003.0
    c *= (0.90 + 0.20*hsh)[:, None]
    crust = pos0[:, 2] > grade - 0.015
    c[crust] *= 1.12
    c *= dict(dry=1.15, moist=1.0, wet=0.85)[moisture]
    if paint is not None:
        for k, col in PAINT.items():
            sel = paint == k
            c[sel] = srgb_to_linear(col)
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def groups(pos, block, pad=0.06):
    """Particle groups reconstructed separately: those over the block footprint (+pad) and the rest (spoil on the tarp,
    spill), so one marching-cubes box does not span the gap between them (wine_pouring's 5 cm cluster pass cost 0.5 s
    on 0.65 M particles)."""
    lo, hi = np.asarray(block["lo"][:2]) - pad, np.asarray(block["hi"][:2]) + pad
    inb = (pos[:, :2] >= lo).all(1) & (pos[:, :2] <= hi).all(1)
    return [np.flatnonzero(inb), np.flatnonzero(~inb)]


def reconstruct_soil(points, h, device='cuda:0', cell=None, radius_factor=(3/(4*np.pi))**(1/3)):
    """Zhu–Bridson reconstruction with an equal-reference-volume grain radius.

    One soil particle carries rho*h**3 mass. The inherited liquid surface used
    radius .55*h (a sphere of only .697*h**3), leaving perforations on separated
    clods. The sphere here has volume h**3. This is a visual support radius, not
    an alteration of particle mass or a fill of arbitrary empty cavities.
    """
    import warp as wp
    from wine_pouring import surface as WS
    pts=np.asarray(points,np.float32).reshape(-1,3)
    if not len(pts):return np.zeros((0,3),np.float32),np.zeros((0,3),np.int32)
    radius=2*h
    lo,hi=pts.min(0)-2*radius,pts.max(0)+2*radius
    cell=max(cell or .5*h,float((np.prod(hi-lo)/WS.MAX_CELLS)**(1/3)))
    n=np.maximum(np.ceil((hi-lo)/cell).astype(int)+1,3)
    with wp.ScopedDevice(device):
        xa=wp.array(pts,dtype=wp.vec3);grid=wp.HashGrid(128,128,128);grid.build(xa,radius)
        phi=wp.zeros(tuple(map(int,n)),dtype=float)
        wp.launch(WS.k_zhu_bridson,dim=tuple(map(int,n)),
                  inputs=[grid.id,xa,wp.vec3(*lo),float(cell),*map(int,n),float(radius),float(radius_factor*h)],outputs=[phi])
        mc=wp.MarchingCubes(*map(int,n),device=device);mc.surface(phi,0.)
        v=(mc.verts.numpy()*cell+lo).astype(np.float32)
        t=mc.indices.numpy().reshape(-1,3)
    if len(t) and np.einsum('ij,ij->i',v[t[:,0]],np.cross(v[t[:,1]],v[t[:,2]])).sum()<0:t=t[:,::-1]
    return v,np.ascontiguousarray(t,np.int32)


def soil_mesh(parts, h, grade, moisture="moist", cell=None, device="cuda:0", lam=0.5, block=None):
    """(vertices, triangles, normals, color, turf, mulch, st) of the soil surface from soil.particles() output."""
    import warp as wp
    from scipy.spatial import cKDTree
    from wine_pouring import surface as WS
    pos = np.asarray(parts["pos"], np.float32)
    keep = np.isfinite(pos).all(1)
    pos = pos[keep]
    mat = np.asarray(parts["mat"])[keep]
    sm = WS.smooth_centres(pos, h, device, lam=lam)
    vs, ts, off = [], [], 0
    for ids in (groups(sm, block) if block is not None else WS.clusters(sm, size=0.05)):
        if len(ids) < 4:
            continue
        v, t = reconstruct_soil(sm[ids], h, device, cell)
        if len(t):
            vs.append(v); ts.append(t + off); off += len(v)
    if not vs:
        z = np.zeros((0, 3), np.float32)
        return dict(points=z, faces=np.zeros((0, 3), np.int32), normals=z, color=z, turf=np.zeros(0, np.float32),
                    mulch=np.zeros(0, np.float32), st=np.zeros((0, 2), np.float32))
    v, t = np.concatenate(vs), np.ascontiguousarray(np.concatenate(ts), np.int32)
    v = WS.taubin(v, t)
    nrm = WS.vertex_normals(v, t)
    cols = particle_colors(mat, np.asarray(parts["pos0"])[keep], grade, None if parts.get("paint") is None else
                           np.asarray(parts["paint"])[keep], moisture)
    tree = cKDTree(pos)
    R = 2.5*h
    d, j = tree.query(v, k=8, distance_upper_bound=R)
    w = np.where(np.isfinite(d), (1.0 - np.minimum(d, R)/R)**2, 0.0)
    j = np.where(np.isfinite(d), j, 0)
    ws = w.sum(1, keepdims=True) + 1e-9
    color = (w[..., None]*cols[j]).sum(1)/ws
    # marking paint covers the mulch / turf texture (the first renders showed no mark on the mulch: the texture
    # blend replaced the painted particles' white)
    unpainted = np.ones(len(mat), bool) if parts.get("paint") is None else (np.asarray(parts["paint"])[keep] == 0)
    pos0 = np.asarray(parts["pos0"], float)[keep]
    grass_skin = (mat == TURF) & (pos0[:, 2] > grade-0.01) & unpainted
    turf = (w*grass_skin[j]).sum(1)/ws[:, 0]
    mulch = (w*((mat[j] == MULCH) & unpainted[j])).sum(1)/ws[:, 0]
    # the hardpan's dry crust (its top 2 cm in the bank) is cracked
    hard = (w*(mat[j] == HARDPAN)).sum(1)/ws[:, 0]
    p0v = (w[..., None]*pos0[j]).sum(1)/ws
    crust = (hard > 0.5) & (p0v[:, 2] > grade - 0.02)
    if crust.any():
        color[crust] *= crack_factor(p0v[crust, :2])[:, None]
    return dict(points=v.astype(np.float32), faces=t, normals=nrm, color=color.astype(np.float32),
                turf=turf.astype(np.float32), mulch=mulch.astype(np.float32), st=v[:, :2].astype(np.float32))
