"""Liquid inventory: where every particle is, in mL per liquid, from the simulated state only.

Classes: inside container c (its exact interior polygon, body frame, below the rim plane); wetting
the outside of container c (within 3 mm of its outer surface: the drip/dribble film); on the bar top; on the floor;
elsewhere (in flight, on the robot). Particle volume is dx^3 (liquid.py fills dx^3 per particle and the settled level
matches the geometric level).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P

OUTSIDE_FILM_M = 0.003
_MASKS = {}


def interior_mask(geo, res=0.5e-3):
    if geo.kind not in _MASKS:
        rmax = float(geo.interior[:, 0].max()) + res
        z0, z1 = float(geo.interior[:, 1].min()), float(geo.rim_z)
        nr, nz = int(rmax/res) + 2, int((z1 - z0)/res) + 2
        R, Z = np.meshgrid((np.arange(nr) + 0.5)*res, z0 + (np.arange(nz) + 0.5)*res, indexing="ij")
        m = geo.interior_contains(R.ravel(), Z.ravel()).reshape(nr, nz)
        _MASKS[geo.kind] = (m, res, z0, nr, nz)
    return _MASKS[geo.kind]


def classify(liquid, world):
    """Per-particle labels: container index (>= 0), or -1 bar, -2 floor, -3 elsewhere; plus outside-film container."""
    x, v, lid = liquid.read()
    n = len(x)
    label = np.full(n, -3, np.int32)
    film = np.full(n, -1, np.int32)
    q = world.state.body_q.numpy()
    conts = world.scene.containers
    # a bottle stands inside its coaster dish: the liquid in the bottle's base is inside both interiors and belongs to
    # the bottle, so coasters are classified last (5.9 mL of a bottle's wine counted as 'in the coaster' otherwise)
    order = sorted(range(len(conts)), key=lambda k: conts[k][1].kind == "coaster")
    for ci in order:
        body, geo, name = conts[ci]
        R = Rotation.from_quat(q[body, 3:])
        pl = R.inv().apply(x - q[body, :3])
        r, z = np.hypot(pl[:, 0], pl[:, 1]), pl[:, 2]
        m, res, z0, nr, nz = interior_mask(geo)
        i = (r/res).astype(np.int64)
        j = ((z - z0)/res).astype(np.int64)
        ok = (i >= 0) & (i < nr) & (j >= 0) & (j < nz)
        inside = np.zeros(n, bool)
        inside[ok] = m[i[ok], j[ok]]
        label[inside & (label == -3)] = ci
        near = (~inside) & (r < float(geo.solid[:, 0].max()) + OUTSIDE_FILM_M + 0.002) & \
            (z > float(geo.solid[:, 1].min()) - OUTSIDE_FILM_M) & (z < geo.rim_z + OUTSIDE_FILM_M)
        if near.any():
            d = liquid._sdf_eval(geo, r[near], z[near])
            idx = np.flatnonzero(near)[d < OUTSIDE_FILM_M]
            film[idx] = np.where(film[idx] < 0, ci, film[idx])
    from .scene import BAR_Y0, BAR_Y1
    bar = (label == -3) & (x[:, 2] < P.BAR_TOP_Z_M + 0.006) & (x[:, 2] > P.BAR_TOP_Z_M - 0.01) & \
        (np.abs(x[:, 0]) < 0.5*P.BAR_LENGTH_M) & (x[:, 1] > BAR_Y0) & (x[:, 1] < BAR_Y1)
    label[bar] = -1
    label[(label == -3) & (x[:, 2] < 0.02)] = -2
    return dict(x=x, v=v, lid=lid, label=label, film=film)


def summarize(liquid, world, c=None):
    """Volumes (mL) per place and liquid; mean/max speed; centre of each container's liquid."""
    c = c or classify(liquid, world)
    V = liquid.V*1e6
    names = list(P.LIQUIDS)
    out = dict(containers={}, bar={}, floor={}, elsewhere={}, film={}, total_mL=float(len(c["x"])*V))
    lid, label = c["lid"], c["label"]
    for ci, (body, geo, name) in enumerate(world.scene.containers):
        sel = label == ci
        out["containers"][name] = {names[k]: float(np.sum(sel & (lid == k))*V) for k in range(len(names)) if np.any(sel & (lid == k))}
        out["containers"][name]["total"] = float(sel.sum()*V)
        fsel = (c["film"] == ci) & (label < 0)
        out["film"][name] = float(fsel.sum()*V)
    for key, code in (("bar", -1), ("floor", -2), ("elsewhere", -3)):
        sel = label == code
        out[key] = {names[k]: float(np.sum(sel & (lid == k))*V) for k in range(len(names)) if np.any(sel & (lid == k))}
        out[key]["total"] = float(sel.sum()*V)
    sp = np.linalg.norm(c["v"], axis=1) if len(c["v"]) else np.zeros(1)
    out["speed_mean_m_s"] = float(sp.mean()) if len(sp) else 0.
    out["speed_max_m_s"] = float(sp.max()) if len(sp) else 0.
    out["speed_mean_by_container_m_s"] = {name: float(sp[label == ci].mean()) for ci, (_, _, name) in
                                          enumerate(world.scene.containers) if np.any(label == ci)}
    out["speed_mean_outside_m_s"] = float(sp[label < 0].mean()) if np.any(label < 0) else 0.
    out["speed_max_by_container_m_s"] = {name: float(sp[label == ci].max()) for ci, (_, _, name) in
                                         enumerate(world.scene.containers) if np.any(label == ci)}
    air = (label == -3) & (c["x"][:, 2] > P.BAR_TOP_Z_M + 0.02)        # in flight: clear of every surface it can rest on
    out["speed_max_airborne_m_s"] = float(sp[air].max()) if np.any(air) else 0.
    return out
