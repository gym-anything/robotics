"""Interior optical concentration: particles -> 3D field -> lossless asset.

Only rendering is sampled. No chemistry update or physics step occurs here.
The volumetric renderer samples world metres, not a surface texture or overlay.
The heterogeneous medium is attached to the original refractive boundary.
"""

import math
import numpy as np
import warp as wp

from .render_payload import encode, decode, MAX_ARRAY_BYTES


@wp.kernel
def sample_interior(grid: wp.uint64, q: wp.array[wp.vec3], c: wp.array[float],
                    origin: wp.vec3, spacing: float, nx: int, ny: int,
                    h: float, out: wp.array[float]):
    i = wp.tid()
    x = i % nx
    y = (i // nx) % ny
    z = i // (nx * ny)
    p = origin + spacing * wp.vec3(float(x), float(y), float(z))
    total, weight = float(0.), float(0.)
    for j in wp.hash_grid_query(grid, p, h):
        r = wp.length(p-q[j])/h
        if r < 1.:
            a = 1.-r
            w = a*a*a*a*(1.+4.*r)
            total += w*c[j]
            weight += w
    # The renderer clips this medium at the actual liquid mesh. Thresholding
    # density here would erode the boundary twice and erase isolated-droplet
    # glyphs (whose optical concentration must not be discarded).
    out[i] = total/weight if weight > 0. else 0.


def sample_field(positions, concentration, support_m, spacing_m, device):
    q = np.asarray(positions, np.float32)
    c = np.asarray(concentration, np.float32)
    if (q.ndim != 2 or q.shape[1] != 3 or c.shape != (len(q),)
            or not np.isfinite(q).all() or not np.isfinite(c).all() or (c < 0).any()
            or not np.isfinite([support_m, spacing_m]).all()
            or min(support_m, spacing_m) <= 0):
        raise ValueError('Invalid interior optical inputs')
    active = q[c > 0]
    if len(active):
        lower = np.floor((active.min(0)-support_m)/spacing_m)*spacing_m
        upper = np.ceil((active.max(0)+support_m)/spacing_m)*spacing_m
        counts = np.rint((upper-lower)/spacing_m).astype(np.int64)+1
    else:
        lower, counts = np.zeros(3), np.full(3, 2)
    size = math.prod(int(n) for n in counts)
    if size*4 > MAX_ARRAY_BYTES:
        raise ValueError('Interior optical field exceeds 32 MiB; refusing boundary-only fallback')
    points = wp.array(q, dtype=wp.vec3, device=device)
    grid = wp.HashGrid(64, 64, 64, device=device)
    grid.build(points, support_m)
    values = wp.zeros(size, dtype=float, device=device)
    wp.launch(sample_interior, dim=size, inputs=[grid.id, points,
        wp.array(c, dtype=float, device=device), wp.vec3(*lower), float(spacing_m),
        int(counts[0]), int(counts[1]), float(support_m)], outputs=[values], device=device)
    return dict(origin_m=lower.tolist(), spacing_m=float(spacing_m),
                counts_xyz=counts.tolist(), values=values.numpy())


def pack(field):
    return {**field, 'values': encode(field['values'], 'float32')}


def unpack(payload):
    if not isinstance(payload, dict) or set(payload) != {'origin_m', 'spacing_m', 'counts_xyz', 'values'}:
        raise ValueError('Invalid optical volume schema')
    origin, spacing, counts = payload['origin_m'], payload['spacing_m'], payload['counts_xyz']
    if (not isinstance(origin, list) or len(origin) != 3 or not np.isfinite(origin).all()
            or type(spacing) not in (int, float) or not math.isfinite(spacing) or spacing <= 0
            or not isinstance(counts, list) or len(counts) != 3
            or any(type(n) is not int or n < 2 for n in counts)):
        raise ValueError('Invalid optical volume coordinates')
    values = decode(payload['values'], 'float32')
    if len(values) != math.prod(counts) or (values < 0).any():
        raise ValueError('Invalid optical volume samples')
    return {**payload, 'values': values}


def clear_field():
    return dict(origin_m=[0., 0., 0.], spacing_m=.001, counts_xyz=[2, 2, 2],
                values=np.zeros(8, np.float32))


def install(stage, field, folder):
    """Archive the full field as an asset; no misleading surface-colour fallback."""
    import hashlib
    import json
    import os
    from pathlib import Path
    from pxr import Sdf, UsdGeom
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError('Titration optical coordinates require a metre-unit stage')
    field = unpack(pack(field))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    description = json.dumps({k:v for k,v in field.items() if k != 'values'},sort_keys=True)
    digest = hashlib.sha256(description.encode()+field['values'].tobytes()).hexdigest()
    path = folder/(digest+'.npz')
    if not path.exists():
        np.savez_compressed(path, **field)
    prim = stage.DefinePrim('/Optics', 'Scope')
    relative = os.path.relpath(path, Path(stage.GetRootLayer().realPath).parent).replace('\\', '/')
    prim.CreateAttribute('titration:concentration',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(relative))
    prim.CreateAttribute('titration:requiredRenderer',Sdf.ValueTypeNames.String).Set('Mitsuba 3.8.0 volpathmis')
    prim.CreateAttribute('titration:fieldSha256',Sdf.ValueTypeNames.String).Set(digest)
    return dict(sha256=digest,counts_xyz=field['counts_xyz'],spacing_m=field['spacing_m'],
                origin_m=field['origin_m'],maximum_molar=float(field['values'].max()),
                representation='world-space float32 concentration; heterogeneous RGB absorption')


def read_field(stage):
    import hashlib
    import json
    prim = stage.GetPrimAtPath('/Optics')
    if not prim:
        return None
    asset = prim.GetAttribute('titration:concentration').Get()
    if not asset or not asset.resolvedPath:
        raise ValueError('Missing archived 3D concentration asset')
    with np.load(asset.resolvedPath,allow_pickle=False) as saved:
        field = dict(origin_m=saved['origin_m'].tolist(),spacing_m=float(saved['spacing_m']),
                     counts_xyz=saved['counts_xyz'].tolist(),values=saved['values'].copy())
    field = unpack(pack(field))
    description = json.dumps({k:v for k,v in field.items() if k != 'values'},sort_keys=True)
    digest = hashlib.sha256(description.encode()+field['values'].tobytes()).hexdigest()
    if digest != prim.GetAttribute('titration:fieldSha256').Get():
        raise ValueError('Archived concentration hash mismatch')
    return field
