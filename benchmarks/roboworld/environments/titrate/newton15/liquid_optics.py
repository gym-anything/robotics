"""Local indicator optics from the transported chemical inventory.

The optical volume carries the transported concentration throughout the liquid,
not a flask-wide colour or a boundary-only estimate. The surface primvar is
retained for diagnostics, but is not the material's absorption input.
"""

import numpy as np
import warp as wp
from .chemistry import equilibrium_ph, INDICATOR_PKA
from .runtime import require_newton

# Mueschke et al., UCRL-JRNL-237116, p.10, Eq.2.14: ln(I0/I), not log10.
# https://www.osti.gov/servlets/purl/965470
GREEN_NAPERIAN_M_INV_MOLAR_INV = 2.935e4*100.


@wp.kernel
def sample_pink(grid: wp.uint64, q: wp.array[wp.vec3], concentration: wp.array[float],
                vertices: wp.array[wp.vec3], h: float, values: wp.array[float]):
    i = wp.tid()
    total, weight = float(0.), float(0.)
    for j in wp.hash_grid_query(grid, vertices[i], h):
        r = wp.length(vertices[i]-q[j])/h
        if r < 1.:
            a = 1.-r
            w = a*a*a*a*(1.+4.*r)
            total += w*concentration[j]
            weight += w
    values[i] = total/weight if weight > 0. else 0.


def apply(surface, positions, moles, parcel_volume_m3, support_m, device="cuda:0"):
    require_newton()
    n = np.asarray(moles)
    q = np.asarray(positions, np.float32)
    if n.shape != (len(q), 5) or not np.isfinite(n).all() or (n < 0).any():
        raise ValueError("Finite nonnegative local five-species inventory required")
    ph = equilibrium_ph((n[:, 0]-n[:, 1])/(1000*parcel_volume_m3))
    pink = n[:, 4]/(1000*parcel_volume_m3)/(1.+10.**(INDICATOR_PKA-ph))
    points = wp.array(q, dtype=wp.vec3, device=device)
    grid = wp.HashGrid(64, 64, 64, device=device)
    grid.build(points, support_m)
    vertices = wp.array(surface.vertices, dtype=wp.vec3, device=device)
    concentration = wp.array(pink.astype(np.float32), dtype=float, device=device)
    values = wp.zeros(len(vertices), dtype=float, device=device)
    wp.launch(sample_pink, dim=len(vertices), inputs=[grid.id, points, concentration,
        vertices, support_m], outputs=[values], device=device)
    surface.pink_molar = values.numpy()
    from .volume_optics import sample_field
    surface.pink_volume = sample_field(q, pink, support_m, parcel_volume_m3**(1/3), device)
    return surface


def bind(stage, material, asset_folder=None):
    from pathlib import Path
    import hashlib
    import shutil
    from pxr import Sdf, UsdShade
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendPath("indicator_volume"))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    source = Path(__file__).with_name("TitrationWater.mdl").resolve()
    destination = (Path(stage.GetRootLayer().realPath).parent if asset_folder is None else asset_folder)/source.name
    if destination.exists() and hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
        raise ValueError("Export folder contains a different optical material source")
    if not destination.exists():
        shutil.copyfile(source, destination)
    shader.SetSourceAsset(Sdf.AssetPath(str(destination.resolve())), "mdl")
    shader.SetSourceAssetSubIdentifier("TitrationWater", "mdl")
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")


def set_primvar(mesh, values):
    from pxr import Sdf, UsdGeom, Vt
    values = np.zeros(len(mesh.GetPointsAttr().Get()), np.float32) if values is None else np.asarray(values, np.float32)
    if values.shape != (len(mesh.GetPointsAttr().Get()),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid local indicator optical field")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("pink_molar", Sdf.ValueTypeNames.FloatArray,
        UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(values))
