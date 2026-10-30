"""Render-only Isaac worker for filleting: the shared `wiping.render_worker` with one state extension.

The shared worker owns the process lifecycle, the authenticated socket, the RTX path-tracing settings, the
warm-up and the capture; it applies recorded rigid poses (`body_poses` -> /Lab/body_i). This module adds, inside
the same `apply_recorded_state` call (so its cost is inside the worker's `usd_update_s` timing):

  paged     {root: fields}  tissue / bone-rod surfaces whose triangle count changes (cutting). Each root
            (/Lab/tissue/<kind>, /Lab/rods/<group>) holds fixed-size pages of `page_triangles` unindexed triangles
            (3 vertices per triangle; faceVertexIndices = 0..3T-1, never rewritten). A frame writes points,
            normals and primvars (st, displayColor, myophase, restpos, aux) of every page; slots past the frame's triangle count
            collapse onto one point (zero-area) and empty pages are made invisible. Pages are created on demand and
            never deleted, so Hydra only ever sees point/primvar updates, not topology changes.
  reauthor  {mesh: fields}  the alternative: one mesh per kind whose
            faceVertexCounts/Indices are rewritten every frame.
  textures  {shader input attribute: png}  e.g. the bench scale's display reading.

No physics is stepped here (the shared worker refuses a stage with a physics scene and reports physics_steps=0).
"""
import math
import os
from pathlib import Path

import numpy as np

FIELDS = {"points": 3, "normals": 3, "st": 2, "color": 3, "myophase": 1, "restpos": 3, "aux": 3}
PRIMVAR = {"st": "primvars:st", "color": "primvars:displayColor", "myophase": "primvars:myophase",
           "restpos": "primvars:restpos", "aux": "primvars:aux"}
FLOAT3 = ("restpos", "aux")              # float3 primvars read by fish_surfaces.mdl (scene::data_lookup_float3)
ROOTS = ("/Lab/tissue", "/Lab/rods")


def _decode(value, width):
    from wiping.render_payload import decode
    return decode(value, "float32", None if width == 1 else width)


def _fields(data):
    if not set(data) <= set(FIELDS) or "points" not in data or "normals" not in data:
        raise ValueError("Invalid surface fields")
    out = {k: _decode(v, FIELDS[k]) for k, v in data.items()}
    n = len(out["points"])
    if n % 3 or any(len(v) != n for v in out.values()):
        raise ValueError("Surface fields must be per-corner arrays of equal length (3 per triangle)")
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError("Non-finite surface field")
    return out


def _define_page(stage, path, page_tris, names):
    from pxr import Sdf, UsdGeom, Vt
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreateSubdivisionSchemeAttr("none")
    g.CreateDoubleSidedAttr(True)
    g.CreatePointsAttr()
    g.CreateNormalsAttr()
    g.SetNormalsInterpolation("vertex")
    g.CreateExtentAttr()
    g.CreateVisibilityAttr("inherited")
    g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(page_tris, 3, np.int32)))
    g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(3*page_tris, dtype=np.int32)))
    pv = UsdGeom.PrimvarsAPI(g)
    for k in names:
        if k == "st":
            pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        elif k == "color":
            pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        elif k == "myophase":
            pv.CreatePrimvar("myophase", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        elif k in FLOAT3:
            pv.CreatePrimvar(k, Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.vertex)
    return g


def _typed(key, value):
    from pxr import Vt
    if key == "st":
        return Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(value, np.float32))
    if key == "myophase":
        return Vt.FloatArray.FromNumpy(np.ascontiguousarray(value, np.float32).ravel())
    return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(value, np.float32))


def paged_updates(stage, root, fields):
    """(attribute spec, value) updates writing `fields` (per-corner arrays) into the pages under `root`."""
    if not root.startswith(ROOTS):
        raise ValueError("Paged surfaces live under /Lab/tissue or /Lab/rods")
    rp = stage.GetPrimAtPath(root)
    if not rp:
        raise ValueError("Missing paged surface root " + root)
    page_tris = int(rp.GetCustomDataByKey("page_triangles"))
    names = [n for n in str(rp.GetCustomDataByKey("fields") or "").split(",") if n]
    if set(fields) - {"points", "normals"} != set(names):
        raise ValueError(f"{root}: fields {sorted(fields)} do not match the declared {names}")
    pv = 3*page_tris
    n = len(fields["points"])
    existing = sorted(c.GetName() for c in rp.GetChildren() if c.GetName().startswith("page_"))
    needed = max(len(existing), (n + pv - 1)//pv)
    layer = stage.GetRootLayer()
    updates = []
    for i in range(needed):
        path = f"{root}/page_{i:04d}"
        if not stage.GetPrimAtPath(path):
            _define_page(stage, path, page_tris, names)
        a, b = i*pv, min(n, (i + 1)*pv)
        cnt = max(0, b - a)
        anchor = fields["points"][a] if cnt else (fields["points"][0] if n else np.zeros(3, np.float32))
        for key, arr in fields.items():
            w = FIELDS[key]
            page = np.empty((pv,) if w == 1 else (pv, w), np.float32)
            if cnt:
                page[:cnt] = arr[a:b].reshape(page[:cnt].shape)
            if key == "points":
                page[cnt:] = anchor                       # unused slots: zero-area triangles at one point
            elif key == "normals":
                page[cnt:] = (0., 0., 1.)
            else:
                page[cnt:] = 0.
            if key == "points":
                ext = page[:cnt] if cnt else page[:1]
                updates.append((layer.GetAttributeAtPath(path + ".extent"),
                                _typed("points", np.array([ext.min(0), ext.max(0)], np.float32))))
            updates.append((layer.GetAttributeAtPath(path + "." + PRIMVAR.get(key, key)), _typed(key, page)))
        from pxr import UsdGeom
        updates.append((layer.GetAttributeAtPath(path + ".visibility"),
                        UsdGeom.Tokens.inherited if cnt else UsdGeom.Tokens.invisible))
    return updates


def reauthor_updates(stage, path, fields):
    """Updates rewriting one mesh's full topology (the measured alternative to pages)."""
    from pxr import UsdGeom, Vt
    if not path.startswith(ROOTS):
        raise ValueError("Re-authored surfaces live under /Lab/tissue or /Lab/rods")
    if not stage.GetPrimAtPath(path):
        raise ValueError("Missing re-authored mesh " + path)
    layer = stage.GetRootLayer()
    n = len(fields["points"])
    updates = [(layer.GetAttributeAtPath(path + ".faceVertexCounts"), Vt.IntArray.FromNumpy(np.full(n//3, 3, np.int32))),
               (layer.GetAttributeAtPath(path + ".faceVertexIndices"), Vt.IntArray.FromNumpy(np.arange(n, dtype=np.int32)))]
    for key, arr in fields.items():
        updates.append((layer.GetAttributeAtPath(path + "." + PRIMVAR.get(key, key)), _typed(key, arr)))
    P = fields["points"] if n else np.zeros((1, 3), np.float32)
    updates.append((layer.GetAttributeAtPath(path + ".extent"), _typed("points", np.array([P.min(0), P.max(0)], np.float32))))
    updates.append((layer.GetAttributeAtPath(path + ".visibility"),
                    UsdGeom.Tokens.inherited if n else UsdGeom.Tokens.invisible))
    return updates


def apply(stage, request, shared):
    from pxr import Sdf
    shared(stage, request)                                   # rigid poses etc. (wiping.render_updates)
    updates = []
    for root, data in request.get("paged", {}).items():
        updates += paged_updates(stage, root, _fields(data))
    for path, data in request.get("reauthor", {}).items():
        updates += reauthor_updates(stage, path, _fields(data))
    if any(not spec for spec, _ in updates):
        raise ValueError("Missing surface attribute")
    with Sdf.ChangeBlock():
        for spec, value in updates:
            spec.default = value
    for attr, file in request.get("textures", {}).items():
        if not attr.startswith(("/DressMaterials/", "/Materials/")) or not Path(file).is_file():
            raise ValueError("Invalid texture update")
        a = stage.GetAttributeAtPath(attr)
        if not a:
            raise ValueError("Missing texture attribute " + attr)
        a.Set(Sdf.AssetPath(str(Path(file).resolve())))
    for path, visible in request.get("visibility", {}).items():
        if not path.startswith(("/Lab/", "/Dressing/")):
            raise ValueError("Invalid visibility path")
        from pxr import UsdGeom
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)


def main():
    import wiping.render_updates as shared_updates
    base = shared_updates.apply_recorded_state
    shared_updates.apply_recorded_state = lambda stage, request: apply(stage, request, base)
    from wiping.render_worker import main as shared_main
    shared_main()


if __name__ == "__main__":
    main()
