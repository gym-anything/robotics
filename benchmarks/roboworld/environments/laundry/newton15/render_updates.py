"""Batch a recorded state's USD edits, with no USD queries inside ChangeBlock."""
import numpy as np


def apply_recorded_state(stage, request):
    from pxr import Gf, Sdf, Vt
    from titrate.newton15.render_payload import decode
    from .details import attribute_values
    values = attribute_values(request.get("details", {}))
    for path, data in request.get("pile", {}).items():
        points = decode(data, "float32", 3)
        values[path+".points"] = Vt.Vec3fArray.FromNumpy(points)
        values[path+".extent"] = Vt.Vec3fArray.FromNumpy(np.array([points.min(0),points.max(0)],np.float32))
    for i,q in enumerate(request["body_poses"]):
        if not np.isfinite(q).all():
            raise ValueError("Non-finite recorded body pose")
        path = f"/Lab/body_{i}"
        values[path+".xformOp:translate"] = Gf.Vec3d(*q[:3])
        values[path+".xformOp:orient"] = Gf.Quatf(q[6], Gf.Vec3f(*q[3:6]))
    for path,data in request["meshes"].items():
        points = decode(data["points"], "float32", 3)
        normals = decode(data["normals"], "float32", 3)
        if not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise ValueError("Non-finite recorded cloth mesh")
        values[path+".points"] = Vt.Vec3fArray.FromNumpy(points)
        values[path+".normals"] = Vt.Vec3fArray.FromNumpy(normals)
        values[path+".extent"] = Vt.Vec3fArray.FromNumpy(np.array([points.min(0),points.max(0)],np.float32))
    # All attributes are authored in the renderer's root layer at export.
    # Resolve handles before the block. Only Sdf APIs are used inside it,
    # following OpenUSD's explicit safety rule for deferred change notices.
    layer = stage.GetRootLayer()
    changes = [(layer.GetAttributeAtPath(path), value) for path,value in values.items()]
    if any(not spec for spec,_ in changes):
        raise ValueError("Recorded update references a missing authored attribute")
    with Sdf.ChangeBlock():
        for spec,value in changes:
            spec.default = value
