"""Shared by the visual modules: which module draws which named part of a layout, the box-shape reader, and references
to the NVIDIA Simple_Warehouse kit. Appearance rules for every module: nothing here changes the simulation; a drawn
part that can be touched stays within its collider's faces (a few millimetres of bevel or surface relief at most);
render-only dressing stays off the routes."""

import math

import numpy as np

KIT = "Isaac/Environments/Simple_Warehouse/Props/"

# static boxes (facility.Layout.boxes) drawn by visuals.dockyard in the dock preset; in the yard preset it draws all
DOCKYARD_BOXES = ("dockwall", "dockheader", "dockface", "bumper", "seal", "pit", "parked_trailer", "curb_",
                  "building_wall")
DOCKYARD_SURFACES = ("apron",)


def owner(world, kind, name):
    """'dockyard' or 'interior': the module that draws a layout box, surface, line or prop."""
    preset = world.scene.c.preset
    if preset == "yard":
        return "dockyard"
    if kind == "box" and name.startswith(DOCKYARD_BOXES):
        return "dockyard"
    if kind == "surface" and name in DOCKYARD_SURFACES:
        return "dockyard"
    if kind == "prop" and name.startswith("dock_door"):
        return "dockyard"
    return "interior"


def box_half(world, s, st, scale):
    """Half-extents of a box shape: a BOX primitive, or a box built as an 8-vertex convex mesh (scene.box_shape, every box
    since the box-box contact defect). None for any other shape."""
    import newton
    t = int(st[s])
    if t == int(newton.GeoType.BOX):
        return np.asarray(scale[s, :3], float)
    if t == int(newton.GeoType.MESH):
        V = np.asarray(getattr(world.model.shape_source[s], "vertices", np.zeros((0, 3))), float)
        if V.shape == (8, 3):
            h = np.abs(V).max(0)
            if np.allclose(np.abs(V), h, atol=1e-6):
                return h * np.asarray(scale[s, :3], float)
    return None


def kit(S, path, name, p, yaw=0.0, scale=None):
    """A Simple_Warehouse kit prop referenced at p (render only)."""
    from ..assets.install import resolve
    from scipy.spatial.transform import Rotation
    prim = S.stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(str(resolve(KIT + name)))
    S.place(prim, p, Rotation.from_euler("z", yaw).as_quat())
    if scale is not None:
        S.UsdGeom.Xformable(prim).AddScaleOp().Set(S.Gf.Vec3f(*map(float, scale)))
    return prim


def layout_box(S, world, bx):
    """A layout box as a plain box in its material (the fallback look of a part no module has modelled yet)."""
    from scipy.spatial.transform import Rotation
    mat = bx["material"] if bx["material"] in S.mats else "machine_grey"
    if bx.get("pitch"):
        qq = Rotation.from_euler("x", bx["pitch"]).as_quat()
    else:
        qq = Rotation.from_euler("z", bx["yaw"]).as_quat()
    return S.box(f"/World/box_{bx['name']}", [2 * h for h in bx["half"]], S.mats[mat], p=bx["centre"], q=qq, uv_per_m=1.0)


def layout_surface(S, world, s):
    """A drivable surface as a slab in its material (fallback look)."""
    from scipy.spatial.transform import Rotation
    cx, cy = s["centre"]; hx, hy = s["half"]
    z = s.get("z_top", 0.0)
    g = s.get("grade", 0.0)
    mat = "asphalt" if s.get("material", "").startswith("asphalt") else \
        "ramp_yellow" if s.get("material") in ("grating",) else \
        "steel" if s.get("material") == "steel" else "floor_concrete"
    if g:
        ax = np.array([*s.get("axis", (1.0, 0.0)), 0.0]); ax /= np.linalg.norm(ax)
        ang = math.atan(g)
        yaw = math.atan2(ax[1], ax[0])
        Rm = Rotation.from_euler("z", yaw) * Rotation.from_euler("y", -ang)
        return S.box(f"/World/surf_{s['name']}", (2 * hx / math.cos(ang), 2 * hy, 0.03), S.mats[mat], p=(cx, cy, z - 0.015),
                     q=Rm.as_quat(), uv_per_m=1.0)
    return S.box(f"/World/surf_{s['name']}", (2 * hx, 2 * hy, s.get("thickness", 0.2)), S.mats[mat],
                 p=(cx, cy, z - s.get("thickness", 0.2) / 2), uv_per_m=0.5)


def layout_line(S, world, ln):
    """A painted floor marking as a thin quad on the surface under it."""
    sc = world.scene
    a, b = np.array(ln["a"]), np.array(ln["b"])
    d = b - a
    Lg = float(np.linalg.norm(d))
    if Lg < 1e-6:
        return None
    u = np.array([d[0] / Lg, d[1] / Lg, 0.0]); v = np.array([-u[1], u[0], 0.0])
    c = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.003 + sc.ground_height((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)])
    return S.quad(f"/World/line_{ln['name']}", c, u, v, (Lg, ln["width"]),
                  S.mats["paint_yellow" if ln["colour"] == "yellow" else "paint_white"])
